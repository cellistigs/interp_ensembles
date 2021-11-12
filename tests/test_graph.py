## Examine graph structure. 
import pytest
from torchviz import make_dot, make_dot_from_trace
import torch
from interpensembles.module import CIFAR10Module,CIFAR10EnsembleModule
from argparse import ArgumentParser

default_args = ["--data_dir", "/home/ubuntu/data/cifar10",
        "--test_phase","0",
        "--dev","0",
        "--logger","tensorboard",
        "--classifier","resnet18",
        "--pretrained","0",
        "--precision","32",
        "--batch_size","256",
        "--max_epochs","100",
        "--num_workers","4",
        "--gpu_id","0",
        "--learning_rate","1e-2",
        "--weight_decay","1e-2",
        "--test_set","CIFAR10"]

@pytest.fixture()
def create_parser():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/ubuntu/data/cifar10")
    parser.add_argument("--download_weights", type=int, default=0, choices=[0, 1])
    parser.add_argument("--test_phase", type=int, default=0, choices=[0, 1])
    parser.add_argument("--dev", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"]
    )

    # TRAINER args
    parser.add_argument("--classifier", type=str, default="resnet18")
    parser.add_argument("--pretrained", type=int, default=0, choices=[0, 1])

    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpu_id", type=str, default="0")

    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--test_set",type = str,default = "CIFAR10",choices = ["CIFAR10","CIFAR10_1"])
    yield parser

def test_resnet18(create_parser):
    parser = create_parser
    args = parser.parse_args(default_args)

    model = CIFAR10Module(args)
    dummy_data = (torch.randn(1,3,32,32),torch.tensor([0]))
    y = model(dummy_data)
    dot = make_dot(y[0].mean(),params = dict(model.named_parameters()))
    dot.render("test_mats/test_resnet18",view = True)

def test_ensembleresnet18(create_parser):
    parser = create_parser
    args = parser.parse_args(default_args)

    model = CIFAR10EnsembleModule(3,args)
    dummy_data = (torch.randn(1,3,32,32),torch.tensor([0]))
    y = model(dummy_data)
    dot = make_dot(y[0].mean(),params = dict(model.named_parameters()))
    dot.render("test_mats/test_ensemble_resnet18_inference",view = True)
    traindot = make_dot(model.training_step(dummy_data,0).mean(),params = dict(model.named_parameters()))
    traindot.render("test_mats/test_ensemble_resnet18_training",view = True)

def test_sum():
    """The training graph from the previous test for ensembles looked weird due to asymmetry in add. Here we just checked the output of adding three scalars to ensure that there is the same assymetry (there is).

    """
    a = torch.tensor(1.0,requires_grad= True)
    b = torch.tensor(0.11,requires_grad = True)
    c = torch.tensor(0.41,requires_grad = True)
    loss = sum([a,b,c])
    dot = make_dot(loss,params = {"a":a,"b":b})
    dot.render("test_mats/test_add",view=True)




