## Try to work with the modules from "divide and conquer" using the same interface we have here.  
import pytest
import numpy as np
import torch
from interpensembles.module import CIFAR10Module
from argparse import ArgumentParser

shakeshake_args = ["--data_dir", "/home/ubuntu/data/cifar10",
        "--test_phase","0",
        "--dev","0",
        "--logger","tensorboard",
        "--classifier","shake_resnet26_2x96d",
        "--pretrained","0",
        "--precision","32",
        "--batch_size","256",
        "--max_epochs","100",
        "--num_workers","4",
        "--gpu_id","0",
        "--learning_rate","1e-2",
        "--weight_decay","1e-2",
        "--test_set","CIFAR10"]

pyramidnet_args = ["--data_dir", "/home/ubuntu/data/cifar10",
        "--test_phase","0",
        "--dev","0",
        "--logger","tensorboard",
        "--classifier","pyramidnet_272",
        "--pretrained","0",
        "--precision","32",
        "--batch_size","256",
        "--max_epochs","100",
        "--num_workers","4",
        "--gpu_id","0",
        "--learning_rate","1e-2",
        "--weight_decay","1e-2",
        "--test_set","CIFAR10"]

efficientnet_args = ["--data_dir", "/home/ubuntu/data/cifar10",
        "--test_phase","0",
        "--dev","0",
        "--logger","tensorboard",
        "--classifier","efficientnet_b0",
        "--pretrained","0",
        "--precision","32",
        "--batch_size","256",
        "--max_epochs","100",
        "--num_workers","4",
        "--gpu_id","0",
        "--learning_rate","1e-2",
        "--weight_decay","1e-2",
        "--test_set","CIFAR10"]

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

def test_default(create_parser):
    """See if we can run shake-shake model code from divide-and-conquer in this module

    """
    parser = create_parser
    args = parser.parse_args(default_args)
    
    model = CIFAR10Module(vars(args))
    dummy_input = torch.tensor(np.random.randn(5,3,32,32)).float()
    dummy_output  = torch.tensor(np.array([1,2,1,1,1])).long()
    dummy_batch = (dummy_input,dummy_output)
    model.eval()
    with torch.no_grad():
        y = model(dummy_batch)

def test_shake_shake(create_parser):
    """See if we can run shake-shake model code from divide-and-conquer in this module

    """
    parser = create_parser
    args = parser.parse_args(shakeshake_args)
    
    model = CIFAR10Module(vars(args))
    dummy_input = torch.tensor(np.random.randn(5,3,32,32)).float()
    dummy_output  = torch.tensor(np.array([1,2,1,1,1])).long()
    dummy_batch = (dummy_input,dummy_output)
    model.eval()
    with torch.no_grad():
        y = model(dummy_batch)

def test_pyramidnet(create_parser):
    """See if we can run pyramidnet model code from divide-and-conquer in this module

    """
    parser = create_parser
    args = parser.parse_args(pyramidnet_args)
    
    model = CIFAR10Module(vars(args))
    dummy_input = torch.tensor(np.random.randn(5,3,32,32)).float()
    dummy_output  = torch.tensor(np.array([1,2,1,1,1])).long()
    dummy_batch = (dummy_input,dummy_output)
    model.eval()
    with torch.no_grad():
        y = model(dummy_batch)

def test_efficientnet(create_parser):
    """See if we can run pyramidnet model code from divide-and-conquer in this module

    """
    parser = create_parser
    args = parser.parse_args(efficientnet_args)
    
    model = CIFAR10Module(vars(args))
    dummy_input = torch.tensor(np.random.randn(5,3,32,32)).float()
    dummy_output  = torch.tensor(np.array([1,2,1,1,1])).long()
    dummy_batch = (dummy_input,dummy_output)
    model.eval()
    with torch.no_grad():
        y = model(dummy_batch)
#def test_ensemble_resnet_calibration(create_parser):
#    parser = create_parser
#    args = parser.parse_args(default_args)
#    
#    model = CIFAR10EnsembleModule(2,args)
#    dummy_input = torch.tensor(np.random.randn(5,3,32,32)).float()
#    dummy_output  = torch.tensor(np.array([1,2,1,1,1])).long()
#    dummy_batch = (dummy_input,dummy_output)
#    model.eval()
#    with torch.no_grad():
#        y = model(dummy_batch)
#        pred,label = model.calibration(dummy_batch)
#    loss = model.criterion(pred,label)    
#    accuracy = model.accuracy(pred,label)*100
#    assert y == (loss,accuracy)
#
#def test_interpensemble_resnet_calibration(create_parser):
#    parser = create_parser
#    args = parser.parse_args(default_args)
#    
#    model = CIFAR10InterEnsembleModule(0.5,args)
#    dummy_input = torch.tensor(np.random.randn(5,3,32,32)).float()
#    dummy_output  = torch.tensor(np.array([1,2,1,1,1])).long()
#    dummy_batch = (dummy_input,dummy_output)
#    model.eval()
#    with torch.no_grad():
#        y = model(dummy_batch)
#        pred,label = model.calibration(dummy_batch)
#    loss = model.criterion(pred,label)    
#    accuracy = model.accuracy(pred,label)*100
#    assert y == (loss,accuracy)
#
#def test_lineargroup_resnet_calibration(create_parser):
#    parser = create_parser
#    args = parser.parse_args(grouplinear_args)
#    
#    model = CIFAR10LinearGroupModule(args)
#    dummy_input = torch.tensor(np.random.randn(5,3,32,32)).float()
#    dummy_output  = torch.tensor(np.array([1,2,1,1,1])).long()
#    dummy_batch = (dummy_input,dummy_output)
#    model.eval()
#    with torch.no_grad():
#        y = model(dummy_batch)
#        pred,label = model.calibration(dummy_batch)
#    loss = model.criterion(pred,label)    
#    accuracy = model.accuracy(pred,label)*100
#    assert y == (loss,accuracy)
