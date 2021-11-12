import os
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from interpensembles.data import CIFAR10Data,CIFAR10_1Data
from interpensembles.module import CIFAR10Module,CIFAR10EnsembleModule,CIFAR10InterEnsembleModule

modules = {"base":CIFAR10Module,
        "ensemble":CIFAR10EnsembleModule,
        "interpensemble":CIFAR10InterEnsembleModule}

script_dir = os.path.abspath(os.path.dirname(__file__))

def main(args):

    if bool(args.download_weights):
        CIFAR10Data.download_weights()
    else:
        seed_everything(0)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

        if args.logger == "wandb":
            logger = WandbLogger(name=args.classifier, project="cifar10")
        elif args.logger == "tensorboard":
            logger = TensorBoardLogger("cifar10", name=args.classifier)

        checkpoint = ModelCheckpoint(monitor="acc/val", mode="max", save_last=False)

        trainer = Trainer(
            default_root_dir = os.path.join(script_dir,"../","models",args.classifier,args.module),    
            fast_dev_run=bool(args.dev),
            logger=logger if not bool(args.dev + args.test_phase) else None,
            gpus=-1,
            deterministic=True,
            weights_summary=None,
            log_every_n_steps=1,
            max_epochs=args.max_epochs,
            checkpoint_callback=checkpoint,
            precision=args.precision,
        )

        if args.module == "base":
            model = modules[args.module](args)
        elif args.module == "ensemble":
            model = modules[args.module](args.nb_models,args)
        elif args.module == "interpensemble":
            model = modules[args.module](args.lamb,args)
        if args.test_phase:
            if args.test_set == "CIFAR10":
                data = CIFAR10Data(args)
            elif args.test_set == "CIFAR10_1":
                data = CIFAR10_1Data(args)
        else:        
            data = CIFAR10Data(args)


        if bool(args.pretrained):
            state_dict = os.path.join(
                "cifar10_models", "state_dicts", args.classifier + ".pt"
            )
            model.model.load_state_dict(torch.load(state_dict))

        if bool(args.test_phase):
            trainer.test(model, data.test_dataloader())
        else:
            trainer.fit(model, data)
            trainer.test()


if __name__ == "__main__":
    parser = ArgumentParser()

    # PROGRAM level args
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
    parser.add_argument("--module", type = str,default = "base",choices = ["base","ensemble","interpensemble"])
    parser.add_argument("--version",type = str,default = "v4",choices = ["v4","v6"])

    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpu_id", type=str, default="0")

    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--test_set",type = str,default = "CIFAR10",choices = ["CIFAR10","CIFAR10_1"])
    parser.add_argument("--nb_models",type = int,default = 4)
    parser.add_argument("--lamb",type = float,default = 0.5)
    parser.add_argument("--scheduler",type = str,default = "cosine",choices = ["cosine","step"])

    args = parser.parse_args()
    main(args)
