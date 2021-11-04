import os
import datetime
import json
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from interpensembles.data import CIFAR10Data,CIFAR10_1Data
from interpensembles.module import CIFAR10Module, CIFAR10EnsembleModule


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

        #model = CIFAR10Module(args)
        model = CIFAR10EnsembleModule.load_from_checkpoint(nb_models=4,checkpoint_path=args.checkpoint)
        cifar10data = CIFAR10Data(args)
        cifar10_1data = CIFAR10_1Data(args,version = "v4")


        if bool(args.pretrained):
            state_dict = os.path.join(
                "cifar10_models", "state_dicts", args.classifier + ".pt"
            )
            model.model.load_state_dict(torch.load(state_dict))

        data = {"in_dist_acc":None,"out_dist_acc":None}
        data["in_dist_acc"] = trainer.test(model, cifar10data.test_dataloader())[0]["acc/test"]
        data["out_dist_acc"] = trainer.test(model, cifar10_1data.test_dataloader())[0]["acc/test"]
        with open("robust_results{}".format(datetime.datetime.now().strftime("%m-%d-%y_%H:%M.%S")),"w") as f:
            json.dump(data,f)


if __name__ == "__main__":
    parser = ArgumentParser()

    # PROGRAM level args
    parser.add_argument("--data_dir", type=str, default="/home/ubuntu/cifar10")
    parser.add_argument("--download_weights", type=int, default=0, choices=[0, 1])
    parser.add_argument("--test_phase", type=int, default=0, choices=[0, 1])
    parser.add_argument("--dev", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"]
    )

    # TRAINER args
    parser.add_argument("--checkpoint", type = str)
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

    args = parser.parse_args()
    main(args)
