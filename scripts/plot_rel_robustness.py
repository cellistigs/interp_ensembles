import os
import datetime
import json
from argparse import ArgumentParser

import torch
import numpy as np
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
    if args.module == "base":
        model = modules[args.module].load_from_checkpoint(checkpoint_path=args.checkpoint)
    elif args.module == "ensemble":
        model = modules[args.module].load_from_checkpoint(nb_models = args.nb_models,checkpoint_path=args.checkpoint)
    elif args.module == "interpensemble":
        model = modules[args.module].load_from_checkpoint(lamb = args.lamb,checkpoint_path=args.checkpoint)
    cifar10data = CIFAR10Data(args)
    cifar10_1data = CIFAR10_1Data(args,version =args.version)


    if bool(args.pretrained):
        state_dict = os.path.join(
            "cifar10_models", "state_dicts", args.classifier + ".pt"
        )
        model.model.load_state_dict(torch.load(state_dict))

    data = {"in_dist_acc":None,"out_dist_acc":None}
    data["in_dist_acc"] = trainer.test(model, cifar10data.test_dataloader())[0]["acc/test"]
    data["out_dist_acc"] = trainer.test(model, cifar10_1data.test_dataloader())[0]["acc/test"]

    all_preds_ind = []
    all_labels_ind = []
    all_preds_ood = []
    all_labels_ood = []

    model.eval()
    with torch.no_grad():
        for idx,batch in enumerate(cifar10data.test_dataloader()):
            ims = batch[0].to("cuda")
            labels = batch[1].to("cuda")
            pred,label = model.calibration((ims,labels))
            ## to cpu
            predarray = pred.cpu().numpy() ## 256x10
            labelarray = label.cpu().numpy() ## 
            all_preds_ind.append(predarray)
            all_labels_ind.append(labelarray)
        for idx,batch in enumerate(cifar10_1data.test_dataloader()):
            ims = batch[0].to("cuda")
            labels = batch[1].to("cuda")
            pred,label = model.calibration((ims,labels))
            ## to cpu
            predarray = pred.cpu().numpy() ## 256x10
            labelarray = label.cpu().numpy() ## 
            all_preds_ood.append(predarray)
            all_labels_ood.append(labelarray)


    results_dir = os.path.join(script_dir,"../results")
    full_path = os.path.join(results_dir,"robust_results{}_{}_{}".format(datetime.datetime.now().strftime("%m-%d-%y_%H:%M.%S"),args.module,args.classifier))
    np.save(full_path+"ind_preds",np.concatenate(all_preds_ind,axis = 0))
    np.save(full_path+"ind_labels",np.concatenate(all_labels_ind,axis = 0))
    np.save(full_path+"ood_preds",np.concatenate(all_preds_ood,axis = 0))
    np.save(full_path+"ood_labels",np.concatenate(all_labels_ood,axis = 0))
    ## write metadata
    with open(full_path+"_meta.json","w") as f:
        json.dump(vars(args),f)


    #with open(os.path.join(results_dir,"robust_results{}_{}_{}".format(datetime.datetime.now().strftime("%m-%d-%y_%H:%M.%S"),args.module,args.classifier)),"w") as f:
    #    json.dump(data,f)


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
    parser.add_argument("--module", type = str,default = "base",choices = ["base","ensemble","interpensemble"])
    parser.add_argument("--version",type = str,default = "v4",choices = ["v4","v6"])
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
    parser.add_argument("--nb_models",type = int,default = 4)
    parser.add_argument("--lamb",type = float,default = 0.5)

    args = parser.parse_args()
    main(args)
