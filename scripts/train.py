import os
from tqdm import tqdm
from argparse import ArgumentParser
import datetime
import torch
import json
import numpy as np
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from interpensembles.data import CIFAR10Data,CIFAR10_1Data
from interpensembles.module import CIFAR10Module,CIFAR10EnsembleModule,CIFAR10InterEnsembleModule
from cifar10_ood.data import CINIC10_Data,CIFAR10_CData

modules = {"base":CIFAR10Module,
        "ensemble":CIFAR10EnsembleModule,
        "interpensemble":CIFAR10InterEnsembleModule}

script_dir = os.path.abspath(os.path.dirname(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def custom_eval(model,ind_data,ood_data,device,softmax = True):   
    """Custom evaluation function to output logits as arrays from models given the trained model, in distribution data and out of distribution data. 

    :param model: a model from interpensembles.modules. Should have a method "calibration" that outputs predictions (logits) and labels given images and labels. 
    :param ind_data: an instance of a data class (like CIFAR10Data,CIFAR10_1Data) that has a corresponding test_dataloader. 
    :param ood_data: an instance of a data class (like CIFAR10Data,CIFAR10_1Data) that has a corresponding test_dataloader.
    :param device: device to run computations on.
    :param softmax: whether or not to apply softmax to predictions. 
    :returns: four arrays corresponding to predictions (array of shape (batch,classes)), and labels (shape (batch,)) for ind and ood data respectively. 

    """
    ## This is the only place where we need to worry about devices. The model should already know what device to use. 
    all_preds_ind = []
    all_labels_ind = []
    all_preds_ood = []
    all_labels_ood = []

    ## model, cifart10data,cifart10_1data,
    model.eval()
    with torch.no_grad():
        for idx,batch in tqdm(enumerate(ind_data.test_dataloader())):
            ims = batch[0].to(device)
            labels = batch[1].to(device)
            pred,label = model.calibration((ims,labels),use_softmax= softmax)
            ## to cpu
            predarray = pred.cpu().numpy() ## 256x10
            labelarray = label.cpu().numpy() ## 
            all_preds_ind.append(predarray)
            all_labels_ind.append(labelarray)
        for idx,batch in tqdm(enumerate(ood_data.test_dataloader())):
            ims = batch[0].to(device)
            labels = batch[1].to(device)
            pred,label = model.calibration((ims,labels),use_softmax = softmax)
            ## to cpu
            predarray = pred.cpu().numpy() ## 256x10
            labelarray = label.cpu().numpy() ## 
            all_preds_ood.append(predarray)
            all_labels_ood.append(labelarray)

    all_preds_ind_array = np.concatenate(all_preds_ind,axis = 0)
    all_labels_ind_array = np.concatenate(all_labels_ind,axis = 0)
    all_preds_ood_array = np.concatenate(all_preds_ood,axis = 0)
    all_labels_ood_array = np.concatenate(all_labels_ood,axis = 0)
    return all_preds_ind_array,all_labels_ind_array,all_preds_ood_array,all_labels_ood_array

def main(args):

    if bool(args.deterministic):
        seed_everything(0)
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.logger == "wandb":
        logger = WandbLogger(name=args.classifier, project="cifar10")
    elif args.logger == "tensorboard":
        logger = TensorBoardLogger("cifar10", name=args.classifier)

    checkpoint = ModelCheckpoint(monitor="acc/val", mode="max", save_last=False, dirpath = os.path.join(script_dir,"../","models",args.classifier,args.module,datetime.datetime.now().strftime("%m-%d-%y"),datetime.datetime.now().strftime("%H_%M_%S")))

    trainerargs = {
        #"default_root_dir":os.path.join(script_dir,"../","models",args.classifier,args.module),    
        "fast_dev_run":bool(args.dev),
        "logger":logger if not bool(args.dev + args.test_phase) else None,
        "deterministic":bool(args.deterministic),
        "weights_summary":None,
        "log_every_n_steps":1,
        "max_epochs":args.max_epochs,
        "checkpoint_callback":checkpoint,
        "precision":args.precision,
        }
    if torch.cuda.is_available():
        print("training on GPU")
        trainerargs["gpus"] = -1  

    trainer = Trainer(**trainerargs)

    ## define arguments for each model class: 
    all_args = {"hparams":args} 
    if args.module == "base":
        pass
    elif args.module == "ensemble":
        all_args["nb_models"] = args.nb_models
    elif args.module == "interpensemble":
        all_args["lamb"] = "lamb"

    

    if bool(args.test_phase) and not bool(args.pretrained): ## if loading from checkpoints: 
        if args.module == "base":
            model = modules[args.module].load_from_checkpoint(checkpoint_path=args.checkpoint,hparams = args)
        elif args.module == "ensemble":    
            model = modules[args.module].load_from_checkpoint(nb_models = all_args["nb_models"],checkpoint_path=args.checkpoint,hparams = args)
        elif args.module == "interpensemble":    
            model = modules[args.module].load_from_checkpoint(lamb = all_args["lamb"],checkpoint_path=args.checkpoint,hparans = args)
    else: ## if training from scratch or loading from state dict:    
        model = modules[args.module](**all_args)
            
    cifar10data = CIFAR10Data(args)
    if args.ood_dataset == "cifar10_1":
        ood_data = CIFAR10_1Data(args,version =args.version)
    elif args.ood_dataset == "cinic10":    
        ood_data = CINIC10_Data(args)
    elif args.ood_dataset == "cifar10_c":    
        assert args.corruption, "for cifar10_c, corruption must be given."
        assert args.level, "for cifar10_c, level must be given"
        ood_data = CIFAR10_CData(args)


    if bool(args.pretrained):
        if args.pretrained_path is None:
            state_dict = os.path.join(
                script_dir,"../","models",
                "cifar10_models", "state_dicts", args.classifier + ".pt"
            )
        else:     
            state_dict = args.pretrained_path
        model.model.load_state_dict(torch.load(state_dict))

    if bool(args.test_phase):
        pass
    else:
        trainer.fit(model, cifar10data)

    data = {"in_dist_acc":None,"out_dist_acc":None}
    data["in_dist_acc"] = trainer.test(model, cifar10data.test_dataloader())[0]["acc/test"]
    data["out_dist_acc"] = trainer.test(model, ood_data.test_dataloader())[0]["acc/test"]

    preds_ind, labels_ind, preds_ood, labels_ood = custom_eval(model,cifar10data,ood_data,device,softmax = bool(args.softmax))

    results_dir = os.path.join(script_dir,"../results")
    full_path = os.path.join(results_dir,"robust_results{}_{}_{}".format(datetime.datetime.now().strftime("%m-%d-%y_%H:%M.%S"),args.module,args.classifier))
    np.save(full_path+"ind_preds",preds_ind)
    np.save(full_path+"ind_labels",labels_ind)
    if args.ood_dataset == "cifar10_1":
        np.save(full_path+"ood_preds",preds_ood)
        np.save(full_path+"ood_labels",labels_ood)
    elif args.ood_dataset == "cinic10":    
        np.save(full_path+"ood_cinic_preds",preds_ood)
        np.save(full_path+"ood_cinic_labels",labels_ood)
    elif args.ood_dataset == "cifar10_c":    
        np.save(full_path+"ood_cifar10_c_{}_{}_preds".format(args.corruption,args.level),preds_ood)
        np.save(full_path+"ood_cifar10_c_{}_{}_labels".format(args.corruption,args.level),labels_ood)
    else:     
        raise Exception("option for ood dataset not recognized.")
    ## write metadata
    metadata = vars(args)
    metadata["save_path"] = trainer.checkpoint_callback.dirpath
    with open(full_path+"_meta.json","w") as f:
        json.dump(metadata,f)


if __name__ == "__main__":
    parser = ArgumentParser()

    # PROGRAM level args
    parser.add_argument("--data_dir", type=str, default="/home/ubuntu/data/cifar10")
    parser.add_argument("--ood_dataset",type = str,default = "cifar10_1",choices = ["cifar10_1","cinic10","cifar10_c"])
    parser.add_argument("--version",type = str,default = "v4",choices = ["v4","v6"]) ## for cifar10.1
    parser.add_argument("--level",type = int,default = None) ## for cifar10_c
    parser.add_argument("--corruption",type = str,default = None) ## for cifar10_c
    parser.add_argument("--deterministic",type = int, default = 0, choices = [0,1])
    parser.add_argument("--test_phase", type=int, default=0, choices=[0, 1],help = "train or evaluation mode. If evaluation, checkpoint must be provided")
    parser.add_argument("--dev", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"]
    )
    parser.add_argument("--checkpoint",type= str,help = "Path to model checkpoint if evaluating")
    parser.add_argument("--softmax",type = int,default = 1,choices = [0,1])
    parser.add_argument("--nb_models",type = int,default = 4)
    parser.add_argument("--module", type = str,default = "base",choices = ["base","ensemble","interpensemble"])

    # TRAINER args
    parser.add_argument("--classifier", type=str, default="resnet18")
    parser.add_argument("--pretrained", type=int, default=0, choices=[0, 1])
    parser.add_argument("--pretrained-path",type = str, default = None)

    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpu_id", type=str, default="0")

    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--test_set",type = str,default = "CIFAR10",choices = ["CIFAR10","CIFAR10_1"])
    parser.add_argument("--lamb",type = float,default = 0.5)
    parser.add_argument("--scheduler",type = str,default = "cosine",choices = ["cosine","step"])

    args = parser.parse_args()
    main(args)
