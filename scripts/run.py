## Copied from interp_ensembles.
import os
import hydra
from tqdm import tqdm
from argparse import ArgumentParser
import datetime
import torch
import json
import numpy as np
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from interpensembles.module import CIFAR10Module,CIFAR10EnsembleModule,CIFAR10RFFModule,CIFAR10RFFSGDModule,MNISTRFFModule,MNISTRFFSGDModule

from cifar10_ood.data import CIFAR10Data,CIFAR10_1Data,CINIC10_Data,CIFAR10_CData,MNISTData


cifar_modules = {
        "base":CIFAR10Module,
        "ensemble":CIFAR10EnsembleModule,
        "rff":CIFAR10RFFModule,
        "rffsgd":CIFAR10RFFSGDModule,
        }

mnist_modules = {
        "rff":MNISTRFFModule,
        "rffsgd":MNISTRFFSGDModule,
        }

script_dir = os.path.abspath(os.path.dirname(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def traindata_eval(model,ind_data,device,softmax = False):   
    """Custom evaluation function to output logits as arrays from models given the trained model on the training data. Used to generate training examples from random labels. 

    :param model: a model from interpensembles.modules. Should have a method "calibration" that outputs predictions (logits) and labels given images and labels. 
    :param ind_data: an instance of a data class (like CIFAR10Data,CIFAR10_1Data) that has a corresponding test_dataloader. 
    :param device: device to run computations on.
    :param softmax: whether or not to apply softmax to predictions. 
    :returns: four arrays corresponding to predictions (array of shape (batch,classes)), and labels (shape (batch,)) for ind and ood data respectively. 

    """
    ## This is the only place where we need to worry about devices. The model should already know what device to use. 
    all_preds= []
    all_labels = []

    ## model, cifart10data,cifart10_1data,
    model.eval()
    with torch.no_grad():
        for idx,batch in tqdm(enumerate(ind_data.train_dataloader(shuffle=False,aug=False))):
            ims = batch[0].to(device)
            ims_flat = ims.reshape(ims.shape[0],-1)
            labels = batch[1].to(device)
            ## TODO change this to not pred, but linear. 
            pred,label = model.calibration((ims_flat,labels),use_softmax=softmax)
            ## to cpu
            predarray = pred.cpu().numpy() ## 256x10
            labelarray = label.cpu().numpy() ## 
            all_preds.append(predarray)
            all_labels.append(labelarray)

    all_preds_array = np.concatenate(all_preds,axis = 0)
    all_labels_array = np.concatenate(all_labels,axis = 0)
    return all_preds_array,all_labels_array

def eval_dataset(model,dataset,device,softmax=True):
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for idx,batch in tqdm(enumerate(dataset.test_dataloader())):
            ims = batch[0].to(device)
            ims_flat = ims.reshape(ims.shape[0],-1)
            print(ims.shape)
            labels = batch[1].to(device)
            pred,label = model.calibration((ims_flat,labels),use_softmax=softmax)
            ## to cpu
            predarray = pred.cpu().numpy() ## 256x10
            labelarray = label.cpu().numpy() ## 
            all_preds.append(predarray)
            all_labels.append(labelarray)
    all_preds_array = np.concatenate(all_preds,axis = 0)
    all_labels_array = np.concatenate(all_labels,axis = 0)
    return all_preds_array,all_labels_array        

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

    ## model, cifart10data,cifart10_1data,
    all_preds_ind,all_labels_ind = eval_dataset(model,ind_data,device,softmax=softmax)
    all_preds_ood,all_labels_ood = eval_dataset(model,ood_data,device,softmax=softmax)
    #model.eval()
    #with torch.no_grad():
    #    for idx,batch in tqdm(enumerate(ind_data.test_dataloader())):
    #        ims = batch[0].to(device)
    #        labels = batch[1].to(device)
    #        pred,label = model.calibration((ims,labels),use_softmax=softmax)
    #        ## to cpu
    #        predarray = pred.cpu().numpy() ## 256x10
    #        labelarray = label.cpu().numpy() ## 
    #        all_preds_ind.append(predarray)
    #        all_labels_ind.append(labelarray)
    #    for idx,batch in tqdm(enumerate(ood_data.test_dataloader())):
    #        ims = batch[0].to(device)
    #        labels = batch[1].to(device)
    #        pred,label = model.calibration((ims,labels),use_softmax=softmax)
    #        ## to cpu
    #        predarray = pred.cpu().numpy() ## 256x10
    #        labelarray = label.cpu().numpy() ## 
    #        all_preds_ood.append(predarray)
    #        all_labels_ood.append(labelarray)

    return all_preds_ind,all_labels_ind,all_preds_ood,all_labels_ood

@hydra.main(config_path = os.path.join(script_dir,"../configs/"),config_name = "run_default_gpu")
def main(args):
    if args.ind_dataset == "cifar10":
        modules = cifar_modules
    elif args.ind_dataset == "mnist":    
        modules = mnist_modules
    else:    
        raise AssertionError("ind_dataset not recognized.")

    ## Set seeds if given.  
    if args.seed is not None:
        seed_everything(args.seed)
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    ## Set up logging. 
    if args.logger == "wandb":
        logger = WandbLogger(name=args.classifier, project=args.ind_dataset)
    elif args.logger == "tensorboard":
        logger = TensorBoardLogger(args.ind_dataset, name=args.classifier)

    ## Configure checkpoint and trainer: 
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
    if args.module.startswith("rff"):
        print("not saving model for rff")
        trainerargs["checkpoint_callback"] = False

    if torch.cuda.is_available():
        print("training on GPU")
        trainerargs["gpus"] = -1  

    trainer = Trainer(**trainerargs)

    ## define arguments for each model class: 
    all_args = {"hparams":args} 
    if args.module == "base":
        pass
    elif args.module == "ensemble":
        pass
    elif args.module == "attention":
        pass

    ## we can load in pretrained models stored as weights. 
    if bool(args.test_phase) and not bool(args.pretrained): ## if loading from checkpoints: 
        if args.module == "base":
            model = modules[args.module].load_from_checkpoint(checkpoint_path=args.checkpoint,hparams = args)
        elif args.module == "ensemble":    
            model = modules[args.module].load_from_checkpoint(checkpoint_path=args.checkpoint,hparams = args)
        elif args.module == "attention":    
            model = modules[args.module].load_from_checkpoint(checkpoint_path=args.checkpoint,hparams = args)
        ## Really should be the case for anything
        else:    
            model = modules[args.module].load_from_checkpoint(checkpoint_path=args.checkpoint,hparams = args)
    else: ## if training from scratch or loading from state dict:    
        model = modules[args.module](**all_args)
        ## if loading from state dictionary instead of checkpoint: 
        if bool(args.pretrained):
            if args.pretrained_path is None:
                state_dict = os.path.join(
                    script_dir,"../","models",
                    "cifar10_models", "state_dicts", args.classifier + ".pt"
                )
            else:     
                state_dict = args.pretrained_path
            model.model.load_state_dict(torch.load(state_dict))
            
    ## what dataset should we evaluate on? 
    if args.ind_dataset == "cifar10":
        ind_data = CIFAR10Data(args)
    elif args.ind_dataset == "mnist":    
        ind_data = MNISTData(args)

    if args.ood_dataset == "cifar10_1":
        ood_data = CIFAR10_1Data(args,version =args.version)
    elif args.ood_dataset == "cinic10":    
        ood_data = CINIC10_Data(args)
    elif args.ood_dataset == "cifar10_c":    
        assert args.corruption, "for cifar10_c, corruption must be given."
        assert args.level, "for cifar10_c, level must be given"
        ood_data = CIFAR10_CData(args)

    ## do we train the model or not? 
    if bool(args.test_phase) or bool(args.random_eval):
        pass
    else:
        trainer.fit(model, ind_data)

    ## testing and evaluation : 
    data = {"in_dist_acc":None,"out_dist_acc":None}
    data["in_dist_acc"] = trainer.test(model, ind_data.test_dataloader())[0]["acc/test"]

    if args.ind_dataset == "mnist":
        print("no ood for mnist")
        preds_ind,labels_ind = eval_dataset(model,ind_data,device,softmax=bool(args.softmax))

    else:
        data["out_dist_acc"] = trainer.test(model, ood_data.test_dataloader())[0]["acc/test"]
        preds_ind, labels_ind, preds_ood, labels_ood = custom_eval(model,ind_data,ood_data,device,softmax = bool(args.softmax))
        ## ood
        if args.ood_dataset == "cifar10_1":
            np.save("ood_preds",preds_ood)
            np.save("ood_labels",labels_ood)
        elif args.ood_dataset == "cinic10":    
            np.save("ood_cinic_preds",preds_ood)
            np.save("ood_cinic_labels",labels_ood)
        elif args.ood_dataset == "cifar10_c":    
            np.save("ood_cifar10_c_{}_{}_preds".format(args.corruption,args.level),preds_ood)
            np.save("ood_cifar10_c_{}_{}_labels".format(args.corruption,args.level),labels_ood)
        else:     
            raise Exception("option for ood dataset not recognized.")

    preds_train,labels_train = traindata_eval(model,ind_data,device,softmax = bool(args.softmax))

    full_path = "." #os.path.join(results_dir,"robust_results{}_{}_{}".format(datetime.datetime.now().strftime("%m-%d-%y_%H:%M.%S"),args.module,args.classifier))
    np.save("ind_preds",preds_ind)
    np.save("ind_labels",labels_ind)
    np.save("train_preds",preds_train)
    np.save("train_labels",labels_train)
    
    ## write metadata
    if not args.module.startswith("rff"):
        metadata = {}
        metadata["model_save_path"] = trainer.checkpoint_callback.dirpath
        with open("meta.json","w") as f:
            json.dump(metadata,f)


if __name__ == "__main__":
    main()

