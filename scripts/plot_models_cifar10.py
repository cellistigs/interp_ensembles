## Plot models that we have on the cifar 10 ER curve. 
import interpensembles.plot as plot
import json
import os
import numpy as np
datapath = os.path.abspath(os.path.join(os.path.dirname(__file__),"../data_v4/"))

"""The models are as follows: 
    - UPDATED v4 ResNet 18: Trained with this repo `train.py`, using deterministic ops and seed_everything = true with default parameters. Only model run with .pt state dict- accessed via `--pretrained 1` flag.  
    - UPDATED v4 Resnet 18_{1-5}: Trained with this repo `train.py`, using non-deterministic ops and seed_everything = false with default parameters. Given as src/interpensembles/cifar10/version10-14 in ptl output.
    - UPDATED v4 Ensemble (n=4) ResNet 18: Trained with this repo, `train_ensemble.py`, using deterministic ops and seed_everything = true with default params. Weights are initialized separately per ensemble member (I checked). given as src/interpensembles/cifar10/version6
    - UPDATED v4 WideResNet 28x10: Results given to me by Geoff for previously trained models- 5000 datapoints withheld for validation.
    - UPDATED v4 Ensemble (n=4) WideResNet 28x10: Results given to me by Geoff for previously trained models- 5000 datapoints withheld for validation. Each member of this ensemble was trained with different minibatch ordering. 
    - UPDATED v4 Ensemble (n=4) ResNet 18_1: Trained with this repo, `train_ensemble.py`, without using deterministic ops and seed everything, october 27th, 2021. (version 21)
    - UPDATED v4 Ensemble (n=4) ResNet 18_2: Trained with this repo, `train_ensemble.py`, without using deterministic ops and seed everything, october 27th, 2021.
    - UPDATED v4 Ensemble (n=4) ResNet 18_3: Trained with this repo, `train_ensemble.py`, without using deterministic ops and seed everything, october 28th, 2021.
    - UPDATED v4 WideResNet 18 x2: Trained with this repo, `train.py --classifier wideresnet18`, without using deterministic ops and seed everything, october 28th, 2021.
    - UPDATED v4 Interpensemble (n=4, x2), ResNet18, lambda = 1: Trained with this repo- should be the same as a wide resnet, using determinstic ops, nov 1st, 2021- scripts/cifar10/resnet18/version19
    - UPDATED v4 Interpensemble (n=4, x2), ResNet18, lambda = 0.5: Trained with this repo using determinstic ops, nov 1st, 2021- scripts/cifar10/resnet18/version20
    - UPDATED v4 Interpensemblee (n = 4,x2), ResNet18, lambda = 0.0: Trained with this repo using deterministic ops, nov1st, 2021. -scripts/cifar10/resnet18/verision22
    - Ensemble (n=4), WideResNet18 x2: Trained with this repo using determinstic ops, nov 3nd, 2021
    - WideResNet18 x4: Trained with this repo using determinstic ops, nov 3rd, 2021
    - WideResNet 18 x4, GroupLinear: Trained with this repo using deterministic ops, nov 3rd, with the group linear output (like mean field).


"""
modelmarkers = {
        "ResNet 18":{"marker":"x","color":"#1b9e77","label":True,"dataset_sizes":[10000,2000]},
        "ResNet 18_1":{"marker":"x","color":"#1b9e77","dataset_sizes":[10000,2000]},
        "ResNet 18_2":{"marker":"x","color":"#1b9e77","dataset_sizes":[10000,2000]},
        "ResNet 18_3":{"marker":"x","color":"#1b9e77","dataset_sizes":[10000,2000]},
        "ResNet 18_4":{"marker":"x","color":"#1b9e77","dataset_sizes":[10000,2000]},
        "ResNet 18_5":{"marker":"x","color":"#1b9e77","dataset_sizes":[10000,2000]},
        "Ensemble (n=4) ResNet 18":{"marker":"+","color":"#d95f02","label":True,"dataset_sizes":[10000,2000]},
        "Ensemble (n=4) ResNet 18_1":{"marker":"+","color":"#d95f02","dataset_sizes":[10000,2000]},
        "Ensemble (n=4) ResNet 18_2":{"marker":"+","color":"#d95f02","dataset_sizes":[10000,2000]},
        "Ensemble (n=4) ResNet 18_3":{"marker":"+","color":"#d95f02","dataset_sizes":[10000,2000]},
        "WideResNet 18 2x":{"marker":"o","color":"#7570b3","label":True,"dataset_sizes":[10000,2000]},
        "WideResNet 28x10":{"marker":"v","color":"#e7298a","label":True,"dataset_sizes":[5000,2000]},
        "Ensemble (n=4) WideResNet 28x10":{"marker":"^","color":"#66a61e","label":True,"dataset_sizes":[5000,2000]},
        "InterpEnsemble (x2), $\lambda = 1$":{"marker": "*","color":"#e6ab02","label":True,"dataset_sizes":[10000,2000]},
        "InterpEnsemble (x2), $\lambda = 0.5$":{"marker": "*","color":"#a6761d","label":True,"dataset_sizes":[10000,2000]},
        "InterpEnsemble (x2), $\lambda = 0.0$":{"marker": "*","color":"#666666","label":True,"dataset_sizes":[10000,2000]},
        "Ensemble (n=4) WideResNet 18 x2":{"marker": "^","color":"#666666","label":True,"dataset_sizes":[10000,2000]},
        "WideResNet 18 x4":{"marker": "v","color":"#666666","label":True,"dataset_sizes":[10000,2000]}, 
        "WideResNet 18 x4 GroupLinear":{"marker": "*","color":"#66a61e","label":True,"dataset_sizes":[10000,2000]}
        }
models = {
        "ResNet 18":"resnet_18_default_deterministic.json",
        "ResNet 18_1":"resnet_18_version10.json",
        "ResNet 18_2":"resnet_18_version11.json",
        "ResNet 18_3":"resnet_18_version12.json",
        "ResNet 18_4":"resnet_18_version13.json",
        "ResNet 18_5":"resnet_18_version14.json",
        "WideResNet 18 2x":"wideresnet_18_x2_version0.json",
        "Ensemble (n=4) ResNet 18":"ensemble_4_resnet_18_default_deterministic.json",
        "Ensemble (n=4) ResNet 18_1":"ensemble_4_resnet_18_version21.json",
        "Ensemble (n=4) ResNet 18_2":"ensemble_4_resnet_18_version22.json",
        "Ensemble (n=4) ResNet 18_3":"ensemble_4_resnet_18_version25.json",
        "InterpEnsemble (x2), $\lambda = 0.0$": "interpensemble_lamb_0.0_deterministic.json",
        "InterpEnsemble (x2), $\lambda = 0.5$": "interpensemble_lamb_0.5_deterministic.json",
        "InterpEnsemble (x2), $\lambda = 1$": "interpensemble_lamb_1_deterministic.json",
        "WideResNet 28x10":"wideresnet_28_x10?_cifar10_val_5000.json",
        "Ensemble (n=4) WideResNet 28x10":"ensemble_4_wideresnet_28_x10?_cifar10_val_5000.json",
        "Ensemble (n=4) WideResNet 18 x2":"ensemble_4_wideresnet_18_2x_deterministic.json",
        "WideResNet 18 x4":"wideresnet_18_x4_default_deterministic.json", 
        "WideResNet 18 x4 GroupLinear":"wideresnet_18_x4_grouplinear_default_deterministic.json"
        }

if __name__ == "__main__":
    ## Load in data
    modelresults = {}
    for m,filename in models.items():
        with open(os.path.join(datapath,filename)) as f:
            modelresults[m] = json.load(f)
    plot.plot_multiple_models(modelresults,"cifar10",range=np.arange(90,100),formatdict = modelmarkers)        

    

