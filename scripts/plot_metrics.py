## Given the models we have generated outputs for, plot interesting combinations of accuracy, calibration and nll. 
import os
import numpy as np
import json
from scipy.special import softmax
import matplotlib.pyplot as plt 
from interpensembles.metrics import AccuracyData,NLLData,CalibrationData

resultsfolder = os.path.join(os.path.abspath(os.path.dirname(__file__)),"../results")
imagesfolder = os.path.join(os.path.abspath(os.path.dirname(__file__)),"../images")

markers = {"ResNet":"rx","Ensemble-2 Synth ResNet":"r*","Ensemble-4 ResNet":"ro","Ensemble-4 Synth ResNet":"ro","WideResNet 18-2":"bx","Ensemble-2 Synth WideResNet 18-2":"b*","Ensemble-4 Synth WideResNet 18-2":"bo","WideResNet 18-4":"gx","Ensemble-2 Synth WideResNet 18-4":"g*","Ensemble-4 Synth WideResNet 18-4":"go","Ensemble-5 Synth ResNet":"r+"}

### Now we define the common names for the data, and their prefixes: 
dataindex = {
            #"ResNet 18":"robust_results11-10-21_23:33.14_base_resnet18",
            #"ResNet 18.1":"robust_results11-10-21_23:34.02_base_resnet18",
            #"ResNet 18.2":"robust_results11-10-21_23:34.24_base_resnet18",
            #"ResNet 18.3":"robust_results11-10-21_23:34.43_base_resnet18",
            #"ResNet 18.4":"robust_results11-10-21_23:35.02_base_resnet18",
            #"ResNet 18.5":"robust_results11-10-21_23:35.21_base_resnet18",
            #"Ensemble-4 Synth ResNet 18":"synth_ensemble_0_resnet18_",
            #"Ensemble-4 Synth ResNet 18.1":"synth_ensemble_1_resnet18_",
            #"Ensemble-4 Synth ResNet 18.2":"synth_ensemble_2_resnet18_",
            #"Ensemble-4 Synth ResNet 18.3":"synth_ensemble_3_resnet18_",
            "ResNet 18":"robust_results11-15-21_02:46.11_base_resnet18",
            "ResNet 18.7":"robust_results11-15-21_03:01.22_base_resnet18",
            "ResNet 18.8":"robust_results11-15-21_03:16.31_base_resnet18",
            "ResNet 18.9":"robust_results11-15-21_03:31.40_base_resnet18",
            "ResNet 18.10":"robust_results11-15-21_03:46.53_base_resnet18",
            "Ensemble-2 Synth ResNet 18":"synth_ensemble_0_base_resnet18_sub2_11_15_",
            "Ensemble-2 Synth ResNet 18.1":"synth_ensemble_1_base_resnet18_sub2_11_15_",
            "Ensemble-2 Synth ResNet 18.2":"synth_ensemble_2_base_resnet18_sub2_11_15_",
            "Ensemble-2 Synth ResNet 18.3":"synth_ensemble_3_base_resnet18_sub2_11_15_",
            "Ensemble-2 Synth ResNet 18.4":"synth_ensemble_4_base_resnet18_sub2_11_15_",
            "Ensemble-2 Synth ResNet 18.5":"synth_ensemble_5_base_resnet18_sub2_11_15_",
            "Ensemble-2 Synth ResNet 18.6":"synth_ensemble_6_base_resnet18_sub2_11_15_",
            "Ensemble-2 Synth ResNet 18.7":"synth_ensemble_7_base_resnet18_sub2_11_15_",
            "Ensemble-2 Synth ResNet 18.8":"synth_ensemble_8_base_resnet18_sub2_11_15_",
            "Ensemble-2 Synth ResNet 18.9":"synth_ensemble_9_base_resnet18_sub2_11_15_",
            "Ensemble-4 Synth ResNet 18":"synth_ensemble_0_base_resnet18_11_15_",
            "Ensemble-4 Synth ResNet 18.6":"synth_ensemble_1_base_resnet18_11_15_",
            "Ensemble-4 Synth ResNet 18.7":"synth_ensemble_2_base_resnet18_11_15_",
            "Ensemble-4 Synth ResNet 18.8":"synth_ensemble_3_base_resnet18_11_15_",
            "Ensemble-4 Synth ResNet 18.9":"synth_ensemble_4_base_resnet18_11_15_",
            #"Ensemble-5 Synth ResNet 18":"synth_ensemble_0_base_resnet18_super5_11_15_",
            #"Ensemble-4 WideResNet 18-2 (mean loss)" :"robust_results11-11-21_20:46.40_ensemble_wideresnet18",
            #"WideResNet 18-2":"robust_results11-11-21_00:22.44_base_resnet18",
            #"Ensemble-4 WideResNet 18-2 (sum loss)":"robust_results11-11-21_20:44.39_ensemble_wideresnet18",
            #"Ensemble-4 ResNet 18":"robust_results11-10-21_23:39.36_ensemble_resnet18",
            #"Ensemble-4 Resnet 18.1":"robust_results11-11-21_00:20.25_ensemble_resnet18",
            #"Ensemble-4 Resnet 18.2":"robust_results11-11-21_00:20.56_ensemble_resnet18",
            #"Ensemble-4 Resnet 18.3":"robust_results11-11-21_00:21.33_ensemble_resnet18", 
            #"InterpEnsemble-4 WideResNet 18-2 $\lambda = 1$":"robust_results11-11-21_17:02.02_interpensemble_wideresnet_18",
            #"InterpEnsemble-4 WideResNet 18-2 $\lambda = 0.5$":"robust_results11-11-21_17:16.27_interpensemble_wideresnet_18",
            #"InterpEnsemble-4 WideResNet 18-2 $\lambda = 0$":"robust_results11-11-21_17:17.19_interpensemble_wideresnet_18"
            "WideResNet 18-2":"robust_results11-14-21_23:49.57_base_wideresnet18",
            "WideResNet 18-2.1":"robust_results11-15-21_00:30.20_base_wideresnet18",
            "WideResNet 18-2.2":"robust_results11-15-21_01:10.29_base_wideresnet18",
            "WideResNet 18-2.3":"robust_results11-15-21_01:50.49_base_wideresnet18",
            "WideResNet 18-2.4":"robust_results11-15-21_02:31.03_base_wideresnet18",
            "Ensemble-2 Synth WideResNet 18-2":"synth_ensemble_0_base_wideresnet18_sub2_11_15_",
            "Ensemble-2 Synth WideResNet 18-2.1":"synth_ensemble_1_base_wideresnet18_sub2_11_15_",
            "Ensemble-2 Synth WideResNet 18-2.2":"synth_ensemble_2_base_wideresnet18_sub2_11_15_",
            "Ensemble-2 Synth WideResNet 18-2.3":"synth_ensemble_3_base_wideresnet18_sub2_11_15_",
            "Ensemble-2 Synth WideResNet 18-2.4":"synth_ensemble_4_base_wideresnet18_sub2_11_15_",
            "Ensemble-2 Synth WideResNet 18-2.5":"synth_ensemble_5_base_wideresnet18_sub2_11_15_",
            "Ensemble-2 Synth WideResNet 18-2.6":"synth_ensemble_6_base_wideresnet18_sub2_11_15_",
            "Ensemble-2 Synth WideResNet 18-2.7":"synth_ensemble_7_base_wideresnet18_sub2_11_15_",
            "Ensemble-2 Synth WideResNet 18-2.8":"synth_ensemble_8_base_wideresnet18_sub2_11_15_",
            "Ensemble-2 Synth WideResNet 18-2.9":"synth_ensemble_9_base_wideresnet18_sub2_11_15_",
            "Ensemble-4 Synth WideResNet 18-2":"synth_ensemble_0_base_wideresnet18_11_15_",
            "Ensemble-4 Synth WideResNet 18-2.1":"synth_ensemble_1_base_wideresnet18_11_15_",
            "Ensemble-4 Synth WideResNet 18-2.2":"synth_ensemble_2_base_wideresnet18_11_15_",
            "Ensemble-4 Synth WideResNet 18-2.3":"synth_ensemble_3_base_wideresnet18_11_15_",
            "Ensemble-4 Synth WideResNet 18-2.4":"synth_ensemble_4_base_wideresnet18_11_15_",
            "WideResNet 18-4":"robust_results11-13-21_01:28.26_base_wideresnet18_4",
            "WideResNet 18-4.1":"robust_results11-13-21_03:38.44_base_wideresnet18_4",
            "WideResNet 18-4.2":"robust_results11-13-21_05:49.19_base_wideresnet18_4",
            "WideResNet 18-4.3":"robust_results11-13-21_07:59.30_base_wideresnet18_4",
            "WideResNet 18-4.4":"robust_results11-13-21_10:10.17_base_wideresnet18_4",
            "Ensemble-2 Synth WideResNet 18-4":"synth_ensemble_0_base_wideresnet18-4_sub2_11_15_",
            "Ensemble-2 Synth WideResNet 18-4.1":"synth_ensemble_1_base_wideresnet18-4_sub2_11_15_",
            "Ensemble-2 Synth WideResNet 18-4.2":"synth_ensemble_2_base_wideresnet18-4_sub2_11_15_",
            "Ensemble-2 Synth WideResNet 18-4.3":"synth_ensemble_3_base_wideresnet18-4_sub2_11_15_",
            "Ensemble-2 Synth WideResNet 18-4.4":"synth_ensemble_4_base_wideresnet18-4_sub2_11_15_",
            "Ensemble-2 Synth WideResNet 18-4.5":"synth_ensemble_5_base_wideresnet18-4_sub2_11_15_",
            "Ensemble-2 Synth WideResNet 18-4.6":"synth_ensemble_6_base_wideresnet18-4_sub2_11_15_",
            "Ensemble-2 Synth WideResNet 18-4.7":"synth_ensemble_7_base_wideresnet18-4_sub2_11_15_",
            "Ensemble-2 Synth WideResNet 18-4.8":"synth_ensemble_8_base_wideresnet18-4_sub2_11_15_",
            "Ensemble-2 Synth WideResNet 18-4.9":"synth_ensemble_9_base_wideresnet18-4_sub2_11_15_",
            "Ensemble-4 Synth WideResNet 18-4":"synth_ensemble_0_base_wideresnet18_4_",
            "Ensemble-4 Synth WideResNet 18-4.1":"synth_ensemble_1_base_wideresnet18_4_",
            "Ensemble-4 Synth WideResNet 18-4.2":"synth_ensemble_2_base_wideresnet18_4_",
            "Ensemble-4 Synth WideResNet 18-4.3":"synth_ensemble_3_base_wideresnet18_4_",
            "Ensemble-4 Synth WideResNet 18-4.4":"synth_ensemble_4_base_wideresnet18_4_"
            }

suffixes = {
        "InD Labels":"ind_labels.npy",
        "InD Probs": "ind_preds.npy",
        "OOD Labels":"ood_labels.npy",
        "OOD Probs": "ood_preds.npy",
        "meta":"_meta.json"
        }

bins = list(np.linspace(0,1,17)[1:-1])

if __name__ == "__main__":
    predfig,predax = plt.subplots(figsize = (10,10))
    calfig,calax = plt.subplots(figsize = (10,10))
    nllfig,nllax = plt.subplots(figsize = (10,10))


    for model,path in dataindex.items():
        modelmetrics = {"ind":[],"ood":[]}
        relfig,relax = plt.subplots(1,2)
        try:
            with open(os.path.join(resultsfolder,path+suffixes["meta"]),"r") as f:
                metadata = json.load(f)
        except FileNotFoundError:
            metadata = {"softmax":True}

        for di,dist in enumerate([["InD Labels","InD Probs"],["OOD Labels","OOD Probs"]]):
            labels = np.load(os.path.join(resultsfolder,path+suffixes[dist[0]]))
            probs = np.load(os.path.join(resultsfolder,path+suffixes[dist[1]]))
            if model.startswith("InterpEnsemble"):
                probs = probs*2
            if bool(metadata.get("softmax",True)) is False:    
                probs = softmax(probs,axis = 1)

            ad = AccuracyData()
            nld = NLLData()
            cd = CalibrationData(bins)
                
            accuracy = ad.accuracy(probs,labels)
            nll = nld.nll(probs,labels,normalize = True)
            ece = cd.ece(probs,labels)
            rel = cd.bin(probs,labels)
            interval = cd.binedges[0][1]-cd.binedges[0][0]
            relax[di].bar([bi[0] for bi in cd.binedges],[bi[0] for bi in cd.binedges],width= interval,color = "red",alpha = 0.3)
            relax[di].bar([bi[0] for bi in cd.binedges],[rel[i]["bin_acc"] for i in cd.binedges],width = interval,color = "blue")
            relax[di].set_aspect("equal")

            if dist[0] == "InD Labels":
                modelmetrics["ind"] = [accuracy,nll,ece]
            if dist[0] == "OOD Labels":
                modelmetrics["ood"] = [accuracy,nll,ece]

        for modelpre,mcand in markers.items():
            if model.startswith(modelpre):
                marker = mcand
                break
            else:    
                pass
        if len(model.split(".")) == 1: 
            predax.plot(modelmetrics["ind"][0],modelmetrics["ood"][0],marker,label = model)
            nllax.plot(modelmetrics["ind"][1],modelmetrics["ood"][1],marker,label = model)
            calax.plot(modelmetrics["ind"][2],modelmetrics["ood"][2],marker,label = model)
        else:    
            predax.plot(modelmetrics["ind"][0],modelmetrics["ood"][0],marker)
            nllax.plot(modelmetrics["ind"][1],modelmetrics["ood"][1],marker)
            calax.plot(modelmetrics["ind"][2],modelmetrics["ood"][2],marker)
        relfig.suptitle("{}".format(model))
        relax[0].set_title("InD Calibration")
        relax[1].set_title("OOD Calibration")
        relax[0].set_xlabel("Confidence")
        relax[1].set_ylabel("Accuracy")
        relfig.savefig(os.path.join(imagesfolder,"reliability_diag_{}.png".format(model)))    
        plt.close(relfig)
    predax.legend()
    nllax.legend()
    calax.legend()

    predax.set_title("Predictive Accuracy")
    predax.set_xlabel("InD Test Accuracy")
    predax.set_ylabel("OOD Test Accuracy")
    nllax.set_title("Negative Log Likelihood")
    nllax.set_xlabel("InD")
    nllax.set_ylabel("OOD")
    calax.set_title("Calibration")
    calax.set_xlabel("InD ECE")
    calax.set_ylabel("OOD ECE")
    predfig.savefig(os.path.join(imagesfolder,"prediction_metrics"))    
    nllfig.savefig(os.path.join(imagesfolder,"nll_metrics"))    
    calfig.savefig(os.path.join(imagesfolder,"calibration_metrics"))    


            






