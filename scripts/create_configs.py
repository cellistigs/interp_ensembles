## create config files for each type of model. 
from plot_metrics import all_dataindices,markers,all_suffixes
import json
import os 
import yaml

here = os.path.dirname(os.path.abspath(__file__))
resultsfolder = os.path.join(here,"../results")
configsfolder = os.path.join(here,"script_configs/")

template = {"plot":False,"ind_stubs":None,"ood_stubs":None,"ood_suffix":None,"gpu":True,"uncertainty":None}

if __name__ == "__main__":
    modelpre_lists = {}
    for oodname,dataindex in all_dataindices.items():
        print(oodname)
        modelpre_lists[oodname] = {}
        for modelpre in markers:
            if not modelpre.startswith("Ensemble"):
                modelpre_lists[oodname][modelpre] = {"ind":[],"ood":[],"ind_softmax":None,"ood_softmax":None}

    ## for ood_dataset: 
    for oodname,dataindex in all_dataindices.items():
        suffixes = all_suffixes[oodname]

        for model,modelid in dataindex.items():

            for modelpre,modeldict in modelpre_lists[oodname].items():

                if model.startswith(modelpre):
                    modeldict["ood"].append(modelid)
                    try:
                        with open(os.path.join(resultsfolder,modelid+suffixes["meta"]),"r") as f:
                            metadata = json.load(f)
                    except FileNotFoundError:
                        metadata = {"softmax":True}

                    modeldict["ood_softmax"] = bool(metadata.get("softmax",True))    


                    if oodname == "cifar10.1":
                        modeldict["ind"].append(modelid)

                        try:
                            with open(os.path.join(resultsfolder,modelid+suffixes["meta"]),"r") as f:
                                metadata = json.load(f)
                        except FileNotFoundError:
                            metadata = {"softmax":True}
                        modeldict["ind_softmax"] = bool(metadata.get("softmax",True))    

                    else:    
                        modeldict["ind"]=modelpre_lists["cifar10.1"][modelpre]["ind"]
                        modeldict["ind_softmax"] = modelpre_lists["cifar10.1"][modelpre]["ind_softmax"] 
        

    try:
        all_ood = os.path.join(configsfolder,"all_ood")
        os.mkdir(all_ood)
    except FileExistsError:    
        pass

    for oodname,dataindex in all_dataindices.items():
        config_group = os.path.join(configsfolder,oodname)
        try:
            os.mkdir(config_group)
        except FileExistsError:    
            pass
        for modelpre,modeldict in modelpre_lists[oodname].items():
            file = {}
            for k,val in template.items():
                file[k] = template[k]
            if modelpre == "Native WideResNet-28-10":
                file["ind_stubs"] = modelpre_lists["cifar10.1"]["Conv WideResNet-28-10"]["ind"]
                file["ind_softmax"] = modelpre_lists["cifar10.1"]["Conv WideResNet-28-10"]["ind_softmax"]
            else:
                file["ind_stubs"] = modeldict["ind"]    
                file["ind_softmax"] = modeldict["ind_softmax"]
            file["ood_stubs"] = modeldict["ood"]    
            file["ood_softmax"] = modeldict["ood_softmax"]
            if oodname == "cifar10.1":
                file["ood_suffix"] = "ood_preds.npy"
            elif oodname == "cinic10":
                file["ood_suffix"] = "ood_cinic_preds.npy"
            else:    
                file["ood_suffix"] = "ood_{}_preds.npy".format(oodname)

            for uncertainty in ["Var","JS","KL"]:
                file["uncertainty"] = uncertainty
                name = "config_{modelpre}_{oodname}_{uncertainty}.yaml".format(modelpre=modelpre,oodname=oodname,uncertainty =uncertainty)
                with open(os.path.join(config_group,name),"w") as f:
                    yaml.dump(file,f)
                with open(os.path.join(all_ood,name),"w") as f:
                    yaml.dump(file,f)



    ### create folder
    ### create base template 
    ### update with lists of relevant data names. 
    ### def update_template(datanames):



