import os 
import hydra
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import yaml
import json 
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
#plt.style.use(os.path.join(here,"../../etc/config/geoff_stylesheet.mplstyle"))
plt.style.use(os.path.join(here,"../../etc/config/geoff_stylesheet.mplstyle"))
permdir = os.path.join(here,"../../results","imagenet_percent_increases_01_24") 
oodname_legend = {"imagenetv2":"ImageNet V2",
        }
oodnames = list(oodname_legend.keys())
modelnames = ["alexnet",
        "resnet50",
        "resnet101",
        ]


def process_config(path):
    """Given a config file, get the uncertainty, base model, and ood dataset corresponding to it.

    """
    with open(path,"r") as f:
        config = yaml.safe_load(f)

    div = config["uncertainty"] 
    ood = config["ood_suffix"]
    modelname = config["ood_hdf5s"][0].split("/")[-1].split("--")[0]
    return div,ood,modelname

def process_signif(path):
    """Given significance path, get the data associated. 

    """
    with open(path,"r") as f: 
        config = json.load(f)
    return config    



def main():
    """Take results of permutation testing and plot in three separate graphs. 

    """
    alldata_dict = {}
    for i in range(6):
        configpath = os.path.join(permdir,str(i),".hydra/config.yaml")
        proppath = os.path.join(permdir,str(i),"increase_proportion.json")
        try:
            div,ood,modelname = process_config(configpath)
        except IndexError:    
            continue
        try:
            prop = process_signif(proppath)["increase_over_ind"]
        except FileNotFoundError:    
            prop = None 
        if not alldata_dict.get(div,False):
            alldata_dict[div] = {}   
        if not alldata_dict[div].get(modelname,False):
            alldata_dict[div][modelname] = {}   
        if not alldata_dict[div][modelname].get(ood,False):
            alldata_dict[div][modelname][ood] = {}   
        alldata_dict[div][modelname][ood] = prop   
        
    for div,divdata in alldata_dict.items():
        modelindex = {mn:i for i,mn in enumerate(modelnames)}
        oodindex = {od:i for i,od in enumerate(oodnames)}
        len_models = len(modelindex)
        len_oodnames = len(oodindex)
        all_ps = np.empty((len_models,len_oodnames))
        all_ps[:] = np.nan
        for m in modelnames:
            for o in oodnames:
                try:
                    all_ps[modelindex[m],oodindex[o]] = divdata[m][o]
                except Exception:    
                    continue
        fig,ax = plt.subplots(figsize = (4,4))
        ax.matshow(all_ps,vmin = -1,vmax = 1,cmap = plt.get_cmap("coolwarm"))        
        for mi,m in enumerate(modelnames):
            for oi,o in enumerate(oodnames):
                text = ax.text(oi, mi, "{:3.3}%".format(100*all_ps[mi, oi]),
                               ha="center", va="center", color="w")
        ax.set_yticks(range(len(modelnames)))
        ax.set_yticklabels(["{}".format(mn) for mn in modelnames],rotation = 30)
        ax.set_xticks(range(len(oodnames)))
        ax.set_xticklabels([oodname_legend[ood] for ood in oodnames],rotation = 30)
        plt.tight_layout()
        plt.savefig("percentage_matrix_imagenet_{}.png".format(div))        


    

if __name__ == "__main__":
    main()
