import os 
import hydra
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import yaml
import json 
import numpy as np
from all_percentage import oodname_legend,oodnames,modelnames

here = os.path.dirname(os.path.abspath(__file__))
permdir = os.path.join(here,"../","permtest_01_25") 
plt.style.use(os.path.join(here,"../../etc/config/geoff_stylesheet.mplstyle"))

def process_config(path):
    """Given a config file, get the uncertainty, base model, and ood dataset corresponding to it.

    """
    with open(path,"r") as f:
        config = yaml.safe_load(f)

    div = config["uncertainty"] 
    ood = config["ood_suffix"]
    try:
        modelname = config["ood_stubs"][0].split("base_")[-1]
        if modelname == "cifar10_wrn28_s1_":
            modelname = "wideresnet28_10" ## some ambiguity here. 
    except Exception:    
        modelname = config["ood_stubs"][0]
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
    for i in range(200):
        configpath = os.path.join(permdir,str(i),".hydra/config.yaml")
        signifpath = os.path.join(permdir,str(i),"signifdata.json")
        try:
            div,ood,modelname = process_config(configpath)
        except IndexError:    
            continue
        try:
            signif = process_signif(signifpath)
        except FileNotFoundError:    
            signif = {"lower":None,"upper":None,"exact":None}
        if not alldata_dict.get(div,False):
            alldata_dict[div] = {}   
        if not alldata_dict[div].get(modelname,False):
            alldata_dict[div][modelname] = {}   
        if not alldata_dict[div][modelname].get(ood,False):
            alldata_dict[div][modelname][ood] = {}   
        alldata_dict[div][modelname][ood] = signif["exact"]    
        
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
        fig,ax = plt.subplots(figsize = (10,10))
        ax.matshow(all_ps,vmin = 0,vmax = 1,cmap = plt.get_cmap("gist_heat"))        
        for mi,m in enumerate(modelnames):
            for oi,o in enumerate(oodnames):
                text = ax.text(oi, mi, "{:3.3}".format(all_ps[mi, oi]),
                               ha="center", va="center", color="w",path_effects=[pe.withStroke(linewidth=0.3, foreground="black")])
        ax.set_yticks(range(len(modelnames)))
        ax.set_yticklabels([mn.replace("_","\_") for mn in modelnames],rotation = 30)
        ax.set_xticks(range(len(oodnames)))
        ax.set_xticklabels([oodname_legend[ood].replace("_","\_") for ood in oodnames],rotation = 30)
        plt.savefig("pval_matrix__01_25_{}.png".format(div))        


if __name__ == "__main__":
    main()
