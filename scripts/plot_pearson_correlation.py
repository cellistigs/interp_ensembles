## take pearson correlation values across models and plot them on a single axis. 
import numpy as np
import matplotlib.pyplot as plt
import joblib 
import os
import analyze_uncertainty 

here = os.path.dirname(os.path.abspath(__file__))
markers = {"ind":"x","cifar10.1":"*","cinic10":"+"}
colors = {"ind":"red","cifar10.1":"blue","cinic10":"green"}
img_dir = os.path.join(here,"../images/")

if __name__ == "__main__":
    
    datasets = ["cifar10.1","cinic10"]
    
    all_pearsonr = {"ind":{},"cifar10.1":{},"cinic10":{}}
    for datas in datasets:
        corrdata = joblib.load(os.path.join(analyze_uncertainty.agg_dir,"corr_data_{}".format(datas)))
        for domain,domaindata in corrdata.items():
            for modelclass,modeldata in domaindata.items(): 
                if domain == "ind":
                    dataclass = "ind"
                elif domain == "ood":    
                    dataclass = datas
                if all_pearsonr[dataclass].get(modelclass,None) is None:
                    if domain == "ind":
                        all_pearsonr[dataclass][modelclass] = []
                    if domain == "ood":
                        all_pearsonr[dataclass][modelclass] = []
                        
                if domain == "ind":
                    all_pearsonr[dataclass][modelclass].append(modeldata[0])    
                else:             
                    all_pearsonr[dataclass][modelclass].append(modeldata[0])
    ## now plot on one axis:                 
    modelnames = list(all_pearsonr["ind"].keys())
    import pdb; pdb.set_trace()
    for domain,datasets in all_pearsonr.items():
            for mi,modelname in enumerate(modelnames):
                try:
                    for di in datasets[modelname]:
                            if mi == 0:
                                plt.plot(mi,di,marker = markers[domain],color = colors[domain],label = domain)
                            else:
                                plt.plot(mi,di,marker = markers[domain],color = colors[domain])
                except KeyError:            
                    print("model not evaluated for this condition.")
    ax = plt.gca()        
    ax.set_xticks(np.arange(len(domaindata)+1))
    ax.set_xticklabels(modelnames,rotation = 30)
    plt.title("Pearson R Values for test dataset uncertainty")
    plt.ylabel("Pearson R")
    plt.xlabel("Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir,"pearsonR"))
