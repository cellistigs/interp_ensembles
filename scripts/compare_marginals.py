import joblib
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import os 
import numpy as np

## compare marginal distributions for aleatoric uncertainty between ensembles and performance matched single models. 
here = os.path.abspath(os.path.dirname(__file__))
agg_dir = os.path.join(here,"../results/aggregated_ensembleresults/")
imagesfolder = os.path.join(here,"../images/marginals")

def main(alldata,data):
    """This function takes a dictionary that looks like:
    {"ind":VarianceData,"ood":VarianceData},
    where VarianceData has fields data, models, and modelclass, indicating the ood/ind, actual labels and predictions and prefix for the model, respectively.
    as well as a string specifying ind vs. ood data.
    """
    for modelclass, modeldata in alldata.items():
        if not np.any([modelclass.startswith(stub) for stub in ["Ensemble"]]):
            fig,ax = plt.subplots(2,6,figsize=(15,10),sharex = True, sharey = True)
            try:
                for di,dataclass in enumerate(["ood","ind"]):
                    ## first get per dataset: 
                    all_likelihoods = []
                    for inst_i,(instancename, instancedata) in enumerate(modeldata[dataclass].models.items()):
                        print(inst_i)
                        preds = instancedata["preds"]
                        targets = instancedata["labels"]
                        likelihoods = preds[np.arange(len(targets)),targets]
                        all_likelihoods.append(likelihoods)
                        ax[di,inst_i+1].hist(likelihoods,bins = 100,density = True, log = True,cumulative = True)
                        ax[di,inst_i+1].set_title("model {}".format(inst_i))
                        ax[di,inst_i+1].axvline(x = np.mean(likelihoods),color = "black")
                    ens_likelihood = modeldata[dataclass].mean_conf()[np.arange(len(targets)),targets]    
                    ax[di,0].hist(ens_likelihood,bins = 100,density = True,log = True,cumulative = True)
                    ax[di,0].axvline(x = np.mean(ens_likelihood),color = "black")
                    ax[di,0].set_title("ensemble ({})".format(dataclass))
                    ax[di,0].set_xlabel("likelihood")
                    ax[di,0].set_ylabel("frequency")
                plt.suptitle("{} cumulative marginal plots: {}".format(modelclass,data))        
                plt.savefig(os.path.join(imagesfolder,"{}_marginal_plots_{}.png".format(modelclass,data)))
                plt.close()
                    
            except Exception as e:   
                print("something went wrong with this model: {}".format(e))
                


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data",type = str,default = "cifar10.1",choices = ["cifar10.1","cinic10","cifar10_c_fog_1","cifar10_c_fog_5","cifar10_c_brightness_1","cifar10_c_brightness_5","cifar10_c_gaussian_noise_1","cifar10_c_gaussian_noise_5","cifar10_c_contrast_1","cifar10_c_contrast_5"])
    args = parser.parse_args()
    datasets = ["cifar10.1","cinic10","cifar10_c_fog_1","cifar10_c_fog_5","cifar10_c_brightness_1","cifar10_c_brightness_5","cifar10_c_gaussian_noise_1","cifar10_c_gaussian_noise_5","cifar10_c_contrast_1","cifar10_c_contrast_5"]
    for datas in datasets:

        data = joblib.load(os.path.join(agg_dir,"ensembledata_{}".format(datas)))
        main(data,datas)

