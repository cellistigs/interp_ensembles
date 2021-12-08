import joblib
from scipy.stats import spearmanr,pearsonr
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from argparse import ArgumentParser
import os 
# Analyze epistemic and aleatoric uncertainty and their degree of correlation:  
plt.rcParams['text.usetex'] = True

here = os.path.dirname(os.path.abspath(__file__))
agg_dir = os.path.join(here,"../results/aggregated_ensembleresults/")
img_dir = os.path.join(here,"../images/variance_quant")

def get_aleatoric_uncertainty(predslabels):
    """Assume predslabels is a dictionary with keys giving modelnames and values dictionaries with keys "labels","preds". 

    we define the the mean ensemble Brier score as: 
    $A = \mathcal{E}[(p_ik-\mu_k)^2]$
    """
    ## first, calculate a mu_k: 
    factors = []
    for model in predslabels.models:
        labels = predslabels.models[model]["labels"] 
        prob = predslabels.models[model]["preds"]
        classes = max(labels+1)
        mu_k = np.array([len(np.where(labels==i)[0])/len(labels) for i in range(classes)]) ## gives mu_k for each for index k. 
        y_onehot = np.zeros(prob.shape)
        y_onehot[np.arange(len(labels)),labels] = 1
        deviance = prob-y_onehot
        brier_factor = deviance**2
        #brier_factor = (predslabels.models[model]["preds"]-mu_k)**2 ## (examples,classes) - (classes,)
        factors.append(brier_factor)
    mean_model_brier = np.mean(np.stack(factors,axis = 0),axis = 0) ## shape (examples,classes)   
    normalization = (mu_k)*(1-mu_k)
    aleatoric_uncertainty = normalization - mean_model_brier ## shape (examples,classes)
    return mean_model_brier,normalization

def main(alldata,data):
    """This function takes a dictionary that looks like: 
    {"ind":VarianceData,"ood":VarianceData},
    where VarianceData has fields data, models, and modelclass, indicating the ood/ind, actual labels and predictions and prefix for the model, respectively.  
    as well as a string specifying ind vs. ood data. 

    """
    exclude = {"cifar10.1":["Ensemble"],"cinic10":["Ensemble"]} 
    corr_data = {"ind":{},"ood":{}}
    for modelclass, modeldata in alldata.items():
        print(modelclass)
        if not np.any([modelclass.startswith(stub) for stub in exclude[data]]):
            fig,ax = plt.subplots(2,2,figsize=(10,10),sharex = True, sharey = True)
            normed = {}
            xmin,xmax = np.inf,-np.inf 
            ymin,ymax = np.inf,-np.inf 
            for di,dataclass in enumerate(["ood","ind"]):
                mean_model_brier,norm = get_aleatoric_uncertainty(modeldata[dataclass])
                variance =modeldata[dataclass].variance()
                normed[dataclass] = (np.mean(mean_model_brier,axis = 1),np.mean(variance,axis = 1))
                xmin,xmax = min(xmin,np.min(normed[dataclass][0])),max(xmax,np.max(normed[dataclass][0]))
                ymin,ymax = min(ymin,np.min(normed[dataclass][1])),max(ymax,np.max(normed[dataclass][1]))

            for di,dataclass in enumerate(["ood","ind"]):
                xx,yy = np.mgrid[xmin:xmax:int(abs(xmax-xmin)*1000)*1j,ymin:ymax:int(abs(ymax-ymin)*1000)*1j]
                eps = 0 #1e-2
                samplepositions = np.vstack([xx.ravel(),yy.ravel()]) 
                kernel = gaussian_kde(normed[dataclass],bw_method = len(normed[dataclass][0])**(-1/4))
                #f = np.flipud(np.reshape(kernel(samplepositions).T,xx.shape))
                f = np.reshape(kernel(samplepositions).T,xx.shape)

                #corr,p = spearmanr(a=normed[dataclass][0],b=normed[dataclass][1])
                corr,p = pearsonr(normed[dataclass][0],normed[dataclass][1])
                corr_data[dataclass][modelclass] = (corr,p)
                ax[di,0].plot(normed[dataclass][1],normed[dataclass][0],"o",label ="{}: pearson r: {}, p = {}".format(dataclass,str(corr)[:5],p),markersize = 0.5)
                idline = np.linspace(xmin,xmax,100)
                ax[di,0].plot(idline,idline,"--",color = "black",label = "y=x")
                axlognorm = ax[di,1].matshow(f,cmap = "RdBu_r",extent = [ymin-eps,ymax+eps,xmin-eps,xmax+eps],norm = SymLogNorm(linthresh = 1e-3,vmin = np.min(f),vmax = np.max(f)),aspect = "auto",origin = "lower")
                divider = make_axes_locatable(ax[di,1])
                cax = divider.append_axes("right",size = "5%",pad = 0.17)
                cax.set_axis_off()
                #asp = np.diff(ax[di,0].get_xlim())[0]/np.diff(ax[di,0].get_ylim())[0]
                #axlognorm = ax[di,1].matshow(f,cmap = "RdBu_r",norm = SymLogNorm(linthresh = 1e-3,vmin = np.min(f),vmax = np.max(f)),aspect = "auto")
                fig.colorbar(axlognorm,ax = cax)
                
                ax[di,0].set_ylabel(r'$\frac{1}{K}\sum_k \mathcal{E}(p_{ik}-\mu_k)$')
                ax[di,0].set_xlabel(r'$\frac{1}{K}\sum_{k}Var(p_{ik})$')
                ax[di,1].set_ylabel(r'$\frac{1}{K}\sum_k \mathcal{E}(p_{ik}-\mu_k)$')
                ax[di,1].set_xlabel(r'$\frac{1}{K}\sum_{k}Var(p_{ik})$')
                ax[di,0].legend()        
            #ax[0].set_xlim(0,1.5*np.mean(norm))
            #ax[0].set_ylim(0,1.5*np.mean(norm))
            #ax[1].set_ylim(np.mean(norm)-0.025,np.mean(norm)+0.01)
            #ax[1].set_xlim(0,0.0002)
            ax[0,0].set_title("scatter (ood)")
            ax[0,1].set_title("kde (ood)")
            ax[1,0].set_title("scatter (ind)")
            ax[1,1].set_title("kde (ind)")
            #plt.tight_layout()

            plt.savefig(os.path.join(img_dir,"mean_brier_vs_variance_{}_{}.png".format(modelclass,data)))        
            plt.close()
    joblib.dump(corr_data,os.path.join(agg_dir,"corr_data_{}".format(data)))        


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data",type = str,default = "cifar10.1",choices = ["cifar10.1","cinic10"])
    args = parser.parse_args()
    datasets = ["cifar10.1","cinic10"]
    for datas in datasets:

        data = joblib.load(os.path.join(agg_dir,"ensembledata_{}".format(datas)))
        main(data,datas)

