import joblib
import tqdm
from interpensembles.mmd import RBFKernel, MMDModule
from interpensembles.uncertainty import BrierScoreMax
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

from interpensembles.metrics import VarianceData
import h5py
from pathlib import Path
import numpy as np


#here = os.path.dirname(os.path.abspath(__file__))
#agg_dir = os.path.join(here,"../results/aggregated_ensembleresults/")
#img_dir = os.path.join(here,"../images/variance_quant")
img_dir = '/data/Projects/linear_ensembles/interp_ensembles/results/imagenet/images/variance_quant'

def get_mean_model_brier(predslabels):
    """Assume predslabels is a dictionary with keys giving modelnames and values dictionaries with keys "labels","preds". 

    we define the the mean ensemble Brier score as: 
    $A = \mathcal{E}[(p_ik-\mu_k)^2]$
    :returns: array of shape 10000, that gives the mean model brier score in each dim. 
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
        #brier_factor = np.sum(deviance**2,axis = 1)
        brier_factor = np.mean(deviance ** 2, axis=1)
        #brier_factor = np.mean(deviance**2,axis = 1)

        #brier_factor = (predslabels.models[model]["preds"]-mu_k)**2 ## (examples,classes) - (classes,)
        factors.append(brier_factor)
    mean_model_brier = np.mean(np.stack(factors,axis = 0),axis = 0) ## shape (examples,classes)   
    #normalization = (mu_k)*(1-mu_k)
    #aleatoric_uncertainty = normalization - mean_model_brier ## shape (examples,classes)
    return mean_model_brier#,normalization


def main(alldata,data,num_classes=10, ensemble_size=5):
    """This function takes a dictionary of dictionaries as primary input, that looks like:.
    {"modelname1":{"ind":VarianceData,"ood":VarianceData},"modelname2":{"ind":VarianceData,"ood":VarianceData}...}

    where VarianceData has fields `data`, `models`, and `modelclass`, indicating the ood/ind labels, predictions, and model class prefix  respectively.  
    :param alldata: dictionary of dicts (see above) 
    :param data: string specifying the ood dataset.

    """
    corr_data = {"ind":{},"ood":{}}
    xmin, xmax = 0, 2  # np.min(maxpoints_uncorr[:,1]),np.max(maxpoints_uncorr[:,1]) + eps
    ymin, ymax = 0, 1  # np.min(maxpoints_uncorr[:,0]),np.max(maxpoints_uncorr[:,0]) + eps
    space_xx = 50j  # int(abs(xmax - xmin) * scale_kernel_grid)* 1j
    space_yy = 50j  # 0.1#int(abs(ymax - ymin) * scale_kernel_grid)* 1j
    xx, yy = np.mgrid[xmin:xmax: space_xx, ymin:ymax:space_yy]
    eps = 0  # 1e-2
    samplepositions = np.vstack([xx.ravel(), yy.ravel()])

    for modelclass, modeldata in alldata.items():
        print(modelclass)
        if not np.any([modelclass.startswith(stub) for stub in ["Ensemble"]]):
            fig,ax = plt.subplots(2,3,figsize=(15, 10),sharex = True, sharey = True)
            normed = {}
            try:
                for di,dataclass in enumerate(["ood","ind"]):
                    mean_model_brier = get_mean_model_brier(modeldata[dataclass])*num_classes
                    variance = modeldata[dataclass].variance()*num_classes
                    normed[dataclass] = (mean_model_brier,np.mean(variance,axis = 1))
                ## Calculate the median distance between points for mmd: . 

                for di,dataclass in enumerate(["ood","ind"]):
                    kernel = gaussian_kde(normed[dataclass])
                    f = np.reshape(kernel(samplepositions).T,xx.shape)
                    corr,p = pearsonr(normed[dataclass][0], normed[dataclass][1])
                    corr_data[dataclass][modelclass] = (corr, p)
                    ax[di,0].plot(normed[dataclass][1], normed[dataclass][0],
                                  "o",
                                  label="{}: pearson r: {}, p = {}".format(dataclass,str(corr)[:5],p),
                                  markersize = 0.5
                                  )
                    idline = np.linspace(ymin, ymax, 100)
                    ax[di,0].plot(idline, idline,"--", color = "black",label = "y=x")
                    axlognorm = ax[di,1].matshow(f,
                                                 cmap = "RdBu_r",
                                                 extent = [ymin-eps,ymax+eps,xmin-eps,xmax+eps],
                                                 norm = SymLogNorm(linthresh = 1e-3,vmin = np.min(f),vmax = np.max(f)),
                                                 aspect = "auto"
                                                 ,origin = "lower"
                                                 )
                    """
                    for mi in range(len(maxpoints_corr)):
                        if mi == 0:
                            ax[di,0].plot(*maxpoints_corr[mi],"*",label = "max variance (corr. errors)")
                            ax[di,0].plot(*maxpoints_uncorr[mi],"^",label = "max variance (uncorr. errors)")
                        else:    
                            ax[di,0].plot(*maxpoints_corr[mi],"*")
                            ax[di,0].plot(*maxpoints_uncorr[mi],"^")
                    """
                    divider = make_axes_locatable(ax[di,1])
                    cax = divider.append_axes("right",size = "5%",pad = 0.17)
                    cax.set_axis_off()
                    fig.colorbar(axlognorm,ax = cax)
                    
                    ax[di,0].set_ylabel(r'$\mathcal{E} \frac{1}{K}\sum_k (p_{ik}-y_i)^2$')
                    ax[di,0].set_xlabel(r'$\frac{1}{K}\sum_{k}Var(p_{ik})$')
                    ax[di,1].set_ylabel(r'$\mathcal{E} \frac{1}{K}\sum_k (p_{ik}-y_i)^2$')
                    ax[di,1].set_xlabel(r'$\frac{1}{K}\sum_{k}Var(p_{ik})$')
                    ax[di,0].legend()        
                ax[0,0].set_title("scatter (ood)")
                ax[0,1].set_title("kde (ood)")
                ax[1,0].set_title("scatter (ind)")
                ax[1,1].set_title("kde (ind)")

                plt.tight_layout()

                plt.savefig(os.path.join(img_dir,"mean_brier_vs_variance_{}_{}.png".format(modelclass,data)))        
                plt.close()
            except Exception as e:   
                print("something went wrong with this model: {}".format(e))
    #joblib.dump(corr_data,os.path.join(agg_dir,"corr_data_{}".format(data)))


if __name__ == "__main__":

    #%%
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, default="imagenetv2mf",
                        choices=["imagenetv2mf"])
    args = parser.parse_args()
    # %% specify what models should start with
    datasets = ['imagenetv2mf']

    def create_cls(data_type, dataset):
        models_to_register = ["deepens1", "deepens2", "deepens3", "deepens4", "deepens5"]
        results_dir = '/data/Projects/linear_ensembles/interp_ensembles/results/imagenet/logits/'
        modelprefix = 'deepens'

        data_cls = VarianceData(modelprefix, data_type)
        # Register multiple models
        model_type = 'resnet50'
        for model in models_to_register:
            folder_name = model_type + '--' + dataset + '--' + model + '.hdf5'
            store_logits_fname = Path(results_dir + folder_name)
            with h5py.File(str(store_logits_fname), 'r') as f:
                logits_out = f['logits'][()]
                labels = f['targets'][()].astype('int')
            # calculate individual probs
            probs_ = np.exp(logits_out) / np.sum(np.exp(logits_out), 1, keepdims=True)
            # register call preds, labels, modelname
            data_cls.register(probs_, labels, model)
        return data_cls

    #%%
    for data in datasets:
        all_data = {}
        data_cls_ind = create_cls('ind', "imagenet")
        all_data['ind'] = data_cls_ind
        data_cls_ind = create_cls('ood', data)
        all_data['ood'] = data_cls_ind

        # alldata = {datas: {"ind": VarianceData, "ood": VarianceData}}
        alldata = {data: all_data}

        main(alldata, data,num_classes=1000)


