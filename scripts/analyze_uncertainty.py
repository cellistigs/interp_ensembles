import joblib
import tqdm
from scipy.spatial.distance import pdist
from interpensembles.mmd import RBFKernel, MMDModule
from interpensembles.uncertainty import BrierScoreMax,LikelihoodMax,ConfidenceMax
from interpensembles.density_estimates import Variance_Decomp
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

nb_classes = 10

here = os.path.dirname(os.path.abspath(__file__))
agg_dir = os.path.join(here,"../results/aggregated_ensembleresults/")
img_dir = os.path.join(here,"../images/variance_quant")

def compute_witnessfunc(normed,normed_incorrect):
    """Given two dictionaries `normed`, normed_incorrect with keys "ind", "ood", computes witness functions for each. 

    """
    xy_samples = [np.stack(normed[dataclass],axis = 1) for dataclass in ["ind","ood"]]
    xy_samples_incorrect = [np.stack(normed_incorrect[dataclass],axis = 1) for dataclass in ["ind","ood"]]
    ## need a fixed n for kde: 

    #all_datapoints = np.concatenate(xy_samples,axis = 0)
    #all_datapoints_incorrect = np.concatenate(xy_samples_incorrect,axis = 0)
    all_datapoints_ind = np.concatenate([xy_samples[0],xy_samples_incorrect[0]],axis = 0)
    all_datapoints_ood = np.concatenate([xy_samples[1],xy_samples_incorrect[1]],axis = 0)
    all_dists = []
    print("calculating pairwise dists")
    dists = pdist(all_datapoints_ind)
    l = np.median(dists)
    #l = 0.001
    rbf = RBFKernel(2,l)
    mmdmod = MMDModule(rbf)
    print("calculating witness function for correct samples")
    witnessfunc = mmdmod.compute_witness(xy_samples[0],xy_samples_incorrect[0])
    print("calculating witness function for incorrect samples")
    witnessfunc_incorrect = mmdmod.compute_witness(xy_samples[1],xy_samples_incorrect[1])
    return witnessfunc,witnessfunc_incorrect

def get_confidence(predslabels,filter=None):
    """Assume predslabels.models is a dictionary with keys giving modelnames and values dictionaries with keys "labels","preds". 

    we get the likelihood: the confidence prediction in the correct class.
    :param filter: filter based on correct or incorrect. 
    :returns: array of shape nb_examples,2, that gives the likelihood of the corresponding example and the variance from models. 
    """
    factors = []
    variance = predslabels.variance()
    for model in predslabels.models:
        labels = predslabels.models[model]["labels"] 
        prob = predslabels.models[model]["preds"]
        factors.append(prob)
    mean_model_probs = np.mean(np.stack(factors,axis = 0),axis = 0) ## shape (examples)   
    confidence = np.max(mean_model_probs,axis = 1)
    aconf = np.argmax(mean_model_probs,axis = 1)
    confidence_variance = variance[np.arange(len(aconf)),aconf]
    if filter == "correct":
        where = np.where(aconf==labels)
        confidence = confidence[where]
        confidence_variance = confidence_variance[where]
    elif filter == "incorrect":
        where = np.where(aconf!=labels)
        confidence = confidence[where]
        confidence_variance = confidence_variance[where]

    return confidence,confidence_variance

def get_likelihood(predslabels,filter = None):
    """Assume predslabels.models is a dictionary with keys giving modelnames and values dictionaries with keys "labels","preds". 

    we get the likelihood: the confidence prediction in the correct class.
    :param filter: filter based on correct or incorrect. 
    :returns: array of shape 10000,2, that gives the likelihood of the corresponding example and the variance from models. 
    """
    factors = []
    probs = []
    variance =predslabels.variance()
    for model in predslabels.models:
        labels = predslabels.models[model]["labels"] 
        prob = predslabels.models[model]["preds"]
        likelihoods = prob[np.arange(len(labels)),labels]
        factors.append(likelihoods)
        probs.append(prob)
    mean_model_likelihood = np.mean(np.stack(factors,axis = 0),axis = 0) ## shape (examples)   
    mean_model_probs = np.mean(np.stack(probs,axis = 0),axis = 0) ## we also need the ensemble predictions to calculate correct vs. incorrect. 
    aconf = np.argmax(mean_model_probs,axis = 1)
    likelihood_variance = variance[np.arange(len(labels)),labels]

    if filter == "correct":
        where = np.where(aconf==labels)
        mean_model_likelihood = mean_model_likelihood[where]
        likelihood_variance = likelihood_variance[where]
    elif filter == "incorrect":
        where = np.where(aconf!=labels)
        mean_model_likelihood = mean_model_likelihood[where]
        likelihood_variance = likelihood_variance[where]
    return mean_model_likelihood,likelihood_variance

def get_mean_model_brier(predslabels):
    """Assume predslabels.models is a dictionary with keys giving modelnames and values dictionaries with keys "labels","preds". 

    we define the the mean ensemble Brier score as: 
    $A = \mathcal{E}[(p_ik-\mu_k)^2]$
    :returns: array of shape 10000, that gives the mean model brier score in each dim. 
    """
    #TODO: output mean variance from here 
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
        brier_factor = np.sum(deviance**2,axis = 1)
        #brier_factor = (predslabels.models[model]["preds"]-mu_k)**2 ## (examples,classes) - (classes,)
        factors.append(brier_factor)
    mean_model_brier = np.mean(np.stack(factors,axis = 0),axis = 0) ## shape (examples,classes)   
    #normalization = (mu_k)*(1-mu_k)
    #aleatoric_uncertainty = normalization - mean_model_brier ## shape (examples,classes)
    return mean_model_brier#,normalization

def main(alldata,data,metric = "Brier"):
    """This function takes a dictionary of dictionaries as primary input, that looks like:.
    {"modelname1":{"ind":VarianceData,"ood":VarianceData},"modelname2":{"ind":VarianceData,"ood":VarianceData}...}

    where VarianceData has fields `data`, `models`, and `modelclass`, indicating the ood/ind labels, predictions, and model class prefix  respectively.  
    :param alldata: dictionary of dicts (see above) 
    :param data: string specifying the ood dataset.
    :param metric: we can work with the brier score or the likelihood and corresponding variance metrics. 

    """
    exclude = {"cifar10.1":["Ensemble"],"cinic10":["Ensemble"]} 
    corr_data = {"ind":{},"ood":{}}
    if metric == "Brier":
        bsm = BrierScoreMax(1)
        maxpoints_uncorr = bsm.get_maxpoints_uncorr(5)
        maxpoints_corr = bsm.get_maxpoints_corr(5)
    elif metric == "Likelihood":   
        lm = LikelihoodMax()
        maxpoints_uncorr = lm.get_maxpoints(5)
    elif metric == "Confidence":   
        cm = ConfidenceMax(nb_classes)
        maxpoints_uncorr = cm.get_maxpoints(5)
    else:     
        raise Exception("Metric not given.")
    eps = 0.01
    xmin,xmax = np.min(maxpoints_uncorr[:,1]),np.max(maxpoints_uncorr[:,1]) + eps
    xmin = 0 
    ymin,ymax = np.min(maxpoints_uncorr[:,0]),np.max(maxpoints_uncorr[:,0]) + eps
    for modelclass, modeldata in alldata.items():
        print(modelclass)
        if not np.any([modelclass.startswith(stub) for stub in ["Ensemble"]]):
            if metric != "Brier":
                fig,ax = plt.subplots(3,5,figsize=(20,15),sharex = True, sharey = True)
            else:    
                fig,ax = plt.subplots(2,4,figsize=(15,10),sharex = True, sharey = True)
            normed = {}
            normed_incorrect = {}
            try:
                for di,dataclass in enumerate(["ood","ind"]):
                    if metric == "Brier":
                        mean_model_brier = get_mean_model_brier(modeldata[dataclass])
                        variance =modeldata[dataclass].variance()
                        normed[dataclass] = (mean_model_brier,np.sum(variance,axis = 1))
                    elif metric == "Likelihood":    
                        confs = modeldata[dataclass].mean_conf()
                        likelihood,variance = get_likelihood(modeldata[dataclass],"correct")
                        normed[dataclass] = (likelihood,variance)
                        likelihood,variance = get_likelihood(modeldata[dataclass],"incorrect")
                        normed_incorrect[dataclass] = (likelihood,variance)
                    elif metric == "Confidence":    
                        confs = modeldata[dataclass].mean_conf()
                        confs,variance = get_confidence(modeldata[dataclass],"correct")
                        normed[dataclass] = (confs,variance)
                        confs,variance = get_confidence(modeldata[dataclass],"incorrect")
                        normed_incorrect[dataclass] = (confs,variance)


                ## Calculate the median distance between points for mmd: . 
                if metric != "Brier":
                    witness,witness_incorrect = compute_witnessfunc(normed,normed_incorrect)
                else:    
                    witness,_ = compute_witnessfunc(normed,normed)
                densities = {}
                conds = []

                for di,dataclass in enumerate(["ood","ind"]):
                    vd = Variance_Decomp(xmin,xmax,ymin,ymax,80,1e-12)
                    #f = vd.joint_kde(normed[dataclass][0],normed[dataclass][1],bw_method = len(normed[dataclass][0])**(-1/4))
                    if metric != "Brier":
                        n_total = len(normed[dataclass][0])+len(normed_incorrect[dataclass][0])/2
                    else:    
                        n_total = len(normed[dataclass][0])
                    d =2 
                    bw = n_total**(-1./(d+4))
                    f = vd.joint_kde(normed[dataclass][0],normed[dataclass][1],bw_method = bw)
                    f_cond = vd.conditional_variance_kde(normed[dataclass][0],normed[dataclass][1],bw_method = len(normed[dataclass][0])**(-1/4))
                    sample_positions = np.vstack([vd.xx.ravel(),vd.yy.ravel()])
                    print("evaluating correct witness func")
                    witness_eval = np.reshape(witness(sample_positions.T),vd.xx.shape)
                    if metric != "Brier":    
                        f_incorrect = vd.joint_kde(normed_incorrect[dataclass][0],normed_incorrect[dataclass][1],bw_method = bw)
                        print("evaluating incorrect witness func")
                        witness_eval_incorrect = np.reshape(witness_incorrect(sample_positions.T),vd.xx.shape)
                    #f_cond = vd.conditional_variance_kde(normed[dataclass][0],normed[dataclass][1])
                    conds.append(f_cond)

                    #corr,p = spearmanr(a=normed[dataclass][0],b=normed[dataclass][1])
                    corr,p = pearsonr(normed[dataclass][0],normed[dataclass][1])
                    corr_data[dataclass][modelclass] = (corr,p)
                    #ax[di,0].plot(normed[dataclass][1],normed[dataclass][0],"o",label ="{}: pearson r: {}, p = {}".format(dataclass,str(corr)[:5],p),markersize = 0.5)
                    ax[di,0].plot(normed[dataclass][1],normed[dataclass][0],"o",markersize = 0.5)
                    #idline = np.linspace(0,xmax,100)
                    #ax[di,0].plot(idline,idline,"--",color = "black",label = "y=x")
                    axlognorm = ax[di,1].matshow(f,cmap = "RdBu_r",extent = [ymin-eps,ymax+eps,xmin-eps,xmax+eps],norm = SymLogNorm(linthresh = 1e-3,vmin = np.min(f),vmax = np.max(f)),aspect = "auto",origin = "lower")
                    ax[di,2].matshow(f_cond,cmap = "RdBu_r",extent = [ymin-eps,ymax+eps,xmin-eps,xmax+eps],norm = SymLogNorm(linthresh = 1e-3,vmin = np.min(f),vmax = np.max(f)),aspect = "auto",origin = "lower")
                    
                    if metric != "Brier":    
                        ax[di,2].plot(normed_incorrect[dataclass][1],normed_incorrect[dataclass][0],"o",markersize = 0.5)
                        #idline = np.linspace(0,xmax,100)
                        #ax[di,2].plot(idline,idline,"--",color = "black",label = "y=x")
                        axlognorm_incorrect = ax[di,3].matshow(f_incorrect,cmap = "RdBu_r",extent = [ymin-eps,ymax+eps,xmin-eps,xmax+eps],norm = SymLogNorm(linthresh = 1e-3,vmin = np.min(f),vmax = 1e3*2),aspect = "auto",origin = "lower")
                    densities["{}_correct".format(dataclass)] = f
                    if metric != "Brier":    
                        densities["{}_incorrect".format(dataclass)] = f_incorrect

                    for mi in range(len(maxpoints_uncorr)):
                        if mi == 0:
                            if metric == "Brier":
                                ax[di,0].plot(*maxpoints_corr[mi],"*",label = "max variance (corr. errors)")
                            else:    
                                ax[di,2].plot(*maxpoints_uncorr[mi],"^",label = "max variance (uncorr. errors)")
                            ax[di,0].plot(*maxpoints_uncorr[mi],"^",label = "max variance (uncorr. errors)")
                        else:    
                            if metric == "Brier":
                                ax[di,0].plot(*maxpoints_corr[mi],"*")
                            else:    
                                ax[di,2].plot(*maxpoints_uncorr[mi],"^")
                            ax[di,0].plot(*maxpoints_uncorr[mi],"^")

                    divider = make_axes_locatable(ax[di,1])
                    if metric != "Brier":    
                        divider_incorrect = make_axes_locatable(ax[di,3])
                    cax = divider.append_axes("right",size = "5%",pad = 0.17)
                    cax.set_axis_off()
                    if metric != "Brier":    
                        cax_incorrect = divider_incorrect.append_axes("right",size = "5%",pad = 0.17)
                        cax_incorrect.set_axis_off()
                    #asp = np.diff(ax[di,0].get_xlim())[0]/np.diff(ax[di,0].get_ylim())[0]
                    #axlognorm = ax[di,1].matshow(f,cmap = "RdBu_r",norm = SymLogNorm(linthresh = 1e-3,vmin = np.min(f),vmax = np.max(f)),aspect = "auto")
                    fig.colorbar(axlognorm,ax = cax)
                    if metric != "Brier":    
                        fig.colorbar(axlognorm_incorrect,ax = cax_incorrect)
                    
                    if metric == "Brier":
                        ax[di,0].set_ylabel(r'$\mathcal{E} \sum_k (p_{ik}-y_i)^2$')
                        ax[di,0].set_xlabel(r'$\sum_{k}Var(p_{ik})$')
                        ax[di,1].set_ylabel(r'$\mathcal{E} \sum_k (p_{ik}-y_i)^2$')
                        ax[di,1].set_xlabel(r'$\sum_{k}Var(p_{ik})$')
                    elif metric == "Likelihood":    
                        ax[di,0].set_ylabel(r'$\mathcal{E} p*$')
                        ax[di,0].set_xlabel(r'$Var(\max_k p*)$')
                        ax[di,1].set_ylabel(r'$\mathcal{E} \max_k p*$')
                        ax[di,1].set_xlabel(r'$Var(\max_k p*)$')
                    elif metric == "Confidence":    
                        ax[di,0].set_ylabel(r'$\max_k \mathcal{E} p_{ik}$')
                        ax[di,0].set_xlabel(r'$Variance$')
                        ax[di,1].set_ylabel(r'$\max_k \mathcal{E} p_{ik}$')
                        ax[di,1].set_xlabel(r'$Variance$')
                        ax[di,2].set_ylabel(r'$\max_k \mathcal{E} p_{ik}$')
                        ax[di,2].set_xlabel(r'$Variance$')
                    ax[di,0].legend()        
                ## diffference densities 
                
                if metric != "Brier":
                    ood_diff= densities["ood_correct"]-densities["ood_incorrect"]
                    ind_diff= densities["ind_correct"]-densities["ind_incorrect"]
                    correct_diff = densities["ood_correct"]-densities["ind_correct"]
                    incorrect_diff = densities["ood_incorrect"]-densities["ind_incorrect"]
                    ax[0,4].set_title("kde ood correct-incorrect")
                    ax[0,4].matshow(ood_diff,cmap = "RdBu_r",extent = [ymin-eps,ymax+eps,xmin-eps,xmax+eps],norm = SymLogNorm(linthresh = 1e-9,vmin = np.min(ood_diff),vmax = np.max(ood_diff)),aspect = "auto",origin = "lower")
                    witness_eval_rescale = (witness_eval+np.min(witness_eval))/(np.max(witness_eval)-np.min(witness_eval))
                    ax[1,4].set_title("kde ind correct-incorrect")
                    #ax[0,4].matshow(witness_eval_rescale,aspect = "auto",origin = "lower")
                    ax[1,4].matshow(ind_diff,cmap = "RdBu_r",extent = [ymin-eps,ymax+eps,xmin-eps,xmax+eps],norm = SymLogNorm(linthresh = 1e-9,vmin = np.min(ind_diff),vmax = np.max(ind_diff)),aspect = "auto",origin = "lower")
                    #ax[1,4].matshow(witness_eval_incorrect,aspect = "auto",origin = "lower")
                    ax[2,1].set_title("kde correct ood-ind")
                    ax[2,1].matshow(correct_diff,cmap = "RdBu_r",extent = [ymin-eps,ymax+eps,xmin-eps,xmax+eps],norm = SymLogNorm(linthresh = 1e-6,vmin = np.min(correct_diff),vmax = np.max(correct_diff)),aspect = "auto",origin = "lower")
                    ax[2,3].set_title("kde incorrect ood-ind")
                    ax[2,3].matshow(incorrect_diff,cmap = "RdBu_r",extent = [ymin-eps,ymax+eps,xmin-eps,xmax+eps],norm = SymLogNorm(linthresh = 1e-6,vmin = np.min(incorrect_diff),vmax = np.max(incorrect_diff)),aspect = "auto",origin = "lower")
                    ax[2,0].axis("off")
                    ax[2,2].axis("off")
                    ax[2,4].axis("off")
                #ax[0].set_xlim(0,1.5*np.mean(norm))
                ax[0,3].matshow(conds[0]-conds[1],cmap = "RdBu_r",extent = [ymin-eps,ymax+eps,xmin-eps,xmax+eps],aspect = "auto",origin = "lower")
                #ax[0].set_ylim(0,1.5*np.mean(norm))
                #ax[1].set_ylim(np.mean(norm)-0.025,np.mean(norm)+0.01)
                #ax[1].set_xlim(0,0.0002)
                if metric != "Brier":
                    ax[0,0].set_title("scatter (ood), correct")
                    ax[0,1].set_title("joint kde (ood),correct")
                    ax[0,2].set_title("scatter (ood), incorrect")
                    ax[0,3].set_title("joint kde (ood) incorrect")

                    ax[1,0].set_title("scatter (ind), correct")
                    ax[1,1].set_title("joint kde (ind),correct")
                    ax[1,2].set_title("scatter (ind), incorrect")
                    ax[1,3].set_title("joint kde (ind) incorrect")
                else:    
                    ax[0,0].set_title("scatter (ood)")
                    ax[0,1].set_title("joint kde (ood)")
                    ax[0,2].set_title("conditional var kde (ood)")
                    ax[1,0].set_title("scatter (ind)")
                    ax[1,1].set_title("joint kde (ind)")
                    ax[1,2].set_title("conditional var kde (ind)")
                    ax[0,3].set_title("conditional var kde ood - conditional var kde ind")
                #plt.tight_layout()

                if metric == "Brier":
                    plt.savefig(os.path.join(img_dir,"mean_brier_vs_variance_correct_{}_{}.png".format(modelclass,data)))        
                elif metric == "Likelihood":    
                    plt.savefig(os.path.join(img_dir,"likelihood_vs_variance_correct_{}_{}.png".format(modelclass,data)))        
                elif metric == "Confidence":    
                    plt.savefig(os.path.join(img_dir,"confidence_vs_variance_correct_{}_{}.png".format(modelclass,data)))        
                plt.close()
            except Exception as e:   
                print("something went wrong with this model: {}".format(e))
                raise
    joblib.dump(corr_data,os.path.join(agg_dir,"corr_data_{}".format(data)))        


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data",type = str,default = "cifar10.1",choices = ["cifar10.1","cinic10","cifar10_c_fog_1","cifar10_c_fog_5","cifar10_c_brightness_1","cifar10_c_brightness_5","cifar10_c_gaussian_noise_1","cifar10_c_gaussian_noise_5","cifar10_c_contrast_1","cifar10_c_contrast_5"])
    parser.add_argument("--metric",type = str,choices = ["Brier","Likelihood","Confidence"])
    args = parser.parse_args()
    datasets = ["cifar10.1","cinic10","cifar10_c_fog_1","cifar10_c_fog_5","cifar10_c_brightness_1","cifar10_c_brightness_5","cifar10_c_gaussian_noise_1","cifar10_c_gaussian_noise_5","cifar10_c_contrast_1","cifar10_c_contrast_5"]
    data = joblib.load(os.path.join(agg_dir,"ensembledata_{}".format(args.data)))
    main(data,args.data,args.metric)

