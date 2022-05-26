## Paper figure version with hydra integration. 

from argparse import ArgumentParser
import hydra
import joblib
import matplotlib.pyplot as plt
import re
import json
from scipy.special import softmax
from scipy.stats import gaussian_kde,pearsonr
from sklearn.metrics import r2_score
import numpy as np
import os
from interpensembles.metrics import NLLData,BrierScoreData

here = os.path.dirname(os.path.abspath(__file__))
plt.style.use(os.path.join(here,"../../etc/config/geoff_stylesheet.mplstyle"))
results = os.path.join(here,"../../results/")
ims = os.path.join(here,"../../images/performance_comp")

@hydra.main(config_path = "compare_performance_configs/cifar10_1_bs",config_name ="config")
def main(args):
    """Calculates the per-datapoint metrics for two different models, and plots them against each other. 
    Optionally, can include a base model which we use as a baseline- calculate increase or decrease in metric performance relative to this base model. 
    Will plot both in and out of distribution changes. 

    :param args: an argument parser object with the fields: "model1", "model2", "metric", "oodname" and optionally "basemodel". 
    """
    all_ordereddata = {"ind_":None}
    basedata = []
    all_ordereddata[args.oodname] = None ## create ood name. 
    maxlen = 10000
    for data in all_ordereddata:
        #1. Get the metric values for each dataset we care about. 
        all_model1_metrics = [get_metrics_outputs(m1,args.metric,data) for m1 in args.model1]
        model1_metrics = np.mean(np.array(all_model1_metrics),axis = 0)
        #model1_metrics = get_ensemble(args.model1,args.metric,data)
        all_model2_metrics = [get_metrics_outputs(m2,args.metric,data) for m2 in args.model2]
        model2_metrics = np.mean(np.array(all_model2_metrics),axis = 0)
        if args.basemodel is not None:
            all_basemodel_metrics = [get_metrics_outputs(bm,args.metric,data) for bm in args.basemodel]
            basemodel_metrics = np.mean(np.array(all_basemodel_metrics),axis = 0)
            basemodel_single = all_basemodel_metrics[args.select]
            basedata.append(basemodel_metrics[:maxlen])
            #basemodel_metrics[np.where(basemodel_metrics < args.thresh_score)] = np.nan
            title = "Change in {}".format(args.metricshowname)
            model1_metrics = model1_metrics[:maxlen]-basemodel_metrics[:maxlen] ## this is ensemble minus average
            model2_metrics = model2_metrics[:maxlen]-basemodel_single[:maxlen]#basemodel_metrics ## this is single minus single 
            #model2_metrics = model2_metrics[:len(basemodel_single)]-basemodel_single#basemodel_metrics ## this is single minus single 
        else:    
            title = "{}".format(args.metric)
        #2. Sort them. 
        #ordered = get_ordered_metricvals(model1_metrics,model2_metrics)    
        ordered = np.stack([model1_metrics,model2_metrics],axis = 0)
        all_ordereddata[data] = ordered
    
    #3. plotting: 
    dataset = [args.indshowname,args.oodshowname]
    markers = ["o","x"] 
    colors = ["C0","C4"]
    fig,ax = plt.subplots(1,2,figsize = (14,5))
    orig_title = title
    for di,(data,datadict) in enumerate(all_ordereddata.items()):
        print("plotting")
        means = np.nanmean(datadict,axis = 1)
        #datadict = datadict[:,~np.any(np.isnan(datadict),axis = 0)]
        z = gaussian_kde(datadict)(datadict)
        #idx = z.argsort()
        ax[di].plot(np.linspace(-10,10),np.linspace(-10,10),alpha = 0.2,linestyle = "--")
        idx = basedata[di].argsort()
        scatterval = ax[di].scatter(datadict[0][idx],datadict[1][idx],marker = markers[di],cmap = "winter",c=basedata[di][idx],label = data,s=1)
        fig.colorbar(scatterval,ax = ax[di])

        #ax[di].scatter(datadict[0],datadict[1],marker = markers[di],cmap = "plasma",label = data,s=1)
        #all_data = np.stack([datadict[0][idx],datadict[1][idx]],axis = 1)
        #print(sum(np.all(all_data<0,axis =1))/len(all_data))
        title = orig_title+": {} ".format(dataset[di])
        if args.basemodel is None:
            ax[di].axvline(means[0])
            ax[di].axhline(means[1])
        else:    
            corr,p = pearsonr(datadict[0],datadict[1])
            title = title+"\n Pearson's R: {:3.3} (p={:3.3})".format(corr,p) 
        ax[di].set_title(title)    
    ax[0].set_xlabel(args.model1showname)
    ax[0].set_ylabel(args.model2showname)
    ax[1].set_xlabel(args.model1showname)
    ax[1].set_ylabel(args.model2showname)
    if args.metric == "BrierScore":
        ax[0].set_xlim(-2,2)
        ax[0].set_ylim(-2,2)
        ax[1].set_xlim(-2,2)
        ax[1].set_ylim(-2,2)
    if args.metric == "Likelihood":
        ax[0].set_xlim(-7,7)
        ax[0].set_ylim(-7,7)
        ax[1].set_xlim(-7,7)
        ax[1].set_ylim(-7,7)
    #plt.savefig(os.path.join(ims,r"compare_avg_avg2_brier_avgbase.png".format(args.model1,args.model2,args.metric,args.basemodel)))    
    plt.savefig("{}_diff_{}.pdf".format(args.dumpname,args.select))
    joblib.dump(all_ordereddata,args.dumpname)


def get_ensemble(stubnames,metric,data):
    """Given a set of dataset stubs that reference a set of logits and a set of labels, will return the corresponding metric value for the ensemble formed from those logits and labels. 

    :param stubname: name of the stub that we will get, containing corresponding predictions and labels.  
    :param metric: either "Likelihood", "Brier_Score", or "Confidence".
    :param data: either "ind" or "ood" or "ood_something"
    :returns: returns an array of shape (number_examples,), giving the metric value for each datapoint. 
    """
    ## TODO: refactor this into metrics. Create/Curry functions as necessary to compute metrics. 
    Confidence = lambda preds,labels: np.max(preds,axis = 1)
    def BrierScore(preds,labels):
        labels_onehot = np.zeros(preds.shape)
        labels_onehot[np.arange(len(labels)),labels] = 1
        return np.sum((preds-labels_onehot)**2,axis = 1)
    NLL = lambda preds,labels: -np.log(preds[np.arange(len(labels)),labels])
    func = {"Likelihood":NLL,"BrierScore":BrierScore,"Confidence":Confidence}

    all_data = []
    for stubname in stubnames:
        ## get name of data and target: 
        dataname = stubname+"{}preds.npy".format(data)
        labelname = stubname+"{}labels.npy".format(data)
        datapath = os.path.join(results,dataname)
        labelpath = os.path.join(results,labelname)

        ## load in 
        data = np.load(datapath)
        labels = np.load(labelpath)
        try:
            with open(os.path.join(results,stubname+"_meta.json"),"r") as f:
                metadata = json.load(f)
        except FileNotFoundError:
            metadata = {"softmax":True}

        ## apply metric:
        if bool(metadata.get("softmax",True)) is False:    
            print("applying softmax to: {}".format(stubname))
            data = softmax(data,axis = 1)
        all_data.append(data)    
    ensdata = np.mean(np.array(all_data),axis = 0)    
    return func[metric](ensdata,labels)

def get_metrics_outputs(stubname,metric,data): 
    """Given the name of a dataset stub that references a set of logits and a set of labels, will return the corresponding metric value for those logits and labels. 

    :param stubname: name of the stub that we will get, containing corresponding predictions and labels.  
    :param metric: either "Likelihood", "Brier_Score", or "Confidence".
    :param data: either "ind" or "ood" or "ood_something"
    :returns: returns an array of shape (number_examples,), giving the metric value for each datapoint. 
    """
    ## TODO: refactor this into metrics. Create/Curry functions as necessary to compute metrics. 
    Confidence = lambda preds,labels: np.max(preds,axis = 1)
    def BrierScore(preds,labels):
        labels_onehot = np.zeros(preds.shape)
        labels_onehot[np.arange(len(labels)),labels] = 1
        return np.sum((preds-labels_onehot)**2,axis = 1)
    NLL = lambda preds,labels: -np.log(preds[np.arange(len(labels)),labels])
    func = {"Likelihood":NLL,"BrierScore":BrierScore,"Confidence":Confidence}

    ## get name of data and target: 
    dataname = stubname+"{}preds.npy".format(data)
    labelname = stubname+"{}labels.npy".format(data)
    datapath = os.path.join(results,dataname)
    labelpath = os.path.join(results,labelname)

    ## load in 
    data = np.load(datapath)
    labels = np.load(labelpath)
    try:
        with open(os.path.join(results,stubname+"_meta.json"),"r") as f:
            metadata = json.load(f)
    except FileNotFoundError:
        metadata = {"softmax":True}

    ## apply metric:
    if bool(metadata.get("softmax",True)) is False:    
        print("applying softmax to: {}".format(stubname))
        data = softmax(data,axis = 1)
    return func[metric](data,labels)

def get_ordered_metricvals(ordering,following):    
    """Given two arrays of metrics that follow the same ordering per datapoint, will sort them both according to the values of the first array given

    :param ordering: a numpy array of shape (nb_examples,) containing metric values for each datapoint. We will sort according to the values in this array. 
    :param following: a numpy array of shape (nb_examples,) conaining metric values for each datapoint. Sorted according to the ordering given by ordering.
    :returns: an array of shape (nb_examples,2) that gives the sorted valeus of ordering and following, respectively. 
    """

    inds = np.argsort(ordering)
    ordered_1 = ordering[inds]
    ordered_2 = following[inds]
    return np.stack([ordered_1,ordered_2],axis = 0)

if __name__ == "__main__":
    #parser = ArgumentParser()
    #parser.add_argument("--model1","-m1",action = "append", help = "first model (or average of first model scores)")
    #parser.add_argument("--model2","-m2",action = "append", help = "second model (or average of second model scores)")
    #parser.add_argument("--metric",choices = ["Likelihood","BrierScore","Confidence"])
    #parser.add_argument("--oodname")
    #parser.add_argument("--basemodel","-bm",action = "append",help = "base model (or average of base model scores) to consider")
    #parser.add_argument("--thresh_score","-t",type = float,help = "threshold score at which to throw out a data point")
    #parser.add_argument("--dumpname","-d",help = "name of file we want to save output to.")
    #parser.add_argument("--select","-s",type = int,help= "which single model to compare ")
    #args = parser.parse_args()
    

    main()
