from argparse import ArgumentParser
import matplotlib.pyplot as plt
import json
from scipy.special import softmax
from scipy.stats import gaussian_kde,pearsonr
import numpy as np
import os
from interpensembles.metrics import NLLData,BrierScoreData

here = os.path.dirname(os.path.abspath(__file__))
results = os.path.join(here,"../results/")
ims = os.path.join(here,"../images/performance_comp")

def main(args):
    """Calculates the per-datapoint metrics for two different models, and plots them against each other. 
    Optionally, can include a base model which we use as a baseline- calculate increase or decrease in metric performance relative to this base model. 
    Will plot both in and out of distribution changes. 

    :param args: an argument parser object with the fields: "model1", "model2", "metric", "oodname" and optionally "basemodel". 
    """
    all_ordereddata = {"ind_":None}
    all_ordereddata[args.oodname] = None ## create ood name. 
    for data in all_ordereddata:
        #1. Get the metric values for each dataset we care about. 
        model1_metrics = get_metrics_outputs(args.model1,args.metric,data)
        model2_metrics = get_metrics_outputs(args.model2,args.metric,data)
        if args.basemodel is not None:
            basemodel_metrics = get_metrics_outputs(args.basemodel,args.metric,data)
            title = "Change in {}".format(args.metric)
            model1_metrics = model1_metrics-basemodel_metrics
            model2_metrics = model2_metrics-basemodel_metrics
        else:    
            title = "{}".format(args.metric)
        #2. Sort them. 
        ordered = get_ordered_metricvals(model1_metrics,model2_metrics)    
        all_ordereddata[data] = ordered
    
    #3. plotting: 
    dataset = ["InD","OOD"]
    markers = ["o","x"] 
    colors = ["C0","C4"]
    fig,ax = plt.subplots(1,2,figsize = (10,5))
    orig_title = title
    for di,(data,datadict) in enumerate(all_ordereddata.items()):
        means = np.mean(datadict,axis = 1)
        z = gaussian_kde(datadict)(datadict)
        idx = z.argsort()
        ax[di].scatter(datadict[0][idx],datadict[1][idx],marker = markers[di],c=np.log2(z[idx]),label = data,s=1)
        title = orig_title+": {} ".format(dataset[di])
        if args.basemodel is None:
            ax[di].axvline(means[0])
            ax[di].axhline(means[1])
        else:    
            corr,p = pearsonr(datadict[0],datadict[1])
            title = title+"\n Pearson's R: {:3.3} (p={:3.3})".format(corr,p) 
        ax[di].set_title(title)    
    ax[0].set_xlabel(args.model1)
    ax[0].set_ylabel(args.model2)
    if args.metric == "Brier_Score":
        ax[0].set_xlim(-2,2)
        ax[0].set_ylim(-2,2)
        ax[1].set_xlim(-2,2)
        ax[1].set_ylim(-2,2)
    plt.savefig(os.path.join(ims,"compare_{}_{}_{}_{}.png".format(args.model1,args.model2,args.metric,args.basemodel)))    

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
    func = {"Likelihood":NLL,"Brier_Score":BrierScore,"Confidence":Confidence}

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
    parser = ArgumentParser()
    parser.add_argument("--model1","-m1")
    parser.add_argument("--model2","-m2")
    parser.add_argument("--metric",choices = ["Likelihood","Brier_Score","Confidence"])
    parser.add_argument("--oodname")
    parser.add_argument("--basemodel")
    args = parser.parse_args()
    

    main(args)