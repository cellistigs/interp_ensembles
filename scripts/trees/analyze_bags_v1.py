## Take existing bagging data, and analyze it to make a plot. 
"""
These are random forest models with depth 1-12 fit to mnist.
Each probs_depth_{i} has all 10,000 predictions for a set of 500 trees in the forest,
and each test_depth_{i} has corresponding labels.
To answer your previous question, I got over the zero probabilities by first averaging the 500 trees in subsets of 125.
This generates valid probabilities, and results in 4 sub-random forests that obey the same decomposition.
Averaging these sub-rfs also recovers the true random forest predictions.
The script analyze_bags.py  run without any parameters will spit out the
CE diversity and average individual single model CE for these 4 sub forests.
"""
import numpy as np
import os 
#here = os.path.abspath(os.path.dirname(__file__))
from pathlib import Path
import pandas as pd

def kl_diversity(probs,labels):
    """Given probs of shape (models,samples, classes) and labels of shape (samples), generate a corresponding diversity measure.  

    """
    correct_class = probs[:,np.arange(len(labels)),labels.astype(int)]+1e-10 # models,samples with a bit of padding. 
    nb_models = len(probs[:,0,0])
    norm = np.sum(correct_class,axis = 0)
    diversity = np.sum(np.log(1/nb_models)-np.log(correct_class/norm),axis = 0)*(1/nb_models)
    return diversity

def get_nll_div(all_tprobs, targets):
    all_probs = []
    for probs in all_tprobs:
        all_probs.append(probs[np.arange(len(targets)), targets])

    array_probs = np.stack(all_probs,axis = 0) # (models,samples)
    norm_term = np.log(np.mean(array_probs,axis = 0))
    diversity = -np.mean(np.log(array_probs),axis = 0)+norm_term
    #diversity = np.mean(diversity)
    return diversity

def get_avg_nll(all_probs, targets):
    all_nll = []
    for probs in all_probs:
        all_nll.append(-np.log(probs[np.arange(len(targets)), targets]))

    array_nll = np.stack(all_nll,axis = 0) # (models,samples)
    #return np.mean(np.mean(array_nll, axis=0))
    return np.mean(array_nll, axis=0)


def compare_savefig(here, depth, num_slices=1):
    """Compute the nll/ce diversity decompositions for individual models, bags and random forest classifiers. 

    """
    ## fitting bags
    ## This is the average of probabilities output by each individual tree. 
    bag_individual_probs, ytest = load_data(here,depth)

    # add offset term
    offset = 1e-10
    bag_individual_probs += offset
    bag_individual_probs /= np.sum(bag_individual_probs, axis=2, keepdims=True)
    avg_nll = get_avg_nll(bag_individual_probs, ytest)
    nll_div = get_avg_nll(bag_individual_probs, ytest)

    # slice me:
    #import pdb; pdb.set_trace()
    num_trees= bag_individual_probs.shape[0]
    slice_size = num_trees//num_slices
    grouped_probs = [np.array(bag_individual_probs[i*slice_size:(i+1)*slice_size]) for i in range(num_slices)]
    bag_individual_nll = [get_avg_nll(group, ytest) for group in grouped_probs]
    bag_diversity = [get_nll_div(group, ytest) for group in grouped_probs]
    #import pdb; pdb.set_trace()
    """    grouped_probs = [np.mean(np.array(bag_individual_probs[i*120:(i+1)*120]),axis = 0) for i in range(5)]
    bag_individual_nll = [-np.log(e[np.arange(len(ytest)),ytest.astype(int)]) for e in grouped_probs]
    grouped_vals = [np.array(bag_individual_probs[i*120:(i+1)*120]) for i in range(5)]
    #bag_diversity = [kl_diversity(e, ytest) for e in grouped_vals]
    bag_diversity = [get_nll_div(e, ytest) for e in grouped_vals]
    """

    #bag_diversity = [kl_diversity[i*120:(i+1)*120]),axis = 0) for i in range(5)]

    ## fitting random forests 
    #rf_score = brier_multi(ytest,rf_clf.predict_proba(xtest))
    #rf_individual_probs = [e.predict_proba(xtest) for e in rf_clf.estimators_]
    #rf_individual_nll = [-np.log(e.predict_proba(xtest)[np.arange(len(ytest)),ytest]) for e in rf_clf.estimators_]
    #rf_diversity = var_diversity(np.stack(rf_individual_probs,axis =0),ytest)
    ## fitting individual models
    bag_individual_nll = np.asarray(bag_individual_nll)
    bag_diversity = np.asarray(bag_diversity)
    #import  pdb; pdb.set_trace()
    return bag_individual_nll, bag_diversity #,rf_individual_nll,rf_diversity

def load_data(datadir,depth):
    probs = np.load(os.path.join(datadir,"probs_depth_{}.npy".format(depth)))
    labels = np.load(os.path.join(datadir,"ytest_depth_{}.npy".format(depth)),allow_pickle = True).astype(int)
    return probs,labels
    

if __name__ == "__main__":
    #%%
    datadir = "./"
    depths = [i+1 for i in range(12)]
    values = []
    for depth in depths:
        rf_indiv, rf_div = compare_savefig(datadir, depth)
        #import pdb; pdb.set_trace()
        #print( (rf_div- rf_indiv))
        #import pdb; pdb.set_trace()
        #print(np.mean(rf_div), np.mean(rf_indiv))
        #print(rf_div-rf_indiv)
        # dump to trees
        x = rf_div.mean(-1)
        y = rf_indiv.mean(-1)
        #x = np.mean(rf_div, 0).mean()
        #y = np.mean(rf_indiv, 0).mean()
        values.append([y, x])
        #values.append([np.mean(rf_div, 0).mean(), np.mean(rf_indiv, 0).mean()])
        #values.append([np.mean(rf_div), np.mean(rf_indiv)])
        print(y-x)
    #import pdb; pdb.set_trace()
    values = np.asarray(values).squeeze()
    outputdir="./"
    output_dir = Path("./")
    if not output_dir.exists():
        os.makedirs(output_dir)
    pd.DataFrame(values).to_csv(output_dir / f"homogeneous.csv")
    #%%

#%5
