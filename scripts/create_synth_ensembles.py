## create synthetic ensembles: 
import scipy.stats
from scipy.special import softmax
import itertools
import os
import numpy as np
from argparse import ArgumentParser

here = os.path.abspath(os.path.dirname(__file__))


def main(args):
    dataset_stubs = args.resultstub 
    ensemblesize = args.ensemblesize
    if args.apply_softmax:
        get_files = lambda f: softmax(np.load(f),axis = 1)
    else:    
        get_files = lambda f: np.load(f)

    subsets = itertools.combinations(range(len(dataset_stubs)),ensemblesize)

    resultspath = os.path.join(here,"../","results")

    for si,s in enumerate(subsets):
        ind_ensemble_results = [get_files(os.path.join(resultspath,dataset_stubs[i]+"ind_preds.npy")) for i in s]
        ood_ensemble_results = [get_files(os.path.join(resultspath,dataset_stubs[i]+"{}preds.npy".format(args.ood_suffix))) for i in s]
        ind_targets = np.load(os.path.join(resultspath,dataset_stubs[0]+"ind_labels.npy"))
        ood_targets = np.load(os.path.join(resultspath,dataset_stubs[0]+"{}labels.npy".format(args.ood_suffix)))
        stacked_ind = np.stack(ind_ensemble_results,axis = 0)
        stacked_ood = np.stack(ood_ensemble_results,axis = 0)
        ind_mean = np.mean(stacked_ind,axis = 0)
        ood_mean = np.mean(stacked_ood,axis = 0)
        np.save(os.path.join(resultspath,"synth_ensemble_{}_{}_ind_preds.npy".format(si,args.nameprefix)),ind_mean)
        np.save(os.path.join(resultspath,"synth_ensemble_{}_{}_{}preds.npy".format(si,args.nameprefix,args.ood_suffix)),ood_mean)
        np.save(os.path.join(resultspath,"synth_ensemble_{}_{}_ind_labels.npy".format(si,args.nameprefix)),ind_targets)
        np.save(os.path.join(resultspath,"synth_ensemble_{}_{}_{}labels.npy".format(si,args.nameprefix,args.ood_suffix)),ood_targets)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-r","--resultstub",action = "append", help = "required, stubs of the files indicating outputs to include, given as names in `../results`.")
    parser.add_argument("-n","--nameprefix",help = "name prefix to add for this ensemble")
    parser.add_argument("-e","--ensemblesize",type = int,help = "size of ensembles to construct from results given")
    parser.add_argument("-s","--apply-softmax",dest = "apply_softmax",action = "store_true",help = "if given, we should apply softmaxes to the given pred.")
    parser.add_argument("-ns","--no-apply-softmax",dest = "apply_softmax",action = "store_false",help = "if given, we should not apply softmaxes to the given pred.")
    parser.add_argument("-os","--ood_suffix",help = "suffix to expect for ood data (default ood_preds)",default = "ood_")

    args = parser.parse_args()
    main(args)
