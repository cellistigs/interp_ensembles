import scipy.stats
import os
import numpy as np

## create synthetic ensembles: 
here = os.path.abspath(os.path.dirname(__file__))

if __name__ == "__main__":

    subsets = [[0,1,2,3],[1,2,3,4],[2,3,4,5],[3,4,5,0]]

    resultspath = os.path.join(here,"../","results")
    dataset_stubs = [
            "robust_results11-10-21_23:33.14_base_resnet18",
            "robust_results11-10-21_23:34.02_base_resnet18",
            "robust_results11-10-21_23:34.24_base_resnet18",
            "robust_results11-10-21_23:34.43_base_resnet18",
            "robust_results11-10-21_23:35.02_base_resnet18",
            "robust_results11-10-21_23:35.21_base_resnet18",
            ]

    for si,s in enumerate(subsets):
        ind_ensemble_results = [np.load(os.path.join(resultspath,dataset_stubs[i]+"ind_preds.npy")) for i in s]
        ood_ensemble_results = [np.load(os.path.join(resultspath,dataset_stubs[i]+"ood_preds.npy")) for i in s]
        ind_targets = np.load(os.path.join(resultspath,dataset_stubs[0]+"ind_labels.npy"))
        ood_targets = np.load(os.path.join(resultspath,dataset_stubs[0]+"ood_labels.npy"))
        stacked_ind = np.stack(ind_ensemble_results,axis = 0)
        stacked_ood = np.stack(ood_ensemble_results,axis = 0)
        ind_mean = np.mean(stacked_ind,axis = 0)
        ood_mean = np.mean(stacked_ood,axis = 0)
        np.save(os.path.join(resultspath,"synth_ensemble_{}_resnet18_ind_preds.npy".format(si)),ind_mean)
        np.save(os.path.join(resultspath,"synth_ensemble_{}_resnet18_ood_preds.npy".format(si)),ood_mean)
        np.save(os.path.join(resultspath,"synth_ensemble_{}_resnet18_ind_labels.npy".format(si)),ind_targets)
        np.save(os.path.join(resultspath,"synth_ensemble_{}_resnet18_ood_labels.npy".format(si)),ood_targets)


