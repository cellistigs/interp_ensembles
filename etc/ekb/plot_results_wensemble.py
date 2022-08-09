"""
Plot results for pretrained models
"""
#%%
import os
from pathlib import Path
import h5py
import numpy as np
from interpensembles.metrics import AccuracyData,NLLData,CalibrationData,VarianceData,BrierScoreData
import matplotlib.pyplot as plt

results_dir='/data/Projects/linear_ensembles/interp_ensembles/results/imagenet/logits/'
folder_names = os.listdir(results_dir)
print(folder_names)
#%%

def get_metrics_ensemble(models, datasets, model_type='resnet50'):
    nll_dpoints = np.zeros((1, len(datasets)))
    bsm_dpoints = np.zeros((1, len(datasets)))
    acc_dpoints = np.zeros((1, len(datasets)))

    for data_idx, dataset in enumerate(datasets):
        probs = []
        for model_idx, model in enumerate(models):
            print('Loading model {}'.format(model))

            folder_name = model_type + '--' + dataset + '--' + model +'.hdf5'
            store_logits_fname = Path(results_dir + folder_name)
            with h5py.File(str(store_logits_fname), 'r') as f:
                logits_out = f['logits'][()]
                labels = f['targets'][()].astype('int')
            # calculate individual probs
            probs_ = np.exp(logits_out) / np.sum(np.exp(logits_out), 1, keepdims=True)

            probs.append(probs_)

        print('Average probs to find ensemble')
        probs = np.asarray(probs)
        probs = probs.sum(0)/len(models)

        #print(probs.shape)
        #print(labels.shape)
        # TODO(ekellbuch): check labels are loaded deterministically
        # calculate metrics
        ad = AccuracyData()
        nld = NLLData()
        bsd = BrierScoreData()

        accuracy = ad.accuracy(probs, labels)
        # nll computed for correct classes
        nll = nld.nll(probs, labels, normalize=True)
        # bs = bsd.brierscore(probs, labels)
        bsm = bsd.brierscore_multi(probs, labels)

        nll_dpoints[0, data_idx] = nll
        bsm_dpoints[0, data_idx] = bsm
        acc_dpoints[0, data_idx] = accuracy
    return nll_dpoints, acc_dpoints, bsm_dpoints


def get_metrics(models, datasets, model_type='resnet50'):
    nll_dpoints = np.zeros((len(models), len(datasets)))
    bsm_dpoints = np.zeros((len(models), len(datasets)))
    acc_dpoints = np.zeros((len(models), len(datasets)))

    for model_idx, model in enumerate(models):
        print('Loading model {}'.format(model))
        for data_idx, dataset in enumerate(datasets):
            folder_name = model_type + '--' + dataset + '--' + model +'.hdf5'
            store_logits_fname = Path(results_dir + folder_name)
            with h5py.File(str(store_logits_fname), 'r') as f:
                logits_out = f['logits'][()]
                labels = f['targets'][()].astype('int')
            # calculate probs
            probs = np.exp(logits_out) / np.sum(np.exp(logits_out), 1, keepdims=True)

            # calculate metrics
            ad = AccuracyData()
            nld = NLLData()
            bsd = BrierScoreData()

            accuracy = ad.accuracy(probs, labels)
            # nll computed for correct classes
            nll = nld.nll(probs, labels, normalize=True)
            # bs = bsd.brierscore(probs, labels)
            bsm = bsd.brierscore_multi(probs, labels)
            nll_dpoints[model_idx, data_idx] = nll
            bsm_dpoints[model_idx, data_idx] = bsm
            acc_dpoints[model_idx, data_idx] = accuracy
    return nll_dpoints, acc_dpoints, bsm_dpoints


def get_metrics_v0(models, datasets):
    nll_dpoints = np.zeros((len(models), len(datasets)))
    bsm_dpoints = np.zeros((len(models), len(datasets)))
    acc_dpoints = np.zeros((len(models), len(datasets)))

    for model_idx, model in enumerate(models):
        print('Loading model {}'.format(model))
        for data_idx, dataset in enumerate(datasets):
            folder_name = model + '--' + dataset + '.hdf5'
            store_logits_fname = Path(results_dir + folder_name)
            with h5py.File(str(store_logits_fname), 'r') as f:
                logits_out = f['logits'][()]
                labels = f['targets'][()].astype('int')
            # calculate probs
            probs = np.exp(logits_out) / np.sum(np.exp(logits_out), 1, keepdims=True)

            # calculate metrics
            ad = AccuracyData()
            nld = NLLData()
            bsd = BrierScoreData()

            accuracy = ad.accuracy(probs, labels)
            # nll computed for correct classes
            nll = nld.nll(probs, labels, normalize=True)
            # bs = bsd.brierscore(probs, labels)
            bsm = bsd.brierscore_multi(probs, labels)
            nll_dpoints[model_idx, data_idx] = nll
            bsm_dpoints[model_idx, data_idx] = bsm
            acc_dpoints[model_idx, data_idx] = accuracy
    return nll_dpoints, acc_dpoints, bsm_dpoints

#%% models_available
models = [ "resnet50", "resnet101", "efficientnet_b0", "wide_resnet50_2",
           "wide_resnet101_2","efficientnet_b1","efficientnet_b2",
            "alexnet" ,"densenet121" ,"googlenet" ,"resnet18" ,"vgg11"]#,"vgg13"]
datasets = ['imagenet', 'imagenetv2mf']

nll_dpoints, acc_dpoints, bsm_dpoints = get_metrics_v0(models, datasets)
#%%
models_de1 = ["deepens1", "deepens2","deepens3","deepens4","deepens5"]
nll_dpoints_de1, acc_dpoints_de1, bsm_dpoints_de1 = get_metrics(models_de1, datasets)

#%%
#TODO(ekellbuch): add plot for ensemble of models
from itertools import combinations
ensemble_groups=list(combinations(models_de1, len(models_de1)-1))
nll_dpoints_esg = np.zeros((len(ensemble_groups), 2))
acc_dpoints_esg = np.zeros((len(ensemble_groups), 2))
bsm_dpoints_esg = np.zeros((len(ensemble_groups), 2))

for egroup_idx, ensemble_group in enumerate(ensemble_groups):
    print(ensemble_group)
    nll_dpoints_e_, acc_dpoints_e_, bsm_dpoints_e_ = get_metrics_ensemble(ensemble_group, datasets)
    nll_dpoints_esg[egroup_idx] = nll_dpoints_e_
    acc_dpoints_esg[egroup_idx] = acc_dpoints_e_
    bsm_dpoints_esg[egroup_idx] = bsm_dpoints_e_
#%%
# colors=['r', 'g', 'b','y','m','k','cyan','pink','purple']*
from matplotlib import colors as mcolors

colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys())

def add_dpoint(f, ax, **kwargs):
    for model_idx, model in enumerate(models):
        #kwargs['color'] = colors[model_idx]
        kwargs['label']=model +"_pretrained"
        kwargs['c']=colors[model_idx]
        ax.plot(f[model_idx, 0], f[model_idx, 1], **kwargs)
    ax.set_xlabel('ind-imagenet')


def add_dpoint_indiv(f, ax, **kwargs):
    for model_idx, model in enumerate(models_de1):
        #kwargs['color'] = colors[model_idx]
        kwargs['label']='resnet50_seeds'#_seed{}'.format(model_idx)
        kwargs['marker']='s'
        kwargs['c']=colors[0]
        ax.plot(f[model_idx, 0], f[model_idx, 1], **kwargs)
    ax.set_xlabel('ind-imagenet')

def add_dpoint_ensemble(f,ax, **kwargs):
    for model_idx, model in enumerate(ensemble_groups):
        #kwargs['color'] = colors[model_idx]
        kwargs['label']='resnet50_ensembles'#.format(model_idx)
        kwargs['marker']='*'
        kwargs['c']=colors[0]
        ax.plot(f[model_idx, 0], f[model_idx, 1], **kwargs)

    ax.set_xlabel('ind-imagenet')

fig, axarr = plt.subplots(1,3, figsize=(10,5))

# plot individual models
kwargs={
    "marker":"o",
    "label":"resnet50",
    "markerfacecolor":"none",
}

# add pretrained models
ax = axarr[0]
add_dpoint(acc_dpoints, ax,**kwargs)
ax.set_title('accuracy')

#%%
ax.set_ylabel('ood-imagenetv2mf')
ax = axarr[1]
add_dpoint(nll_dpoints, ax,**kwargs)
ax.set_title('nll')
#%%
ax = axarr[2]
add_dpoint(bsm_dpoints, ax,**kwargs)
ax.set_title('brier score')

# add the components of the ensemble
ax = axarr[0]
add_dpoint_indiv(acc_dpoints_de1, ax, **kwargs)
ax = axarr[1]
add_dpoint_indiv(nll_dpoints_de1, ax, **kwargs)
ax = axarr[2]
add_dpoint_indiv(bsm_dpoints_de1, ax, **kwargs)

# add plot of ensembles
ax = axarr[0]
add_dpoint_ensemble(acc_dpoints_esg, ax, **kwargs)
ax = axarr[1]
add_dpoint_ensemble(nll_dpoints_esg, ax, **kwargs)
ax = axarr[2]
add_dpoint_ensemble(bsm_dpoints_esg, ax, **kwargs)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(),bbox_to_anchor=(1.05, 1.0), loc='upper left')


plt.tight_layout()
plt.show()
#%%
