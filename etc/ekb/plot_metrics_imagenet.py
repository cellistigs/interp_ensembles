"""
Plot results for pretrained models
"""
#%%
import os
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, '/data/Projects/linear_ensembles/interp_ensembles/etc/ekb/')
#%%
from results_class import Model, EnsembleModel
results_dir='/data/Projects/linear_ensembles/interp_ensembles/results/imagenet/logits/'
folder_names = os.listdir(results_dir)
#%% pretrained models
"""
models = [ "resnet50",  "resnet18", "vgg11", "wide_resnet50_2",
           "alexnet", "resnet101", "efficientnet_b0",
           "wide_resnet101_2","efficientnet_b1","efficientnet_b2",
            "densenet121" ,"googlenet"]


datasets = ['imagenet', 'imagenetv2mf']
data_types = ['','']
"""
models = [ "resnet50", "resnet18", "vgg11", "wide_resnet50_2", "alexnet"]
#data_type_ood = '--brightness--1'
dataset_ood = str(sys.argv[1])
data_type_ood = str(sys.argv[2])

print('hey',dataset_ood, data_type_ood, 'bye')
datasets = ['imagenet', dataset_ood]
data_types = ['',data_type_ood]
#"""

fileout='/data/Projects/linear_ensembles/interp_ensembles/results/imagenet/images/metrics/{}{}.png'.format(dataset_ood,
                                                                                                           data_type_ood)
print(fileout)

#%%
nll_dpoints = np.zeros((len(models), len(datasets)))
bsm_dpoints = np.zeros((len(models), len(datasets)))
acc_dpoints = np.zeros((len(models), len(datasets)))

for model_idx, model in enumerate(models):
    for data_idx, dataset in enumerate(datasets):
        folder_name = model + '--' + dataset + data_types[data_idx] + '.hdf5'
        print('Register {}'.format(folder_name))
        store_logits_fname = Path(results_dir + folder_name)
        model_cls = Model(model, dataset)
        model_cls.register(str(store_logits_fname))
        nll_dpoints[model_idx, data_idx] = model_cls.get_nll()
        bsm_dpoints[model_idx, data_idx] = model_cls.get_brier()
        acc_dpoints[model_idx, data_idx] = model_cls.get_accuracy()

#%% scratch models for ensemble
model_type = 'resnet50'
models_de1 = ["deepens1", "deepens2", "deepens3", "deepens4", "deepens5"]
nll_dpoints_de1 = np.zeros((len(models_de1), len(datasets)))
bsm_dpoints_de1 = np.zeros((len(models_de1), len(datasets)))
acc_dpoints_de1 = np.zeros((len(models_de1), len(datasets)))

for model_idx, model in enumerate(models_de1):
    for data_idx, dataset in enumerate(datasets):
        folder_name = model_type + '--' + dataset + '--' + model + data_types[data_idx] +'.hdf5'
        print('Register {}'.format(folder_name))
        store_logits_fname = Path(results_dir + folder_name)
        model_cls = Model(model, dataset)
        model_cls.register(str(store_logits_fname))
        nll_dpoints_de1[model_idx, data_idx] = model_cls.get_nll()
        bsm_dpoints_de1[model_idx, data_idx] = model_cls.get_brier()
        acc_dpoints_de1[model_idx, data_idx] = model_cls.get_accuracy()
#%% Compute metrics for ensembles
from itertools import combinations
ensemble_groups=list(combinations(models_de1, len(models_de1)-1))
nll_dpoints_esg = np.zeros((len(ensemble_groups), 2))
acc_dpoints_esg = np.zeros((len(ensemble_groups), 2))
bsm_dpoints_esg = np.zeros((len(ensemble_groups), 2))

for egroup_idx, ensemble_group in enumerate(ensemble_groups):
    for data_idx, dataset in enumerate(datasets):
        model_cls = EnsembleModel(model_type, dataset)
        for model_idx, model in enumerate(ensemble_group):
            folder_name = model_type + '--' + dataset + '--' + model + data_types[data_idx] +'.hdf5'
            store_logits_fname = Path(results_dir + folder_name)
            model_cls.register(str(store_logits_fname), model)
        nll_dpoints_esg[egroup_idx, data_idx] = model_cls.get_nll()
        bsm_dpoints_esg[egroup_idx, data_idx] = model_cls.get_brier()
        acc_dpoints_esg[egroup_idx, data_idx] = model_cls.get_accuracy()
#%%
from matplotlib import colors as mcolors
colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys())

def add_dpoint(f, ax, **kwargs):
    for model_idx, model in enumerate(models):
        kwargs['label']=model +"_pretrained"
        kwargs['c']=colors[model_idx]
        kwargs['linestyle']='None'
        ax.plot(f[model_idx, 0], f[model_idx, 1], **kwargs)

def add_dpoint_indiv(f, ax, **kwargs):
    for model_idx, model in enumerate(models_de1):
        kwargs['label']='{}_seeds'.format(model_type)#_seed{}'.format(model_idx)
        kwargs['marker'] ='s'
        kwargs['c']  =colors[0]
        kwargs['linestyle']='None'
        ax.plot(f[model_idx, 0], f[model_idx, 1], **kwargs)

def add_dpoint_ensemble(f,ax, **kwargs):
    for model_idx, model in enumerate(ensemble_groups):
        kwargs['label']='{}_ensembles'.format(model_type)
        kwargs['marker']='*'
        kwargs['c']=colors[0]
        kwargs['linestyle']='None'
        ax.plot(f[model_idx, 0], f[model_idx, 1], **kwargs)


#%%
fig, axarr = plt.subplots(1,3, figsize=(10,5))

# plot individual models
kwargs={
    "marker":"o",
    "label":"resnet50",
    "markerfacecolor":"none",
}
xlabel_='ind-{}'.format(datasets[0] + data_types[0])
ylabel_='ood-{}'.format(datasets[1] + data_types[1])

# add pretrained models
ax = axarr[0]
add_dpoint(acc_dpoints, ax,**kwargs)
ax.set_title('accuracy')
ax.set_xlabel(xlabel_)
ax.set_ylabel(ylabel_)

ax = axarr[1]
add_dpoint(nll_dpoints, ax,**kwargs)
ax.set_title('nll')
ax.set_xlabel(xlabel_)
ax = axarr[2]
add_dpoint(bsm_dpoints, ax,**kwargs)
ax.set_title('brier score')
ax.set_xlabel(xlabel_)
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
plt.savefig(fileout)
plt.close(fig)
#plt.show()
#%%
