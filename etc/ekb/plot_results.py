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

results_dir='/data/Projects/linear_ensembles/interp_ensembles/results/imagenet/'
folder_names = os.listdir(results_dir)
print(folder_names)
#%%
# models_available
models = [ "resnet101", "efficientnet_b0", "wide_resnet50_2", "wide_resnet101_2","efficientnet_b1","efficientnet_b2"]
datasets = ['imagenet', 'imagenetv2mf']
nll_dpoints = np.zeros((len(models), len(datasets)))
bsm_dpoints = np.zeros((len(models), len(datasets)))
acc_dpoints = np.zeros((len(models), len(datasets)))

for model_idx, model in enumerate(models):
    print('Loading model {}'.format(model))
    for data_idx, dataset in enumerate(datasets):
        folder_name=model +'--'+ dataset +'.hdf5'
        store_logits_fname = Path(results_dir + folder_name)
        with h5py.File(str(store_logits_fname), 'r') as f:
            logits_out = f['logits'][()]
            labels = f['targets'][()].astype('int')
        # calculate probs
        probs = np.exp(logits_out)/np.sum(np.exp(logits_out), 1, keepdims=True)

        # calculate metrics
        ad = AccuracyData()
        nld = NLLData()
        bsd = BrierScoreData()

        accuracy = ad.accuracy(probs, labels)
        # nll computed for correct classes
        nll = nld.nll(probs, labels, normalize=True)
        #bs = bsd.brierscore(probs, labels)
        bsm = bsd.brierscore_multi(probs, labels)
        nll_dpoints[model_idx, data_idx] = nll
        bsm_dpoints[model_idx, data_idx] = bsm
        acc_dpoints[model_idx, data_idx] = accuracy
#%%

def add_dpoint(f,ax ):
    for model_idx, model in enumerate(models):
        ax.plot(f[model_idx, 0], f[model_idx, 1], 'o', label=model)
    ax.set_xlabel('ind-imagenet')


fig, axarr = plt.subplots(1,3)
ax = axarr[0]
add_dpoint(acc_dpoints, ax)
ax.set_title('accuracy')
ax.set_xlim([0.65,0.85])
ax.set_ylim([0.65,0.85])

ax.set_ylabel('ood-imagenetv2mf')
ax = axarr[1]
add_dpoint(nll_dpoints, ax)
ax.set_title('nll')
ax.set_xlim([0.65,1.8])
ax.set_ylim([0.65,1.8])

ax = axarr[2]
add_dpoint(bsm_dpoints, ax)
ax.set_title('brier score')
ax.set_xlim([0.25,0.55])
ax.set_ylim([0.25,0.55])

plt.tight_layout()
plt.show()
#%%
