"""
want to use the class Taiga created called VarianceData
"""
from interpensembles.metrics import VarianceData
import h5py
from pathlib import Path
import numpy as np

# TODO: support multiple types of ood classes
#%% specify what models should start with
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


# %%
for data in datasets:
    all_data = {}
    data_cls_ind = create_cls('ind', "imagenet")
    all_data['ind'] = data_cls_ind
    data_cls_ind = create_cls('ood', data)
    all_data['ood'] = data_cls_ind

    # alldata = {datas: {"ind": VarianceData, "ood": VarianceData}}
    alldata = {data: all_data}

#%% debug variance data is not suscr

predslabels = all_data['ind']

#%%

def get_mean_model_brier(predslabels):
    """Assume predslabels is a dictionary with keys giving modelnames and values dictionaries with keys "labels","preds".

    we define the the mean ensemble Brier score as:
    $A = \mathcal{E}[(p_ik-\mu_k)^2]$
    :returns: array of shape 10000, that gives the mean model brier score in each dim.
    """
    #%% first, calculate a mu_k:
    factors = []
    probs = []
    for model in predslabels.models:
        labels = predslabels.models[model]["labels"]
        prob = predslabels.models[model]["preds"]
        classes = max(labels+1)
        #print(classes)
        mu_k = np.array([len(np.where(labels==i)[0])/len(labels) for i in range(classes)]) ## gives mu_k for each for index k.
        y_onehot = np.zeros(prob.shape)
        y_onehot[np.arange(len(labels)), labels] = 1
        deviance = prob-y_onehot
        #brier_factor = np.sum(deviance**2,axis = 1)
        brier_factor = np.mean(deviance**2,axis = 1)

        #brier_factor = (predslabels.models[model]["preds"]-mu_k)**2 ## (examples,classes) - (classes,)
        factors.append(brier_factor)
        probs.append(prob)
    #%%
    mean_model_brier = np.mean(np.stack(factors,axis = 0),axis = 0) ## shape (examples,classes)
    #normalization = (mu_k)*(1-mu_k)
    #aleatoric_uncertainty = normalization - mean_model_brier ## shape (examples,classes)
    return mean_model_brier#,normalization

#%%
modeldata = all_data
dataclass = 'ind'
#%%
predslabels = modeldata[dataclass]
normed = {}
mean_model_brier = get_mean_model_brier(predslabels)*1000#/1000
variance = modeldata[dataclass].variance()*1000
variance_model = np.mean(variance, axis=1)
normed[dataclass] = (mean_model_brier, variance_model)

#%%
from scipy.stats import gaussian_kde
kernel = gaussian_kde(normed[dataclass])
#%%
import matplotlib.pyplot as plt
fig, axarr = plt.subplots(1,2)

ax = axarr[0]
ymin = 0
ymax= 1#/1000
ax.plot(normed[dataclass][1] , normed[dataclass] [0],'o', markersize=0.5)
idline = np.linspace(ymin, ymax, 100)
ax.plot(idline, idline, "--", color="black", label="y=x")
# now the kde plot
from interpensembles.uncertainty import BrierScoreMax
from matplotlib.colors import SymLogNorm

num_classes = 1#000
scale_kernel_grid = 0.1  # 500
ensemble_size = 5
bsm = BrierScoreMax(num_classes)
eps = 0
maxpoints_uncorr = bsm.get_maxpoints_uncorr(ensemble_size)
maxpoints_corr = bsm.get_maxpoints_corr(ensemble_size)
xmin, xmax = 0, 2 #np.min(maxpoints_uncorr[:, 1]), np.max(maxpoints_uncorr[:, 1]) + eps
ymin, ymax = 0, 1 #np.min(maxpoints_uncorr[:, 0]), np.max(maxpoints_uncorr[:, 0]) *1000+ eps
space_xx = 50j #abs(xmax - xmin) * scale_kernel_grid)* 1j
space_yy = 50j #int(abs(ymax - ymin) * scale_kernel_grid)* 1j
xx, yy = np.mgrid[xmin:xmax: space_xx , ymin:ymax:space_yy ]

#%%
print(xx)
eps = 0  # 1e-2
samplepositions = np.vstack([xx.ravel(), yy.ravel()])

kernel = gaussian_kde(normed[dataclass])
f = np.reshape(kernel(samplepositions).T, xx.shape)

ax = axarr[1]
axlognorm = ax.matshow(f,
                       cmap="RdBu_r",
                       #extent=[ymin - eps, ymax + eps, xmin - eps, xmax + eps],
                       norm=SymLogNorm(linthresh=1e-3, vmin=np.min(f), vmax=np.max(f)),
                       aspect="auto",
                       origin="lower",
                       )


plt.tight_layout()
plt.show()

#%%
import seaborn as sns

df= {}
df['mean'] = mean_model_brier
df['variance'] = variance_model

sns.jointplot(x=df["variance"], y=df["mean"], kind='kde')

plt.show()

#%%
# Plot marginal:
sns.jointplot(x=df["variance"], y=df["mean"], kind='hex', marginal_kws=dict(bins=10, fill=True))

plt.show()
#%%

#%
# marginal kde estimation on for alleatoric uncertainty
#plt.hist(variance_model, density=True)
#plt.show()
plt.hist(variance_model)
plt.show()
#%%