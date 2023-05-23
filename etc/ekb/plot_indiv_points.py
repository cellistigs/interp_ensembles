"""
Per data point log likelihoood
"""
#%%
# picka  network and pick an ensemble
#%%

import os
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, '/data/Projects/linear_ensembles/interp_ensembles/etc/ekb/')

from results_class import Model, EnsembleModel

data_type="val"
results_dir='/data/Projects/linear_ensembles/interp_ensembles/results/imagenet/logits/'
#results_dir="/datahd3a/imagenet_testbest_outputs/logits/val/"
folder_names = os.listdir(results_dir)
# pretrained models

models = [ "resnet50",  "resnet18", "vgg11", "wide_resnet50_2",
           "alexnet", "resnet101", "efficientnet_b0",
           "wide_resnet101_2","efficientnet_b1","efficientnet_b2",
            "densenet121" ,"googlenet"]


datasets = ['imagenet', 'imagenetv2mf']
data_types = ['deepens1','']
"""
models = [ "resnet50", "resnet18", "vgg11", "wide_resnet50_2", "alexnet"]
#data_type_ood = '--brightness--1'
dataset_ood = str(sys.argv[1])
data_type_ood = str(sys.argv[2])

print('hey',dataset_ood, data_type_ood, 'bye')
datasets = ['imagenet', dataset_ood]
data_types = ['',data_type_ood]

fileout='/data/Projects/linear_ensembles/interp_ensembles/results/imagenet/images/metrics/{}{}.png'.format(dataset_ood,
                                                                                                           data_type_ood)
print(fileout)

"""


#%%
model = "resnet50"
data_idx = 0
dataset = datasets[data_idx]
folder_name = model + '--' + dataset + '--' + data_types[data_idx] + '.hdf5'
print('Register {}'.format(folder_name))
store_logits_fname = Path(results_dir + folder_name)
model_cls = Model(model, dataset)
model_cls.register(str(store_logits_fname))

base_model = model_cls

#%%
model = "resnet101"
data_idx = 0
dataset = datasets[data_idx]
folder_name = model + '--' + dataset + data_types[data_idx] + '.hdf5'
print('Register {}'.format(folder_name))
store_logits_fname = Path(results_dir + folder_name)
model_cls = Model(model, dataset)
model_cls.register(str(store_logits_fname))

single_model = model_cls
# %% scratch models for ensemble
model_type = 'resnet50'
models_de1 = ["deepens1", "deepens2", "deepens3", "deepens4", "deepens5"]

model_cls = EnsembleModel(model_type, dataset)
for model_idx, model in enumerate(models_de1):
    folder_name = model_type + '--' + dataset + '--' + model + data_types[data_idx] + '.hdf5'
    store_logits_fname = Path(results_dir + folder_name)
    model_cls.register(str(store_logits_fname), model)

ensemble_model = model_cls
#%%

x_var = single_model.probs() #.max(1)[::10]
y_var = ensemble_model.probs() #.max(1)[::10]
target = single_model.labels()
#%%
x = np.log(x_var[np.arange(len(target)), target])*-1
y = np.log(y_var[np.arange(len(target)), target])*-1


#%%
from scipy.stats import gaussian_kde

# Calculate the point density
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

fig, ax = plt.subplots()
ax.scatter(x, y, c=z, s=1)
ax.set_title('indiv point nll {}'.format(dataset))
ax.set_xlabel('resnet 101')
ax.set_ylabel('ensemble resnet 50')
plt.show()
#%%
# We can make this much more explicit,
# by also considering the per-datapoint nll of a single vgg 11 as a baseline,
# and subtracting it off. We are then asking,
# how much does A) ensembling, and B)
# the choice of a better model improve the nll of each data point?
b_var = base_model.probs() #.max(1)[::10]
baseline = np.log(b_var[np.arange(len(target)), target])*-1

x = np.log(x_var[np.arange(len(target)), target])*-1 - baseline
y = np.log(y_var[np.arange(len(target)), target])*-1 - baseline

#%%
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

fig, ax = plt.subplots()
ax.scatter(x, y, c=z, s=1)
ax.set_title('indiv point nll {}'.format(dataset))
ax.set_xlabel('resnet 101 - resnet 50')
ax.set_ylabel('ensemble resnet 50 - resnet 50')
plt.show()

#%% per data point nll

#histogram definition
bins = [500, 500] # number of bins

# histogram the data
hh, locx, locy = np.histogram2d(x, y, bins=bins)

# Sort the points by density, so that the densest points are plotted last
z = np.array([hh[np.argmax(a<=locx[1:]),np.argmax(b<=locy[1:])] for a,b in zip(x,y)])
idx = z.argsort()
x2, y2, z2 = x[idx], y[idx], z[idx]

plt.figure(1,figsize=(8,8)).clf()
s = plt.scatter(x2, y2, c=z2, cmap='viridis', marker='.')
plt.show()
#%%
import matplotlib.pyplot as plt, numpy as np, numpy.random, scipy

#histogram definition
xyrange = [[-1,30],[-1,30]] # data range
bins = [100,100] # number of bins
thresh = 3  #density threshold

#data definition
N = int(1e5)
#xdat = np.random.normal(size=N)
#ydat=np.random.normal(1, 0.6, size=N)
xdat = x#np.random.normal(size=N)
ydat=y#np.random.normal(1, 0.6, size=N)

# histogram the data
hh, locx, locy = scipy.histogram2d(xdat, ydat, range=xyrange, bins=bins)
posx = np.digitize(xdat, locx)
posy = np.digitize(ydat, locy)

#select points within the histogram
ind = (posx > 0) & (posx <= bins[0]) & (posy > 0) & (posy <= bins[1])
hhsub = hh[posx[ind] - 1, posy[ind] - 1] # values of the histogram where the points are
xdat1 = xdat[ind][hhsub < thresh] # low density points
ydat1 = ydat[ind][hhsub < thresh]
hh[hh < thresh] = np.nan # fill the areas with low density by NaNs

plt.imshow(np.flipud(hh.T),cmap='jet',extent=np.array(xyrange).flatten(), interpolation='none', origin='upper')
plt.colorbar()
plt.plot(xdat1, ydat1, '.',color='darkblue')
plt.show()
#%%