"""
Having used
/data/Projects/linear_ensembles/interp_ensembles/scripts/paperfigs/estimate_biasvar_ce_allmodels.py
to store the bias and variance estimates for different ensembles read those and plot them

Plot the bias-variance tradeoff for the different ensembles for cifar-10 and imagenet
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
import matplotlib as mpl

here = "/data/Projects/linear_ensembles/interp_ensembles/" #os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

plt.style.use(os.path.join(here,"etc/config/stylesheet.mplstyle"))

#%%
imagenet_data_dir = Path("/data/Projects/linear_ensembles/interp_ensembles/results/biasvar/Imagenet Heterogeneous")
cifar10_data_dir = Path("/data/Projects/linear_ensembles/interp_ensembles/results/biasvar/CIFAR 10")
mnist_data_dir = Path("/data/Projects/linear_ensembles/interp_ensembles/results/biasvar/MNIST")

#%%

def read_dataset_imagenet(data_dir):
  datas = []
  files = os.listdir(data_dir)
  files = [f for f in files if f.endswith(".csv") and f.startswith("values_ens4")]

  for file in files:
    datafile = data_dir / file
    data = pd.read_csv(datafile).iloc[:, 1:].values
    datas.append(data)
  datas = np.concatenate(datas)
  # %
  # read homogeneous values
  datafile = data_dir / "homogeneous_values_ens4.csv"
  data = pd.read_csv(datafile).iloc[:, 1:].values

  # % ret het binned ensemble
  datafile = data_dir / "heterogeneous_binned_values6_ens4.csv"
  data2 = pd.read_csv(datafile).iloc[:, 1:].values
  datafile = data_dir / "heterogeneous_binned_values3_ens4.csv"
  data3 = pd.read_csv(datafile).iloc[:, 1:].values
  datafile = data_dir / "heterogeneous_binned_values5_ens4.csv"
  data4 = pd.read_csv(datafile).iloc[:, 1:].values
  # datas= np.concatenate((datas, data2, data3, data4))
  # datas = np.concatenate((datas, data3, data4))
  # datas = np.concatenate((datas, data2, data3, data4))
  datas = np.concatenate((datas, data2, data3, data4))
  return data, datas


def read_dataset_cifar10(data_dir):
  datas = []
  files = os.listdir(data_dir)
  files = [f for f in files if f.endswith(".csv") and f.startswith("values_ens4_")]

  for file in files:
    datafile = data_dir / file
    data = pd.read_csv(datafile).iloc[:,1:].values
    datas.append(data)
  datas = np.concatenate(datas)

  # read homogeneous values
  datafile = data_dir / "homogeneous_values_ens4.csv"
  data = pd.read_csv(datafile).iloc[:, 1:].values

  return data, datas

def get_maxxy(data, datas):
  max_y_value = np.ceil(np.max([data[:,0].max(),datas[:,0].max()]))
  max_x_value = np.ceil(np.max([data[:,1].max(),datas[:,1].max()]))
  max_x_value = np.max([max_y_value/2.5, max_x_value])
  return max_x_value, max_y_value
#%

data, datas = read_dataset_imagenet(imagenet_data_dir)
data0, datas0 = read_dataset_cifar10(cifar10_data_dir)

#%
marker_size=12
#%
max_x_value, max_y_value = get_maxxy(data, datas)
best_index = np.argmin(datas[:,2])
slope = datas[best_index,1]/datas[best_index,0]
best_offset= datas[best_index,0] - datas[best_index,1]


# fit all these points using linear regression
from sklearn.linear_model import LinearRegression
def linear_fit(x, y):
  if x.ndim == 1:
    X = x[:, None]
  lm = LinearRegression()
  lm.fit(X, y)
  params = np.append(lm.intercept_, lm.coef_)
  predictions = lm.predict(X)
  return predictions, params

all_datas = np.concatenate((data, datas))
all_datas = all_datas[1:]
prediction, params = linear_fit(all_datas[:,1], all_datas[:,0])
#%
fig, axarr = plt.subplots(1, 3, figsize=(12, 5))
ax = axarr[2]
ax.scatter(data[:, 1], data[:, 0], s=10, label="homogeneous")
ax.scatter(datas[:, 1], datas[:, 0], color="orange", s=marker_size, label="homogeneous", alpha=0.5)
#ax.scatter(data[:, 1], data[:, 0], s=10, color='tab:blue')
#ax.scatter(datas[:, 1], datas[:, 0],  s=10, color='tab:blue')#color="orange", s=10, label="homogeneous", alpha=0.5)

#ax.scatter(datas[:, 1], datas[:, 0], color="orange", s=10, label="heterogeneous", alpha=0.5)
#ax.plot(np.linspace(0, 3, 100), np.linspace(0, 3, 100), label="perfect ensemble")

line = np.linspace(0, 100, 100)
for i in range(19):
  ax.plot(line, line + i*0.5, alpha=0.1, color="black")
  ax.plot(line + i*0.5, line, alpha=0.1, color="black")
ax.plot(line, line + best_offset + 1e-10, "--", color="black", label="best ensemble", alpha=0.5)

#ax.plot(line, line*params[1]+ params[0], "--", color="red", label="all ensemble fit", alpha=0.5)

#ax.set_xlabel("CE Jensen gap (pred. diversity)")
#ax.set_ylabel("Avg. single model loss (CE)")
ax.set_xlim([0, 0.7])
ax.set_ylim([0, max_y_value])
ax.set_title('ImageNet \n Deep Ensembles')
ax.xaxis.set_major_locator(MaxNLocator(3, prune='lower'))
ax.yaxis.set_major_locator(MaxNLocator(3))


#% same plot for cifar10
data = data0.copy()
datas = datas0.copy()
nan_mask = np.any(np.isnan(data),-1)
data= data[~nan_mask]
nan_mask = np.any(np.isnan(datas),-1)
datas= datas[~nan_mask]
max_x_value, max_y_value = get_maxxy(data, datas)
best_index = np.argmin(datas[:,2])
slope = datas[best_index,1]/datas[best_index,0]
best_offset= datas[best_index,0] - datas[best_index,1]
#%
all_datas = np.concatenate((data, datas))
all_datas = all_datas[1:]
prediction, params = linear_fit(all_datas[:,1], all_datas[:,0])
#%
ax = axarr[1]
ax.scatter(data[:, 1], data[:, 0], s=marker_size, label="homogeneous")
ax.scatter(datas[:, 1], datas[:, 0], s=marker_size, label="heterogeneous", alpha=0.5, color="orange")
#ax.scatter(data[:, 1], data[:, 0], s=10, color='tab:blue')#, label="homogeneous")
#ax.scatter(datas[:, 1], datas[:, 0], s=10, color='tab:blue')#, label="heterogeneous", alpha=0.5, color="orange")
line = np.linspace(0, 100, 100)
line_rate = 0.2
for i in range(19):
  ax.plot(line, line + i*line_rate, alpha=0.1, color="black")
  ax.plot(line + i*line_rate, line, alpha=0.1, color="black")

ax.plot(line, line + best_offset, "--", color="black", label="best ensemble", alpha=0.5)

#ax.plot(line, line*params[1]+ params[0], "--", color="red", label="all ensemble fit", alpha=0.5)

#ax.set_xlabel("CE Jensen gap (pred. diversity)")
#ax.set_ylabel("Avg. single model CE")
ax.set_xlim([0, 0.25])
ax.set_ylim([0, 0.7])
ax.set_title('CIFAR10 \n Deep Ensembles')

ax.xaxis.set_major_locator(MaxNLocator(3, prune='lower'))
ax.yaxis.set_major_locator(MaxNLocator(4))
#$ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False)
#ax.legend(loc="lower right",bbox_to_anchor=(0.0, 0.0), ncol=3, frameon=True)
#ax.legend(loc="upper left",bbox_to_anchor=(0.0, 0.0), ncol=3, frameon=True)
#ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5), ncol=3, frameon=True)
#ax.legend(loc='lower left', bbox_to_anchor=(0, 1.05, 0, 0.2), ncol=3, frameon=True, mode="expand")
#plt.legend()
#ax.legend(loc='lower center')#, bbox_to_anchor=(0.7, 0.0),
          #fancybox=True, shadow=False, ncol=3)
handles, labels = ax.get_legend_handles_labels()

#fig.legend(handles, labels, bbox_to_anchor=(0.9, 0), loc="lower right",
#                bbox_transform=fig.transFigure, ncol=3, borderaxespad=0,
#                panhandled=0, columnspacing=0.2, labelspacing=0.2)
"""
fig.legend(handles, labels, bbox_to_anchor=(0.53, 0.94), loc="center",
                bbox_transform=fig.transFigure, ncol=3, borderaxespad=0,
                handletextpad=0, columnspacing=0.25, labelspacing=0.2,
                fancybox=True,
              markerscale=3., handlelength=0.9)
"""
#ax.legend(loc='upper left')
#plt.tight_layout()
#fig.tight_layout()
#%

#%
#fig, axarr = plt.subplots(1,2, figsize=(10,5))
ax = axarr[0]
data_mnist = pd.read_csv(mnist_data_dir/"homogeneous.csv")
data_mnist = np.asarray(data_mnist.values)[:,1:][:10]
# drop worse to zxoom in

# make blue to red
#ax.scatter(data_mnist[:,1], data_mnist[:,0], s=marker_size, label="homogeneous", cmap='coolwarm')
z= np.arange(1,11,1)*10#+10
norm = mpl.colors.Normalize(vmin=1, vmax=12)

ax.scatter(data_mnist[:,1], data_mnist[:,0],
           #c=z,
           s=z,#s=marker_size,
           label="homogeneous",
           #cmap='coolwarm', vmin=1, vmax=10,
           alpha=1)

ensemble_p = data_mnist[:,0] - data_mnist[:,1]
best_indices = np.argsort(ensemble_p)#[::1]
best_index = best_indices[0]
#best_index = np.argmin(ensemble_p)
slope = data_mnist[best_index,0]/data_mnist[best_index,1]
best_offset= ensemble_p[best_index] #data_mnist[best_index,0] - data_mnist[best_index,1]
#best_offset=-4
line = np.linspace(0, 100, 100)
line_rate = 2.0
for i in range(19):
  ax.plot(line, line + i*line_rate, alpha=0.1, color="black")
  ax.plot(line + i*line_rate , line, alpha=0.1, color="black")

#best_index=12
#ax.plot(data_mnist[best_index,1], data_mnist[best_index,0], 'ro')
ax.plot(line, line + best_offset, "--", color="black", label="best ensemble", alpha=0.5)
ax.set_ylabel("Avg. single model CE")
ax.set_title("MNIST \n Random Forests")
print(data_mnist.max(0))
ax.set_xlim([0, 5])
ax.set_ylim([0, 5])
ax.yaxis.set_major_locator(MaxNLocator(3))
ax.xaxis.set_major_locator(MaxNLocator(3, prune='lower'))

#ax.legend()
fig.legend(handles, labels, bbox_to_anchor=(0.5, 1.05), loc="center",
                bbox_transform=fig.transFigure,
                ncol=3,
                borderaxespad=0,
                handletextpad=1,
                columnspacing=0.25, labelspacing=0.2,
                fancybox=True,
              markerscale=3., handlelength=0.9)
fig.supxlabel('CE Jensen gap (pred. diversity)',  y=0.12)
fig.subplots_adjust(bottom=0.1)
#fig.subplots_adjust(bottom=0.2, top=0.62, left=0.1, right=0.9, wspace=0.3)
#plt.savefig("biasvar_cifar10_imagenet_mnist.pdf", bbox_inches='tight')

#fig.tight_layout()
plt.tight_layout()
#fig.tight_layout(h_pad=0.25, w_pad=0.5, top=0.9)
plt.savefig("biasvar_cifar10_imagenet_mnist.pdf", bbox_inches='tight')
plt.show()
#%%
