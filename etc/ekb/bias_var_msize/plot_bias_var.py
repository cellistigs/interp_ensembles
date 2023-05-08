"""
Having used
/data/Projects/linear_ensembles/interp_ensembles/scripts/paperfigs/estimate_biasvar_ce_allmodels.py
to store the bias and variance estimates for different ensembles read those and plot them

Plot the bias-variance tradeoff for the different ensembles for cifar-10 and imagenet
"""

#%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
from matplotlib.ticker import MaxNLocator
import fire

import matplotlib

BASE_DIR = Path("/data/Projects/linear_ensembles/interp_ensembles")
plt.style.use(os.path.join(str(BASE_DIR), "etc/config/stylesheet.mplstyle"))

output_dir = BASE_DIR / "results"

#%
def read_dataset_imagenet(datafile):
  # % ret het binned ensemble
  data3 = pd.read_csv(datafile, index_col=False, header=0)
  # filter het
  data3 = data3[data3['type']=='het']
  return data3


def get_maxxy(data, datas):
  max_y_value = np.ceil(np.max([data[:,0].max(),datas[:,0].max()]))
  max_x_value = np.ceil(np.max([data[:,1].max(),datas[:,1].max()]))
  max_x_value = np.max([max_y_value/2.5, max_x_value])
  return max_x_value, max_y_value

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

#%%

def main(dataset='imagenet', foldername="bias_var_msize/ens_binned_values_scored_parallel"):

  #%%
  #dataset="imagenet_c_gaussian_noise_5"
  #foldername="bias_var_msize/ens_binned_values_scored_parallel"
  #%%
  data_filename = output_dir / foldername / "{}.csv".format(dataset)
  all_datas = read_dataset_imagenet(data_filename)
  output_filename = output_dir / "figures" / foldername / "{}.pdf".format(dataset)

  os.makedirs(os.path.dirname(output_filename), exist_ok=True)
  print('Output stored at {}'.format(output_filename))
  #%
  #max_x_value, max_y_value = get_maxxy(data, datas)
  #best_index = np.argmin(datas[:,2])
  #slope = datas[best_index,1]/datas[best_index,0]
  #best_offset= datas[best_index,0] - datas[best_index,1]


  prediction, params = linear_fit(all_datas['bias'].values, all_datas['var'].values)

  #%%
  all_subdatas=all_datas.copy()

  # color given ensemble performance
  """
  colors = ['red', 'green','blue']
  cmap = matplotlib.colors.ListedColormap(colors)
  assign_cls = lambda x: np.digitize(x, [0, 0.8, 1.6])
  label_flag = assign_cls(all_subdatas['perf'].values)
  kwargs={
        's': 15,
        'alpha': 0.5,
        'c': label_flag,
        'cmap': cmap,
  }
  """
  ens_size = all_subdatas['ens_size'].values
  ens_size = np.unique(ens_size)
  assert len(ens_size) == 1
  ens_size = ens_size[0]

  # color given score
  colors = ['blue', 'purple','green','red']
  #colors = ['blue','red']
  cmap = matplotlib.colors.ListedColormap(colors)
  #cmap = matplotlib.cm.get_cmap('cool')
  # here 40 bc ensemble size is 4
  max_score = ens_size*10
  assign_cls = lambda x: np.digitize(x, np.linspace(0, max_score, 10)) #[5, 9, 11, 12, 14])
  label_flag = assign_cls(all_subdatas['score'].values)
  #label_flag = all_subdatas['score'].values/16
  kwargs={
        's': 15,
        'alpha': 0.3,
        'c': label_flag,
        'cmap': cmap,
        #'vmin':0,
        #'vmax': 10,
  }
  print(label_flag)

  #%
  bias_min = (all_subdatas['bias'].values).min().round(1) - 0.5
  bias_max = (all_subdatas['bias'].values).max().round(2) + 0.5
  var_max = (all_subdatas['var'].values).max().round(2) + 0.5
  per_max = (all_subdatas['perf'].values).max().round(2) + 0.5
  #%
  fig, axarr = plt.subplots(1, 6, figsize=(25,5))
  # avg single model performancce vs diversity
  ax = axarr[0]
  ax.scatter(all_subdatas['var'].values, all_subdatas['bias'].values, **kwargs)

  offset = max(np.round(bias_max / 5, 1), 0.5)
  line = np.linspace(0, 10, 10)
  for i in range(19):
    ax.plot(line, line + i*offset, alpha=0.1, color="black")
    ax.plot(line + i*offset, line, alpha=0.1, color="black")
  ax.set_xlim([0, var_max])
  ax.set_ylim([0, bias_max])
  ax.xaxis.set_major_locator(MaxNLocator(3, prune='lower'))
  ax.yaxis.set_major_locator(MaxNLocator(3))
  ax.set_xlabel('Jensen gap')#,  y=0.10)
  ax.set_title('Avg. model CE')#,  y=0.45)

  handles, labels = ax.get_legend_handles_labels()

  # --------------------------------------------
  #1. [] avg. single model vs. ens. performance
  ax = axarr[1]
  ax.scatter(all_subdatas['perf'].values, all_subdatas['bias'].values, **kwargs)
  ax.set_title('Avg. model CE')
  ax.set_xlabel('Ensemble \n CE')#,  y=0.45)
  ax.plot(np.linspace(0, 100, 100), np.linspace(0, 100, 100))
  ax.set_xlim([0, per_max])
  ax.set_ylim([0, bias_max])
  ax.xaxis.set_major_locator(MaxNLocator(3, prune='lower'))
  ax.yaxis.set_major_locator(MaxNLocator(3))

  # --------------------------------------------
  # diversity vs ens performance
  ax = axarr[2]
  ax.scatter(all_subdatas['perf'].values,all_subdatas['var'].values, **kwargs)
  ax.set_title('Jensen gap')#,  y=0.10)
  ax.set_xlabel('Ensemble \n CE')#,  y=0.45)
  ax.plot(np.linspace(0, 100, 100), np.linspace(0, 100, 100))
  ax.set_xlim([0, per_max])
  ax.set_ylim([0, var_max])
  ax.xaxis.set_major_locator(MaxNLocator(3, prune='lower'))

  # --------------------------------------------
  # player score vs ens performance
  ax = axarr[3]
  ax.scatter(all_subdatas['perf'].values, all_subdatas['score'].values, **kwargs)
  ax.set_title(r'$\Sigma$ player scores')  # ,  y=0.10)
  ax.set_xlabel('Ensemble \n CE ')  # ,  y=0.45)
  ax.set_ylim([0, max_score])
  #--------------------------------------------
  # num_params vs ens performance
  ax = axarr[5]
  ax.scatter(all_subdatas['perf'].values,all_subdatas['num_params'].values ,**kwargs)
  ax.set_title('N params')#,  y=0.10)
  ax.set_xlabel('Ensemble \n CE ')#,  y=0.45)
  #ax.set_yscale('log')
  #ax.set_xscale('log')
  #ax.set_xlim([0, 3])
  ax.xaxis.set_major_locator(MaxNLocator(3, prune='lower'))
  ax.yaxis.set_major_locator(MaxNLocator(3))

  ax = axarr[4]
  ax.scatter(all_subdatas['var'].values, all_subdatas['score'].values, **kwargs)
  ax.set_xlabel('Jensen gap')#,  y=0.10)
  ax.set_title(r'$\Sigma$ player scores')  # ,  y=0.10)
  #ax.set_yscale('log')
  #ax.set_xscale('log')
  #ax.set_xlim([0, 3])
  ax.set_ylim([0, max_score])
  ax.set_xlim([0, var_max])
  ax.yaxis.set_major_locator(MaxNLocator(3))
  ax.xaxis.set_major_locator(MaxNLocator(3, prune='lower'))

  fig.tight_layout()
  fig.subplots_adjust(top=0.8)
  #fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.95), frameon=False)
  plt.savefig(output_filename, bbox_inches='tight')
  #plt.show()
  plt.close()
  #%%
  return

#%%
if __name__ == "__main__":
    fire.Fire(main)
#%%

