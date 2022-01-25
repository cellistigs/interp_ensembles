"""
There are some models which fall outside a line,
How to find these? Fit Gaussian Mixture Model.
"""
#!/usr/bin/env python
# coding: utf-8
import matplotlib as mpl
mpl.rc_file_defaults()
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib as mpl
from scipy import stats

#%%

def clean_df(results):
    # assert only one dataset
    # drop data name and filename
    data_names_dict = results.groupby('Distribution').apply(lambda x: x['Dataset'].unique())
    #assert data_names_dict['InD'].size == 1
    #assert data_names_dict['OOD'].size == 1

    results = results.drop(columns=["Dataset"])
    results = results.drop(columns=["Filename"])
    results = results.drop(columns=["Model Name"])
    results = results.drop(columns=["Model Seed"])
    results = results.drop(columns=["Brier",'Top 1\% Err','NLL'])
    results["ID"] = pd.Categorical(results["ID"])
    results["Type"] = pd.Categorical(results["Type"])
    results["Distribution"] = pd.Categorical(results["Distribution"])
    results.set_index(["ID", "Type",  "Distribution"], inplace=True)
    results.columns.name = "Metric"
    results = results.unstack("Distribution").stack("Metric")
    return results

#%%
results = pd.read_csv("/data/Projects/linear_ensembles/interp_ensembles/results/metrics/imagenetv2mf_metrics.csv")
results2 = pd.read_csv(
    "/data/Projects/linear_ensembles/interp_ensembles/results/metrics/imagenetv2mf_ensemble_metrics.csv")
results = results.append(results2)

results3 = pd.read_csv("/data/Projects/linear_ensembles/interp_ensembles/results/metrics/testbed_val.csv")
results4 = pd.read_csv(
    "/data/Projects/linear_ensembles/interp_ensembles/results/metrics/testbed_imagenetv2-matched-frequency-format-val_metrics.csv")
results3 = results3.append(results4)

# drop aug
results3 = results3.iloc[[ not('aug' in mname) for mname in results3['Model Name']]]
#%%
results = clean_df(results)
results3 = clean_df(results3)

#%%
plt.plot(results['InD'].values, results['OOD'].values,'o')
plt.plot(results3['InD'].values, results3['OOD'].values,'o')

plt.tight_layout()
plt.show()
#%%
# all results

results5 = results.copy()
results5 = results5.append(results3)
results5 = results5.dropna()
#%%
plt.plot(results5['InD'].values, results5['OOD'].values,'o')

plt.tight_layout()
plt.show()
#%%


#%%
from sklearn.mixture import GaussianMixture
X =np.stack((results5['InD'].values, results5['OOD'].values)).T
gm = GaussianMixture(n_components=2, random_state=0).fit(X)

labels = gm.predict(X)
fig, ax  = plt.subplots(1,2, figsize=(8,4))

ax[1].set_title('All Models classified')
scatter = ax[1].scatter(X[:,0],X[:,1], c=labels, s=50, alpha=0.5,)
handles_, labels_ = scatter.legend_elements(prop="colors", alpha=0.6)
legend2 = ax[1].legend(handles_, labels_, loc="upper right", title="Class")

ax[0].set_title('All Models')
ax[0].scatter(results5['InD'].values, results5['OOD'].values, c='r', s=50, alpha=0.5, label='testbed')
ax[0].scatter(results['InD'].values, results['OOD'].values, c='b', s=50, alpha=0.5, label='ours')
ax[0].legend()
plt.tight_layout()
plt.show()

#%%
# which of the testbed
# use these labels to find models which agree and models which do not:
print('Models that are okay')
print(results5.iloc[labels==1], flush=True)
print('\n\n')
print('Models that are not in line')
print(results5.iloc[labels==0], flush=True)
#%%