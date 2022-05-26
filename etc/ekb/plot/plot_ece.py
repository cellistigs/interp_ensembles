"""
Plot ece plots
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
import os
import fire


def clean_df(results):
    # assert only one dataset
    # drop data name and filename
    data_names_dict = results.groupby('Distribution').apply(lambda x: x['Dataset'].unique())
    #assert data_names_dict['InD'].size == 1
    #assert data_names_dict['OOD'].size == 1
    results = results.drop(columns=["Top 1\% Err"])
    results = results.drop(columns=["NLL"])
    results = results.drop(columns=["Brier"])
    results = results.drop(columns=["Dataset"])
    results = results.drop(columns=["Filename"])
    results = results.drop(columns=["Model Name"])
    results = results.drop(columns=["Model Seed"])
    results["ID"] = pd.Categorical(results["ID"])
    results["Type"] = pd.Categorical(results["Type"])
    results["Distribution"] = pd.Categorical(results["Distribution"])
    results.set_index(["ID", "Type",  "Distribution"], inplace=True)
    results.columns.name = "Metric"
    results = results.unstack("Distribution").stack("Metric")
    return results

mpl.rc_file_defaults()


def _linear_fit_deprecated(X, Y):
    if X.ndim == 1:
        X = X[:,None]
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y)  # perform linear regression
    Y_pred = linear_regressor.predict(X)  # make predictions

    return Y_pred


def linear_fit(x, y):
    if x.ndim == 1:
        X = x[:, None]
    lm = LinearRegression()
    lm.fit(X,y)
    params = np.append(lm.intercept_,lm.coef_)
    predictions = lm.predict(X)
    newX = np.append(np.ones((len(X),1)), X, axis=1)
    MSE = (sum((y-predictions)**2))/(len(newX)-len(newX[0]))
    var_b = MSE*(np.linalg.inv(np.dot(newX.T, newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params/ sd_b
    #print(ts_b)
    p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-len(newX[0])))) for i in ts_b]
    #print(p_values)
    return predictions, params, sd_b, ts_b, p_values


def z_score_difference(beta1, beta2, se_1, se_2):
    z = (beta1 - beta2)/np.sqrt(se_1**2 + se_2**2)
    p_value = stats.norm.sf(np.abs(z))
    return z, p_value

#%%

def main(ensemble_type='ensemble_homog'):
    # ensemble can be ensemble_homog ensemble_heter or ensemble_all (homo+her as one ensemble type)
    # if homo plot only homo
    #%%
    print('Plotting ece {}'.format(ensemble_type))
    #ensemble_type = 'ensemble_all'
    savefigdir = '/data/Projects/linear_ensembles/interp_ensembles/images/'
    resultsdir = '/data/Projects/linear_ensembles/interp_ensembles/results/metrics'
    savefigname = os.path.join(savefigdir, 'ece_metrics_{}.pdf'.format(ensemble_type))
    #%%
    all_datasets = ['cifar10.1', 'cinic10', 'imagenetv2']
    ind_datasets = ['cifar10', 'cifar10', 'imagenet']
    all_plot_results = pd.DataFrame([])
    for data_idx, run_dataset in enumerate(all_datasets):
        #run_dataset='cinic10'
        datasets = {}
        datasets['OOD'] = run_dataset
        datasets['InD'] = 'imagenet' if 'imagenet' in run_dataset else 'cifar10'

        allfiles = os.listdir(resultsdir)
        resultfiles = [file_ for file_ in allfiles if (run_dataset in file_) ]
        # Filter out heterogeneous ensembles
        if ensemble_type == 'ensemble_homog':
            resultfiles = [file_ for file_ in resultfiles if not('het_ensemble' in file_) ]

        print(resultfiles)
        # filter out heter

        # get name of ood dataset
        results = pd.DataFrame([])
        for r_ in resultfiles:
            results_ = pd.read_csv(os.path.join(resultsdir, r_))
            results = results.append(results_)

        # filter out model's which didn't achieve min performance
        # import pdb; pdb.set_trace()
        results = results[results['Top 1\% Err'] < 0.8]

        results = clean_df(results)
        results = results.dropna()
        plot_results = results.reset_index().set_index("ID")
        plot_results['OOD Data'] = run_dataset
        all_plot_results = all_plot_results.append(plot_results)

    if ensemble_type == 'ensemble_all':
        all_plot_results = all_plot_results.replace('Het. Ensemble', 'Ensemble')

    m_sizes={
        'Single Model': 25,
        'Ensemble' : 50,
        'Het. Ensemble': 50,
    }
    if ensemble_type == 'ensemble_heter':
        hue_order=['Single Model', 'Ensemble', 'Het. Ensemble']
    else:
        hue_order=['Single Model', 'Ensemble']

    colors = ['yellowgreen', 'tomato', 'dodgerblue']
    # hue is
    g= sns.FacetGrid(
        data=all_plot_results,
        #x="InD",
        # y="OOD",
        col="OOD Data", hue="Type", #style="Type",
        hue_order=hue_order,
        #kind="scatter",
        #sizes=m_sizes,
        #size="Type",
        palette=colors,
        #alpha=0.5,
        height=4,
        sharex=False,
        sharey=False,
    )

    g.map_dataframe(sns.scatterplot, x="InD", y="OOD")
    for d in range(len(all_datasets)):
        g.axes[0, d].set_xlabel('InD {}'.format(ind_datasets[d].upper()))
        g.axes[0, d].set_ylabel('OOD {}'.format(all_datasets[d].upper()))
        g.axes[0, d].set_title('ECE')


    g.add_legend()
    g.tight_layout()

    plt.savefig(savefigname)
    plt.close()

    #%%

if __name__ == "__main__":
    fire.Fire(main)
#%%


#%%
import os
dirName="/data/Projects/linear_ensembles/interp_ensembles/data/logits_miller"
listOfFiles = list()
for (dirpath, dirnames, filenames) in os.walk(dirName):
    listOfFiles += [os.path.join(dirpath, file) for file in filenames if ".npz" in file]

#%%