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
    results = results.drop(columns=["ECE"])

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

def main(run_dataset,ensemble_type='ensemble_homog'):
    #%%
    print('Running {} {}'.format(run_dataset, ensemble_type))
    #run_dataset='cinic10'
    datasets = {}
    datasets['OOD'] = run_dataset
    datasets['InD'] = 'imagenet' if 'imagenet' in run_dataset else 'cifar10'

    savefigdir = '/data/Projects/linear_ensembles/interp_ensembles/images/'
    resultsdir = '/data/Projects/linear_ensembles/interp_ensembles/results/metrics'

    savefigname = os.path.join(savefigdir, run_dataset + '_metrics_{}.pdf'.format(ensemble_type))
    savetabname1 = os.path.join(savefigdir, run_dataset + '_r2metrics_{}.tex'.format(ensemble_type))
    savetabname2 = os.path.join(savefigdir, run_dataset + '_diffmetrics_{}.tex'.format(ensemble_type))

    allfiles = os.listdir(resultsdir)
    #resultfiles = [file_ for file_ in allfiles if (run_dataset in file_) or (datasets['InD']+'_' in file_)]
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

    # drop row where error  80%
    # import pdb; pdb.set_trace()
    # filter out model's which didn't achieve min performance
    #import pdb; pdb.set_trace()
    results = results[results['Top 1\% Err'] < 0.8]

    metrics = ["0-1 Error", "NLL", "Brier", "rESCE"]
    results = results.rename(columns={"Top 1\% Err": '0-1 Error'})

    #import pdb; pdb.set_trace()
    results = clean_df(results)
    results = results.dropna()

    #metrics = ["Top 1\% Err", "NLL", "Brier", "rESCE"]
    plot_results = results.reset_index().set_index("ID")

    if ensemble_type == 'ensemble_all':
        plot_results = plot_results.replace('Het. Ensemble', 'Ensemble')

    # Plot metrics


    #metrics = ["Top 1\% Err", "NLL", "Brier", "ECE", "rESCE"]
    #model_types=['Single Model', 'Ensemble', 'Het. Ensemble']

    if ensemble_type == 'ensemble_heter':
        model_types=['Single Model', 'Ensemble', 'Het. Ensemble']
    else:
        model_types=['Single Model', 'Ensemble']

    #colors =['yellowgreen', 'tomato']
    colors = ['yellowgreen', 'tomato', 'dodgerblue']

    #colors=['deepskyblue', 'hotpink']
    #palette = "Set2"
    g = sns.FacetGrid(
        data=plot_results, col="Metric", hue="Type",
        hue_order=model_types,
        despine=False, legend_out=False,
        aspect=1, height=4.,
        sharey=False,
        sharex=False,
        col_order=metrics,
        palette=colors,
    )
    g.map_dataframe(sns.scatterplot, x="InD", y="OOD")
    #g.set(xlim=(0, None), ylim=(0, None))
    #g = g.map(plt.scatter, "Metric.Top 1% Err", "Metric.Top 1% Err")
    g.set_xlabels("InD {}".format(datasets['InD'].upper()))
    g.set_ylabels("OOD {}".format(datasets['OOD'].upper()))

    g.add_legend()
    #plt.show()

    all_scores = []
    all_pdiff = []
    for m_idx, metric_type in enumerate(metrics):
        single_models = plot_results.loc[(plot_results['Metric'] == metric_type)]
        x_total = single_models['InD'].values
        y_total = single_models['OOD'].values

        y_pred_total, params_total, sd_b_total, ts_b_total, p_values_total = linear_fit(x_total, y_total)

        r2_val_all = r2_score(y_total, y_pred_total)

        if r2_val_all > 0.1:
            g.axes[0, m_idx].plot(x_total, y_pred_total, linewidth=1, c='k',alpha=0.5, label="x=y")


        current_results = [metric_type, 'All', params_total[0], sd_b_total[0], ts_b_total[0], p_values_total[0], r2_val_all, len(y_total)]

        all_scores.append(current_results)

        # fix metric

        for t_idx, model_type in enumerate(model_types):
            # Fit to each model type
            model_metric = single_models.loc[(single_models['Type'] == model_type)]
            x = model_metric['InD'].values
            y = model_metric['OOD'].values

            y_pred, params, sd_b, ts_b, p_values = linear_fit(x, y)
            r2_val = r2_score(y, y_pred)

            if r2_val_all > 0.1:
                g.axes[0, m_idx].plot(x, y_pred, linewidth=1, c=colors[t_idx], alpha=0.5)

            current_results = [metric_type, model_type, params[0], sd_b[0], ts_b[0],p_values[0], r2_val, len(y)]
            all_scores.append(current_results)

            z_value_diff, p_value_diff = z_score_difference(params_total[0], params[0], sd_b_total[0], sd_b[0])

            current_pdiff = [metric_type, model_type, params_total[0], params[0], sd_b_total[0], sd_b[0], z_value_diff, p_value_diff, len(y)]
            all_pdiff.append(current_pdiff)

        # ax = g.axes[0, m_idx]
        # xy_ = np.linspace(*ax.get_xlim())
        # g.axes[0, m_idx].plot(xy_, xy_, 'k--')
        lims_x = min(x_total.min(), y_total.min())
        lims_y = min(x_total.max(), y_total.max())
        g.axes[0, m_idx].axline( [lims_x, lims_x], [lims_y, lims_y], color='k', linestyle='--', linewidth=1)  #, alpha=0.75, linewidth=0.75)
        #g.axes[0,m_idx].axline([1, 1], [2, 2], color='k', linestyle='--', linewidth=1)  #, alpha=0.75, linewidth=0.75)

    g.add_legend()
    plt.tight_layout()
    #plt.show()
    #%%
    plt.savefig(savefigname)
    plt.close()

    def polish_pd(all_pdiff1):
        all_pdiff1 = all_pdiff1.round(3)
        all_pdiff1["Metric"] = pd.Categorical(all_pdiff1["Metric"])
        all_pdiff1["Type"] = pd.Categorical(all_pdiff1["Type"])
        all_pdiff1.set_index(["Metric", "Type"], inplace=True)
        return all_pdiff1

    all_scores1 = pd.DataFrame(all_scores, columns=['Metric','Type','Coefficient','Std. error','t-statistic', 'p-value','R^2', 'Number of models'])
    all_scores1 = polish_pd(all_scores1)
    print('\n\n' + run_dataset + '\n\n')
    print(all_scores1['R^2'].unstack()['All'], results.shape[0]/4)
    print('\n\n' + run_dataset + '\n\n')
    print(all_scores1.to_latex(savetabname1, multirow=True,sparsify=True,multicolumn=True))
    all_pdiff1 = pd.DataFrame(all_pdiff, columns=['Metric','Type','beta_all','beta_type','Std. error','se_type','z-score','p-value','Number of models'])
    all_pdiff1 = polish_pd(all_pdiff1)
    print('\n\n' + run_dataset + '\n\n')
    print(all_pdiff1.to_latex(savetabname2,multirow=True,sparsify=True,multicolumn=True))

    #%%

if __name__ == "__main__":
    fire.Fire(main)


#%%
"""
def clean_df(results):
    # assert only one dataset
    # drop data name and filename
    data_names_dict = results.groupby('Distribution').apply(lambda x: x['Dataset'].unique())
    #assert data_names_dict['InD'].size == 1
    #assert data_names_dict['OOD'].size == 1

    results = results.drop(columns=["Dataset"])
    results = results.drop(columns=["Filename"])
    #results = results.drop(columns=["Model Name"])
    results = results.drop(columns=["Model Seed"])
    results["ID"] = pd.Categorical(results["ID"])
    results["Type"] = pd.Categorical(results["Type"])
    results["Model Name"] = pd.Categorical(results["Model Name"])
    results["Distribution"] = pd.Categorical(results["Distribution"])
    results.set_index(["ID", "Model Name", "Type",  "Distribution"], inplace=True)
    #results.set_index(["ID", "Type",  "Distribution"], inplace=True)
    results.columns.name = "Metric"
    results = results.unstack("Distribution").stack("Metric")
    return results

# check for Taiga


for model_name in [0]:#['resnet50','alexnet','resnet101']:
    results = pd.read_csv('/data/Projects/linear_ensembles/interp_ensembles/results/metrics/imagenetv2mf_metrics.csv')
    data2 = pd.read_csv(
        '/data/Projects/linear_ensembles/interp_ensembles/results/metrics/imagenetv2mf_ensemble_metrics.csv')
    results = results.append(data2)

    # filter results only for alexne
    #results = results[results['Model Name']==model_name]

    results = clean_df(results)
    results = results.dropna()
    plot_results = results.reset_index().set_index("ID")

    metrics = ["Top 1\% Err", "NLL", "Brier", "ECE", "rESCE"]
    # model_types=['Single Model', 'Ensemble', 'Het. Ensemble']
    model_types = ['Single Model', 'Ensemble']

    g = sns.FacetGrid(
        data=plot_results, col="Metric", hue="Type",
        hue_order=model_types,
        despine=False, legend_out=False,
        aspect=1, height=4.,
        sharey='col',
        sharex='col',
        col_order=metrics,
        row='Model Name'
    )
    g.map_dataframe(sns.scatterplot, x="InD", y="OOD")
    #g.set(xlim=(0, 0.7), ylim=(0, 0.7))
    # g = g.map(plt.scatter, "Metric.Top 1% Err", "Metric.Top 1% Err")
    #g.set_xlabels("InD {}".format(model_name))
    #g.set_ylabels("OOD {}".format(model_name))
    g.add_legend()
    plt.tight_layout()
    plt.show()
"""
#%%