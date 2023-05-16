"""
Make ensembles for imagenet, and calculate bias/variance for each ensemble size.
"""
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np
from itertools import combinations
import random
from tqdm import tqdm
from interpensembles.predictions import EnsembleModel
import multiprocessing
from functools import partial
import fire
import yaml
import os

BASE_DIR = Path("/data/Projects/linear_ensembles/interp_ensembles")


output_dir = BASE_DIR / "results"

# read size of models trained on imagenet
def read_model_size_file(dataset):
    if 'imagenet' in dataset:
        dataset = 'imagenet'
    model_size_file = BASE_DIR / "scripts/script_configs/datasets/{}/model_num_params.yaml".format(dataset)
    with open(model_size_file) as f:
        my_dict = yaml.safe_load(f)
    return my_dict


def get_player_score(perf):
    player_perf_bins = np.linspace(0, 1, 10)
    return np.digitize(perf, player_perf_bins)


#%%
def main(binning=6, ensemble_size=4, max_mpclass=11, max_enspclass=11, serial_run=False, seed=None,
         dataset='imagenet', out_name="bias_var_msize/ens_binned_values_scored_parallel"):

    #%%
    # get a random seed
    if seed is None:
        seed = np.random.randint(0, 100000)
    random.seed(seed)
    #%% read metrics_file
    metrics_file = BASE_DIR / "results/model_performance/{}.csv".format(dataset)

    ind_models = pd.read_csv(metrics_file)

    #%% Filter out models in terms of performance
    err_values = 1 - ind_models['acc'].values

    # Split models in 10% err rate performance groups
    if binning == 3:
        err_bins = np.linspace(0, 1, 6)
    elif binning == 1:
        # for het_ensemble_2 now use smaller bins
        err_bins = np.linspace(0, err_values.max() + 0.01, 11)
    elif binning == 3:
        # for het_ensemble_2 now use smaller bins
        err_bins = np.linspace(0, 1, 4)
    elif binning == 4:
        # for het_ensemble_2 now use smaller bins
        err_bins = np.linspace(0, err_values.max() + 0.01, 5)
    elif binning == 5:
        # for het_ensemble_2 now use smaller bins
        err_bins = np.linspace(0.34, 1, 2)
    elif binning == 6:
        # for het_ensemble_2 now use smaller bins
        err_bins = np.asarray([0.3, 0.8, 1])
    elif binning == 7:
        # for het_ensemble_2 now use smaller bins
        err_bins = np.asarray([0.18, 0.20, 0.25, 0.35, 0.45, 0.9, 1])
    elif binning == 8:
        # for het_ensemble_2 now use smaller bins
        err_bins = np.asarray([0.18, 0.20, 0.25, 0.35, 0.6, 1])
    elif binning ==9:
        # for het_ensemble_2 now use smaller bins
        err_bins = np.asarray([0.18, 0.8, 1])

    #%%
    model_cl_assignment = np.digitize(err_values, err_bins)
    model_classes, num_models_p_class = np.unique(model_cl_assignment, return_counts=True)

    print('Bins ', err_bins.round(2), num_models_p_class, flush=True)

    num_mclasses = np.sum(num_models_p_class > ensemble_size)
      #int(50/num_mclasses)

    #print(ind_models["Model Name"].iloc[model_cl_assignment>=1])
    #%% subsample model with similar performance
    #plt.title('Model error')
    #plt.hist(err_values, bins=20, rwidth=0.4)
    #plt.show()`
    #%%
    """
    # randomly filter models in each bin
    for model_class_id in model_classes:
        # Find all models in group
        models_in_cls = np.argwhere(model_cl_assignment==model_class_id).flatten()
        if len(models_in_cls) > max_mpclass:
            new_models_id = random`.sample(list(models_in_cls), max_mpclass)
        # Find all combinations of models in group:
    """
    all_biasvar = []
    bin_id_list = []
    results = []
    model_num_params = read_model_size_file(dataset)['models']
    #%%
    for model_class_id in model_classes:
        #%% Find all models in group
        models_in_cls = np.argwhere(model_cl_assignment==model_class_id).flatten()
        if len(models_in_cls) <= ensemble_size:
            print('Skipping bin {}'.format(err_bins[model_class_id]), flush=True)
        #else:
            #print('Building ensembles w {} models {} {}'.format(model_class_id, model_classes, num_models_p_class), flush=True)
        #%% control number of models in each bin:
        if len(models_in_cls) > max_mpclass:
            models_in_cls = random.sample(list(models_in_cls), max_mpclass)
        #%%
        ensemble_groups = set()
        num_ensembles = min(max_enspclass, len(list(combinations(models_in_cls, ensemble_size))))
        while len(ensemble_groups) < num_ensembles:
            ensemble_groups.add(tuple(sorted(random.sample(list(models_in_cls), ensemble_size))))
        try:
            err_bins[model_class_id]
        except:
            import pdb; pdb.set_trace()
        print('\nBuilding {} ensembles in bin {}\n'.format(len(ensemble_groups), err_bins[model_class_id]), flush=True)
        #%% this could run in parallel
        if serial_run:
            #"""
            for egroup_idx, ensemble_group in enumerate(tqdm(ensemble_groups)):
                tmp_outs = get_ensemble_metrics(ind_models, model_num_params, ensemble_group)
                all_biasvar.append(tmp_outs)
                bin_id_list.append([model_class_id])
            #"""
        else:
            get_ens = partial(get_ensemble_metrics, ind_models, model_num_params )
            pool = multiprocessing.Pool(20)
            for result in tqdm(pool.imap_unordered(get_ens, ensemble_groups, chunksize=2), total=len(ensemble_groups)):
             results.append([model_class_id] + result)

    #%%
    if serial_run:
        #biasvar_array = np.array(all_biasvar)
        #bin_id_list = np.array(bin_id_list)
        #values = np.hstack([bin_id_list, biasvar_array])
        values = [bin_id_list, all_biasvar]
    else:
        #values = np.array(results)
        values = results
    print('The end, now storing data', flush=True)
    df_results = pd.DataFrame(values,
                              columns=['bin_id', 'name', 'bias', 'var', 'perf', 'num_params', 'score',]
                                       )
    #
    num_rows = df_results.shape[0]
    ens_size= np.ones(num_rows) * ensemble_size
    df_results['ens_size'] = ens_size
    df_results['type'] = np.asarray(['het']*num_rows)
    df_results['binning'] = np.ones(num_rows) * binning
    #df_results['seed'] = np.ones(num_rows) * seed
    #df_results['seed'] = np.ones(num_rows) * seed

    results_filename = output_dir / out_name / "{}.csv".format(dataset)
    os.makedirs(results_filename.parent, exist_ok=True)
    if os.path.exists(results_filename):
        df_results = pd.concat([pd.read_csv(results_filename, index_col=False), df_results], axis=0)
    df_results.to_csv(results_filename, index=False)
    return

#%%
result_list = []
def log_result(result):
    result_list.append(result)


def get_ensemble_metrics(ind_models, model_num_params, ensemble_group):

    #model_num_params = read_model_size_file()['models']

    e_models = []
    for m_idx, m_ind_idx in enumerate(ensemble_group):
        model_ = ind_models.iloc[m_ind_idx]
        e_models.append(model_)
    # Ensemble name
    name = "--".join([m["models"] for m in e_models])
    # Make an ensemble:
    num_params = 0
    ens = EnsembleModel(name, "ind")
    player_score = 0
    for i, m in enumerate(e_models):
        model_name = m["models"] + "_" + str(i)
        #print(m["Filename"], flush=True)
        ens.register(filename=m["filepaths"],
                     modelname=model_name,
                     labelpath=m['labelpaths'],
                     )
        # search for size of model
        acc = ens.models[model_name].get_accuracy()
        player_score += get_player_score(acc)
        num_params += model_num_params[m["models"]]['num_params']

    bias, var, perf = ens.get_avg_nll(), ens.get_nll_div(), ens.get_nll()
    print('Built ensemble {}: \n nparams {} player_score {} with bias {:.3f}  var {:.3f}  perf {:.3f}'.format(
        name,num_params,player_score,bias, var, perf), flush=True)
    del ens
    return [name, bias, var, perf, num_params, player_score]


#%%
if __name__ == "__main__":
    fire.Fire(main)