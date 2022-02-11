"""
Build homogeneous ensembles by averaging model seeds selecting using csv
"""
import os
import sys
from itertools import combinations

# %%
import pandas as pd
import torch
import fire
from metrics import calculate_metrics
from model_class import EnsembleModel
import numpy as np
import random
from tqdm import tqdm

def make_ensemble_from_files(ensemble_name, dataset, model_file_list):
    # We speedup by not realoading targets for every model
    # Build a model
    model_cls = EnsembleModel(ensemble_name, dataset)
    # Build an ensemble
    for model in model_file_list:
        model_cls.register(str(model), ensemble_name)


    return model_cls.probs(), model_cls.labels()


def preprocess_heterogeneous_ensemble(metrics_file,
                                    output_name,
                                    ood_file=None,
                                    num_bins=15,
                                    ensemble_size=4,
                                    binning=2,
                                    ):

    # TODO: accept separate ind and ood files
    results = pd.read_csv(metrics_file)
    if ood_file is not None:
        results = results.append(pd.read_csv(ood_file))

    # filter out any non single models.
    results = results.loc[results['Type'] == 'Single Model']

    # check the ind/ood datasets available:
    data_names_dict = results.groupby('Distribution').apply(lambda x: x['Dataset'].unique())
    # data_names = results['Dataset'].unique()
    for data_type, dataset in data_names_dict.items():
        assert len(dataset) == 1
        data_names_dict[data_type] =dataset[0]
    #%% File out ind models, and find models which have the same performance.
    ind_models = results[results['Distribution'] == 'InD']
    ood_models = results[results['Distribution'] == 'OOD']

    err_values = ind_models['Top 1\% Err'].values
    # Split models in 10% err rate performance groups
    if binning ==2:
        err_bins = np.linspace(0, 1, 6)
    else:
        # for het_ensemble_2 now use smaller bins
        err_bins = np.linspace(0, err_values.max() + 0.01, 11)

    model_cl_assignment = np.digitize(err_values, err_bins)
    model_classes, num_models_p_class = np.unique(model_cl_assignment, return_counts=True)

    print('Bins ', err_bins.round(2), num_models_p_class)

    num_mclasses = np.sum(num_models_p_class > ensemble_size)
    max_pclass = int(100/num_mclasses)
    #%%
    heter_id = 0
    ensemble_results = []
    typ = 'Het. Ensemble'
    #%%
    for model_class_id in model_classes:
        #%%
        # Find all models in group
        models_in_cls = np.argwhere(model_cl_assignment==model_class_id).flatten()
        if len(models_in_cls) <= ensemble_size:
            print('Skipping bin {}'.format(err_bins[model_class_id]))
            continue
        ensemble_groups = set()
        num_ensembles = min(max_pclass, len(list(combinations(models_in_cls, ensemble_size))))
        while len(ensemble_groups) < num_ensembles:
            ensemble_groups.add(tuple(sorted(random.sample(list(models_in_cls), ensemble_size))))
        print('Building {} ensembles'.format(len(ensemble_groups)))
        # this could run in parallel
        for egroup_idx, ensemble_group in enumerate(tqdm(ensemble_groups)):
            for data_type, dataset in data_names_dict.items():
                e_files = []
                for m_idx, m_ind_idx in enumerate(ensemble_group):
                    model_ = ind_models.iloc[m_ind_idx]

                    if data_type == 'OOD':
                        model_ = ood_models[ood_models['ID'] == model_['ID']]
                        assert len(model_) == 1
                        model_ = model_.iloc[0]

                    e_files.append(model_['Filename'])

                # calculate ensemble
                short_prefix ='heter_{}_{}'.format(heter_id, dataset)
                _probs, target = make_ensemble_from_files(short_prefix, dataset, e_files)

                # Calculate metrics,
                nll, top1err, brier, ece, resce = calculate_metrics(_probs, target, is_logits=False,
                                                                  num_bins=num_bins)

                dist = data_type
                if binning==2:
                    ensemble_id = 'ensemble2_' + str(heter_id) + '--' + str(egroup_idx)
                    model_name = 'ensemble2_' + str(heter_id)
                else:
                    ensemble_id = 'ensemble_' + str(heter_id) + '--' + str(egroup_idx)
                    model_name = 'ensemble_' + str(heter_id)
                ensemble_results.append([ensemble_id, short_prefix, dataset, model_name, egroup_idx, typ, dist, nll, top1err, brier, ece, resce])

            heter_id+=1

            #%%

    ensemble_results = pd.DataFrame(ensemble_results,columns=[
        "ID", "Filename", "Dataset", "Model Name", "Model Seed", "Type", "Distribution", "NLL", "Top 1\% Err", "Brier", "ECE", "rESCE"
    ])
    ensemble_results.to_csv(output_name, index=False)


if __name__ == "__main__":
    fire.Fire(preprocess_heterogeneous_ensemble)