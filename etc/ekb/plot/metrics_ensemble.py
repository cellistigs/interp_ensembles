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


def make_ensemble_from_files(ensemble_name, dataset, model_file_list):
    # Build a model
    model_cls = EnsembleModel(ensemble_name, dataset)
    # Build an ensemble
    for model in model_file_list:
        model_cls.register(str(model), ensemble_name)

    # Return probs and targets for ensemble
    return model_cls.probs(), model_cls.labels()


def preprocess_homogeneous_ensemble(metrics_file,
                                    output_name,
                                    num_bins=15,
                                    ensemble_size=4,
                                    ):

    results = pd.read_csv(metrics_file)
    # make sure we are using only single models to build ensembles
    assert results['Type'].unique() == 'Single Model'
    # check the ind/ood datasets available:
    data_names_dict = results.groupby('Distribution').apply(lambda x: x['Dataset'].unique())
    model_names = results['Model Name'].unique()
    # data_names = results['Dataset'].unique()
    assert data_names_dict['InD'].size == 1
    # TODO(ekb): support multiple OOD
    assert data_names_dict['OOD'].size == 1

    # For each model
    typ ='Ensemble'
    ensemble_results = []
    for model_name in model_names:
        # filter models according to model_names, and which exist for ind dataset
        model_full = results.loc[(results['Model Name'] == model_name)]
        # Find seeds for model class
        model_seeds = model_full['Model Seed'].unique()
        # list of lists which contains the seeds for each ensemble
        ensemble_groups = list(combinations(model_seeds, ensemble_size))
        #%% for each group we need to combine
        for egroup_idx, ensemble_group in enumerate(ensemble_groups):
            # print(egroup_idx, ensemble_group)
            for dataset in results['Dataset'].unique():
                tmp_model = model_full.loc[model_full['Dataset']==dataset]

                # get all files in ensemble
                e_files = []
                for e_ in ensemble_group:
                    e_files.append(tmp_model[tmp_model['Model Seed'] == e_]['Filename'].item())

                # calculate ensemble
                short_prefix ='homogeneous_{}_{}_{}'.format(model_name,''.join(map(str,ensemble_group)),dataset)
                _probs, target = make_ensemble_from_files(short_prefix, dataset, e_files)

                # Calculate metrics,
                nll, top1err, brier, ece, resce = calculate_metrics(_probs, target, is_logits=False,
                                                                  num_bins=num_bins)

                assert tmp_model['Distribution'].unique().size == 1
                dist = tmp_model['Distribution'].unique()[0]

                ensemble_id = 'ensemble_' + model_name + '--' + str(egroup_idx)
                ensemble_results.append([ensemble_id, short_prefix, dataset, model_name, egroup_idx, typ, dist, nll, top1err, brier, ece, resce])

    ensemble_results = pd.DataFrame(ensemble_results,columns=[
        "ID", "Filename", "Dataset", "Model Name", "Model Seed", "Type", "Distribution", "NLL", "Top 1\% Err", "Brier", "ECE","rESCE"
    ])
    ensemble_results.to_csv(output_name, index=False)


if __name__ == "__main__":
    fire.Fire(preprocess_homogeneous_ensemble)