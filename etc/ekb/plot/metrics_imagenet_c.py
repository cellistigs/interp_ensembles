"""
Plot for imagenet
"""
import fire
import os
import tqdm

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
import torch
import h5py
import sys
from itertools import combinations

from model_class import Model, read_values
from metrics import calculate_metrics


def process_metrics_from_file(prefix, is_logits=True, num_bins=15, dataset=None):

    _probs, target = read_values(prefix, dataset=dataset)
    return calculate_metrics(_probs, target,is_logits=is_logits, num_bins=num_bins)


def process_folder_imagenet(folder, output_name, distortion_name, logits_ood_folder=None,
                            num_bins=15):
    # there are no ensembles
    #%%
    # Read InD filenames and filter out imagenetv2mf
    filenames = sorted([os.path.join(folder, basename) for basename in os.listdir(folder)])
    filenames = [file_ for file_ in filenames if 'imagenetv2mf' not in file_]

    #%%
    # Read ood filenames and filter out files with distortion
    if not(logits_ood_folder is None):
        filenames1 = sorted([os.path.join(logits_ood_folder, basename) for basename in os.listdir(logits_ood_folder)])
        prefixes = [file_ for file_ in filenames1 if distortion_name in file_]

    #%%
    # aggregate ind + ood filenames for distortion_name
    prefixes = prefixes + filenames

    #%%
    results = []
    prefixes = tqdm.tqdm(prefixes, desc="Models")
    is_ensemble = False  # "ensemble" in prefix.lower()
    typ = "Ensemble" if is_ensemble else "Single Model"

    for idx, prefix in enumerate(prefixes):
        short_prefix = os.path.basename(prefix)
        dist = "OOD" if ("imagenetc" in prefix.lower()) else "InD"

        # here is where things get tricky
        if dist == 'InD':
            model_name, dataset, model_seed = short_prefix.rsplit('--')
            model_seed = int(model_seed.rsplit('.', 1)[0][7:])
            model_id = model_name + '--' + str(model_seed)
        elif dist == 'OOD':
            model_name, dataset, model_seed, distortion_type, distortion_level = short_prefix.rsplit('--')
            model_seed = int(model_seed.rsplit('.', 1)[0][7:])
            model_id = model_name + '--' + str(model_seed)

        nll, top1err, brier, ece, resce = process_metrics_from_file(prefix,
                                                               is_logits=(not is_ensemble),
                                                               num_bins=num_bins,
                                                                dataset=dataset)

        results.append([model_id, prefix, dataset, model_name, model_seed, typ, dist, nll, top1err, brier, ece, resce])

    # 'filename', 'imagenet' 'alexnet' 1, 'Single Model', 'InD', #, #, #, #
    results = pd.DataFrame(results, columns=["ID", "Filename", "Dataset", "Model Name", "Model Seed", "Type", "Distribution", "NLL", "Top 1\% Err", "Brier", "ECE","rESCE"])

    results.to_csv(output_name, index=False)


if __name__ == "__main__":
    #fire.Fire(process_folder)
    fire.Fire(process_folder_imagenet)