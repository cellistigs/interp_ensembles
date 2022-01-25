"""
Plot for imagenet
"""
import fire
import os

import torch
import tqdm

import pandas as pd

from model_class import Model, read_values, get_imagenet_tb_targets
from metrics import calculate_metrics
from model_types import is_standard

# preload targets to speed up processing?


def process_metrics_from_file(prefix, is_logits=True, num_bins=15, dataset=None):

    _probs, target = read_values(prefix, dataset=dataset)
    return calculate_metrics(_probs, target,is_logits=is_logits, num_bins=num_bins)


def process_folder_imagenet(folder,
                            output_name,
                            ind_folder=None,
                            distortion_name=None,
                            severity=None,
                            num_bins=15):

    # read ood data for a given distortion and severity
    filenames = sorted([os.path.join(folder, basename) for basename in os.listdir(folder)])

    # include ind_folder if ind data is not in folder!
    if not(ind_folder is None):
        prefixes = sorted([os.path.join(ind_folder, basename) for basename in os.listdir(ind_folder)])
        #prefixes = [file_ for file_ in filenames1 if 'imagenetv2mf' not in file_]

    # aggregate ind + ood filenames for distortion_name
    prefixes = prefixes + filenames

    #%%
    # preload targets to speed up processing
    values = ['imagenet', 'imagenetv2mf']
    targets = {}
    for val in values:
        targets[val] = get_imagenet_tb_targets(val)
    # the format val dataset from paper: The format-val versions are variants of the original
    # dataset encoded with jpeg settings similar to the original one
    # Unless otherwise stated, results in
    # our paper referring to imagenetv2 are for imagenetv2-matched-frequency-format-val.
    #targets['imagenetv2-matched-frequency-format-val'] = targets['imagenetv2-matched-frequency']
    #%%
    if distortion_name is not None:
        targets['imagenetc'] = get_imagenet_tb_targets("imagenetc", distortion_name, str(severity))
    #%%
    is_ensemble = False  # "ensemble" in prefix.lower()
    typ = "Ensemble" if is_ensemble else "Single Model"
    model_seed = 0
    is_logits = True
    #%%
    results = []
    prefixes = tqdm.tqdm(prefixes, desc="Models")

    for idx, prefix in enumerate(prefixes):
        short_prefix = os.path.basename(prefix)
        data = pd.read_pickle(prefix)
        model_name = data['model']

        if not is_standard(model_name):
            continue;
        #%%
        # rename datasets so they are compatible with our models
        dataset = data['eval-setting']
        if 'imagenet-c' in dataset:
            dataset = 'imagenetc'
        elif 'imagenetv2-matched-frequency' in dataset:
            dataset = 'imagenetv2mf'
        elif 'val' == dataset:
            dataset = 'imagenet'

        _probs = torch.Tensor(data['logits']).double()
        target = targets[dataset]
        dist = "InD" if ("imagenet" == dataset) else "OOD"
        model_id = model_name + '--' + str(model_seed)

        nll, top1err, brier, ece, resce = calculate_metrics(_probs,
                                                          target,
                                                          is_logits=is_logits,
                                                          num_bins=num_bins)

        results.append([model_id, prefix, dataset, model_name, model_seed, typ, dist, nll, top1err, brier, ece, resce])

    # 'filename', 'imagenet' 'alexnet' 1, 'Single Model', 'InD', #, #, #, #
    results = pd.DataFrame(results, columns=[
                "ID", "Filename", "Dataset", "Model Name", "Model Seed", "Type", "Distribution", "NLL", "Top 1\% Err", "Brier", "ECE", "rESCE"
                ])

    results.to_csv(output_name, index=False)


if __name__ == "__main__":
    #fire.Fire(process_folder)
    fire.Fire(process_folder_imagenet)