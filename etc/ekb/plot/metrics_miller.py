import fire
import os
import tqdm

import numpy as np
import pandas as pd
import torch
import glob
from torchvision.datasets import CIFAR10
from cifar10_ood.data import CINIC10, CIFAR10_C
from torchvision.transforms import ToTensor
import sys
sys.path.insert(0, '/data/Projects/linear_ensembles/interp_ensembles/etc/ekb/plot')
# from metrics import calculate_metrics
from metrics import calculate_metrics
import glob


def get_targets(dataset_name):

    if dataset_name == 'cifar10':
        targets = CIFAR10(root='/datahd3a/datasets/CIFAR10/', train=False, download=True,
                                       transform=ToTensor())
    elif dataset_name == 'cinic10':
        targets = CINIC10('/datahd3a/datasets/CINIC-10/cinic-10', "test")

    else:
        raise NotImplementedError("No targets available for {}".format(dataset_name))
    return torch.tensor(targets.targets)

#%%

def process_folder(folder, output_name, num_bins=15):
    assert 'cinic10' in output_name #ood_name == 'cinic10'
    #%%
    datasets = ['cifar10', 'cinic10']
    filenames = glob.glob(folder + '/*/*.npz')
    typ = "Single Model"
    results = []

    # avoid
    targets = {}
    targets['cifar10'] = get_targets('cifar10')
    targets['cinic10'] = get_targets('cinic10')

    #%%
    for filename in tqdm.tqdm(filenames):
        data = np.load(filename)
        for dataset in datasets:
            logits = torch.tensor(data[dataset])
            target = targets[dataset]
            #print(logits.shape, len(target))
            nll, top1err, brier, ece, resce = calculate_metrics(logits, target, is_logits=True, num_bins=num_bins)

            model_name, model_seed = filename.rsplit('/', 2)[-2:]
            model_seed = int(model_seed.rsplit('.')[0][4:])
            model_id = model_name + '--' + str(model_seed)
            dist = "InD" if ("cifar10" == dataset ) else "OOD"
            results.append([model_id, filename, dataset, model_name, model_seed, typ, dist, nll, top1err, brier, ece, resce])

    results = pd.DataFrame(results, columns=["ID", "Filename", "Dataset", "Model Name", "Model Seed", "Type", "Distribution", "NLL", "Top 1\% Err", "Brier", "ECE","rESCE"])
    results.to_csv(output_name, index=False)


if __name__ == "__main__":
    fire.Fire(process_folder)
