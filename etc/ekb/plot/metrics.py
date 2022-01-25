import fire
import os
import tqdm

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
import torch


def compute_nll(probs,target,normalize=True):
    probs = probs[torch.arange(target.size(0)), target]
    logprobs = probs.log()
    nll = -logprobs.sum()
    if normalize:
        nll = nll / logprobs.size(0)
    return nll


def compute_brier(fs, ys):
    deviance = fs - ys.to(fs.dtype)
    return (deviance).pow(2.).mean(dim=-1)


def compute_brier_multi_np(prob, target):
    """The "original" brier score definition that accounts for other classes explicitly. Note the range of this test is 0-K, where k is the number of classes.
    :param prob: array of probabilities per class.
    :param target: list/array of true targets.

    """
    prob = prob.numpy()
    target= target.numpy()
    target_onehot = np.zeros(prob.shape)
    target_onehot[np.arange(len(target)), target] = 1  ## onehot encoding.
    deviance = prob - target_onehot
    return np.mean(np.sum(deviance ** 2, axis=1))


def compute_brier_multi(prob, target):
    """The "original" brier score definition that accounts for other classes explicitly. Note the range of this test is 0-K, where k is the number of classes.
    :param prob: array of probabilities per class.
    :param target: list/array of true targets.

    """
    target_onehot = torch.zeros(prob.shape)
    target_onehot[torch.arange(len(target)), target] = 1  ## onehot encoding.
    deviance = prob - target_onehot
    return torch.mean(torch.sum(deviance.pow(2), axis=1))


def compute_ece(fs, ys, power=1, num_bins=15):
    ys = ys.to(fs.dtype)
    bin_boundaries = torch.linspace(0, 1, num_bins + 1, dtype=fs.dtype, device=fs.device)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # Expand
    fs = fs[..., None]
    ys = ys[..., None]

    # Get mask for samples in bin
    in_bin = (fs.ge(bin_lowers) * fs.lt(bin_uppers)).to(fs.dtype).transpose(-1, -2)
    tot_in_bin = in_bin.sum(dim=-1).clamp(1., None)
    prob_in_bin = (in_bin @ fs).squeeze(-1)
    true_in_bin = (in_bin @ ys).squeeze(-1)
    esces = (prob_in_bin - true_in_bin).div(tot_in_bin).abs().pow(power)

    return torch.mul(esces, tot_in_bin / ys.size(-2)).sum(dim=-1)


def compute_esce_lb(fs, ys):
    gs = torch.tensor(IsotonicRegression().fit(fs, ys).predict(fs)).to(dtype=fs.dtype, device=fs.device)
    brier_f = compute_brier(fs, ys)
    brier_g = compute_brier(gs, ys)
    return brier_f - brier_g


def calculate_metrics(_probs, target, is_logits=True, num_bins=15):

    if is_logits:
        _probs = _probs.softmax(dim=-1)

    fs, _preds = _probs.max(dim=-1)  # N, N

    ys = torch.eq(target, _preds)  # N

    acc = (ys.sum() / ys.size(0)).item()

    top1err = 1-acc
    nll = compute_nll(_probs, target).item()

    # brier = compute_brier(fs, ys).item()
    brier = compute_brier_multi(_probs, target).item()

    ece = compute_ece(fs, ys, power=1, num_bins=num_bins).item()

    resce = compute_ece(fs, ys, power=2, num_bins=num_bins).sqrt().item()

    # resce_lb = compute_esce_lb(fs, ys).sqrt().item()

    return nll, top1err, brier, ece, resce


def process_metrics(prefix, is_logits=True, num_bins=15):
    _probs = torch.tensor(np.load(f"{prefix}_preds.npy")).double()  # N x C
    target = torch.tensor(np.load(f"{prefix}_labels.npy").astype('int'))  # N

    return calculate_metrics(_probs, target, is_logits=is_logits, num_bins=num_bins)


def process_folder(folder, output_name, num_bins=15):
    filenames = sorted([os.path.join(folder, basename) for basename in os.listdir(folder)])
    prefixes = [basename.replace("_preds.npy", "") for basename in filenames if "_preds.npy" in basename]
    # Filter out CINIC data
    prefixes = [prefix for prefix in prefixes if "cinic" not in prefix]
    results = []
    prefixes = tqdm.tqdm(prefixes, desc="Models")
    for idx, prefix in enumerate(prefixes):
        short_prefix = os.path.basename(prefix).replace("ind", "").replace("ood", "")
        is_ensemble = "ensemble" in prefix.rsplit('/',1)[-1]
        typ = "Ensemble" if is_ensemble else "Single Model"
        dist = "InD" if ("ind" in prefix.rsplit('/',1)[-1]) else "OOD"
        print(idx, prefix, dist)
        #ece, resce, resce_lb = process(prefix, is_logits=(not is_ensemble), num_bins=num_bins)
        nll, top1err, brier, ece, resce = process_metrics(prefix, is_logits=(not is_ensemble), num_bins=num_bins)

        dataset = 'cifar10' if (dist == "InD") else "cifar10.1"
        model_name = 'unknown'
        model_seed = 'uknown'
        results.append([short_prefix, prefix, dataset, model_name, model_seed, typ, dist, nll, top1err, brier, ece, resce])

    results = pd.DataFrame(results, columns=[
        "ID", "Filename", "Dataset", "Model Name", "Model Seed", "Type", "Distribution", "NLL", "Top 1\% Err", "Brier", "ECE", "rESCE"
    ]
                           )
    results.to_csv(output_name, index=False)


if __name__ == "__main__":
    fire.Fire(process_folder)
