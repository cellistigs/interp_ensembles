"""
Class which contains the results
"""
import h5py
import numpy as np
from interpensembles.metrics import AccuracyData, NLLData, BrierScoreData
from itertools import combinations
from torchvision.datasets import CIFAR10
from cifar10_ood.data import CINIC10, CIFAR10_C
from torchvision.transforms import ToTensor
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import torch
import os
from pathlib import Path
import pandas as pd

BASEDATA_DIR= str(Path(os.path.expanduser("~") ) / "pytorch_datasets")

# include imagenet testbed naming convention
DATASET_DIRECTORIES = {
    'imagenetv2-matched-frequency': "{}/{}/{}".format(BASEDATA_DIR,"imagenetv2-b-33",'val'),
    'val': "{}/{}/{}".format(BASEDATA_DIR, "imagenet",'val'),
    'imagenetc': "{}/{}".format(BASEDATA_DIR,"imagenetc"),
    'imagenetv2mf': "{}/{}/{}".format(BASEDATA_DIR,"imagenetv2-b-33",'val'),
    'imagenet': "{}/{}/{}".format(BASEDATA_DIR, "imagenet",'val'),
    'imagenetv2-matched-frequency-format-val': "{}/{}/{}".format(BASEDATA_DIR,"imagenetv2-b-33",'val'),
    'imagenet-c.brightness.1_on-disk':"{}/{}/{}/{}".format(BASEDATA_DIR, "imagenetc","brightness",str(1)),
    'imagenet-c.brightness.3_on-disk':"{}/{}/{}/{}".format(BASEDATA_DIR, "imagenetc", "brightness",str(3)),
    'imagenet-c.brightness.5_on-disk': "{}/{}/{}/{}".format(BASEDATA_DIR, "imagenetc", "brightness", str(5)),
    'imagenet-c.contrast.1_on-disk': "{}/{}/{}/{}".format(BASEDATA_DIR, "imagenetc", "contrast", str(1)),
    'imagenet-c.contrast.3_on-disk': "{}/{}/{}/{}".format(BASEDATA_DIR, "imagenetc", "contrast", str(3)),
    'imagenet-c.contrast.5_on-disk': "{}/{}/{}/{}".format(BASEDATA_DIR, "imagenetc", "contrast", str(5)),
    'imagenet-c.gaussian_noise.1_on-disk': "{}/{}/{}/{}".format(BASEDATA_DIR, "imagenetc", "gaussian_noise", str(1)),
    'imagenet-c.gaussian_noise.3_on-disk': "{}/{}/{}/{}".format(BASEDATA_DIR, "imagenetc", "gaussian_noise", str(3)),
    'imagenet-c.gaussian_noise.5_on-disk': "{}/{}/{}/{}".format(BASEDATA_DIR, "imagenetc", "gaussian_noise", str(5)),
    'imagenet-c.fog.1_on-disk': "{}/{}/{}/{}".format(BASEDATA_DIR, "imagenetc", "fog", str(1)),
    'imagenet-c.fog.3_on-disk': "{}/{}/{}/{}".format(BASEDATA_DIR, "imagenetc", "fog", str(3)),
    'imagenet-c.fog.5_on-disk': "{}/{}/{}/{}".format(BASEDATA_DIR, "imagenetc", "fog", str(5)),
}


def get_targets(dataset_name):
    if dataset_name == 'cifar10':
        targets = CIFAR10(root='/datahd3a/datasets/CIFAR10/', train=False, download=True,
                                       transform=ToTensor())

    elif dataset_name == 'cinic10':
        targets = CINIC10('/datahd3a/datasets/CINIC-10/cinic-10', "test")

    else:
        raise NotImplementedError("No targets available for {}".format(dataset_name))
    return torch.tensor(targets.targets)


def get_imagenet_tb_targets(dataset_name, distortion_name=None, severity=None):
    if distortion_name is not None:
        valdir = os.path.join(DATASET_DIRECTORIES[dataset_name], distortion_name, severity)
    else:
        valdir = os.path.join(DATASET_DIRECTORIES[dataset_name])

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir,
                             transforms.Compose([
                             transforms.Resize(256),
                             transforms.CenterCrop(224),
                             transforms.ToTensor(),
                             normalize,
        ])),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    return torch.tensor(val_loader.dataset.targets)


def read_values(prefix, dataset=None, as_tensor=True):
    # read imagenet test bed models
    if prefix.endswith("pickle"):
        # data = pd.read_pickle(filename)
        # data['model'] is the model name
        # data['eval-setting'] is the dataset
        # data['logits'] are the logits
        data = pd.read_pickle(prefix)
        logits = data['logits']
        dataset = data['eval-setting']
        target = get_imagenet_tb_targets(dataset).numpy()
    # Read imagenet models
    elif prefix.endswith('hdf5'):
        with h5py.File(str(prefix), 'r') as f:
            logits = f['logits'][()]
            target = f['targets'][()].astype('int')
    # read miller logits
    elif prefix.endswith('npz'):
        assert dataset is not None
        data = np.load(prefix)
        logits = data[dataset]
        target = get_targets(dataset).numpy()
    # read cifar10c logits
    else:
        try:
            logits = np.load(f"{prefix}_preds.npy")  # N x C
            target = np.load(f"{prefix}_labels.npy").astype('int')  # N
        except:
            raise NotImplementedError('Can\'t read logits from mode {} in {}'.format(prefix))

    if as_tensor:
        logits = torch.tensor(logits).double()
        target = torch.tensor(target.astype('int'))
    return logits, target


class EnsembleModel(object):
    def __init__(self, modelprefix, data):
        self.modelprefix = modelprefix
        self.models = {}  ## dict of dicts- key is modelname, value is dictionary of preds/labels.
        self.data = data

    def register(self, filename, modelname):
        self._logits, self._labels = read_values(filename, self.data)
        self._probs = torch.exp(self._logits) / torch.sum(torch.exp(self._logits), 1, keepdims=True)
        self.models[modelname] = {"preds": self._probs,
                                  "labels":self._labels,
                                  #"logits": self._logits
                                  }
        #print('Registered models in ensemble')

    def probs(self):
        """Calculates mean confidence across all softmax output.

        :return: array of shape samples, classes giving per class variance.
        """
        all_probs = []
        for model, modeldata in self.models.items():
            probs = modeldata["preds"]
            all_probs.append(probs)
        array_probs = torch.stack(all_probs, axis=0)
        self._probs = torch.mean(array_probs, axis = 0)
        return self._probs

    def labels(self):
        for model, modeldata in self.models.items():
            labels = modeldata["labels"]
            break
        return labels


class Model(object):
    def __init__(self,  modelprefix, data):
        self.modelprefix = modelprefix
        self.data = data

    def register(self, filename):
        self.filename = filename
        self._logits, self._labels = read_values(filename, self.data)
        self._probs = np.exp(self._logits) / np.sum(np.exp(self._logits), 1, keepdims=True)

    def probs(self):
        # n x c
        return self._probs

    def labels(self):
        # n
        return self._labels

    def logits(self):
        # n x c
        return self._logits


