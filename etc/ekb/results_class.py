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
import torch

def get_targets(dataset_name):
    if dataset_name == 'cifar10':
        targets = CIFAR10(root='/datahd3a/datasets/CIFAR10/', train=False, download=True,
                                       transform=ToTensor())
    elif dataset_name == 'cinic10':
        targets = CINIC10('/datahd3a/datasets/CINIC-10/cinic-10', "test")

    else:
        raise NotImplementedError("No targets available for {}".format(dataset_name))
    return torch.tensor(targets.targets)


def read_values(mode, prefix, dataset=None, as_tensor=False):
    # Read imagenet models
    if mode == 'ekb_imagenet':
        assert prefix.endswith('hdf5')
        with h5py.File(str(prefix), 'r') as f:
            logits = f['logits'][()]
            target = f['targets'][()].astype('int')
    # read miller logits
    elif mode == 'miller_cin':
        assert prefix.endswith('npz')
        assert dataset is not None
        data = np.load(prefix)
        logits = data[dataset]
        target = get_targets(dataset)
    # read cifar logits
    elif mode == 'ta_cifar':
        logits = np.load(f"{prefix}_preds.npy")  # N x C
        target = np.load(f"{prefix}_labels.npy").astype('int')  # N
    else:
        raise NotImplementedError('Can\'t read logits from mode {} in {}'.format(mode, prefix))
    if as_tensor:
        logits = torch.tensor(logits).double()
        target = torch.tensor(target)
    return logits, target

class EnsembleModel(object):

    def __init__(self, modelprefix, data):
        self.modelprefix = modelprefix
        self.models = {}  ## dict of dicts- key is modelname, value is dictionary of preds/labels.
        self.data = data

    def register(self, filename, modelname, dataset=None):
        # read imagenet logits
        if filename.endswith('hdf5'):
            with h5py.File(str(filename), 'r') as f:
                self._logits = f['logits'][()]
                self._labels = f['targets'][()].astype('int')
                self._probs = np.exp(self._logits) / np.sum(np.exp(self._logits), 1, keepdims=True)
        # read miller logits
        #elif filename.endswith('npz'):
        else:
            raise NotImplementedError('')
        self.models[modelname] = {"preds": self._probs,
                                  "labels":self._labels,
                                  "logits": self._logits}

    def probs(self):
        """Calculates mean confidence across all softmax output.

        :return: array of shape samples, classes giving per class variance.
        """
        all_probs = []
        for model,modeldata in self.models.items():
            probs = modeldata["preds"]
            all_probs.append(probs)
        array_probs = np.stack(all_probs,axis = 0)
        self._probs = np.mean(array_probs, axis = 0)
        return self._probs


    def labels(self):
        for model, modeldata in self.models.items():
            labels = modeldata["labels"]
            break
        return labels

    def get_accuracy(self):
        ad = AccuracyData()
        return ad.accuracy(self.probs(), self.labels())

    def get_nll(self, normalize=True):

        nld = NLLData()
        return nld.nll(self.probs(), self.labels(), normalize=normalize)

    def get_brier(self):
        bsd = BrierScoreData()
        return bsd.brierscore_multi(self.probs(), self.labels())


class Model(object):
    def __init__(self,  modelprefix, data):
        self.modelprefix = modelprefix
        self.data = data

    def register(self, filename):
        self.filename = filename
        with h5py.File(str(self.filename), 'r') as f:
            self._logits = f['logits'][()]
            self._labels = f['targets'][()].astype('int')
            self._probs = np.exp(self._logits) / np.sum(np.exp(self._logits), 1, keepdims=True)

    def mean_conf(self):
        # n x c
        return self._probs

    def probs(self):
        # n x c
        return self._probs

    def labels(self):
        # n
        return self._labels

    def logits(self):
        # n x c
        return self._logits

    def get_accuracy(self):
        ad = AccuracyData()
        return ad.accuracy(self.probs(), self.labels())

    def get_nll(self, normalize=True):

        nld = NLLData()
        return nld.nll(self.probs(), self.labels(), normalize=normalize)

    def get_brier(self):
        bsd = BrierScoreData()
        return bsd.brierscore_multi(self.probs(), self.labels())


