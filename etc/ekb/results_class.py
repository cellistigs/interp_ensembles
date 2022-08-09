"""
Class which contains the results
"""
import h5py
import numpy as np
from interpensembles.metrics import AccuracyData, NLLData, BrierScoreData

class EnsembleModel(object):

    def __init__(self, modelprefix, data):
        self.modelprefix = modelprefix
        self.models = {}  ## dict of dicts- key is modelname, value is dictionary of preds/labels.
        self.data = data

    def register(self, filename, modelname):
        with h5py.File(str(filename), 'r') as f:
            self._logits = f['logits'][()]
            self._labels = f['targets'][()].astype('int')
            self._probs = np.exp(self._logits) / np.sum(np.exp(self._logits), 1, keepdims=True)

        self.models[modelname] = {"preds":self._probs, "labels":self._labels,"logits": self._logits}

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
        for model,modeldata in self.models.items():
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