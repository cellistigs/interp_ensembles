"""
Classes which contains the predictions from a given network and ensembles. 
"""
import os 
import h5py
import numpy as np
from .metrics import AccuracyData, NLLData, BrierScoreData

class EnsembleModel(object):
    """Collect the outputs of a series of models to allow ensemble based analysis.  

    :param modelprefix: string prefix to identify this set of models. 
    :param data: string to identify the dataset that set of models is evaluated on. 
    """

    def __init__(self, modelprefix, data):
        self.modelprefix = modelprefix
        self.models = {}  ## dict of dicts- key is modelname, value is dictionary of preds/labels.
        self.data = data

    def register(self, filename, modelname, inputtype=None,labelpath = None):
        """Register a model's predictions to this ensemble object. 
        :param filename: (string) path to file containing logit model predictions. 
        :param modelname: (string) name of the model to identify within an ensemble
        :param inputtype: (optional) [h5,npy] h5 inputs or npy input, depending on if we're looking at imagenet or cifar10 datasets. If not given, will be inferred from filename extension.   
        :param labelpath: (optional) if npy format files, labels must be given. 
        """
        if inputtype is None:
            _,ext = os.path.splitext(filename)
            inputtype = ext[1:] 
            assert inputtype in ["h5","hdf5","npy"], "inputtype inferred from extension must be `h5` or `npy` if not given, not {}.".format(inputtype)
            
        if inputtype in ["h5","hdf5"]:
            with h5py.File(str(filename), 'r') as f:
                self._logits = f['logits'][()]
                self._labels = f['targets'][()].astype('int')
                self._probs = np.exp(self._logits) / np.sum(np.exp(self._logits), 1, keepdims=True)
        
        if inputtype == "npy":
            assert labelpath is not None, "if npy, must give labels."
            self._logits = np.load(filename)
            self._labels = np.load(labelpath)
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

    def get_variance(self):
        """Get variance across ensemble members. Estimate sample variance across the dataset with unbiased estimate, not biased.  
        """
        all_probs = []
        for model,modeldata in self.models.items():
            probs = modeldata["preds"]
            all_probs.append(probs)
        array_probs = np.stack(all_probs,axis = 0) # (models,samples,classes)
        var = np.var(array_probs,axis = 0,ddof = 1)
        return np.mean(np.sum(var,axis = -1))
    
    def get_bias_bs(self):
        """Given a brier score, estimate bias across the dataset.   
        """
        all_probs = []
        for model,modeldata in self.models.items():
            probs = modeldata["preds"]
            all_probs.append(probs)
        array_probs = np.stack(all_probs,axis = 0)
        bsd = BrierScoreData()
        model_bs = np.array([bsd.brierscore_multi_vec(m["preds"],m["labels"]) for m in self.models.values()])
        return np.mean(np.mean(model_bs,axis = 0))


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
