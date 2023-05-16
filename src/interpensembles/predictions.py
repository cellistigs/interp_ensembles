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

    def register(self, filename, modelname, inputtype=None,labelpath = None, logits = True,npz_flag = None):
        """Register a model's predictions to this ensemble object. 
        :param filename: (string) path to file containing logit model predictions. 
        :param modelname: (string) name of the model to identify within an ensemble
        :param inputtype: (optional) [h5,npy] h5 inputs or npy input, depending on if we're looking at imagenet or cifar10 datasets. If not given, will be inferred from filename extension.   
        :param labelpath: (optional) if npy format files, labels must be given. 
        :param logits: (optional) we assume logits given, but probs can also be given directly. 
	:param npz_flag: if filetype is .npz, we asume that we need to pass a dictionary key to retrieve either `cifar10` or `cinic10` logits.
        """
        if inputtype is None:
            _,ext = os.path.splitext(filename)
            inputtype = ext[1:] 
            assert inputtype in ["h5","hdf5","npy","npz"], "inputtype inferred from extension must be `h5` or `npy`, or `npz` if not given, not {}.".format(inputtype)
            
        if inputtype in ["h5","hdf5"]:
            with h5py.File(str(filename), 'r') as f:
                self._logits = f['logits'][()]
                self._labels = f['targets'][()].astype('int')
                self._probs = np.exp(self._logits) / np.sum(np.exp(self._logits), 1, keepdims=True)
        
        elif inputtype == "npy":
            assert labelpath is not None, "if npy, must give labels."
            if logits:  
                self._logits = np.load(filename)
                self._labels = np.load(labelpath)
                self._probs = np.exp(self._logits) / np.sum(np.exp(self._logits), 1, keepdims=True)
            else:               
                self._logits = None 
                self._labels = np.load(labelpath)
                self._probs = np.load(filename) 

        elif inputtype == "npz":   
            assert labelpath is not None, "if npz must give labels."
            assert npz_flag is not None, "if npz must give flag for which logits to retrieve."
            if logits:  
                self._logits = np.load(filename)[npz_flag]
                self._labels = np.load(labelpath)
                self._probs = np.exp(self._logits) / np.sum(np.exp(self._logits), 1, keepdims=True)
            else:               
                self._logits = None 
                self._labels = np.load(labelpath)
                self._probs = np.load(filename) 
        

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

    def get_variance_vec(self):
        """Get variance across ensemble members. Estimate sample variance across the dataset with unbiased estimate, not biased.  
        """
        all_probs = []
        for model,modeldata in self.models.items():
            probs = modeldata["preds"]
            all_probs.append(probs)
        array_probs = np.stack(all_probs,axis = 0) # (models,samples,classes)
        var = np.var(array_probs,axis = 0,ddof = 1)
        return np.sum(var,axis = -1)

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
    
    def get_avg_certainty(self):
        """Get average certainty across ensemble members. 
        """
        all_probs = []
        for model, modeldata in self.models.items():
            probs = modeldata["preds"]
            all_probs.append(probs)
        array_probs = np.stack(all_probs,axis = 0) # (models,samples,classes)
        norms = np.linalg.norm(array_probs,axis = -1)
        avg_norms = np.mean(norms,axis = 0)
        return avg_norms

    def get_bias_bs_vec(self):
        """Given a brier score, estimate bias across the dataset.   
        """
        all_probs = []
        for model,modeldata in self.models.items():
            probs = modeldata["preds"]
            all_probs.append(probs)
        array_probs = np.stack(all_probs,axis = 0)
        bsd = BrierScoreData()
        model_bs = np.array([bsd.brierscore_multi_vec(m["preds"],m["labels"]) for m in self.models.values()])
        return np.mean(model_bs,axis = 0)

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

    def get_avg_nll(self):
        """estimate the average NLL across the ensemble. 

        """
        all_nll = []
        for model,modeldata in self.models.items():
            probs = modeldata["preds"]
            targets = modeldata["labels"]
            all_nll.append(-np.log(probs[np.arange(len(targets)),targets]))

        array_nll = np.stack(all_nll,axis = 0) # (models,samples)
        return np.mean(np.mean(array_nll,axis = 0))

    def get_nll_div(self):    
        """estimate diversity between ensembles members corresponding to the jensen gap between ensemble and single
        model nll

        """
        all_probs = []
        for model,modeldata in self.models.items():
            probs = modeldata["preds"]
            targets = modeldata["labels"]
            all_probs.append(probs[np.arange(len(targets)),targets])
            

        array_probs = np.stack(all_probs,axis = 0) # (models,samples)
        norm_term = np.log(np.mean(array_probs,axis = 0))
        diversity = np.mean(-np.mean(np.log(array_probs),axis = 0)+norm_term)
        return diversity 

    def get_avg_nll_vec(self):
        """estimate the average NLL across the ensemble per datapoint. 

        """
        all_nll = []
        for model,modeldata in self.models.items():
            probs = modeldata["preds"]
            targets = modeldata["labels"]
            all_nll.append(-np.log(probs[np.arange(len(targets)),targets]))

        array_nll = np.stack(all_nll,axis = 0) # (models,samples)
        return np.mean(array_nll,axis = 0)

    def get_nll_div_vec(self):    
        """estimate diversity between ensembles members corresponding to the jensen gap between ensemble and single
        model nll per datapoint 

        """
        all_probs = []
        for model,modeldata in self.models.items():
            probs = modeldata["preds"]
            targets = modeldata["labels"]
            all_probs.append(probs[np.arange(len(targets)),targets])
            

        array_probs = np.stack(all_probs,axis = 0) # (models,samples)
        norm_term = np.log(np.mean(array_probs,axis = 0))
        diversity = -np.mean(np.log(array_probs),axis = 0)+norm_term
        return diversity 

    def get_pairwise_corr(self):
        """Get pairwise correlation between ensemble members:

        """
        all_probs = []
        M = len(self.models)
        for model,modeldata in self.models.items():
            probs = modeldata["preds"]
            targets = modeldata["labels"]
            all_probs.append(probs)
        array_probs = np.stack(all_probs,axis = 0) # (models,samples,classes)
        array_pred_labels = np.argmax(array_probs,axis = -1) # (models,samples)
        compare = array_pred_labels[:,None,:] == array_pred_labels # (models,models,samples)
        means = np.mean(compare.astype(int),axis = -1) ## (models,models)
        mean_means = np.sum(np.tril(means,k=-1))/((M*(M-1))/2) ## (average across all pairs)
        return 1-mean_means

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
