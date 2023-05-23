"""
Classes which contains the predictions from a given network and ensembles. 
"""
import itertools
import os 
import h5py
import numpy as np
from .metrics import AccuracyData, NLLData, BrierScoreData, quadratic_uncertainty
import pandas as pd


class Model(object):
    def __init__(self,  modelprefix, data):
        self.modelprefix = modelprefix
        self.data = data

    def register(self,
                 filename,
                 inputtype=None,
                 labelpath=None,
                 logits=True,
                 npz_flag=None,
                 mask_array=None):
        """Register a model's predictions to this model object.
    :param filename: (string) path to file containing logit model predictions.
    :param modelname: (string) name of the model to identify within an ensemble
    :param inputtype: (optional) [h5,npy] h5 inputs or npy input, depending on if we're looking at imagenet or cifar10 datasets. If not given, will be inferred from filename extension.
    :param labelpath: (optional) if npy format files, labels must be given.
    :param logits: (optional) we assume logits given, but probs can also be given directly.
    :param npz_flag: if filetype is .npz, we asume that we need to pass a dictionary key to retrieve logits
          used for `cifar10` or `cinic10` logits.
    """

    def __init__(self, modelprefix, data, dtype='float32'):
        self.modelprefix = modelprefix
        self.models = {}  ## dict of dicts- key is modelname, value is dictionary of preds/labels.
        self.data = data
        self.dtype = dtype

        if inputtype is None:
            _, ext = os.path.splitext(filename)
            inputtype = ext[1:]
            assert inputtype in [
                "h5", "hdf5", "npy", "npz", "pickle"
            ], "inputtype inferred from extension must be `h5` or `npy`, or `npz` if not given, not {}.".format(
                inputtype)

        if mask_array is None:
            mask_array = ()

        if inputtype in ["h5", "hdf5"]:
            with h5py.File(str(filename), 'r') as f:
                self._logits = f['logits'][mask_array]
                self._labels = f['targets'][mask_array].astype('int')
                self._probs = np.exp(self._logits) / np.sum(
                    np.exp(self._logits), 1, keepdims=True)

        elif inputtype == "npy":
            assert labelpath is not None, "if npy, must give labels."
            if logits:
                self._logits = np.load(filename)[mask_array]
                self._labels = np.load(labelpath)[mask_array]
                self._probs = np.exp(self._logits) / np.sum(
                    np.exp(self._logits), 1, keepdims=True)
            else:
                self._logits = None
                self._labels = np.load(labelpath)
                self._probs = np.load(filename)

        elif inputtype == "npz":
            assert labelpath is not None, "if npz must give labels."
            assert npz_flag is not None, "if npz must give flag for which logits to retrieve."
            if logits:
                self._logits = np.load(filename)[npz_flag][mask_array]
                self._labels = np.load(labelpath)[mask_array]
                self._probs = np.exp(self._logits) / np.sum(
                    np.exp(self._logits), 1, keepdims=True)
            else:
                self._logits = None
                self._labels = np.load(labelpath)[mask_array]
                self._probs = np.load(filename)[mask_array]
        elif inputtype == 'pickle':
            if logits:
                self._logits = pd.read_pickle(filename)['logits']
                self._labels = np.load(labelpath)
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

    def get_qunc(self, as_vec=False):
        """estimate the average single model uncertainty.

        """
        all_probs = self.probs()  # (samples,classes)
        return quadratic_uncertainty(all_probs, as_vec=as_vec)



class EnsembleModel(Model):
    """Collect the outputs of a series of models to allow ensemble based analysis.  

    :param modelprefix: string prefix to identify this set of models. 
    :param data: string to identify the dataset that set of models is evaluated on. 
    """

    def __init__(self, modelprefix, data):
        self.modelprefix = modelprefix
        self.models = {}  ## dict of dicts- key is modelname, value is dictionary of preds/labels.
        self.data = data

    def register(self,
                 filename,
                 modelname,
                 inputtype=None,
                 labelpath=None,
                 logits=True,
                 npz_flag = None,
                 mask_array=None):
        """Register a model's predictions to this ensemble object. 
        :param filename: (string) path to file containing logit model predictions. 
        :param modelname: (string) name of the model to identify within an ensemble
        :param inputtype: (optional) [h5,npy] h5 inputs or npy input, depending on if we're looking at imagenet or cifar10 datasets. If not given, will be inferred from filename extension.   
        :param labelpath: (optional) if npy format files, labels must be given. 
        :param logits: (optional) we assume logits given, but probs can also be given directly. 
	      :param npz_flag: if filetype is .npz, we asume that we need to pass a dictionary key to retrieve either `cifar10` or `cinic10` logits.
        """
        model = Model(modelname, 'data')
        model.register(filename=filename,
                     inputtype=inputtype,
                     labelpath=labelpath,
                     logits=logits,
                     npz_flag=npz_flag,
                     mask_array=mask_array
        )
        self.models[modelname] = model

    def probs(self):
        """Calculates mean confidence across all softmax output.

        :return: array of shape samples, classes giving per class variance.
        """
        all_probs = []
        for model,modeldata in self.models.items():
            probs = modeldata.probs()
            all_probs.append(probs)
        array_probs = np.stack(all_probs,axis = 0)
        self._probs = np.mean(array_probs, axis = 0)
        return self._probs


    def labels(self):
        for model,modeldata in self.models.items():
            labels = modeldata.labels()
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
            probs = modeldata.probs()
            all_probs.append(probs)
        all_probs = np.stack(all_probs,axis = 0) # (models,samples,classes)
        var = np.var(all_probs,axis = 0,ddof = 1)
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
            probs = modeldata.probs()
            all_probs.append(probs)
        array_probs = np.stack(all_probs,axis = 0)
        bsd = BrierScoreData()
        model_bs = np.array([bsd.brierscore_multi_vec(m.probs(),m.labels()) for m in self.models.values()])
        return np.mean(model_bs,axis = 0)

    def get_bias_bs(self):
        """Given a brier score, estimate bias across the dataset.   
        """
        all_probs = []
        for model,modeldata in self.models.items():
            probs = modeldata.probs()
            all_probs.append(probs)
        array_probs = np.stack(all_probs,axis = 0)
        bsd = BrierScoreData()
        model_bs = np.array([bsd.brierscore_multi_vec(m.probs(),m.labels()) for m in self.models.values()])
        return np.mean(np.mean(model_bs,axis = 0))
    
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

    def get_avg_nll(self):
        """estimate the average NLL across the ensemble. 

        """
        all_nll = []
        for model,modeldata in self.models.items():
            probs = modeldata.probs()
            targets = modeldata.labels()
            all_nll.append(-np.log(probs[np.arange(len(targets)),targets]))

        array_nll = np.stack(all_nll,axis = 0) # (models,samples)
        return np.mean(np.mean(array_nll,axis = 0))

    def get_nll_div(self):    
        """estimate diversity between ensembles members corresponding to the jensen gap between ensemble and single
        model nll

        """
        all_probs = []
        for model,modeldata in self.models.items():
            probs = modeldata.probs()
            targets = modeldata.labels()
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
            

        all_probs = np.stack(all_probs,axis = 0) # (models,samples)
        norm_term = np.log(np.mean(all_probs,axis = 0))
        diversity = np.mean(-np.mean(np.log(all_probs),axis = 0)+norm_term)
        return diversity 

    def get_pairwise_corr(self):
        """Get pairwise correlation between ensemble members:

        """
        all_probs = []
        M = len(self.models)
        for model,modeldata in self.models.items():
            probs = modeldata.probs()
            targets = modeldata.labels()
            all_probs.append(probs)
        array_probs = np.stack(all_probs,axis = 0) # (models,samples,classes)
        array_pred_labels = np.argmax(array_probs,axis = -1) # (models,samples)
        compare = array_pred_labels[:,None,:] == array_pred_labels # (models,models,samples)
        means = np.mean(compare.astype(int),axis = -1) ## (models,models)
        mean_means = np.sum(np.tril(means,k=-1))/((M*(M-1))/2) ## (average across all pairs)
        return 1-mean_means

    def get_avg_qunc(self, as_vec=False):
        """estimate the average single model uncertainty.
        """
        all_unc = []
        for model, modeldata in self.models.items():
            probs = modeldata.probs()
            model_unc = quadratic_uncertainty(probs, as_vec=True)  # (samples)
            all_unc.append(model_unc)

        all_unc = np.stack(all_unc, axis=0)  # (models,samples)
        all_unc = np.mean(all_unc, axis=0)  # (samples)
        if as_vec:
            return all_unc
        else:
            return np.mean(all_unc)

    def get_ens_qunc(self, as_vec=False):
        """estimate the average single model uncertainty.

        """
        all_probs = self.probs()  # (samples,classes)
        return quadratic_uncertainty(all_probs, as_vec=as_vec)
    def _get_diversity_score(self, metric):
        """Get average disagreement between ensemble members:

        """

        def disagreement(logits_1, logits_2):
            """Disagreement between the predictions of two classifiers."""
            preds_1 = np.argmax(logits_1, axis=-1)
            preds_2 = np.argmax(logits_2, axis=-1)
            return preds_1 != preds_2

        def cosine_distance(x, y):
            """Cosine distance between vectors x and y."""
            x_norm = np.sqrt(np.sum(np.power(x, 2), axis=-1))
            x_norm = np.reshape(x_norm, (-1, 1))
            y_norm = np.sqrt(np.sum(np.power(y, 2), axis=-1))
            y_norm = np.reshape(y_norm, (-1, 1))
            normalized_x = x / x_norm
            normalized_y = y / y_norm
            return np.sum(normalized_x * normalized_y, axis=-1)

        def kl_divergence(p, q):
            return np.sum(p * np.log(p / q), axis=-1)
        diversity_metric = {'avg_disagreement': disagreement,
                            'cosine_similarity': cosine_distance,
                            'kl_divergence': kl_divergence}

        all_probs = []
        all_disagreements = []
        M = len(self.models)
        for model,modeldata in self.models.items():
            probs = modeldata["preds"]
            targets = modeldata["labels"]
            all_probs.append(probs)
        array_probs = np.stack(all_probs,axis = 0) # (models,samples,classes)
        array_pred_labels = np.argmax(array_probs,axis = -1) # (models,samples)
        for pair in list(itertools.combinations(range(M), 2)):
            all_disagreements.append(np.sum(diversity_metric[metric](array_probs[pair[0]],array_probs[pair[1]])))
        return np.mean(all_disagreements)/array_probs.shape[1]
    def get_diversity_score(self, metric='pairwise_corr'):

        if metric == 'pairwise_corr':
            return self.get_pairwise_corr()
        else:
            return self._get_diversity_score(metric=metric)

    def get_avg_qunc(self, as_vec=False):
        """estimate the average single model uncertainty.

        """
        all_unc = []
        for model, modeldata in self.models.items():
            probs = modeldata["preds"]
            model_unc = quadratic_uncertainty(probs, as_vec=True)  #(samples)
            all_unc.append(model_unc)

        all_unc = np.stack(all_unc, axis=0)  # (models,samples)
        all_unc = np.mean(all_unc, axis=0)  # (samples)
        if as_vec:
            return all_unc
        else:
            return np.mean(all_unc)

    def get_ens_qunc(self, as_vec=False):
        """estimate the average single model uncertainty.

        """
        all_probs = self.probs()  # (samples,classes)
        return quadratic_uncertainty(all_probs, as_vec=as_vec)


class Model(object):
    def __init__(self,  modelprefix, data):
        self.modelprefix = modelprefix
        self.data = data

    def register(self, filename, inputtype=None,labelpath = None, logits = True,npz_flag = None):
        """Register a model's predictions to this ensemble object. 
        :param filename: (string) path to file containing logit model predictions. 
        :param inputtype: (optional) [h5,npy] h5 inputs or npy input, depending on if we're looking at imagenet or cifar10 datasets. If not given, will be inferred from filename extension.   
        :param labelpath: (optional) if npy format files, labels must be given. 
        :param logits: (optional) we assume logits given, but probs can also be given directly. 
	:param npz_flag: if filetype is .npz, we asume that we need to pass a dictionary key to retrieve either `cifar10` or `cinic10` logits.
        """
        if inputtype is None:
            _,ext = os.path.splitext(filename)
            inputtype = ext[1:] 
            assert inputtype in ["h5","hdf5","npy","npz","pickle"], "inputtype inferred from extension must be either (`h5`, `npy`,`npz` `pickle`) if not given, not {}.".format(inputtype)
            
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
        elif inputtype == 'pickle':
            if logits:
                self._logits = pd.read_pickle(filename)['logits']
                self._labels = np.load(labelpath)
                self._probs = np.exp(self._logits) / np.sum(np.exp(self._logits), 1, keepdims=True)
            else:
                raise NotImplementedError("If inputtype is pickle, logits must be True.")

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

    def get_qunc(self):
        return quadratic_uncertainty(self.probs())
