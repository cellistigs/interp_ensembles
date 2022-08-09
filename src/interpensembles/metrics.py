## Tools to help calculate calibration related metrics given a predictions and labels.  
import numpy as np

class BrierScoreData(object):
    """Calculates brier score. 

    """
    def __init__(self):
        pass

    def brierscore(self,prob,target):
        """Given an array of probabilities, `prob`, (batch,dim) and an array of targets `target` (dim), calculates the brier score as if we were in the binary case: take the predicted probability of the target class, and just calculate based on that.
        :param prob: array of probabilities per class. 
        :param target: list/array of true targets. 

        """
        probs = prob[np.arange(len(target)),target]
        deviance = probs-np.ones(probs.shape)
        
        return np.mean(deviance**2)

    def brierscore_multi(self,prob,target):
        """The "original" brier score definition that accounts for other classes explicitly. Note the range of this test is 0-2. 
        :param prob: array of probabilities per class. 
        :param target: list/array of true targets. 

        """
        target_onehot = np.zeros(prob.shape)
        target_onehot[np.arange(len(target)),target] = 1 ## onehot encoding. 
        deviance = prob-target_onehot
        return np.mean(np.sum(deviance**2,axis = 1))

class VarianceData(object):
    """Calculates variance/related metrics. In particular, this is the variance in the confidence of the top predicted label. 

    """
    def __init__(self,modelprefix,data):
        """Takes a modelprefix that specifies the kinds of models over which we will be calculating variance. 

        :param modelprefix: a string specifying what model names should start with. 
        :param data: ind/ood, specifying if this class is calculating variance for in or out of distribution data. 
        """
        self.modelprefix = modelprefix
        self.models = {} ## dict of dicts- key is modelname, value is dictionary of preds/labels.
        self.data = data
    
    def register(self,preds,labels,modelname):
        """Takes predictions and labels. as a rough check, will assert that the labels match those that have already been registered.  
        :param preds: assumes softmax applied. 
        :param labels: vector of integers. 
        :param modelname: enforce that models we are calculating diversity over have the same model prefix. 
        """
        assert modelname.startswith(self.modelprefix); "modelname must start with modelprefix"
        for model,modeldata in self.models.items():
            assert np.all(labels == modeldata["labels"]); "labels must match already registered." 
            assert np.allclose(np.sum(preds,axis = 1), 1); "predictions must be probabilities"
        self.models[modelname] = {"preds":preds,"labels":labels}    
    
    def mean_conf(self):
        """Calculates mean confidence across all softmax output. 

        :return: array of shape samples, classes giving per class variance. 
        """
        all_probs = []
        for model,modeldata in self.models.items():
            probs = modeldata["preds"]
            all_probs.append(probs)
        array_probs = np.stack(all_probs,axis = 0)    
        return np.mean(array_probs, axis = 0)

    def variance(self):
        """Calculates variance in confidence across all softmax output. 

        :return: array of shape samples,classes giving per class variance. 
        """
        all_probs = []
        for model,modeldata in self.models.items():
            probs = modeldata["preds"]
            all_probs.append(probs)
        array_probs = np.stack(all_probs,axis = 0)    
        return np.var(array_probs, axis = 0)

    def variance_confidence(self):
        """Calculates an array of shape (sample, 2) that gives the mean and variance of confidence estimates together per datapoint.   

        """
        target = self.models[list(self.models.keys())[0]]["labels"]

        all_vars = self.variance()
        all_conf = self.mean_conf()
        tprob_vars = all_vars[np.arange(len(target)),target]
        tprob_confs = all_conf[np.arange(len(target)),target]
        return np.stack([tprob_confs,tprob_vars],axis = 1)

    def expected_variance(self):
        """Calculates expected variance across y|x and x. This is just selecting the variance in the top probability for y|x, and then averaging over all examples for x. 

        :return: scalar expected variance
        """
        target = self.models[list(self.models.keys())[0]]["labels"]

        all_vars = self.variance()
        tprob_vars = all_vars[np.arange(len(target)),target]
        return np.mean(tprob_vars)

class AccuracyData(object):
    """Calculates accuracy related metrics. 

    """
    def __init__(self):
        pass
    
    def accuracy(self,prob,target):
        """Given predictions (example,class) and targets (class), will calculate the accuracy.  

        """
        selected = np.argmax(prob, axis = 1)
        correct = target == selected
        accuracy = sum(correct)/len(target)
        return accuracy

class NLLData(object):
    """Calculates the negative log likelihood of the data. 

    """
    def __init__(self):
        pass
    
    def nll(self,prob,target,normalize = False):
        """Given predictions (example,class) and targets (class), will calculate the negative log likelihood. Important here that the probs are expected to be outputs of softmax functions.   

        """
        probs = prob[np.arange(len(target)),target]
        logprobs = np.log(probs)

        nll = -sum(logprobs)
        if normalize:
            nll= nll/len(logprobs)
        return nll

class CalibrationData(object):
    """Initializes an object to bin predictions that are fed to it in batches.  

    """
    def __init__(self,binedges):
        """Initialize with a set of floats giving the interval spacing between different bins

        :param binedges: list of edges of the bins, not including 0 and 1. Will create intervals like `[[0,binedges[0]),[binedges[0],binedges[1]),...,[binedges[-1],100]]`  
        """
        assert binedges[0] > 0 and binedges[-1] < 1, "bin edges must be strictly within limits."
        assert np.all(np.diff(binedges)> 0), "bin edges must be ordered" 
        assert type(binedges) == list
        padded = [0] + binedges + [1] 
        self.binedges = [(padded[i],padded[i+1]) for i in range(len(padded)-1)]

    def bin(self,prob,target):
        """Given predictions  (example, class) and targets  (class), will bin them according to the binedges parameter.
        Returns a dictionary with keys giving bin intervals, and values another dictionary giving the accuracy, confidence, and number of examples in the bin. 
        """
        data = self.analyze_batch(prob,target)
        ## first let's divide the data by bin: 
        bininds = np.array(list(data["bin"]))
        bin_assigns = [np.where(bininds == i) for i in range(len(self.binedges))]
        ## now we want per-bin stats:
        all_stats = {} 
        for ai,assignments in enumerate(bin_assigns):
            bin_card = len(assignments[0]) 
            name = self.binedges[ai]
            if bin_card == 0:
                bin_conf = np.nan
                bin_acc = np.nan
            else:    
                bin_conf = sum(data["maxprob"][assignments])/bin_card
                bin_acc = sum(data["correct"][assignments])/bin_card
            all_stats[name] = {"bin_card":bin_card,"bin_conf":bin_conf,"bin_acc":bin_acc}
        return all_stats    

    def ece(self,prob,target):
        """Calculate the expected calibration error across bins given a probability and target. 

        """
        all_stats = self.bin(prob,target)
        ece_nonnorm = 0 
        for interval,intervalstats in all_stats.items():
            if intervalstats["bin_card"] == 0:
                continue
            else:
                factor = intervalstats["bin_card"]*abs(intervalstats["bin_acc"]-intervalstats["bin_conf"])
                ece_nonnorm += factor
        ece = ece_nonnorm/len(target)         
        return ece


    def getbin(self,data):
        """Halt and return the index where your maximum prediction fits into a bin. 

        """
        index = len(self.binedges)-1 
        for b in self.binedges[::-1]: ## iterate in reverse order
            if data >= b[0]:
                break
            else:
                index -= 1
        return index        

    def analyze_batch(self,prob,target):
        """Given a matrix of class probabilities (batch, class) and a target (class), returns calibration related info about that datapoint: 
        {"prob":prob,"target":target,"correct":bool,"bin":self.binedges[index]}

        """
        assert len(prob.shape) == 2
        assert prob.shape[0] == len(target)
        maxprob,maxind = np.amax(prob,axis = 1),np.argmax(prob,axis= 1)
        correct = maxind == target
        binind = map(self.getbin,maxprob)
        return {"maxprob":maxprob,"maxind":maxind,"target":target,"correct":correct,"bin":binind}



