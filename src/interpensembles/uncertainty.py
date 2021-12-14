import numpy as np
## calculate uncertainty related quantities that are not data metrics. 
"""
We want to calculate quantities that give some intuition for metrics of interest in terms of ensemble performance.  

For example: given an ensemble, given a mean data likelihood of p, what is the variance in the data likelihood we would expect if models were perfectly confident one way or another? 

Or: given an ensemble, fixing the model brier score, what is the expected variance in probabilities across all classes? 

"""

def variance_c_perclass(c,e):
    """Given an ensemble of e networks and all models perfectly confident, fix a data class i. What is the variance across the ensemble in the probability of data class i given c models are confident with p = 1 about data class i? 

    """
    return (c*(1-c/e)**2+(e-c)*(-c/e)**2)/e ## variance term from the correct class. 

class BrierScoreMax():
    """Variance quantities related to the model brier score. The relevant variance quantity in this case is the average variance across all classes. 

    We can consider two cases: 
    1. errors perfectly correlated. This is close to what we observe. 
    2. errors perfectly uncorrelated. This should in theory also be possible. 

    """
    def __init__(self,k):
        """Give the number of classes, k to initialize. 

        """
        self.k = k

    def get_maxpoints_corr(self,e):    
        """Given the ensemble size, e, will calculate the ensemble brier score and variance corresponding to cases where c/e of the models are correct, and the errors are perfectly correlated.   

        :param e: ensemble size
        :returns: an array of shape (e+1,2) giving the variance and brier score, with array[c] giving the metrics for c networks correct. 
        """
        all_points = []
        for c in range(e+1):
            mean_ens_variance = (variance_c_perclass(c,e) + variance_c_perclass(e-c,e))/self.k
            brier_score = 2*(e-c)/e
            all_points.append([mean_ens_variance,brier_score])
        return np.array(all_points)    

    def get_maxpoints_uncorr(self,e):
        """Given the ensemble size, e, will calculate the ensemble brier score and variance corresponding to cases where c/e of the models are correct, and the errors are perfectly uncorrelated.   
        **Note: assumes that e < self.k + 1. If this is not true, the bound will be loose. 

        :param e: ensemble size
        :returns: an array of shape (e+1,2) giving the variance and brier score, with array[c] giving metrics for c networks correct. 
        """
        all_points = []
        for c in range(e+1):
            mean_ens_variance = (variance_c_perclass(c,e) + (e-c)*variance_c_perclass(1,e))/self.k
            brier_score = 2*(e-c)/e
            all_points.append([mean_ens_variance,brier_score])
        return np.array(all_points)    

