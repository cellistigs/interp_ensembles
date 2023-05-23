import numpy as np
import os 
from joblib import Memory
from scipy.stats import gaussian_kde

here = os.path.dirname(os.path.abspath(__file__))

cachedir = os.path.join(here,"../../etc/")
memory = Memory(cachedir,verbose = 0)
## Density estimation module. 

class Variance_Decomp():
    """Variance decomposition kde estimates. KDE for joint, marginals, and conditional distributions.  

    """

    def __init__(self,metricmin,metricmax,varmin,varmax,sample_points,cutoff = 1e-6):
        """Give the minimum and maximum values of the metric and variance you want to consider when generating this kde plot. Also give the total number of sample points that you want, and the class will try to approximately distribute these points into a grid that respects the aspect ratio between the metric range and the variance range. 

        :param metricmin: min value of metric when gridding
        :param metricmax: max value of metric when gridding
        :param varmin: min value of variance when gridding
        :param varmax: max value of variance when gridding
        :param sample_points: base number of sample points per dimension. Will be scaled by the range of each dimension. 
        :param cutoff: density which we should consider 0. important when calculating our conditional distributions. 
        """
        self.metricrange = (metricmin,metricmax)
        self.varrange = (varmin,varmax)
        self.sample_points = sample_points
        ## we want to 
        xx,yy = np.mgrid[metricmin:metricmax:int(abs(metricmax-metricmin)*sample_points)*1j,varmin:varmax:int(abs(varmax-varmin)*sample_points)*1j]
        self.xx = xx ## repeated grid
        self.yy = yy ## repeated grid 
        self.cutoff = 1e-6

    def joint_kde(self,metric,var,bw_method = None):    
        """ Return the joint kde estimate of metric and variance. 

        :param metric: one dimensional array of data giving metric values. 
        :param var: one dimeionsional array of data giving corresponding variance values.
        :param bw_method: method for choosing bandwidth. 
        """
        sample_positions = np.vstack([self.xx.ravel(),self.yy.ravel()])
        kernel = gaussian_kde(np.stack([metric,var],axis = 0),bw_method = bw_method)
        f = np.reshape(kernel(sample_positions).T,self.xx.shape)
        f[f< self.cutoff] = 0
        return f
        

    def marginal_metric_kde(self,metric,bw_method= None):
        """ Return the marginal kde estimate of metric. 

        :param metric: one dimensional array of data giving metric values. 
        :param var: one dimeionsional array of data giving corresponding variance values.
        :param bw_method: method for choosing bandwidth. 
        """
        sample_positions = np.unique(self.xx)
        kernel = gaussian_kde(metric,bw_method = bw_method)
        f = kernel(sample_positions)
        f[f<self.cutoff] = 0
        return f

    def conditional_variance_kde(self,metric,var,bw_method = None):
        """Return the conditional kde estimate of variance given metric. 

        """
        joint = self.joint_kde(metric,var,bw_method = bw_method)    
        marg = self.marginal_metric_kde(metric,bw_method = bw_method)    
        ## this gives one array of shape len(metrics),len(var) and one of shape len(metrics)
        conditional = (joint.T/marg).T
        conditional[np.isnan(conditional)] = 0 
        conditional[conditional == np.inf] = 0 
        return conditional





