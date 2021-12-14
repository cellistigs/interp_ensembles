"""
Module to implement MMD related code. 
Includes: 
- kernel functions for feature maps.  
- mmd test statistic computation 
- significance threshold computation 
- witness function computation 

"""
import numpy as np 

class MMDKernel(): 
    """Base class of different kernel functions for MMD. 

    """
    def __init__(self,dimension):
        """Initialize with parameters. Set the dimension of data we are working with.

        """
        assert type(dimension) == int
        assert dimension > 0 
        self.dimension = dimension
    
    def __call__(self,data1,data2):
        """Given two arrays, each of shape (batch size, dimension), calculates the kernel value evaluated at each row. 

        :param data1: a numpy array of shape batch size, dimension
        :param data2: a numpy array of shape batch size, dimension
        :returns: an array of kernel evaluations of shape (batch size,)
        """
        raise NotImplementedError
        
class RBFKernel(MMDKernel):
    """RBF kernel. Takes parameter `l`, giving the length scale of the data (assume scalar for now.)

    :param dimension: data dimensionality
    :param l: length scale > 0
    :ivar K: the maximum value this kernel can take. 
    """
    def __init__(self,dimension,l):
        super().__init__(dimension)
        assert l > 0 
        self.l = l 
        self.K = 1

    def __call__(self,data1,data2):    
        """ Given two arrays, calculates: 

        $exp(-\frac{\|array1-array2\|^2}{2l^2})$

        :param data1: a numpy array of shape (...,batch size, dimension)
        :param data2: a numpy array of shape (...,batch size, dimension)
        :returns: an array of kernel evaluations of shape (...,batch size,)
        """
        assert data1.shape[-1] == self.dimension
        assert data2.shape[-1] == self.dimension
        logrbf = -(np.linalg.norm(data1-data2,axis = -1)**2)/(2*self.l**2)
        rbf = np.exp(logrbf)
        return rbf

class MMDModule(): 
    """Module to compute mmd test statistics, thresholds, and witness functions given data. Initialized with an MMDKernel, and given the following methods:  
    - compute_witness: compute the witness function given two samples. Return the witness function so that it can be evaluated on data of choice. 
    - compute_mmd2: compute $MMD_u^2$ from Gretton 2021a- an unbiased (not necessarily non-negative) estimate of the squared MMD between samples for potentially different numbers of samples. 
    - compute_threshold_distribution_free
    - todo: compute threshold based on asymptotic distribution of test. 

    """
    def __init__(self,kernel):
        """Initialize with the kernel that we want to use. 
        """
        self.kernel = kernel
        self.K = kernel.K # max value 
        self.dimension = kernel.dimension
    
    def compute_witness(self,data1,data2):
        """The witness function can be estimated as the inner product of the feature map and the difference in mean embeddings. We can do finite sample estimates of this quantity: 

        """
        ## assuming correctly shaped inputs: 
        sample1_factor = lambda t: np.mean(self.kernel(data1,t),axis = 1)
        sample2_factor = lambda t: np.mean(self.kernel(data2,t),axis = 1)

        def witness(factor):
            """We are going to project the evaluation point into a numpy array of the correct shape. 

            :param factor: an array of shape (evalpoints,dim)
            :returns: an array of shape (evalpoints,)
            """
            repeated1 = np.repeat(factor.reshape(len(factor),1,self.dimension),repeats = len(data1),axis =1) 
            repeated2 = np.repeat(factor.reshape(len(factor),1,self.dimension),repeats = len(data2),axis =1) 
            witnesseval = sample1_factor(repeated1) - sample2_factor(repeated2)
            return witnesseval

        return witness

    def compute_mmd2():
        """TODO

        """
    
    def compute_threshold_distribution_free():
        """TODO

        """



