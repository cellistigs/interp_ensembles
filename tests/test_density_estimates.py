## 
import numpy as np
import matplotlib.pyplot as plt
import interpensembles.density_estimates as density_estimates
import os 

here = os.path.dirname(os.path.abspath(__file__))
test_mats = os.path.join(here,"test_mats")

class Test_Variance_Decomp():
    def test_init(self):
        xmin = 0
        xmax = 1
        ymin = 0 
        ymax = 1
        sample_points = 10
        de = density_estimates.Variance_Decomp(xmin,xmax,ymin,ymax,sample_points) 
    
    def test_joint_kde(self):
        xmin = 0
        xmax = 1
        ymin = 0 
        ymax = 2
        sample_points = 100
        de = density_estimates.Variance_Decomp(xmin,xmax,ymin,ymax,sample_points) 
        a = np.sin(np.linspace(0,1))
        b = np.cos(np.linspace(0,1))

        joint = de.joint_kde(a,b)
        plt.imshow(joint,extent = [ymin,ymax,xmin,xmax],origin = "lower")
        plt.savefig(os.path.join(test_mats,"joint_kde"))
        plt.close()

    def test_marg_kde(self):    
        xmin = 0
        xmax = 1
        ymin = 0 
        ymax = 2
        sample_points = 10
        de = density_estimates.Variance_Decomp(xmin,xmax,ymin,ymax,sample_points) 
        a = np.sin(np.linspace(0,1))
        b = np.cos(np.linspace(0,1))

        marg = de.marginal_metric_kde(a)
        plt.plot(marg)
        plt.savefig(os.path.join(test_mats,"marg_kde"))
        plt.close()

    def test_cond_kde(self):    
        xmin = 0  
        xmax = 1
        ymin = 0 
        ymax = 2
        sample_points = 100
        de = density_estimates.Variance_Decomp(xmin,xmax,ymin,ymax,sample_points) 
        a = np.sin(np.linspace(0,1))
        b = np.cos(np.linspace(0,1))

        cond = de.conditional_variance_kde(a,b)
        plt.imshow(cond,extent = [ymin,ymax,xmin,xmax],origin = "lower")
        plt.savefig(os.path.join(test_mats,"cond_kde"))
        plt.close()
