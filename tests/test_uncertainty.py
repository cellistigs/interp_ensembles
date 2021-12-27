import interpensembles.uncertainty as uncertainty
import numpy as np

def test_variance_c_perclass():
    e = 5
    assert uncertainty.variance_c_perclass(0,e) == uncertainty.variance_c_perclass(0,e) == 0 
    assert uncertainty.variance_c_perclass(0,e) == uncertainty.variance_c_perclass(0,e) == 0 
    assert np.isclose(uncertainty.variance_c_perclass(0,e),0)
    assert np.isclose(uncertainty.variance_c_perclass(1,e),uncertainty.variance_c_perclass(e-1,e))
    assert np.isclose(uncertainty.variance_c_perclass(e-1,e),0.16)
    assert np.isclose(uncertainty.variance_c_perclass(2,e),uncertainty.variance_c_perclass(e-2,e))
    assert np.isclose(uncertainty.variance_c_perclass(e-2,e),0.24)

class Test_ConfidenceMax():
    e = 5
    k = 10

    def test_init(self):
        uncertainty.ConfidenceMax(self.k)

    def test_get_maxpoints(self):        
        cm = uncertainty.ConfidenceMax(self.k)
        corr = cm.get_maxpoints(self.e)
        assert corr[0,0] == 0
        assert corr[-1,0] == 0
        assert np.isclose(corr[1,0],0.24)
        assert np.isclose(corr[2,0],0.16)

class Test_LikelihoodMax():
    e = 5
    k = 10

    def test_init(self):
        uncertainty.LikelihoodMax()

    def test_get_maxpoints(self):        
        lm = uncertainty.LikelihoodMax()
        corr = lm.get_maxpoints(self.e)
        assert corr[0,0] == 0
        assert corr[-1,0] == 0
        assert np.isclose(corr[1,0],0.16)
        assert np.isclose(corr[2,0],0.24)
        assert np.isclose(corr[3,0],0.24)
        assert np.isclose(corr[4,0],0.16)

class Test_BrierScoreMax():    
    e = 5
    k = 10

    def test_init(self):
        uncertainty.BrierScoreMax(self.k)
        
    def test_get_maxpoints_corr(self):    
        bsm = uncertainty.BrierScoreMax(self.k)
        corr = bsm.get_maxpoints_corr(self.e)
        assert corr[0,0] == 0
        assert corr[-1,0] == 0
        assert np.isclose(corr[1,0],0.32/10)
        assert np.isclose(corr[-2,0],0.32/10)
        assert np.isclose(corr[-3,0],0.48/10)

    def test_get_maxpoints_uncorr(self):
        bsm = uncertainty.BrierScoreMax(self.k)
        corr = bsm.get_maxpoints_uncorr(self.e)
        assert np.isclose(corr[0,0],(0.16)/2)
        assert corr[-1,0] == 0
        assert np.isclose(corr[1,0],0.16/2)
        assert np.isclose(corr[2,0],(0.24+3*0.16)/10)

