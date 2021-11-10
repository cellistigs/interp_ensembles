import pytest
import numpy as np
from interpensembles.calibration import CalibrationData

class Test_CalibrationData():
    @pytest.mark.parametrize("interval",([0,1],[0,0.40,1],[0,0.40,101],[0,0.40,1],[-10,0.30,0.50,0.90]))
    def test_init_interval(self,interval):
        with pytest.raises(AssertionError):
            CalibrationData(interval)
    def test_init_formatting(self):
        interval = [0.01,0.04,0.50,0.80]
        cd = CalibrationData(interval)
        assert cd.binedges == [(0,0.01),(0.01,0.04),(0.04,0.50),(0.50,0.80),(0.80,1)] 
    @pytest.mark.parametrize("testprob,bin",((1,3),(0.9,3),(0.6,3),(0.59,2),(0.039,1),(0.01,1),(0.005,0),(0,0)))
    def test_getbin(self,testprob,bin):    
        interval = [0.01,0.04,0.6]
        cd = CalibrationData(interval)
        assert cd.getbin(testprob) == bin

    @pytest.mark.parametrize("target,correct",(([0,0],[False,False]),([2,1],[True,False]),([0,2],[False,True])))
    def test_analyze_datapoint(self,target,correct):    
        interval = [0.01,0.04,0.50,0.80]
        cd = CalibrationData(interval)

        probs = np.array([[0.1,0.1,0.8],[0.2,0.3,0.5]])
        batchinfo = cd.analyze_batch(probs,target)
        assert np.all(batchinfo["maxprob"] == np.array([0.8,0.5]))
        assert np.all(batchinfo["maxind"] == np.array([2,2]))
        assert np.all(batchinfo["target"] == target)
        assert np.all(batchinfo["correct"] == correct)
        assert np.all(list(batchinfo["bin"]) == [4,3])
        

        probs = np.array([[0.2,0.3,0.5]])
        batchinfo = cd.analyze_batch(probs,target[1:])
        assert np.all(batchinfo["maxprob"] == np.array([0.5]))
        assert np.all(batchinfo["maxind"] == np.array([2]))
        assert np.all(batchinfo["target"] == target[-1:])
        assert np.all(batchinfo["correct"] == correct[-1:])
        assert np.all(list(batchinfo["bin"]) == [3])

    def test_bin(self):
        interval = [0.01,0.04,0.50,0.80]
        cd = CalibrationData(interval)

        probs = np.array([[0.15,0.15,0.7],[0.1,0.1,0.8],[0.3,0.3,0.4],[0.2,0.3,0.5]])
        target = [0,0,2,2]
        binned_stats = cd.bin(probs,target)
        assert binned_stats[(0.50,0.80)]["bin_conf"] == 0.6 
        assert binned_stats[(0.50,0.80)]["bin_acc"] == 0.5 
        assert binned_stats[(0.80,1)]["bin_conf"] == 0.8 
        assert binned_stats[(0.80,1)]["bin_acc"] == 0 
        assert binned_stats[(0.04,0.50)]["bin_conf"] == 0.4 
        assert binned_stats[(0.04,0.50)]["bin_acc"] == 1 
        assert np.isnan(binned_stats[(0,0.01)]["bin_acc"]) 
        assert np.isnan(binned_stats[(0,0.01)]["bin_conf"]) 

    def test_ece(self):
        interval = [0.01,0.04,0.50,0.80]
        cd = CalibrationData(interval)

        probs = np.array([[0.15,0.15,0.7],[0.1,0.1,0.8],[0.3,0.3,0.4],[0.2,0.3,0.5]])
        target = [0,0,2,2]
        ece = cd.ece(probs,target)
        assert ece == (0.1*2+0.8*1+0.6*1)/4

        