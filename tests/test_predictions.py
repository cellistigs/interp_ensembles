import pytest
import numpy as np
from interpensembles.predictions import EnsembleModel
from hydra import compose, initialize

class Test_EnsembleModel():
    """Test the ability to cross analyze both imagenet and pytorch ensembles

    """
    with initialize(config_path="./test_mats/test_configs/test_predictions",job_name="test_app"):
        cfg_img = compose(config_name="imagenet")
        cfg_cifar = compose(config_name="cifar")

    def test_init_imagenet(self):
        ens = EnsembleModel("imagenet_test","ind")

    def test_init_cifar(self):
        ens = EnsembleModel("cifar_test","ind")

    def test_register_imagenet_explicit(self):    
        """explicitly give the inputtype and register models. 

        """
        ens = EnsembleModel("imagenet_test","ind")
        filenames = [f for f in self.cfg_img.filenames]
        [ens.register(f,i,"hdf5") for i,f in enumerate(filenames)]
        assert np.allclose(np.sum(ens.models[1]["preds"],axis = -1),1)
        assert not np.allclose(np.sum(ens.models[1]["logits"],axis = -1),1)
    
    def test_register_imagenet_implicit(self):    
        """implicitly register models. 

        """
        ens = EnsembleModel("imagenet_test","ind")
        filenames = [f for f in self.cfg_img.filenames]
        [ens.register(f,i) for i,f in enumerate(filenames)]
        assert np.allclose(np.sum(ens.models[1]["preds"],axis = -1),1)
        assert not np.allclose(np.sum(ens.models[1]["logits"],axis = -1),1)

    def test_register_cifar_explicit(self):
        """explicitly give the inputtype and register models. 

        """
        ens = EnsembleModel("cifar_test","ind")
        filenames = [f for f in self.cfg_cifar.filenames]
        labelpaths = [f for f in self.cfg_cifar.labelpaths]
        [ens.register(f,i,"npy",l) for i,(f,l) in enumerate(zip(filenames,labelpaths))]
        assert np.allclose(np.sum(ens.models[1]["preds"],axis = -1),1)
        assert not np.allclose(np.sum(ens.models[1]["logits"],axis = -1),1)
        
    def test_register_cifar_implicit(self):
        """implicitly register models. 

        """
        ens = EnsembleModel("cifar_test","ind")
        filenames = [f for f in self.cfg_cifar.filenames]
        labelpaths = [f for f in self.cfg_cifar.labelpaths]
        [ens.register(f,i,labelpath=l) for i,(f,l) in enumerate(zip(filenames,labelpaths))]
        assert np.allclose(np.sum(ens.models[1]["preds"],axis = -1),1)
        assert not np.allclose(np.sum(ens.models[1]["logits"],axis = -1),1)

    def test_biasvar_imagenet(self):
        """Calculate bias and variance from imagenet models. 

        """
        ens = EnsembleModel("imagenet_test","ind")
        filenames = [f for f in self.cfg_img.filenames]
        [ens.register(f,i) for i,f in enumerate(filenames)]
        var = ens.get_variance()
        bias = ens.get_bias_bs()
        print(var,bias)
        assert 0 


    def test_biasvar_cifar(self):
        """Calculate bias and variance from cifar models. 

        """
        ens = EnsembleModel("cifar_test","ind")
        filenames = [f for f in self.cfg_cifar.filenames]
        labelpaths = [f for f in self.cfg_cifar.labelpaths]
        [ens.register(f,i,labelpath=l) for i,(f,l) in enumerate(zip(filenames,labelpaths))]
        var = ens.get_variance()
        bias = ens.get_bias_bs()
        print(var,bias)
        assert 0 

