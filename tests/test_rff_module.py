## test the rff module
from hydra import compose, initialize
import interpensembles
from interpensembles.module import CIFAR10RFFModule

class Test_CIFAR10RFFModule():
    """Run the following tests: 

    1. are individual submodels different in initialization from each other? 
    2. are the learning rates set properly? 
    """
    with initialize(config_path="test_mats/test_configs/", job_name="test_app"):
        cfg = compose(config_name="run_default_rff", overrides=["nb_models=3"])
    
    def test_init(self):
        ens = CIFAR10RFFModule(self.cfg)
    def test_opt(self):
        ens = CIFAR10RFFModule(self.cfg)
        opt = ens.configure_optimizers()
        print(opt.__dict__)

