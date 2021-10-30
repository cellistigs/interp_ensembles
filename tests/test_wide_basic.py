## 
import interpensembles.cifar10_models.resnet as resnets
from torchviz import make_dot, make_dot_from_trace
import torch
import numpy as np

def test_resnet():
    
    model = resnets.resnet18().double()
    dummy_input = torch.tensor(np.random.randn(1,3,32,32))
    y = model(dummy_input.double())
    dot = make_dot(y[0].mean(),params = dict(model.named_parameters()))
    dot.render("test_mats/test_resnet18_basic",view = True)

def test_wresnet():
    
    model = resnets.wideresnet18().double()
    dummy_input = torch.tensor(np.random.randn(1,3,32,32))
    y = model(dummy_input.double())
    dot = make_dot(y[0].mean(),params = dict(model.named_parameters()))
    dot.render("test_mats/test_resnet18_2x",view = True)

def test_paramcount():    
    """Test that the wide resnet has approximately 4x the number of trainable parameters. 

    """
    def count_parameters(model):

        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    model = resnets.resnet18().double()
    widemodel = resnets.wideresnet18().double()
    ## the parameter count should be dominated by square convolutions. Figure out a better way to count these later. 
    assert count_parameters(model)*4/count_parameters(widemodel) -1 < 0.001
