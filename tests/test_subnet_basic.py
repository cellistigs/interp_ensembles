## 
import interpensembles.cifar10_models.resnet as resnets
from torchviz import make_dot, make_dot_from_trace
import torch
import numpy as np


def test_subresnet_graph():
    """Generates a graph. shared weights should be indicated as "baseresnet"
    """
    
    basemodel = resnets.WideResNet(resnets.BasicBlock,[2,2,2,2],k = 2).double()
    model = resnets.widesubresnet18(basemodel,1).double()
    dummy_input = torch.tensor(np.random.randn(1,3,32,32))
    y = model(dummy_input.double())
    dot = make_dot(y[0].mean(),params = dict(model.named_parameters()))
    dot.render("test_mats/test_subresnet18_basic",view = True)

def test_subresnet_zeroes_one():
    """Test that each layer and the final layer return the right number of zeroes. 

    """
    basemodel = resnets.WideResNet(resnets.BasicBlock,[2,2,2,2],k = 2).double()
    model = resnets.widesubresnet18(basemodel,0).double()
    dummy_input = torch.tensor(np.random.randn(1,3,32,32))
    y = model.before_fc(dummy_input.double())
    all_vals = np.prod(np.shape(y.detach().numpy()))
    assert len(np.where(y.detach().numpy())[0]) == all_vals/2 
    
def test_subresnet_zeroes():
    """Test that each layer and the final layer return the right number of zeroes. 

    """
    basemodel = resnets.WideResNet(resnets.BasicBlock,[2,2,2,2],k = 2).double()
    model = resnets.widesubresnet18(basemodel,1).double()
    dummy_input = torch.tensor(np.random.randn(1,3,32,32))
    y = model.before_fc(dummy_input.double())
    all_vals = np.prod(np.shape(y.detach().numpy()))
    assert len(np.where(y.detach().numpy())[0]) == all_vals/2 

def test_subresnet_zeroes_three():
    """Test that each layer and the final layer return the right number of zeroes. 

    """
    basemodel = resnets.WideResNet(resnets.BasicBlock,[2,2,2,2],k = 2).double()
    model = resnets.widesubresnet18(basemodel,2).double()
    dummy_input = torch.tensor(np.random.randn(1,3,32,32))
    y = model.before_fc(dummy_input.double())
    all_vals = np.prod(np.shape(y.detach().numpy()))
    assert len(np.where(y.detach().numpy())[0]) == all_vals/2 

def test_subresnet_zeroes_four():
    """Test that each layer and the final layer return the right number of zeroes. 

    """
    basemodel = resnets.WideResNet(resnets.BasicBlock,[2,2,2,2],k = 2).double()
    model = resnets.widesubresnet18(basemodel,3).double()
    dummy_input = torch.tensor(np.random.randn(1,3,32,32))
    y = model.before_fc(dummy_input.double())
    all_vals = np.prod(np.shape(y.detach().numpy()))
    assert len(np.where(y.detach().numpy())[0]) == all_vals/2 

