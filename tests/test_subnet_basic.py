## 
import interpensembles.cifar10_models.resnet as resnets
import pytest
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

def test_subresnet_zeroes():
    """Test that each layer and the final layer return the right number of zeroes. 

    """
    basemodel = resnets.WideResNet(resnets.BasicBlock,[2,2,2,2],k = 2).double()
    model = resnets.widesubresnet18(basemodel,1).double()
    dummy_input = torch.tensor(np.random.randn(1,3,32,32))
    y = model.before_fc(dummy_input.double())
    all_vals = np.prod(np.shape(y.detach().numpy()))
    assert len(np.where(y.detach().numpy())[0]) == all_vals/2 

def test_subresnet_zeroes_one():
    """Test that each layer and the final layer return the right number of zeroes. 

    """
    basemodel = resnets.WideResNet(resnets.BasicBlock,[2,2,2,2],k = 2).double()
    model = resnets.widesubresnet18(basemodel,0).double()
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


@pytest.mark.parametrize("model0,model1", [(0, 1), (1, 2), (2, 3), (1,0), (2,1),(3,2),(0,2),(0,3),(1,3)])
def test_gradient_coupling(model0,model1):
    """Get responses from both before and after taking a gradient step. 
    Update one with gradients. Get responses from both after. 

    """
    basemodel = resnets.WideResNet(resnets.BasicBlock,[2,2,2,2],k = 2)
    model0 = resnets.widesubresnet18(basemodel,model0)
    model1 = resnets.widesubresnet18(basemodel,model1)
    dummy_input = torch.tensor(np.random.randn(1,3,32,32)).float()
    ## ## before gradient step: 
    y0 = model0.before_fc(dummy_input)
    y1 = model1.before_fc(dummy_input)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(list(model0.parameters())+list(model1.parameters()),lr = 0.1,momentum=0.9)
    optimizer.zero_grad() 
    loss_fn(y0.mean(),torch.tensor([0.0])).backward()
    optimizer.step()
    optimizer.step()
    y0_post = model0.before_fc(dummy_input)
    y1_post = model1.before_fc(dummy_input)
    assert not torch.all(y0_post == y0)
    assert torch.all(y1_post == y1)
    

    


