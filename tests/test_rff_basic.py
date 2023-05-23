## 
import interpensembles.cifar10_models.rff as rff
import pytest
import torch
from torch import nn
import numpy as np


def test_rff_graph():
    """Generates a graph. shared weights should be indicated as "baseresnet"
    """
    batch_size = 10
    in_dim = 100
    out_dim = 500
    classes = 2
   
    model = rff.RFF(in_dim,out_dim,classes)
    dummy_input = torch.tensor(np.random.randn(batch_size,in_dim))
    y = model(dummy_input.float())
    assert y.detach().numpy().shape == (batch_size,classes)

def test_grad():
    batch_size = 10
    in_dim = 100
    out_dim = 500
    classes = 2
   
    model = rff.RFF(in_dim,out_dim,classes)
    old_weights = model.project.weights.detach().numpy()
    old_offset = model.project.offset.detach().numpy()
    old_proj = model.regress.weight.detach().numpy()
    old_bias = model.regress.bias.detach().numpy()
    print(old_weights[0,0],"old")
    print(old_offset[0],"old")
    print(old_proj[0,0],"old (change)")
    print(old_bias[0],"old (change)")
    dummy_input = torch.tensor(np.random.randn(batch_size,in_dim))
    y = model(dummy_input.float())
    loss_fn = torch.nn.NLLLoss()
    optimizer = torch.optim.SGD(list(model.parameters()),lr = 0.1,momentum=0.9)
    optimizer.zero_grad() 
    loss = loss_fn(y.mean(axis = 0),torch.tensor([0,1])).backward()
    optimizer.step()
    optimizer.step()
    new_weights = model.project.weights.detach().numpy()
    new_offset = model.project.offset.detach().numpy()
    new_proj = model.regress.weight.detach().numpy()
    new_bias = model.regress.bias.detach().numpy()
    print(new_weights[0,0],"new")
    print(new_offset[0],"new")
    print(new_proj[0,0],"new (change)")
    print(new_bias[0],"new (change)")
    assert np.all(new_weights == old_weights)
    assert np.all(new_offset == old_offset)
    assert np.all(new_proj != old_proj)
    assert np.all(new_bias != old_bias)
