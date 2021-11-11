## 
import interpensembles.cifar10_models.resnet as resnets
import pytest
from torchviz import make_dot, make_dot_from_trace
import torch
from torch import nn
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
    assert not torch.all(y1_post == y1)

def test_gradient_coupling_subs():
    """Get responses from base models and child model before and after taking a gradient step. 
    Update one with gradients. Get responses from both after. 

    """
    basemodel = resnets.WideResNet(resnets.BasicBlock,[2,2,2,2],k = 2)
    model0 = resnets.widesubresnet18(basemodel,0)
    model1 = resnets.widesubresnet18(basemodel,1)
    model2 = resnets.widesubresnet18(basemodel,2)
    model3 = resnets.widesubresnet18(basemodel,3)
    dummy_input = torch.tensor(np.random.randn(1,3,32,32)).float()
    ## ## before gradient step: 
    ybase = basemodel(dummy_input)
    y0 = model0.before_fc(dummy_input)
    y1 = model1.before_fc(dummy_input)
    y2 = model2.before_fc(dummy_input)
    y3 = model3.before_fc(dummy_input)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(list(basemodel.parameters())+list(model0.parameters())+list(model1.parameters())+list(model2.parameters())+list(model3.parameters()),lr = 0.1,momentum=0.9)
    optimizer.zero_grad() 
    loss_fn(sum([y0.mean(),y1.mean(),y2.mean(),y3.mean()]),torch.tensor([0.0])).backward()
    optimizer.step()
    optimizer.step()
    ybase_post = basemodel(dummy_input)
    assert not torch.all(ybase == ybase_post)
def test_gradient_coupling_base():
    """Get responses from base models and child model before and after taking a gradient step. 
    Update one with gradients. Get responses from both after. 

    """
    basemodel = resnets.WideResNet(resnets.BasicBlock,[2,2,2,2],k = 2)
    model0 = resnets.widesubresnet18(basemodel,0)
    model1 = resnets.widesubresnet18(basemodel,1)
    model2 = resnets.widesubresnet18(basemodel,2)
    model3 = resnets.widesubresnet18(basemodel,3)
    dummy_input = torch.tensor(np.random.randn(1,3,32,32)).float()
    ## ## before gradient step: 
    ybase = basemodel(dummy_input)
    y0 = model0.before_fc(dummy_input)
    y1 = model1.before_fc(dummy_input)
    y2 = model2.before_fc(dummy_input)
    y3 = model3.before_fc(dummy_input)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(list(basemodel.parameters())+list(model0.parameters())+list(model1.parameters())+list(model2.parameters())+list(model3.parameters()),lr = 0.1,momentum=0.9)
    optimizer.zero_grad() 
    loss_fn(ybase.mean(),torch.tensor([0.0])).backward()
    optimizer.step()
    optimizer.step()
    y0_post = model0.before_fc(dummy_input)
    y1_post = model1.before_fc(dummy_input)
    y2_post = model2.before_fc(dummy_input)
    y3_post = model3.before_fc(dummy_input)
    assert not torch.all(y0_post == y0)
    assert not torch.all(y1_post == y1)
    assert not torch.all(y2_post == y2)
    assert not torch.all(y3_post == y3)
    
    
def test_model_parity():
    """If we have constant initialization, make sure you get the same responses from the subnet and standard models. 

    """
    basemodel_wide = resnets.WideResNet(resnets.BasicBlock,[2,2,2,2],k = 2)
    basemodel_narrow = resnets.ResNet(resnets.BasicBlock,[2,2,2,2])
    basemodels = [basemodel_wide,basemodel_narrow]
    dummy_input = torch.tensor(np.random.randn(1,3,32,32)).float()

    ## initialize as constants.
    for model in basemodels:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight, 0.01)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    ## we need to do some hacking here to get the input from conv1 of the narrow model and feed it to the base model.
    conv1_narrow = basemodel_narrow.conv1(dummy_input)
    narrow_shape = conv1_narrow.shape
    conv1_narrow_padded = torch.cat([conv1_narrow,torch.zeros(narrow_shape)],axis =1)

    submodel = resnets.widesubresnet18(basemodel_wide,0)            

    y0_layer1 = submodel.layer1(conv1_narrow_padded)
    ynarrow_layer1 = basemodel_narrow.layer1(conv1_narrow)
    y0_layer2 = submodel.layer2(y0_layer1)
    ynarrow_layer2 = basemodel_narrow.layer2(ynarrow_layer1)
    y0_layer3 = submodel.layer3(y0_layer2)
    ynarrow_layer3 = basemodel_narrow.layer3(ynarrow_layer2)
    y0_layer4 = submodel.layer4(y0_layer3)
    ynarrow_layer4 = basemodel_narrow.layer4(ynarrow_layer3)
    y0_before_fc = submodel.avgpool(y0_layer4).reshape(1,-1)
    ynarrow_before_fc = basemodel_narrow.avgpool(ynarrow_layer4).reshape(1,-1)

    assert torch.all(y0_layer1[:,:64]==ynarrow_layer1)
    assert torch.all(y0_layer2[:,:128]==ynarrow_layer2)
    assert torch.all(torch.isclose(y0_layer3[:,:256],ynarrow_layer3,1e-4))
    assert torch.all(torch.isclose(y0_layer4[:,:512],ynarrow_layer4,1e-4))
    assert torch.all(torch.isclose(y0_before_fc[:,:512],ynarrow_before_fc,1e-4))
    
def test_model_parity_w_grads():
    """If we have constant initialization, make sure you get the same responses from the subnet and standard models. 

    """
    basemodel_wide = resnets.WideResNet(resnets.BasicBlock,[2,2,2,2],k = 2)
    basemodel_narrow = resnets.ResNet(resnets.BasicBlock,[2,2,2,2])
    basemodels = [basemodel_wide,basemodel_narrow]
    dummy_input = torch.tensor(np.random.randn(1,3,32,32)).float()

    ## initialize as constants.
    for model in basemodels:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight, 1)
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    ## we need to do some hacking here to get the input from conv1 of the narrow model and feed it to the base model.
    conv1_narrow = basemodel_narrow.conv1(dummy_input)
    narrow_shape = conv1_narrow.shape
    conv1_narrow_padded = torch.cat([conv1_narrow,torch.zeros(narrow_shape)],axis =1)

    submodel = resnets.widesubresnet18(basemodel_wide,0)            

    y0_layer1 = submodel.layer1(conv1_narrow_padded)
    ynarrow_layer1 = basemodel_narrow.layer1(conv1_narrow)
    y0_layer2 = submodel.layer2(y0_layer1)
    ynarrow_layer2 = basemodel_narrow.layer2(ynarrow_layer1)
    y0_layer3 = submodel.layer3(y0_layer2)
    ynarrow_layer3 = basemodel_narrow.layer3(ynarrow_layer2)
    y0_layer4 = submodel.layer4(y0_layer3)
    ynarrow_layer4 = basemodel_narrow.layer4(ynarrow_layer3)
    y0_before_fc = submodel.avgpool(y0_layer4).reshape(1,-1)
    ynarrow_before_fc = basemodel_narrow.avgpool(ynarrow_layer4).reshape(1,-1)



    assert torch.all(y0_layer1[:,:64]==ynarrow_layer1)
    assert torch.all(y0_layer2[:,:128]==ynarrow_layer2)
    assert torch.all(torch.isclose(y0_layer3[:,:256],ynarrow_layer3))
    assert torch.all(torch.isclose(y0_layer4[:,:512],ynarrow_layer4))
    assert torch.all(torch.isclose(y0_before_fc[:,:512],ynarrow_before_fc))


