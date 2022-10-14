import torch
import torch.nn as nn
from ensemble_attention.layers import create_subnet_params,create_subnet_params_output_only,Conv2d_subnet_layer,ChannelSwitcher,LogSoftmaxGroupLinear
import os

__all__ = [
    "ResNet",
    "wideresnet18"
    "wideresnet18_4"
    "resnet18",
    "resnet34",
    "resnet50",
]
def subnetconv(params,layer):
    """subnet version. 

    """
    if layer.kernel_size == (3,3):
        return Conv2d_subnet_layer(
            params,   
            layer.in_channels,
            layer.out_channels,
            kernel_size=3,
            stride=layer.stride,
            padding=layer.padding,
            groups=layer.groups,
            bias=False,
            dilation=layer.dilation,
            padding_mode = layer.padding_mode,
            device = None,
            dtype = None
        )
    elif layer.kernel_size == (1,1):    
        return Conv2d_subnet_layer(params,layer.in_channels, layer.out_channels, kernel_size=1, stride=layer.stride, bias=False,
            padding=layer.padding,
            groups=layer.groups,
            dilation=layer.dilation,
            padding_mode = layer.padding_mode,
            device = None,
            dtype = None)
    else:    
        raise Exception("unexpected kernel size")

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        #width = int(planes * (base_width / 64.0)) * groups
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """

    :param inplanes: the size of the input channels for the entire residual block (usually size 64)
    :param planes: the size of the output to expect without expansion factor (given as a class attribute)
    """
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width) ## output is B,H,W,Width
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation) ## B,H,W,Width
        self.bn2 = norm_layer(width) 
        self.conv3 = conv1x1(width, planes * self.expansion) ## B,H,W,planes*4
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class WideResNet_GroupLinear(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=10,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        k = 1
    ):
        """
        WideResNet with a width parameter k, and grouping of output. For now we group output into same number as width parameter. Note differences from the original wideresnet- we expand the initial layer in addition to all the others, and do not use dropout. 

        :param block: The type of unit. 
        :param layers: a list of layer counts per block.  
        :param num_classes:
        :param zero_init_residual:
        :ivar inplanes: init_var: 64*k.  This variable is the operant one for width determination. It updated by each make_layer function to be the "planes" argument passed to make_layer. Its state is changed through successive calls to "planes". 
        """
        super(WideResNet_GroupLinear, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64*k  
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # CIFAR10: kernel_size 7 -> 3, stride 2 -> 1, padding 3->1
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        # END

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64*k, layers[0])
        self.layer2 = self._make_layer(
            block, 128*k, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256*k, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512*k, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = LogSoftmaxGroupLinear(512 * block.expansion * k, num_classes,k)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        """Make one of the 4 residual blocks. 
        :param block: the unit type: basic (ResNet 18/32) or bottleneck (ResNet 50+)
        :param planes: the number of channels to have in the output 
        :param blocks: the number of units to repeat. This is the distinguishing factor between different resnets. 
        :param stride: stride of convolution
        :param dilate: whether to replace stride with dilations ? . 

        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion: ## why is this not triggered?
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        ## first append a unit that takes self.inplanes on its own.
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        ## then expand the input dimension to planes*block.expansion for the following unit and all subsequent ones.
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x
class WideResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=10,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        k = 1
    ):
        """
        WideResNet with a width parameter k. Note differences from the original wideresnet- we expand the initial layer in addition to all the others, and do not use dropout. 

        :param block: The type of unit. 
        :param layers: a list of layer counts per block.  
        :param num_classes:
        :param zero_init_residual:
        :ivar inplanes: init_var: 64*k.  This variable is the operant one for width determination. It updated by each make_layer function to be the "planes" argument passed to make_layer. Its state is changed through successive calls to "planes". 
        """
        super(WideResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64*k  
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # CIFAR10: kernel_size 7 -> 3, stride 2 -> 1, padding 3->1
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        # END

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64*k, layers[0])
        self.layer2 = self._make_layer(
            block, 128*k, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256*k, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512*k, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion * k, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        """Make one of the 4 residual blocks. 
        :param block: the unit type: basic (ResNet 18/32) or bottleneck (ResNet 50+)
        :param planes: the number of channels to have in the output 
        :param blocks: the number of units to repeat. This is the distinguishing factor between different resnets. 
        :param stride: stride of convolution
        :param dilate: whether to replace stride with dilations ? . 

        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion: ## why is this not triggered?
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        ## first append a unit that takes self.inplanes on its own.
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        ## then expand the input dimension to planes*block.expansion for the following unit and all subsequent ones.
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x

class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=10,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # CIFAR10: kernel_size 7 -> 3, stride 2 -> 1, padding 3->1
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        # END

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        """Make one of the 4 residual blocks. 
        :param block: the unit type: basic (ResNet 18/32) or bottleneck (ResNet 50+)
        :param planes: the number of channels to have in the output 
        :param blocks: the number of units to repeat. This is the distinguishing factor between different resnets. 
        :param stride: stride of convolution
        :param dilate: whether to replace stride with dilations ? . 

        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion: ## why is this not triggered?
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        ## first append a unit that takes self.inplanes on its own.
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        ## then expand the input dimension to planes*block.expansion for the following unit and all subsequent ones.
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def before_fc(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)

        return x

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


class SubResNet(nn.Module):
    """A potentially wide resnet that acts as a subnetwork of an existing network. initialized with all the normal parameters but also a separate existing resnet model. 
    The hard part about this is making sure that subnet indices work correctly in the "cross terms"- those subnets which have weights which alternate their non-zero component betweeen different channels.   

    :param: replace_params
    """
    replace_params = {"conv1"}
    def __init__(
        self,
        block,
        layers,
        num_classes=10,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        k = None,
        baseresnet = None,
        nb_subnets = None,
        indices = None 
    ):
        super(SubResNet, self).__init__()
        self.baseresnet = baseresnet
        self.nb_subnets = k**2
        self.indices = indices
        self.alt = indices["layer1"][0] in [1,2]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64*k
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # CIFAR10: kernel_size 7 -> 3, stride 2 -> 1, padding 3->1
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        ### Replace with indexed replacement factor: 
        #base = self.baseresnet.conv1 ## base Conv2d module we want to use for replacement. 
        params = create_subnet_params_output_only(self.conv1.weight,self.nb_subnets) ## returns an indexed dictionary of weight params. 
        conv_subnet = subnetconv(params[self.indices["conv1"]],self.conv1) ## create subnet. ## TODO add index
        self.conv1=conv_subnet ## we have to set attribute directly. 

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64*k, layers[0])
        self.replace(self.layer1,self.baseresnet.layer1, indices = self.indices["layer1"])
        ## this function first looks at len(model.layer) to see how many blocks there are. 
        ## then, it looks at the type of block: is it a BottleNeck or BasicBlock. 
        ## if bottleneck, it looks for 3 convolutions, and if basic it looks for 2
        self.layer2 = self._make_layer(
            block, 128*k, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.replace(self.layer2,self.baseresnet.layer2, indices = self.indices["layer2"])
        self.layer3 = self._make_layer(
            block, 256*k, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.replace(self.layer3,self.baseresnet.layer3, indices = self.indices["layer3"])
        self.layer4 = self._make_layer(
            block, 512*k, layers[3], stride=2,  dilate=replace_stride_with_dilation[2]
        )
        self.replace(self.layer4,self.baseresnet.layer4,indices = self.indices["layer4"])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*k * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                pass
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def replace(self,layerreplace,layersource,indices): 
        """Takes the "layer" (I call it a block) indicated, and replaces the convolutional weights given in the `layerreplace` with a masked version from `layersource`. The masking depends on `indices` and `nb_blocks`, which indicate how many subnets we want, and what index subnet we should generate the corresponding mask for. 
        Assumes that `layerreplace` and `layersource` both correspond to a nn.Sequential set of BasicBlock or Bottleneck units. 

        """
        ## First, let's confirm that the indices parameter corresponds to the layers given. 
        assert len(layerreplace) == len(layersource), "the two residual blocks contain different numbers of units. replacement impossible."
        nb_units = len(layerreplace)
        if isinstance(layerreplace[0],BasicBlock):
            layertype = "Basic"
            layernb = 2
        elif isinstance(layereplace[0],Bottleneck):    
            layertype = "Bottleneck"
            layernb =3 
        else:
            raise Exception("layer is not instance of `BasicBlock` or `Bottleneck`")    

        ## handle downsampling (only in the first unit if it exists)
        if layerreplace[0].downsample is not None:
            replace_downsample = True
        else:    
            replace_downsample = False

        nb_layers = nb_units*layernb + int(replace_downsample) ## add one more for the 1x1 layer if we need to downsample. 
        assert nb_layers == len(indices), "The `indices` you passed do not cover all "
        ## now we do replacement
        replaceindex = 0
        for unit_ind in range(nb_units):
            if layertype == "Basic":
                convs = {
                        "conv1":{"source":layersource[unit_ind].conv1,"target":layerreplace[unit_ind]}, ## source is the actual layer, target is the unit. target here is pretty worthless- you can remove later. 
                        "conv2":{"source":layersource[unit_ind].conv2,"target":layerreplace[unit_ind]}
                        }
            elif layertype == "Bottleneck":    
                convs = {
                        "conv1":{"source":layersource[unit_ind].conv1,"target":layerreplace[unit_ind]},
                        "conv2":{"source":layersource[unit_ind].conv2,"target":layerreplace[unit_ind]},
                        "conv3":{"source":layersource[unit_ind].conv3,"target":layerreplace[unit_ind]}
                        }
                    
            for convname,convset in convs.items():    
                base = convset["source"] ## base Conv2d module we want to use for replacement. 
                params = create_subnet_params(base.weight,self.nb_subnets) ## returns an indexed dictionary of weight params. 
                conv_subnet = subnetconv(params[indices[replaceindex]],base) ## create subnet. 
                setattr(convset["target"],convname,conv_subnet) ## we have to set attribute directly. 
                replaceindex +=1

            if unit_ind == 0:    
                if replace_downsample:
                    base = layersource[unit_ind].downsample[0]
                    params = create_subnet_params(base.weight,self.nb_subnets)
                    conv_subnet = subnetconv(params[indices[replaceindex]],base) ## create subnet. 
                    new_downsample_layers = [conv_subnet,convset["target"].downsample[1]]
                    if self.alt:
                        channelswitcher = ChannelSwitcher(base.weight.shape[0])
                        new_downsample_layers.append(channelswitcher)
                    new_sequential = nn.Sequential(*new_downsample_layers)
                    convset["target"].downsample = new_sequential
                    replaceindex +=1
        assert replaceindex == len(indices)        

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        """Make one of the 4 residual blocks. 
        :param block: the unit type: basic (ResNet 18/32) or bottleneck (ResNet 50+)
        :param planes: the number of channels to have in the output 
        :param blocks: the number of units to repeat. This is the distinguishing factor between different resnets. 
        :param indices: a list of integers indexing into the individual convolutional layers inside this residual blocks. The list must match the number of convolutional layers available given as number_of_conv_layers_per(block)*blocks. 
        :param stride: stride of convolution
        :param dilate: whether to replace stride with dilations ? . 

        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion: ## why is this not triggered?
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        ## first append a unit that takes self.inplanes on its own.
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        ## then expand the input dimension to planes*block.expansion for the following unit and all subsequent ones.
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def before_fc(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)

        return x

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x

def _widesubresnet(arch, block, layers, pretrained, progress, device, k, baseresnet, indices, **kwargs):

    model = SubResNet(block, layers, k = k, baseresnet = baseresnet, indices = indices,**kwargs)
    if pretrained:
        script_dir = os.path.dirname(__file__)
        state_dict = torch.load(
            script_dir + "/state_dicts/" + arch + ".pt", map_location=device
        )
        model.load_state_dict(state_dict)
    return model

def _wideresnet_grouplinear(arch, block, layers, pretrained, progress, device, k, **kwargs):
    model = WideResNet_GroupLinear(block, layers, k = k,**kwargs)
    if pretrained:
        script_dir = os.path.dirname(__file__)
        state_dict = torch.load(
            script_dir + "/state_dicts/" + arch + ".pt", map_location=device
        )
        model.load_state_dict(state_dict)
    return model

def _wideresnet(arch, block, layers, pretrained, progress, device, k, **kwargs):
    model = WideResNet(block, layers, k = k,**kwargs)
    if pretrained:
        script_dir = os.path.dirname(__file__)
        state_dict = torch.load(
            script_dir + "/state_dicts/" + arch + ".pt", map_location=device
        )
        model.load_state_dict(state_dict)
    return model

def _resnet(arch, block, layers, pretrained, progress, device, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        script_dir = os.path.dirname(__file__)
        state_dict = torch.load(
            script_dir + "/state_dicts/" + arch + ".pt", map_location=device
        )
        model.load_state_dict(state_dict)
    return model

def widesubresnet18(baseresnet, index, pretrained=False, progress=True, device="cpu", **kwargs):
    """Constructs a wide (2x) ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        baseresnet: base resnet model to share weights with. 
        index: index of subresnet  
    """
    indices = { ## subnet indices for resnet 18
        0:{
            "conv1":0,
            "layer1":[0,0,0,0],
            "layer2":[0,0,0,0,0],
            "layer3":[0,0,0,0,0],
            "layer4":[0,0,0,0,0],
            },
        1:{
            "conv1":0,  
            "layer1":[2,1,2,1],
            "layer2":[2,1,1,2,1],
            "layer3":[2,1,1,2,1],
            "layer4":[2,1,1,2,1],
            },
        2:{
            "conv1":1, ## shifts to the second set of channels: 
            "layer1":[1,2,1,2],
            "layer2":[1,2,2,1,2],
            "layer3":[1,2,2,1,2],
            "layer4":[1,2,2,1,2],
            },
        3:{
            "conv1":1, ## shifts to the second set of channels: 
            "layer1":[3,3,3,3],
            "layer2":[3,3,3,3,3],
            "layer3":[3,3,3,3,3],
            "layer4":[3,3,3,3,3],
            },
        }

    return _widesubresnet(
        "wideresnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, device, k = 2, baseresnet = baseresnet, indices = indices[index], **kwargs
    )

def wideresnet18_4_grouplinear(pretrained=False, progress=True, device="cpu", **kwargs):
    """Constructs a wide (4x) ResNet-18 model with grouping of linear output channels.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _wideresnet_grouplinear(
        "wideresnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, device, k = 4, **kwargs
    )

def wideresnet18_4(pretrained=False, progress=True, device="cpu", **kwargs):
    """Constructs a wide (4x) ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _wideresnet(
        "wideresnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, device, k = 4, **kwargs
    )

def wideresnet18(pretrained=False, progress=True, device="cpu", **kwargs):
    """Constructs a wide (2x) ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _wideresnet(
        "wideresnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, device, k = 2, **kwargs
    )

def resnet18(pretrained=False, progress=True, device="cpu", **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, device, **kwargs
    )


def resnet34(pretrained=False, progress=True, device="cpu", **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, device, **kwargs
    )


def resnet50(pretrained=False, progress=True, device="cpu", **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, device, **kwargs
    )
