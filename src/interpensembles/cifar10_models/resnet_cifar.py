## Implement the original cifar architectures ResNets, not adaptations of the imagenet architectures to cifar. 
import torch.nn as nn
from .resnet import BasicBlock,conv3x3,conv1x1,_resnet

class BasicBlockCIFAR(nn.Module):
    """Basic Block for CIFAR style 

    """
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=16,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlockCIFAR, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 16:
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
class ResNetCIFAR(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=10,
        zero_init_residual=False,
        groups=1,
        width_per_group=16,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super(ResNetCIFAR, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 16 
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
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(
            block, 32, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 64, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

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
                elif isinstance(m, BasicBlockCIFAR):
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
        #x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)

        return x

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x

def _resnet(arch, block, layers, pretrained, progress, device, **kwargs):
    model = ResNetCIFAR(block, layers, **kwargs)
    if pretrained:
        script_dir = os.path.dirname(__file__)
        state_dict = torch.load(
            script_dir + "/state_dicts/" + arch + ".pt", map_location=device
        )
        model.load_state_dict(state_dict)
    return model

def resnet8_cf(pretrained=False, progress=True, device="cpu", **kwargs):
    """Constructs a ResNet-8 model (CIFAR architecture).
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet8", BasicBlockCIFAR, [1, 1, 1], pretrained, progress, device, **kwargs
    )

def resnet14_cf(pretrained=False, progress=True, device="cpu", **kwargs):
    """Constructs a ResNet-14 model (CIFAR architecture).
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet14", BasicBlockCIFAR, [2, 2, 2], pretrained, progress, device, **kwargs
    )

def resnet20_cf(pretrained=False, progress=True, device="cpu", **kwargs):
    """Constructs a ResNet-20 model (CIFAR architecture).
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet20", BasicBlockCIFAR, [3, 3, 3], pretrained, progress, device, **kwargs
    )

def resnet26_cf(pretrained=False, progress=True, device="cpu", **kwargs):
    """Constructs a ResNet-26 model (CIFAR architecture).
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet26", BasicBlockCIFAR, [4, 4, 4], pretrained, progress, device, **kwargs
    )

def resnet32_cf(pretrained=False, progress=True, device="cpu", **kwargs):
    """Constructs a ResNet-32 model (CIFAR architecture).
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet32", BasicBlockCIFAR, [5, 5, 5], pretrained, progress, device, **kwargs
    )

def resnet38_cf(pretrained=False, progress=True, device="cpu", **kwargs):
    """Constructs a ResNet-38 model (CIFAR architecture).
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet38", BasicBlockCIFAR, [6, 6, 6], pretrained, progress, device, **kwargs
    )

def resnet44_cf(pretrained=False, progress=True, device="cpu", **kwargs):
    """Constructs a ResNet-44 model (CIFAR architecture).
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet44", BasicBlockCIFAR, [7, 7, 7], pretrained, progress, device, **kwargs
    )
