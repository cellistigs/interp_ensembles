# Work log 

## 10/20/21
Today I was going to start implementing networks, but I wanted to check more implementation details about relevant and similar models like Wide-Resnet and ResNeXt. Wide-Resnet does something similar to what we propose for shallower networks: ResNet 18 and 32 use basic blocks, and double all layers along the channel dimension, making them well suited to our needs. Once they get to 50 however, they only increase the bottleneck layer's channels, not the 1x1 convolutions. Likewise, ResNeXt is more interested in adding group convolutions to the bottleneck layer, and less in a standard doubling of width. The model to start with is wide ResNet-18-2 with basic blocks on CIFAR 10, but I don't know if this exists in many model zoos. Start with a single layer tomorrow and see if you can get here. 

