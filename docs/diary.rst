Progress Diary 
==============

10/20/21
--------
Today I was going to start implementing networks, but I wanted to check more implementation details about relevant and similar models like Wide-Resnet and ResNeXt. Wide-Resnet does something similar to what we propose for shallower networks: ResNet 18 and 32 use basic blocks, and double all layers along the channel dimension, making them well suited to our needs. Once they get to 50 however, they only increase the bottleneck layer's channels, not the 1x1 convolutions. Likewise, ResNeXt is more interested in adding group convolutions to the bottleneck layer, and less in a standard doubling of width. The model to start with is wide ResNet-18-2 with basic blocks on CIFAR 10, but I don't know if this exists in many model zoos. Start with a single layer tomorrow and see if you can get here. 

10/24/21
--------

Figured out the gradient issues with our model. We needed to assign a list of modules that could then expose parameters for gradient descent.  
## TODO: it might be better to replace the conv2d factory that inherits from nn.Module with a layer that inherits directly from conv2d- we can have a special subnets attribute there instead. The benefits of this are that it might be easier to build subnets and pass parameters to parents, and it looks cleaner.  


10/25/21
--------

Today, we got our basic evaluation framework going- using a github repo devoted to Pytorch/CIFAR 10 we downloaded a pretrained model, evaluated it on CIFAR 10's test set, and then also imported the CIFAR 10.1 test set into pytorch and evaluated on that. The results look good- we are very much on the robustness curve defined by previous work, and we know we can now evaluate on CIFAR 10.1 reliably.  

The next step is to train an ensemble on CIFAR 10 data, and then evaluate performance with it. We can do this with independent training or with "joint" training. 
If we do joint training we might want to examine the graph afterwards. 

Code you wrote today: 

- to compare performance of a pretrained resnet 18 on CIFAR 10 and CIFAR 10.1

```
python plot_rel_robustness.py --test_phase 1 --pretrained 1 --classifier resnet18 --data_dir ./ --num_workers 4 --test
```

This outputs a json file with a timestamp identifier. 

- to plot the results against the ER curve: 

```
from interpensembles.plot import plot_model_performance
>>> plot_model_performance({in_dist_acc:number,out_dist_acc:number},"cifar10")
```
This outputs a png file plotting the resulting performance. 

TODO: clean up your results saving: `plot_rel_robustness` outputs a two field json file with the output results, but some metadata would be nice too. 


10/26/21
--------

Today, I set up a gpu environment on AWS for this project, and wrote a ptl module to train an ensemble independently. This turns out to be pretty easy- you just have to modify te forward and train_step appropriately and you should be all set. I want to check that the models trained this way are being trained as intended, and there are a bunch of check I can do with models tomorrow to make sure that this is the case. As of right now though, the training accuracy and the loss are mirroring well throughout training. It would be good to see the final test accuracy tomorrow too. 

Once these are done, we should take off the "deterministic" flag and see how the ensemble performs compared to a bunch of individual networks. Figure out if there is a reason the validation accuracy should be better- my concept is that this should be the exact same performance as a single model- same training batch order, etc. 

We should also write a script to convert the ptl checkpoint and do testing on that. 

- it's possible that seeding everything does not mean that the different networks have the same weights- we're drawing multiple times after all. 
  this "version 3" might be interesting just for that reason alone...
  
10/27/21
--------

Today I am checking the generated network structure with torchviz. It looks like the network we trained on yesterday was not a true ensemble, but rather an instance where you had weight sharing between all ensemble members, processed independently in four different streams. The graph structure of this network for two members is given in `tests/test_mats/test_weightshare_resnet18_10_26.pdf`. We checked that the weights between two different streams of this network are identical at initialization. What is strange to note is that this model should have the exact same initialization and training schema as the original resnet 18 model, but the test accuracy was better by about .06 %.  This is a tiny margin that probably doesn't matter, but given everything is deterministic it's a little weird. This is saved as `interpensembles/cifar10/resnet10/version3` on your remote instance. 

TODO: follow up on this result. 
  
Note- we don't randomize the minibatch ordering between ensemble members. This is a difference from the most "standard" ensembles. 
As batch size increases, members of "standard ensembles" should become more similar simply bc you lose one source of data diversity. 

Today we also added 5 additional resnet 18s, Geoff's wide resnets, and started training resnet ensembles. We see a similar trend- ensemble performance is pretty much smack dab on the ER = 0 line. Tomorrow we can add our other ensembles, and work with layers. 

10/28/21
--------

Go back to Mania and Sur 2021. These ensembling results are predicted by this paper, no?
