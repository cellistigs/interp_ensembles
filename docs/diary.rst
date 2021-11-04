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
- in some ways, yes. Think about the perspective of dominance probability- this is exactly the idea of "reliability" that John said ensembles are designed to address. 

11/03/21  
--------
## Interpolation Model Progress
Some time has passed since the last update. Since then, we figured out the interensembles model architecture- we learned how to use either getattr/setattr or direct assignment to share weights between different models, as well as the nn.Module class- you used this to implement a `ChannelSwitcher` module that has no trainable parameters and simply switches the first half and second half of the activations in the channel dimension (necessary for identity blocks).  

Training this model is slow- in the future, it might be better to instantiate from individual subresnets, and then construct a wide model by summing together the activations of the subnetworks per layer. The difference is that right now, you have a lot of dimensions zeroed out, which look like they are effectively making it so that you train a model 5x bigger than the big model. 

For now, it will also be useful to implement stochastic training- flip a coin and choose either the big model or the ensemble to train. This is more efficient if we can choose where to propagate gradients, and reduces the risk of seeing weird things due to optimization dynamics. 

## "Mean Field" optimization. 

On 11/02 Geoff suggested a different way to address the question of how ensembles and large networks might differ- in the costs that they are being trained with. We can see the cost used to train an ensemble of neural networks as an effective "mean field" approximation to the cost of training a large network- we are training on the average loss of all of the networks, instead of training on the big network loss. One interesting thing to inspect is if this is the source of ensemble performance- is it the case that ensembles do better because they have a different loss? One way to test this is to apply the "ensemble loss" to a big network and see how it does. At the very least this seems to train very well. Take a look at old papers like [this](http://www.cs.columbia.edu/~blei/fogm/2018F/materials/SaulJaakkolaJordan1996.pdf) and see if there's a connection to the work you're doing here. 

## Ensemble training. 

One thing to be careful of- when taking the average, we need to be very careful of the model learning rate. We were before leaving the learning rate unchanged, but this means that each model has an effective learning rate that is divided by 4. We can change this learning rate, but we have to be careful because we have warmup and cosine annealing on- these schedulers have fixed hyperparameters that will lead to nonlinear changes in the learning rate schedule when paired with a scalar increase in the base learning rate. 

We see in training an ensemble of WideResNet 2x that training on the sum of losses, as opposed to the average, seems to lead to comparable training accuracy during training, and better validation accuracy. The curves look like this:   

.. image:: images/acc_train_ensembles.png
   :width: 800

This is during training. The orange curve is the training accuracy of a single wideresnet 2x model. The dark blue is the training accuracy of an ensemble of 4 wideresnet 2x models using the average loss and default hyperparameters. The light blue curve is the training accuracy (so far) of an ensemble of 4 wideresnet 2x models using the summed loss and default hyperparameters. See below for the corresponding validation: 

.. image:: images/acc_val_ensembles.png
   :width: 800

We can see that training accuracy of the light blue matches that of the single model, and that the validation accuracy is consistently better. Note also, of interest is the fact that when you train with the average loss (effective 4x lower learning rate), the ensemble first does worse, then better, and then converges to basically the same solution as a single network. This is something we saw on a different occasion when we accidentally shared weights between different ensemble members as well- first they underperform, then they outperform on validation accuracy, and eventually they end up looking the same. This is pretty strange- what do we know about the effect of training a model with 4x as many parameters with 1/4th the learning rate?  

One clear implication this has is that our interpensembles are being trained weird. Their learning rates need to be altered somehow, but we should be careful about how. 


