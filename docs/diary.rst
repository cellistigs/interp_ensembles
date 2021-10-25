Progress Diary 
==============

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

