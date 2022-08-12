# Readme for script directory. 

This directory and subdirectories contain scripts that reference the source code in `src`. The scripts here reference config files, stored in `scripts/script_configs`. These config files contain run specific information, and can be referenced at the command line. 

## Bias-Variance

To run bias variance estimates for a bunch of models, first we create a config file. This file should be placed in `scripts/script_configs/biasvar/`. If you look in this directory, you should find a config file for `cifar10` called `cifar10/cifar10.yaml`. This file references all of the model related information needed to run the corresponding script. To run the script with this config file, you can reference the config file as follows:  

```
python estimate_biasvar.py +cifar10@=cifar10.yaml
```
Here the first `cifar10` references a subdirectory within `scripts/script_configs/biasvar` and the second the specific file to run. 

If you wanted to analyze cinic0 data, using a config file in the same directory, you would run: 
```
python estimate_biasvar.py +cifar10@=cifar10_cinic.yaml
```

If you wanted to analyze imagenet data, you could run the following instead: 
```
python estimate_biasvar.py +imagenet@=imagenet.yaml
```

