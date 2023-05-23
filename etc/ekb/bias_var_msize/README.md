
## Extension of Fig 1.

First we look at the distribution of model errors for ~100 models trained on imagenet.

<img src=model_error.png width="250" height="250">

Given this distribution, we can assign scores to each player according to its performance, and add up the individual scores to 
get the team score.


## Code
1. [x] Make file with single model metrics:
``` 
#See instructions in https://github.com/cellistigs/interp_ensembles/blob/imagenet_pl/etc/ekb/plot/store_.sh
python scripts/calculate_model_performance.py --config_name=imagenet
```

2. [x] Make file ensemble metrics
```
python etc/ekb/bias_var_msize/metrics_het_ensemble_parallel.py --dataset=imagenet
```

3. [x] Make plot avg. single model performance vs. diversity, avg. single model vs. ens. performance, diversity vs. ens. performance 
```
python etc/ekb/bias_var_msize/plot_bias_var.py --dataset=imagenet
```
