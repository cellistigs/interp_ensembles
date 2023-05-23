# Reconsidering Deep Ensembles 

This repository contains code for the manuscript "Reconsidering Deep Ensembles". Here we provide information for how to regenerate figures 1, 2 and 4 of the corresponding paper. 
## Project structure: 

Please follow these guidelines when contributing to this repo. 
- source code (like the ensemble and model class, or calculating bias and variance) goes in `src/interpensembles`. We assume users have installed this package. I will also attempt to write tests for all code that lives in `src`.
- scripts (to generate plots, w/hardcoded references to datasets) live in `scripts` and import `interpensembles` as a package.

## Old project premise:

We all love deep ensembles because they are great for out of distribution detection and covering a diverse set of solutions. However, these benefits, that are supposedly selectively targeted by ensembles (due to diversity among ensemble members, independent errors, etc) are also usually accompanied by an increase in performance on in-distribution data. Do we actually do better in a qualitatively different way when we ensemble, as opposed to when we just properly train a much larger model? 

## Installation 
We highly recommend usage of a GPU backed machine in order to reproduce the figures here. CPU only based usage is not tested.  

You will need to install the dependencies found in the requirements.txt file in order to reproduce figures. We recommend doing so in a [conda](https://www.anaconda.com) virtual environment. Once you have installed conda, check your installation by running:

```
conda list
```

Then create a new environment as follows: 

```
conda create -n env_name python=3.7
```

Now move into the root directory of this repo:
```
cd /path/to/this/repo
```

Activate your new environment, install dependencies and python package: 
```
conda activate env_name
conda install pip 
pip install -r requirements.txt
pip install -e ./src
```


## Figures 1 and 2 setup:
To regenerate Figures 1 and 2, please use the repo linked as "compare_performance" in the manuscript.

### Retrieving data
In order to replicate figures, you will need to load the datasets given at [this zenodo link](https://doi.org/10.5281/zenodo.6582653). Please add upload the contents into the "results" directory of this repo. 

### Figure 1 
Figure 1 is generated using the script `scripts/plot_conditional_variance.py`. Please navigate to the directory `scripts` before running the commands below.
Figure 1 results can be regenerated in two steps, corresponding to the two sets of ensembles used. 
- Row 1: 
```
python plot_conditional_variance.py
```
- Row 2: 
```
python plot_conditional_variance.py "+all_imagenet_ood@=config_AlexNet_imagenetv2mf_Var.yaml"
```

### Figure 2 
Figure 2 is generated using the script `scripts/paperfigs/compare_performance.py`
Figure 2 results can be generated in two steps, corresponding to the two pairs of ensemble/single models used. 
- Left Column: 
```
python compare_performance.py "+cifar10_1_bs@=config.yaml"
```
- Right Column: 
```
python compare_performance.py "+cinic10_bs@=config.yaml"
```

## Figure 4 setup:
To regenerate Figure 4, please use the repo linked as "imagenet_pl" in the manuscript

### Retrieving data
Please see the instructions in `etc/ekb/README.md` on how to download and prepare data. 

## Figure 4 
Navigate to the directory `etc/ekb/plot`, and please run the following:

```
store_.sh
```


