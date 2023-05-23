# Pathologies of Predictive Diversity in Deep Ensembles

This repository contains code for the manuscript "Pathologies of Predictive Diversity in Deep Ensembles". Please see the paper for details on how figures were generated.  
## Project structure: 

Please follow these guidelines when contributing to this repo. 
- source code (like the ensemble and model class, or calculating bias and variance) goes in `src/interpensembles`. We assume users have installed this package. I will also attempt to write tests for all code that lives in `src`.
- scripts (to generate plots, w/hardcoded references to datasets) live in `scripts` and import `interpensembles` as a package.

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


