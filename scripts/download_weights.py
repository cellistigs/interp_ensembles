"""Downloads weights for pretrained models and creates a directory called `cifar10_models` that contains their state dictionaries. 
Will write in to the "models" directory- requests for pretrained models should look there too. 
"""
from interpensembles.data import CIFAR10Data,CIFAR10_1Data
import os 

here = os.path.abspath(os.path.dirname(__file__))
modeldir = os.path.join(here,"../models/")

if __name__ == "__main__": 
    os.chdir(modeldir)

    CIFAR10Data.download_weights()

