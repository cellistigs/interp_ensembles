# Estimate the bias and variance of a variety of models. 
import hydra
import numpy as np
from interpensembles.predictions import EnsembleModel 
import os 
here = os.path.abspath(os.path.dirname(__file__))

import matplotlib.pyplot as plt

@hydra.main(config_path = "script_configs/biasvar",config_name = "cifar10")
def main(args):
    all_biasvar = []
    for modelname,model in args.models.items():
        ens = EnsembleModel(modelname,"ind")
        [ens.register(os.path.join(here,m),i,None,os.path.join(here,l)) for i,(m,l) in  enumerate(zip(model.modelnames,model.labelpaths))]
        bias,var = ens.get_bias_bs(),ens.get_variance()
        all_biasvar.append([bias,var])
    
    biasvar_array = np.array(all_biasvar)
    plt.scatter(biasvar_array[:,1],biasvar_array[:,0])
    plt.plot(np.linspace(0,0.1,100),np.linspace(0,0.1,100))
    plt.title(args.title)
    plt.xlabel("variance")
    plt.ylabel("bias")
    plt.show()

    
if __name__ == "__main__":
    main()
