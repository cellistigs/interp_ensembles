# Estimate the bias and variance of a variety of models. 
import hydra
import numpy as np
from interpensembles.predictions import EnsembleModel 
import os 
here = os.path.abspath(os.path.dirname(__file__))

import matplotlib.pyplot as plt

@hydra.main(config_path = "script_configs/biasvar",config_name = "default")
def main(args):
    all_biasvar = []
    for modelname,model in args.models.items():
        ens = EnsembleModel(modelname,"ind")
        [ens.register(os.path.join(here,m),i,None,os.path.join(here,l)) for i,(m,l) in  enumerate(zip(model.modelnames,model.labelpaths))]
        bias,var = ens.get_bias_bs(),ens.get_variance()
        all_biasvar.append([bias,var])
    
    biasvar_array = np.array(all_biasvar)

    fig,ax = plt.subplots()
    ax.scatter(biasvar_array[:,1],biasvar_array[:,0])
    ax.plot(np.linspace(0,args.line_extent,100),np.linspace(0,args.line_extent,100))
    ax.set_title(args.title)
    ax.set_xlabel("variance")
    ax.set_ylabel("bias")
    fig.savefig("{}_biasvar.pdf".format(args.title))
    plt.show()

    
if __name__ == "__main__":
    main()
