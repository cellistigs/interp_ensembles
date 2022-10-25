# Estimate the bias and variance of a variety of models. 
from omegaconf.errors import ConfigAttributeError
import hydra
import numpy as np
from interpensembles.predictions import EnsembleModel 

import os 
here = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import h5py
plt.style.use(os.path.join(here,"../etc/config/stylesheet.mplstyle"))
from sklearn.linear_model import LinearRegression
import pandas as pd
from pathlib import Path
from itertools import combinations


def proportion(p,M,r):
    """Given the following model :

    f_i ~ {Cat(p,(1-p)/(M-1),...,(1-p)/(M-1)) with probability r}
          {Cat((1-p)/(M-1),p,...,(1-p)/(M-1)) with probability 1-r}

    get the average single model brier score and average variance. This means calculating the brier score and variance for each instance of r, and averaging quantites over r.
    """

    BS_r = (p-1)**2+(1-p)/(M-1) # brier score for correct proportion.
    BS_1_r = ((1-p)/(M-1)-1)**2+p**2+(M-2)*((1-p)/(M-1))**2 ## brier score for incorrect proportion.
    BS = r*BS_r+(1-r)*(BS_1_r)
    var_first = p*(1-p)*r+(1-p)/(M-1)*(1-(1-p)/(M-1))*(1-r) #expected variance of the first two entries.
    var_others = (1-p)/(M-1)*(1-(1-p)/(M-1)) #expected variance of all other entries.
    sum_var = 2*var_first+(M-2)*var_others
    return BS+sum_var,sum_var

# Make a heterogeneous ensemble. 
def get_heterogeneous_biasvar(models, seed = 0, ensemble_size = 4):
    """

    """
    all_biasvar = []
    all_biasvarperf = []
    all_models = []
    for arch, modelinfo in models.items():
        for mi, modelpath in enumerate(modelinfo.modelnames):
            modeldict = {"arch":arch,"modelpath":modelpath,"labelpath":modelinfo.labelpaths[mi]}
            try: 
                modeldict["logits"] = modelinfo.logits
            except ConfigAttributeError:    
                modeldict["logits"] = True ## if not given, assume true
            try: 
                modeldict["npz_flag"] = modelinfo.npz_flag
            except ConfigAttributeError:    
                modeldict["npz_flag"] = None
            all_models.append(modeldict)
    # ensemble_sizes = [len(modelinfo.modelnames) for m, modelinfo in models.items()]
    ensemble_sizes = [ensemble_size]*int(len(models)//ensemble_size)
    np.random.seed(seed)
    permed = np.random.permutation(all_models)
    # all_ensembles = []
    for ensemble_size in ensemble_sizes:
        chunk = permed[:ensemble_size]
        name = "Model:_".join([m["arch"] for m in chunk])
        ens = EnsembleModel(name,"ind") 
        [ens.register(os.path.join(here,m["modelpath"]),i,None,os.path.join(here,m["labelpath"]),logits= m["logits"],npz_flag = m["npz_flag"]) for i,m in enumerate(chunk)]
        # all_ensembles.append(ens)
        permed = permed[ensemble_size:]
        #bias,var,perf = ens.get_bias_bs(),ens.get_variance(),ens.get_brier()
        bias, var, perf = ens.get_avg_nll(),ens.get_nll_div(),ens.get_nll()
        print("Permuted {}: Bias: {}, Variance: {}, Performance: {}".format(name,bias,var,perf))
        all_biasvar.append([bias, var])
        all_biasvarperf.append([perf, bias/var])
    
    biasvar_array = np.array(all_biasvar)
    biasvarperf_array = np.array(all_biasvarperf)
    return biasvar_array, biasvarperf_array

# get bias, variance, and performance to plot
def get_arrays_toplot(models, ensemble_size = 4):
    """
    Takes as argument a dictionary of models: keys giving model names, values are dictionaries with paths to individual entries. 
    :param models: names of individual models. 
    """
    all_biasvar = []
    all_biasvarperf = []
    for modelname, model in models.items():
        if len(model.modelnames) == 1 and len(model.modelnames) < ensemble_size:
            # does not have enough members to form a homogeneous ensemble.
            continue
        ens = EnsembleModel(modelname,"ind")
        kwargs = {}
        try: 
            kwargs["logits"] = model.logits
        except ConfigAttributeError:    
            pass
        try: 
            kwargs["npz_flag"] = model.npz_flag
        except ConfigAttributeError:    
            pass

        # sample combinations of homogeneous ensembles
        ensemble_groups = list(combinations(range(len(model.modelnames)), ensemble_size))
        for ensemble_group in ensemble_groups:
            models_ = [model.modelnames[ens_mem] for ens_mem in ensemble_group]
            labels_ = [model.labelpaths[ens_mem] for ens_mem in ensemble_group]
            print(models_, labels_)
            for i, (m, l) in enumerate(zip(models_, labels_)):
                ens.register(filename=os.path.join(here,m),
                             modelname=i,
                             inputtype=None,
                             labelpath=os.path.join(here,l),
                             **kwargs)
            #bias,var,perf = ens.get_bias_bs(),ens.get_variance(),ens.get_brier()
            bias, var, perf = ens.get_avg_nll(),ens.get_nll_div(),ens.get_nll()
            print("{}: Bias: {}, Variance: {}, Performance: {}".format(modelname,bias,var,perf))
            all_biasvar.append([bias,var])
            all_biasvarperf.append([perf,bias/var])
    
    biasvar_array = np.array(all_biasvar)
    biasvarperf_array = np.array(all_biasvarperf)
    return biasvar_array, biasvarperf_array


@hydra.main(config_path = "../script_configs/biasvar/imagenet", config_name="imagenet_het1")
def main(args):
    # Set up results directory
    results_dir = Path( here) / "../results/biasvar/{}".format(args.title)
    results_dir.mkdir(parents=True, exist_ok=True)
    ensemble_size = 4
    # Set up figure
    fig, ax = plt.subplots(figsize=(9, 8))

    # Get homogeneous ensembles:
    biasvar_array, biasvarperf_array = get_arrays_toplot(args.models, ensemble_size= ensemble_size)

    values = np.hstack([biasvar_array, biasvarperf_array])
    pd.DataFrame(values).to_csv(results_dir / "homogeneous_values_ens{}.csv".format(ensemble_size))
    # Get heterogeneous ensembles:

    for seed in range(25):
        biasvar_array_permed, biasvarperf_array_permed= get_heterogeneous_biasvar(args.models, seed=seed, ensemble_size=ensemble_size)

        # store values
        values = np.hstack([biasvar_array_permed, biasvarperf_array_permed])
        pd.DataFrame(values).to_csv(results_dir / "values_ens{}_seed{}.csv".format(ensemble_size, seed))
        if seed == 0:
            ax.scatter(biasvar_array_permed[:,1],biasvar_array_permed[:,0],color = "orange",s=10,label =
                    "heterogeneous",alpha =0.5)
        else:    
            ax.scatter(biasvar_array_permed[:,1],biasvar_array_permed[:,0],color = "orange",s=10,alpha = 0.5)

    defaultline = np.array([proportion(pi,10,0.98) for pi in np.linspace(0.87,1,100)])
    #plt.plot(defaultline[:,1],defaultline[:,0],"--",color = "black",label="sim frontier")
    ax.plot(np.linspace(0,3,100),np.linspace(0,3,100),label="perfect ensemble")
    line = np.linspace(0,100,100)
    if args.get("colormap", False) is True:  ## plot assuming scatters are ordered
        colors = cm.rainbow(np.linspace(0, 1, len(biasvar_array)))
        ax.scatter(biasvar_array[:, 1], biasvar_array[:, 0], s=20, c=colors, label="homogeneous")
    else:
        ax.scatter(biasvar_array[:, 1], biasvar_array[:, 0], s=20, label="homogeneous")

    if 'CIFAR' in args.title:
        for i in range(19):
            ax.plot(line, line+i*0.05-0.45,alpha = 0.1,color = "black")
        ax.plot(line,line+0.108,"--",color = "black",label = "best ensemble",alpha = 0.5)
        ax.set_xlim([0, 0.3])
        ax.set_ylim([0.1, 0.4])
    elif 'Imagenet' in args.title:
        # find level set for best model
        best_ensemble_idx = np.argmin(biasvarperf_array_permed[:,0])
        best_offset = biasvar_array_permed[best_ensemble_idx, 0] - biasvar_array_permed[best_ensemble_idx, 1]
        ax.plot(line, line+best_offset,"--",color = "black",label = "best ensemble",alpha = 0.5)
        for i in range(19):
            ax.plot(line, line + i * 0.5, alpha=0.1, color="black")
            ax.plot(line + i * 0.5, line, alpha=0.1, color="black")
        ax.set_xlim([0, 1.2])
        ax.set_ylim([0, 3])

    ax.set_title(args.title)

    ax.set_xlabel("CE Jensen gap (pred. diversity)")
    ax.set_ylabel("Avg. single model loss (CE)")
    plt.legend()
    fig.savefig( results_dir / "biasvar_ce2_ens{}.pdf".format(ensemble_size))

    plt.show()
    #fig,ax = plt.subplots()
    #ax.scatter(biasvarperf_array[:,0],biasvarperf_array[:,1],s=0.5)
    ##ax.scatter(biasvarperf_array_permed[:,0],biasvarperf_array_permed[:,1],s=0.5,color = "orange")
    #ax.set_title("Performance to Bias/Variance Ratio")
    #ax.set_xlabel("Performance")
    #ax.set_ylabel("bias/variance")
    #fig.savefig("{}_perf_biasvar.pdf".format(args.title))

    
if __name__ == "__main__":
    main()
