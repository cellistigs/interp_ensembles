# Estimate the bias and variance of a variety of models. 
from omegaconf.errors import ConfigAttributeError
import hydra
import numpy as np
from interpensembles.predictions import EnsembleModel 

import os 
here = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

plt.style.use(os.path.join(here,"../etc/config/geoff_stylesheet.mplstyle"))

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
def get_heterogeneous_biasvar(models,seed = 0):
    """

    """
    all_biasvar = []
    all_biasvarperf = []
    all_models = []
    for arch,modelinfo in models.items():
        for mi,modelpath in enumerate(modelinfo.modelnames):  
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
    ensemble_sizes = [len(modelinfo.modelnames) for m,modelinfo in models.items()]

    np.random.seed(seed)
    permed = np.random.permutation(all_models)
    all_ensembles = []
    for ensemble_size in ensemble_sizes:
        chunk = permed[:ensemble_size]
        name = "Model:_".join([m["arch"] for m in chunk])
        ens = EnsembleModel(name,"ind") 
        [ens.register(os.path.join(here,m["modelpath"]),i,None,os.path.join(here,m["labelpath"]),logits= m["logits"],npz_flag = m["npz_flag"]) for i,m in enumerate(chunk)]
        all_ensembles.append(ens)
        permed = permed[ensemble_size:]
        bias,var,perf = ens.get_bias_bs(),ens.get_variance(),ens.get_brier()
        #print("Permuted {}: Bias: {}, Variance: {}, Performance: {}".format(name,bias,var,perf))
        all_biasvar.append([bias,var])
        all_biasvarperf.append([perf,bias/var])
    
    biasvar_array = np.array(all_biasvar)
    biasvarperf_array = np.array(all_biasvarperf)
    return biasvar_array,biasvarperf_array

# get bias, variance, and performance to plot
def get_arrays_toplot(models):
    """
    Takes as argument a dictionary of models: keys giving model names, values are dictionaries with paths to individual entries. 
    :param models: names of individual models. 
    """
    all_biasvar = []
    all_biasvarperf = []
    gammas = []
    for modelname,model in models.items():
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
 
        [ens.register(os.path.join(here,m),i,None,os.path.join(here,l),**kwargs) for i,(m,l) in  enumerate(zip(model.modelnames,model.labelpaths))]
        bias,var,perf = ens.get_bias_bs(),ens.get_variance(),ens.get_brier()
        print(modelname,var,bias)
        if modelname in ["DKL_gamma_1","ResNet18"]:
            base_biasvar = [bias,var]
        else:    
            #print("{}: Bias: {}, Variance: {}, Performance: {}".format(modelname,bias,var,perf))
            all_biasvar.append([bias,var])
            all_biasvarperf.append([perf,bias/var])
            gammas.append(modelname.split("DKL_gamma_")[-1])
    
    biasvar_array = np.array(all_biasvar)
    biasvarperf_array = np.array(all_biasvarperf)
    return biasvar_array,biasvarperf_array,base_biasvar,np.array(gammas)


@hydra.main(config_path = "../script_configs/biasvar/cifar10",config_name = "cifar10_dkl")
def main(args):
    print(len(args.models))
    biasvar_array,biasvarperf_array,base_biasvar,gammas = get_arrays_toplot(args.models)

    fig,ax = plt.subplots(figsize=(9,7))
    #for seed in range(10):
    #    biasvar_array_permed,biasvarperf_array_permed= get_heterogeneous_biasvar(args.models,seed=seed)
    #    if seed == 0:
    #        ax.scatter(biasvar_array_permed[:,1],biasvar_array_permed[:,0],color = "orange",s=1,label = "heterogeneous")
    #    else:    
    #        ax.scatter(biasvar_array_permed[:,1],biasvar_array_permed[:,0],color = "orange",s=1)
    #defaultline = np.array([proportion(pi,10,0.98) for pi in np.linspace(0.87,1,100)])        
    #plt.plot(defaultline[:,1],defaultline[:,0],"--",color = "black",label="sim frontier")
    if args.get("colormap",False) is True: ## plot assuming scatters are ordered
        colors =np.concatenate([ cm.coolwarm(np.linspace(0, 0.5, 8)[:-1]) ,
            cm.coolwarm(np.linspace(0.5,1,16)[1:])])
        ax.scatter(biasvar_array[:,1],biasvar_array[:,0],s=60,c= colors,edgecolors =
                "black",linewidths = 0.5)
    else:    
        ax.scatter(biasvar_array[:,1],biasvar_array[:,0],s=60,edgecolor = "black", linewidths =
                0.5)
    for i in range(15):    
        ax.plot(np.linspace(0,args.line_extent,100),0.055+0.005*i+np.linspace(0,args.line_extent,100),color = "black",alpha = 0.2)
    ax.scatter(base_biasvar[1],base_biasvar[0],marker = "x",color = "black",s=30,label = "$\gamma=1$")
    ax.set_xlim([0.01,0.05])
    ax.set_ylim([0.1,0.14])
    ax.set_box_aspect(1)
    ax.set_xlabel("Variance")
    ax.set_ylabel("Avg. Single Model")
    #fig.colorbar(sc,ax = ax)
    ax2 = fig.add_axes([0.85, 0.1, 0.03, 0.8])
    ## cmap stuff 
    cmap = cm.coolwarm
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', colors, cmap.N)
    norm = mpl.colors.BoundaryNorm(gammas.astype(float),cmap.N)
    cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,
            ticks=gammas.astype(float), norm = norm,boundaries=gammas.astype(float),format= "%.2e")
    ax.set_title("CIFAR 10: ResNet18 $\gamma$ models")
    #plt.legend()
    fig.savefig("{}_biasvar_zoom.pdf".format(args.title))
    plt.show()

    
if __name__ == "__main__":
    main()
