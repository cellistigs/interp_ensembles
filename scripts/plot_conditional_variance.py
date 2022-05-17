## Plots the conditional expectation of variance as a function of aleatoric uncertainty. Based off of `Conditional_Variance_Aleatoric` in this notebook. 

## Cell 0 
import hydra
import time
from hydra import utils
import json
import gc 
from statsmodels.stats.proportion import proportion_confint 
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
from scipy.stats import gaussian_kde
here = os.path.dirname(os.path.abspath(__file__))
import seaborn as sns
from sklearn.metrics import r2_score
import torch

import matplotlib as mpl

plt.style.use(os.path.join(here,"../etc/config/geoff_stylesheet.mplstyle"))

def Var(probs,labels):
    """Calculates average and diversity decomposition for variance

    :param probs: probabilities (softmax) with shape (sample,model,class)
    :param labels: labels
    """
    avg = 1-probs.pow(2).sum(dim=-1).mean(dim=-1)
    div = probs.var(dim=-2).sum(dim=-1)
    return avg,div

def JS(probs,labels):
    """Calculates average and diversity decomposition for jensen-shannon. 

    :param probs: probabilities (softmax) with shape (sample,model,class)
    :param labels: labels
    """
    ensemble_probs = probs.mean(dim=-2)
    ensemble_info = ensemble_probs*ensemble_probs.log()
    ensemble_entropy = -ensemble_info.sum(dim = -1)

    # Now get entropy for each individual model member. 
    single_model_info = probs*probs.log()
    avg_single_model_entropy = -single_model_info.sum(dim=-1).mean(dim = -1)
    ## then, the average single model entropy is: 
    avg = avg_single_model_entropy
    div = ensemble_entropy - avg_single_model_entropy ## this is Jensen Shannon divergence
    
    return avg,div

def KL(probs,labels):
    """Calculates average and diversity decomposition for kl divergence. 

    :param probs: probabilities (softmax) with shape (sample,model,class)
    :param labels: labels
    """
    ensemble_probs = probs.mean(dim=-2)
    ensemble_like = ensemble_probs[range(ensemble_probs.shape[0]),labels]
    ensemble_nll = -ensemble_like.log()

    all_single = []
    for si in range(probs.shape[1]):
        single_model_like = probs[range(probs.shape[0]),si,labels]
        single_model_nll = -single_model_like.log()
        all_single.append(single_model_nll)
    avg_single_model_nll = torch.stack(all_single,axis = 0).mean(0)
    avg = avg_single_model_nll
    div = -(ensemble_nll-avg_single_model_nll)
    print(avg,div)
    return avg,div

div_funcs = {"Var":Var,"JS":JS,"KL":KL}

div_names = {"imagenetv2":"ImageNet V2 (OOD)",
        "ood_cinic_preds.npy":"CINIC-10 (OOD)",
        "ood_preds.npy":"CIFAR10.1 (OOD)",
        "ood_cifar10_c_fog_1_preds.npy":"CIFAR10-C Fog 1 (OOD)",
        "ood_cifar10_c_fog_5_preds.npy":"CIFAR10-C Fog 5 (OOD)",
        "ood_cifar10_c_brightness_1_preds.npy":"CIFAR10-C Brightness 1 (OOD)",
        "ood_cifar10_c_brightness_5_preds.npy":"CIFAR10-C Brightness 5 (OOD)",
        "ood_cifar10_c_gaussian_noise_1_preds.npy":"CIFAR10-C Gauss Noise 1 (OOD)",
        "ood_cifar10_c_gaussian_noise_5_preds.npy":"CIFAR10-C Gauss Noise 5 (OOD)",
        "ood_cifar10_c_contrast_1_preds.npy":"CIFAR10-C Contrast 1 (OOD)",
        "ood_cifar10_c_contrast_5_preds.npy":"CIFAR10-C Contrast 5 (OOD)",
        }

quantity_names = {"Var":
            {"avg":r"Avg. Uncertainty (1-$ E[ \Vert f \Vert^2 ]$)",
            "div":"Variance"},
        "JS":{
            "avg":"Avg. Entropy",
            "div":"JS Divergence"},
        "KL":{
            "avg":"Avg. NLL",
            "div":"KL Divergence"}}

def refresh_cuda_memory():
    """
    Re-allocate all cuda memory to help alleviate fragmentation
    """
    # Run a full garbage collect first so any dangling tensors are released
    gc.collect()

    # Then move all tensors to the CPU
    locations = {}
    for obj in gc.get_objects():
        if not isinstance(obj, torch.Tensor):
            continue

        locations[obj] = obj.device
        obj.data = obj.data.cpu()
        if isinstance(obj, torch.nn.Parameter) and obj.grad is not None:
            obj.grad.data = obj.grad.cpu()

    # Now empty the cache to flush the allocator
    torch.cuda.empty_cache()

    # Finally move the tensors back to their associated GPUs
    for tensor, device in locations.items():
        tensor.data = tensor.to(device)
        if isinstance(tensor, torch.nn.Parameter) and tensor.grad is not None:
            tensor.grad.data = tensor.grad.to(device)

# @hydra.main(config_path = "script_configs",config_name ="testcinicgaussian") from remote
@hydra.main(config_path = "script_configs",config_name ="modeldata_test.yaml")
def main(cfg):
    """Function to create ensembles from groups of logits on the fly, and compare their conditional variances. 

    args:
    :param plot: whether or not to plot output. 
    :param ind_stubs: a list of in distribution stub names 
    :param ood_stubs: a list of out of distribution stub names
    :param ood_suffix: a list of suffixes to append for ood data 
    :param gpu: if we should use gpu or not: 
    """
    ## formatting limits
    kde_exp = -0.125 ## 1/4*nb_dims
    xranges = {"Var":[0.,1.],"JS":[0,np.log(cfg.nclasses)],"KL":[0,11]}
    xrange = xranges[cfg.uncertainty]

    ## Cell 1: formatting filenames 
    basedir = os.path.join(here,"../results/")


    if (cfg.ind_stubs is not None and cfg.ood_stubs is not None):
        ind_prob_paths = [os.path.join(basedir,s_in+"ind_preds.npy") for s_in in cfg.ind_stubs]
        ood_prob_paths = [os.path.join(basedir,s_in+cfg.ood_suffix) for s_in in cfg.ood_stubs]
        if cfg.ind_softmax is False:
            ind_probs = torch.stack([
                torch.tensor(np.load(ind_prob_path)).float()
                for ind_prob_path in ind_prob_paths
            ], dim=-2).softmax(dim=-1)
        else:
            ind_probs = torch.stack([
                torch.tensor(np.load(ind_prob_path)).float()
                for ind_prob_path in ind_prob_paths
            ], dim=-2)    
        ## Cell 2: formatting data

        ind_labels = torch.tensor(np.load(ind_prob_paths[0].replace("preds", "labels"))).long()

        if cfg.ood_softmax is False:
            ood_probs = torch.stack([
                torch.tensor(np.load(ood_prob_path)).float()
                for ood_prob_path in ood_prob_paths
            ], dim=-2).softmax(dim=-1)
        else:    
            ood_probs = torch.stack([
                torch.tensor(np.load(ood_prob_path)).float()
                for ood_prob_path in ood_prob_paths
            ], dim=-2)
        ood_labels = torch.tensor(np.load(ood_prob_paths[0].replace("preds", "labels"))).long()
    elif (cfg.ind_hdf5s is not None and cfg.ood_hdf5s is not None):    
        ind_prob_paths = [os.path.join(basedir,s_in) for s_in in cfg.ind_hdf5s]
        ood_prob_paths = [os.path.join(basedir,s_in) for s_in in cfg.ood_hdf5s]

        ind_probs = []
        for ind_prob in ind_prob_paths:
            with h5py.File(str(ind_prob), 'r') as f:
                ind_logits_out = f['logits'][()]
                ind_labels = f['targets'][()].astype('int')
                # calculate individual probs
                ind_probs.append(np.exp(ind_logits_out) / np.sum(np.exp(ind_logits_out), 1, keepdims=True))
        ind_probs = torch.stack([
            torch.tensor(ip) for ip in ind_probs],dim = -2)        
        ind_ind_labels = torch.tensor(ind_labels).long()

        ood_probs = []
        for ood_prob in ood_prob_paths:
            with h5py.File(str(ood_prob), 'r') as f:
                ood_logits_out = f['logits'][()]
                ood_labels = f['targets'][()].astype('int')
                # calculate oodividual probs
                ood_probs.append(np.exp(ood_logits_out) / np.sum(np.exp(ood_logits_out), 1, keepdims=True))
        ood_probs = torch.stack([
            torch.tensor(ip) for ip in ood_probs],dim = -2)        
        ood_ood_labels = torch.tensor(ood_labels).long()

    ## Cell 2: formatting data

    ind_labels = torch.tensor(np.load(ind_prob_paths[0].replace("preds", "labels"))).long()
    ind_indices =  torch.randperm(len(ind_probs))[:10000]

    if cfg.ood_softmax is False:
        ood_probs = torch.stack([
            torch.tensor(np.load(ood_prob_path)).float()
            for ood_prob_path in ood_prob_paths
        ], dim=-2).softmax(dim=-1)
    else:    
        ood_probs = torch.stack([
            torch.tensor(np.load(ood_prob_path)).float()
            for ood_prob_path in ood_prob_paths
        ], dim=-2)
    ood_labels = torch.tensor(np.load(ood_prob_paths[0].replace("preds", "labels"))).long()
    ood_indices =  torch.randperm(len(ood_probs))[:20000]

    ## Cell 3: 
    div_func = div_funcs[cfg.uncertainty]

    ind_avg, ind_div = div_func(ind_probs[ind_indices],ind_labels[ind_indices])
    ood_avg, ood_div = div_func(ood_probs[ood_indices],ood_labels[ood_indices])

    min_avg = np.min((np.min(ind_avg.numpy()),np.min(ood_avg.numpy())))
    max_avg = np.max((np.max(ind_avg.numpy()),np.max(ood_avg.numpy())))


    #ind_avg = ind_probs[ind_indices].pow(2).sum(dim=-1).mean(dim=-1)
    #ood_avg = ood_probs[ood_indices].pow(2).sum(dim=-1).mean(dim=-1)

    ## Cell 4: 

    #ind_div = ind_probs[ind_indices].var(dim=-2).sum(dim=-1)
    #ood_div = ood_probs[ood_indices].var(dim=-2).sum(dim=-1)

    ## Cell 5: 

    ind_avg_kde = gaussian_kde(ind_avg, bw_method=len(ind_indices) ** (kde_exp))
    ood_avg_kde = gaussian_kde(ood_avg, bw_method=len(ood_indices) ** (kde_exp))

    ## Cell 6: 
    ind_joint_kde = gaussian_kde(np.stack([ind_avg, ind_div]), bw_method=len(ind_indices) ** (kde_exp))
    ood_joint_kde = gaussian_kde(np.stack([ood_avg, ood_div]), bw_method=len(ood_indices) ** (kde_exp))

    ## Cell 7: 
    xs = np.linspace(xrange[0],xrange[1], 101)
    x_grid, y_grid = np.meshgrid(xs, xs)
    joint = np.stack([x_grid.reshape(-1), y_grid.reshape(-1)])
    ind_condition = np.logical_or(x_grid < ind_avg.min().numpy(),x_grid > ind_avg.max().numpy())
    ood_condition = np.logical_or(x_grid < ood_avg.min().numpy(),x_grid > ood_avg.max().numpy())
    
    ## Cell 8: 
    import gpytorch

    class GPModel(gpytorch.models.ExactGP):
        def __init__(self, metrics, variances,gpu = False):
            if gpu:
                ## multigpu settings
                output_device = torch.device('cuda:0')
                n_devices = torch.cuda.device_count()
                likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
            else:    
                likelihood = gpytorch.likelihoods.GaussianLikelihood()
            super().__init__(metrics, variances, likelihood)
            self.mean_module = gpytorch.means.ZeroMean()
            if gpu:
                base_covar_module = gpytorch.kernels.RBFKernel()
                self.covar_module = gpytorch.kernels.MultiDeviceKernel(
                base_covar_module, device_ids=range(n_devices),
                output_device=output_device
            )
            else:    
                self.covar_module = gpytorch.kernels.RBFKernel()
            self.covar_module.initialize(lengthscale=(len(variances) ** (kde_exp)))
            self.likelihood.initialize(noise=1e-4)

        def forward(self, x):
            mean = self.mean_module(x)
            covar = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean, covar)

    if cfg.signiftest is True:


        def cond_expec(ef2,var,num_classes):
            """Takes in two 1-d arrays representing expected norm squared (confidence) and variance. 
            Given these, returns the conditional expectation evaluated on int(1-(1/M)*100)+1 points between 1/M and 1.  

            """
            if cfg.gpu:
                ef2 = ef2.cuda()
                var = var.cuda()
                model = GPModel(ef2.double(),var.double()).double()
                model = model.cuda()
                max_cholesky_size = 800
            else:    
                model = GPModel(ef2.double(),var.double()).double()
                max_cholesky_size = 800 
            model.eval()
            start_conf = 1. / num_classes
            if cfg.gpu:
                cond_expec_xs = torch.linspace(min_avg,max_avg, int((1. - start_conf) * 100) + 1).cuda()
                #cond_expec_xs = torch.linspace(0,1, int((1. - start_conf) * 100) + 1).cuda()
            else:    
                cond_expec_xs = torch.linspace(min_avg,max_avg, int((1. - start_conf) * 100) + 1)
            with torch.no_grad(), gpytorch.settings.skip_posterior_variances():
                with gpytorch.settings.max_cholesky_size(max_cholesky_size):
                    cond_expec = model(torch.tensor(cond_expec_xs).double()).mean
                    if cfg.gpu:
                        cond_expec_final = cond_expec.cpu()
                        cond_expec_xs_final = cond_expec_xs.cpu()
                        del cond_expec
                        del cond_expec_xs
                        del model.likelihood
                        del model
                        del ef2
                        del var
                        return cond_expec_final,cond_expec_xs_final
            
            return cond_expec,cond_expec_xs

        def dstat(sample1,sample2,nclasses,return_ce = False,xrange = xrange):
            """Takes a distance statistic measuring how much greater the ood sample is than the ind sample.  

            :param sample1: a tuple containing conditional expectation and variance
            :param sample2: a tuple containing conditional expectation and variance
            """
            c1,cx = cond_expec(*sample1,cfg.nclasses)
            c2,cx = cond_expec(*sample2,cfg.nclasses)
            avg_abs = np.mean(c2.numpy()-c1.numpy())
            if return_ce is False:
                return 1/(1-(1/cfg.nclasses))*avg_abs
            else:
                return 1/(1-(1/cfg.nclasses))*avg_abs,c1,c2,cx

        def shuffle(sample1,sample2):
            """Given two tuples each containing paired data, aggregates them, permutes them, and then splits them once again into two samples of the same size as the original data. 

            :param sample1: a tuple containing conditional expectation and variance
            :param sample2: a tuple containing conditional expectation and variance
            """
            ## create numpy arrays for easy manipulation:
            s1_stacked = np.stack(sample1,axis = 1)
            s2_stacked = np.stack(sample2,axis = 1)
            ## get the shape of each: 
            s1_shape = np.shape(s1_stacked)
            s2_shape = np.shape(s2_stacked)
            assert s1_shape[1] == 2
            assert s2_shape[1] == 2
            ## we will split based on this shape:
            n1 = s1_shape[0]
            n2 = s2_shape[0]
            ## now concatenate:
            agg_sample = np.concatenate([s1_stacked,s2_stacked],axis = 0)
            perm_sample = np.random.permutation(agg_sample)
            perm_s1 = (torch.tensor(perm_sample[:n1,0]),torch.tensor(perm_sample[:n1,1]))
            perm_s2 = (torch.tensor(perm_sample[n1:,0]),torch.tensor(perm_sample[n1:,1]))
            return perm_s1,perm_s2


        N = 5 
        cfg.nclasses = 10
        sample1 = (ind_avg.double(), ind_div.double())
        sample2 = (ood_avg.double(), ood_div.double())

        ## calculate original stat
        print("calculating orig stat")
        d_orig,ind_cond_expec,ood_cond_expec,cond_expec_xs = dstat(sample1,sample2,cfg.nclasses,return_ce = True,xrange=xrange)
        dorig_val = d_orig.item()
        print("calculating shuffle stats")

        dprimes = []
        ces = []
        for n in range(N):
            print("split {}".format(n))
            dp,ce1,ce2,cex = dstat(*shuffle(sample1,sample2),cfg.nclasses,return_ce = True,xrange=xrange)


            dprimes.append(dp)
            ces.append(ce1)
            ces.append(ce2)
            print(dorig_val,"orig d")
            print(dprimes,"prime d")

            if n % 10 == 0: 
                refresh_cuda_memory()    
            print(f"After emptying cache: {torch.cuda.memory_allocated()}")
            print(f"After emptying cache: {torch.cuda.memory_cached()}")

        count = sum([dp> dorig_val for dp in dprimes])
        data = {"lower":None,"upper":None,"exact":None}
        interval = proportion_confint(count,N)
        data["lower"] =interval[0]
        data["upper"] = interval[1]
        data["exact"] = count/N
        with open("signifdata.json","w") as f:
            json.dump(data,f)





    if cfg.signiftest is False:
        ind_model = GPModel(ind_avg.double(), ind_div.double()).double()
        ood_model = GPModel(ood_avg.double(), ood_div.double()).double()
        ind_model.eval()
        ood_model.eval()

        start_conf = 1. / cfg.nclasses
        cond_expec_xs = torch.linspace(start_conf, 1., int((1. - start_conf) * 100) + 1)
        with torch.no_grad(), gpytorch.settings.skip_posterior_variances():
            with gpytorch.settings.max_cholesky_size(1e6):
                ind_cond_expec = ind_model(torch.tensor(cond_expec_xs).double()).mean
                ood_cond_expec = ood_model(torch.tensor(cond_expec_xs).double()).mean

                ind_preds_ind = ind_model(torch.tensor(ind_avg.double())).mean
                ood_preds_ood = ood_model(torch.tensor(ood_avg.double())).mean
                ind_preds_ood = ind_model(torch.tensor(ood_avg.double())).mean
                ood_preds_ind = ood_model(torch.tensor(ind_avg.double())).mean
                print("R^2 ind predicts ind: {}".format(r2_score(ind_div,ind_preds_ind)))
                print("R^2 ood predicts ind: {}".format(r2_score(ind_div,ood_preds_ind)))
                print("R^2 ood predicts ood: {}".format(r2_score(ood_div,ood_preds_ood)))
                print("R^2 ind predicts ood: {}".format(r2_score(ood_div,ind_preds_ood)))

    ## Cell 9:        


    varlims = {"Var":17.,"JS":5.}
    if cfg.plot:
        fig, (var_ax, ind_cond_ax, ood_cond_ax, cond_exp_ax) = plt.subplots(
            1, 4, figsize=(12, 3), sharex=False, sharey=False
        )
        levels = np.linspace(-3., 3., 51)

        sns.kdeplot(ind_div, ax=var_ax)
        sns.kdeplot(ood_div, ax=var_ax)
        var_ax.set(xlabel=quantity_names[cfg.uncertainty]["div"], title="Marginal {} Dist.\nComparison".format(cfg.uncertainty), ylim=(0., varlims[cfg.uncertainty]))

        ind_vals = ind_joint_kde(joint).reshape(x_grid.shape)
        ind_vals = ind_vals / ind_avg_kde(x_grid.ravel()).reshape(x_grid.shape)
        #ind_vals = np.where(x_grid < (1. / num_classes), 0., ind_vals)
        ind_vals = np.where(ind_condition, 0.,ind_vals)
        f = ind_cond_ax.contourf(
            x_grid, y_grid, ind_vals.clip(0., 10.),
            cmap="Blues",
            levels=np.linspace(0., 10., 50),
        )
        ind_cond_ax.set(
            xlim=(xrange[0],xrange[1]), ylim=(xrange[0],1), xlabel=r"{}".format(quantity_names[cfg.uncertainty]["avg"]),
            ylabel=r"{}".format(cfg.uncertainty), title="Conditional {}. Dist.\nImageNet (InD)".format(cfg.uncertainty)
        )
        fig.colorbar(f, ax=ind_cond_ax)

        ood_vals = ood_joint_kde(joint).reshape(x_grid.shape)
        ood_vals = ood_vals / ood_avg_kde(x_grid.ravel()).reshape(x_grid.shape)
        ood_vals = np.where(ood_condition, 0., ood_vals)
        f = ood_cond_ax.contourf(
            x_grid, y_grid, ood_vals.clip(0., 10.),
            cmap="Oranges",
            levels=np.linspace(0., 10., 50),
        )
        ood_cond_ax.set(
            xlim=(xrange[0],xrange[1]), ylim=(xrange[0],1),
            xlabel=r"{}".format(quantity_names[cfg.uncertainty]["avg"]),
            title="Conditional {}. Dist.\n{}".format(cfg.uncertainty,div_names[cfg.ood_suffix]),
            yticks=[]
        )
        fig.colorbar(f, ax=ood_cond_ax)
        cond_expec_show = np.where(ind_cond_expec)

        if cfg.signiftest is True:
            [cond_exp_ax.plot(cond_expec_xs,ce,color = "black",alpha= 0.05) for ce in ces]
        cond_exp_ax.plot(cond_expec_xs, ind_cond_expec, label="CIFAR10 (InD)")
        cond_exp_ax.plot(cond_expec_xs, ood_cond_expec, label="{}".format(div_names[cfg.ood_suffix]))
        cond_exp_ax.set(
            xlim=(xrange[0],xrange[1]),
            ylim=(xrange[0],xrange[1]),
            ylabel=r"$E [ \textrm{Diversity} \mid \textrm{Avg} ]$",
            xlabel=r"{}".format(quantity_names[cfg.uncertainty]["avg"]),
            title="Conditionally Expected \n{}: Comparison".format(quantity_names[cfg.uncertainty]["div"])
        )
        cond_exp_ax.legend(loc="best")
        fig.tight_layout()

        plt.savefig("cond_expected_{}.png".format(cfg.uncertainty))

        #fig,ax = plt.subplots(figsize= (4,4))
        #r2_matrix = np.array([[ind_r2_ind,ood_r2_ind],[ind_r2_ood,ood_r2_ood]])
        #ax.matshow(r2_matrix,vmin = 0, vmax = 1)
        #ax.set_title(r"$R^2$ values:"+"\n"+"InD OOD conditional variance")
        #ax.set_xticklabels(["","InD","OOD"])
        #ax.set_ylabel("Model")
        #ax.set_xlabel("Data")
        #ax.set_yticklabels(["","InD","OOD"])

        #for i in range(2):
        #    for j in range(2):
        #        c = r2_matrix[j,i]
        #        ax.text(i, j, "{:3.3}".format(c), va='center', ha='center')


        #plt.tight_layout()
        #plt.savefig("r2_matrix.png")        

if __name__ == "__main__":
    main()

