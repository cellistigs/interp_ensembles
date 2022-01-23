## Plots the conditional expectation of variance as a function of aleatoric uncertainty. Based off of `Conditional_Variance_Aleatoric` in this notebook. 

## Cell 0 
import hydra
import time
from hydra import utils
import json
from statsmodels.stats.proportion import proportion_confint 
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import gaussian_kde
here = os.path.dirname(os.path.abspath(__file__))
import seaborn as sns
from sklearn.metrics import r2_score
import torch

import matplotlib as mpl

plt.style.use(os.path.join(here,"../etc/config/geoff_stylesheet.mplstyle"))

@hydra.main(config_path = "script_configs",config_name ="modeldata_test")
def main(cfg):
    """Function to create ensembles from groups of logits on the fly, and compare their conditional variances. 

    args:
    :param plot: whether or not to plot output. 
    :param ind_stubs: a list of in distribution stub names 
    :param ood_stubs: a list of out of distribution stub names
    :param ood_suffix: a list of suffixes to append for ood data 
    :param gpu: if we should use gpu or not: 
    """

    ## Cell 1: formatting filenames 
    basedir = os.path.join(here,"../results/")
    ind_prob_paths = [
        os.path.join(basedir, basename)
        for basename in os.listdir(basedir)
        if "base_wideresnet28_10ind_preds" in basename
    ]
    ind_prob_paths = [os.path.join(basedir,s_in+"ind_preds.npy") for s_in in cfg.ind_stubs]
    # basedir = "../models/cinic_wrn28_10/"
    ood_prob_paths = [
        os.path.join(basedir, basename)
        for basename in os.listdir(basedir)
        if "base_wideresnet28_10ood_cinic_preds" in basename
    ]
    ood_prob_paths = [os.path.join(basedir,s_in+cfg.ood_suffix) for s_in in cfg.ood_stubs]
    num_classes = 10
    kde_exp = -0.125 ## 1/4*nb_dims

    ## Cell 2: formatting data

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
    ood_indices =  torch.randperm(len(ood_probs))[:10000]

    ## Cell 3: 
    ind_ef2 = ind_probs[ind_indices].pow(2).sum(dim=-1).mean(dim=-1)
    ood_ef2 = ood_probs[ood_indices].pow(2).sum(dim=-1).mean(dim=-1)

    ## Cell 4: 

    ind_var = ind_probs[ind_indices].var(dim=-2).sum(dim=-1)
    ood_var = ood_probs[ood_indices].var(dim=-2).sum(dim=-1)

    ## Cell 5: 

    ind_ef2_kde = gaussian_kde(ind_ef2, bw_method=len(ind_indices) ** (kde_exp))
    ood_ef2_kde = gaussian_kde(ood_ef2, bw_method=len(ood_indices) ** (kde_exp))

    ## Cell 6: 
    ind_joint_kde = gaussian_kde(np.stack([ind_ef2, ind_var]), bw_method=len(ind_indices) ** (kde_exp))
    ood_joint_kde = gaussian_kde(np.stack([ood_ef2, ood_var]), bw_method=len(ood_indices) ** (kde_exp))

    ## Cell 7: 
    xs = np.linspace(0., 1., 101)
    x_grid, y_grid = np.meshgrid(xs, xs)
    joint = np.stack([x_grid.reshape(-1), y_grid.reshape(-1)])

    ## Cell 8: 
    import gpytorch

    class GPModel(gpytorch.models.ExactGP):
        def __init__(self, metrics, variances,gpu = False):
            if gpu:
                likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
            else:    
                likelihood = gpytorch.likelihoods.GaussianLikelihood()
            super().__init__(metrics, variances, likelihood)
            self.mean_module = gpytorch.means.ZeroMean()
            self.covar_module = gpytorch.kernels.RBFKernel()
            self.covar_module.initialize(lengthscale=(len(variances) ** (kde_exp)))
            self.likelihood.initialize(noise=1e-4)

        def forward(self, x):
            mean = self.mean_module(x)
            covar = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean, covar)

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
            max_cholesky_size = 1e6
        model.eval()
        start_conf = 1. / num_classes
        if cfg.gpu:
            cond_expec_xs = torch.linspace(start_conf, 1., int((1. - start_conf) * 100) + 1).cuda()
        else:    
            cond_expec_xs = torch.linspace(start_conf, 1., int((1. - start_conf) * 100) + 1)
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

    def dstat(sample1,sample2,nclasses,return_ce = False):
        """Takes a distance statistic between conditional expectation of two given samples.  

        :param sample1: a tuple containing conditional expectation and variance
        :param sample2: a tuple containing conditional expectation and variance
        """
        c1,cx = cond_expec(*sample1,nclasses)
        c2,cx = cond_expec(*sample2,nclasses)
        avg_abs = np.mean(np.abs(c1-c2).numpy())
        if return_ce is False:
            return 1/(1-(1/nclasses))*avg_abs
        else:
            return 1/(1-(1/nclasses))*avg_abs,c1,c2,cx

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


    N = 100 
    nclasses = 10
    sample1 = (ind_ef2.double(), ind_var.double())
    sample2 = (ood_ef2.double(), ood_var.double())

    ## calculate original stat
    print("calculating orig stat")
    d_orig,ind_cond_expec,ood_cond_expec,cond_expec_xs = dstat(sample1,sample2,nclasses,return_ce = True)
    print("calculating shuffle stats")

    dprimes = []
    ces = []
    for n in range(N):
        print("split {}".format(n))
        dp,ce1,ce2,cex = dstat(*shuffle(sample1,sample2),nclasses,return_ce = True)


        dprimes.append(dp)
        ces.append(ce1)
        ces.append(ce2)
        print(d_orig,"orig d")
        print(dprimes,"prime d")

        del dp 
        del ce1
        del ce2
        del cex

        torch.cuda.empty_cache()
        print(f"After emptying cache: {torch.cuda.memory_allocated()}")
        print(f"After emptying cache: {torch.cuda.memory_cached()}")

    count = sum([dp> d_orig for dp in dprimes])
    data = {"lower":None,"upper":None,"exact":None}
    interval = proportion_confint(count,N)
    data["lower"] =interval[0]
    data["upper"] = interval[1]
    data["exact"] = count/N
    with open("signifdata.json","w") as f:
        json.dump(data,f)




    #ind_model = GPModel(ind_ef2.double(), ind_var.double()).double()
    #ood_model = GPModel(ood_ef2.double(), ood_var.double()).double()
    #ind_model.eval()
    #ood_model.eval()

    #start_conf = 1. / num_classes
    #cond_expec_xs = torch.linspace(start_conf, 1., int((1. - start_conf) * 100) + 1)
    #with torch.no_grad(), gpytorch.settings.skip_posterior_variances():
    #    with gpytorch.settings.max_cholesky_size(1e6):
    #        ind_cond_expec = ind_model(torch.tensor(cond_expec_xs).double()).mean
    #        ood_cond_expec = ood_model(torch.tensor(cond_expec_xs).double()).mean

    #        ind_preds_ind = ind_model(torch.tensor(ind_ef2.double())).mean
    #        ood_preds_ood = ood_model(torch.tensor(ood_ef2.double())).mean
    #        ind_preds_ood = ind_model(torch.tensor(ood_ef2.double())).mean
    #        ood_preds_ind = ood_model(torch.tensor(ind_ef2.double())).mean
    #        print("R^2 ind predicts ind: {}".format(r2_score(ind_var,ind_preds_ind)))
    #        print("R^2 ood predicts ind: {}".format(r2_score(ind_var,ood_preds_ind)))
    #        print("R^2 ood predicts ood: {}".format(r2_score(ood_var,ood_preds_ood)))
    #        print("R^2 ind predicts ood: {}".format(r2_score(ood_var,ind_preds_ood)))


    ## Cell 9:        
    if cfg.plot:
        fig, (var_ax, ind_cond_ax, ood_cond_ax, cond_exp_ax) = plt.subplots(
            1, 4, figsize=(12, 3), sharex=True, sharey=False
        )
        levels = np.linspace(-3., 3., 51)

        sns.kdeplot(ind_var, ax=var_ax)
        sns.kdeplot(ood_var, ax=var_ax)
        var_ax.set(xlabel="Var.", title="Marginal Var. Dist.\nComparison", ylim=(0., 15.))

        ind_vals = ind_joint_kde(joint).reshape(x_grid.shape)
        ind_vals = ind_vals / ind_ef2_kde(x_grid.ravel()).reshape(x_grid.shape)
        ind_vals = np.where(x_grid < (1. / num_classes), 0., ind_vals)
        f = ind_cond_ax.contourf(
            x_grid, y_grid, ind_vals.clip(0., 10.),
            cmap="Blues",
            levels=np.linspace(0., 10., 50),
        )
        ind_cond_ax.set(
            xlim=(0., 1.), ylim=(0., 1.), xlabel=r"Avg. Conf. ($ E[ \Vert f \Vert^2 ]$)",
            ylabel=r"Variance", title="Conditional Var. Dist.\nCIFAR10 (InD)"
        )
        fig.colorbar(f, ax=ind_cond_ax)

        ood_vals = ood_joint_kde(joint).reshape(x_grid.shape)
        ood_vals = ood_vals / ood_ef2_kde(x_grid.ravel()).reshape(x_grid.shape)
        ood_vals = np.where(x_grid < (1. / num_classes), 0., ood_vals)
        f = ood_cond_ax.contourf(
            x_grid, y_grid, ood_vals.clip(0., 10.),
            cmap="Oranges",
            levels=np.linspace(0., 10., 50),
        )
        ood_cond_ax.set(
            xlabel=r"Avg. Conf. ($ E[ \Vert f \Vert^2 ]$)",
            title="Conditional Var. Dist.\n{}".format(cfg.ood_suffix),
            yticks=[]
        )
        fig.colorbar(f, ax=ood_cond_ax)

        [cond_exp_ax.plot(cond_expec_xs,ce,color = "black",alpha= 0.05) for ce in ces]
        cond_exp_ax.plot(cond_expec_xs, ind_cond_expec, label="CIFAR10 (InD)")
        cond_exp_ax.plot(cond_expec_xs, ood_cond_expec, label="CINIC10 (OOD)")
        cond_exp_ax.set(
            ylim=(0., 1.),
            ylabel=r"$E [ \textrm{Var} \mid \textrm{Conf} ]$",
            xlabel=r"Avg. Conf. ($ E[ \Vert f \Vert^2 ]$)",
            title="Conditionally Expected Var.\nComparison"
        )
        cond_exp_ax.legend(loc="best")
        fig.tight_layout()

        plt.savefig("cond_expected_var.png")

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

