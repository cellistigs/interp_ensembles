import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm
import copy
from scipy.special import softmax

here = os.path.dirname(__file__)
#results = os.path.join(here,"results","mnist_no_reg")
results = os.path.join(here,"results","mnist_no_reg_3")
extraresults = os.path.join(here,"results","mnist_no_reg_2")
plt.style.use(os.path.join(here,"../../etc/config/stylesheet.mplstyle"))

nb_classes = 10
nb_examples = 10000
interp_thresh = 250
size_ensemble = 10
coolwarm = cm.get_cmap("coolwarm",interp_thresh)

def get_all_vecs(M,C):
    all_vecs = []
    init = np.zeros(C)
    init[0] = M
    all_vecs.append(copy.deepcopy(init))
    all_vecs.append(copy.deepcopy(np.roll(init,1)))
    all_vecs.append(copy.deepcopy(np.roll(init,-1)))
    while init[0] > 1:
        for i in range(C-1):
            if i == 0 and max(init[:1])<= 1:
                init[i] -= 1
                init[i+1] +=1
                all_vecs.append(copy.deepcopy(np.roll(init,1)))
                all_vecs.append(copy.deepcopy(np.roll(init,-1)))

            else:
                if init[i] > 1 :
                    init[i] -= 1
                    init[i+1] +=1
                    all_vecs.append(copy.deepcopy(init))
                    all_vecs.append(copy.deepcopy(np.roll(init,1)))
                    all_vecs.append(copy.deepcopy(np.roll(init,-1)))
                else:    
                    pass
    return all_vecs        

def get_all_dirichlet_predict(M,C):
    """get the scores and variances corresponding to concentration of prediction into the corners of the simplex.
    """
    vecs = get_all_vecs(M,C)
    print(vecs)
    all_vars = []
    all_avg = []
    ## variances: 
    for vec in vecs: 
        mean = vec/M
        neg_vec = M - vec
        var = (vec*(1-mean)**2 + neg_vec*(mean)**2)/(M)
        all_vars.append(var)
        score = neg_vec[0]/M*2
        all_avg.append(score)
    vars_array = np.array(all_vars).sum(axis = 1)    
    return vars_array,all_avg    
        
def proportion(p,M = 2):
    BS = (p-1)**2+((1-p)**2)/(M-1)
    sum_var = p*(1-p)+(1-p)*(M-2+p)/(M-1)
    return BS+sum_var,sum_var 

def analytic(classes):
    """Show analytic results with samplewise uncorrelated errors. output a curve per classes. 

    from code: 
    for classes,colors,label in [(1000000,"purple","999999 \n(ImageNet independent)"),(10,"blue","9 \n(CIFAR-10 independent)"),(8,"green",7),(6,"yellow",5),(4,"orange",3),(2,"red","1 (binary)")]:
        output = np.array([(proportion(ai,M = classes)) for ai in a])
        plt.plot(output[:,1],output[:,0],color = colors,label = "Error Classes: {}".format(label))
    """
    a = np.linspace(0,1,100)
    output = np.array([(proportion(ai,M = classes)) for ai in a])
    return output


def get_filenames(results):
    result_cands = os.listdir(results)
    widths = [r for r in result_cands if not r.endswith(".json")]
    width_dirs = {int(w):{"dirname":os.path.join(results,w),"preds":[os.path.join(results,w,"{}_probs.npy".format(i)) for i in range(size_ensemble)],"label":os.path.join(results,w,"labels.npy")} for w in widths} 
    #extraresult_cands = os.listdir(extraresults)
    #extrawidths = [r for r in extraresult_cands if not r.endswith(".json")]
    #for ew in extrawidths:
    #    for i in range(5):
    #        width_dirs[int(ew)]["preds"].append(os.path.join(extraresults,ew,"{}_probs.npy".format(i)))
    return width_dirs

def get_data(width_dirs):
    data = {}
    for w,wdict in width_dirs.items():
        data[w] = {
                "preds":[np.load(path) for path in wdict["preds"]],
                "label": np.array([int(i) for i in np.load(wdict["label"],allow_pickle=True)])
                }
    return data    

def permute_samples_inclass(preds,labels):
    """
    """
    args = np.argsort(labels)
    # sort into classes: 
    sorted_labels,sorted_preds = labels[args],preds[args]
    for i in range(max(labels)):
        # get all samples with a given class output:
        where_label = np.where(sorted_labels==i)[0]
        # permute all samples with a given class output: 
        permuted_inclass = np.random.permutation(where_label)
        sorted_preds[where_label] = sorted_preds[permuted_inclass]
    return sorted_preds,sorted_labels    

def generate_class_permute_indices(permindex):
    """
    """
    indices = [np.arange(nb_classes) for i in range(nb_examples)]
    all_true_indices = np.array(indices).T
    for p in permindex:
        all_true_indices[:,p] = np.random.permutation(nb_classes)
    return all_true_indices    

def var_diversity(probs,labels,reduce_mean = True):
    """Given probs of shape (models,samples, classes) and labels of shape (samples), generate a corresponding diversity measure.  

    """
    variances = np.sum(np.var(probs,axis = 0,ddof = 0),axis = -1)
    if reduce_mean:
        return np.mean(variances)
    else:
        return variances

def brier_multi(probs,labels,reduce_mean = True): 
    labels_onehot = np.zeros(np.shape(probs))
    for i,li in enumerate(labels):
        labels_onehot[i,int(li)] = 1
    if reduce_mean:    
        return np.mean(np.sum((probs-labels_onehot)**2,axis = 1))    
    else:
        return np.sum((probs-labels_onehot)**2,axis = 1)

def compare_at_depth(probs,labels,reduce_mean = True):
    """Compute the bias variance decompositions and scores for individual models, bags, and random forest classifiers. 

    """

    ## fitting bags
    ## This is the average of probabilities output by each individual tree. 
    ens_individual_score = [brier_multi(np.stack(probs[i],axis = 0),labels,reduce_mean) for i in range(len(probs))]
    ens_variance = var_diversity(np.stack(probs,axis = 0),labels,reduce_mean)
    return ens_individual_score,ens_variance

def rescale_softmax(softmax,temp = 100):
    """take softmax probability output and rescale with a new temperature. 
    """
    


if __name__ == "__main__":

    indices = [i for i in range(size_ensemble)]
    filenames = get_filenames(results)
    data = get_data(filenames)

    to_plot = {}
    for w,wdata in data.items():
        labels = wdata["label"]
        ## get the indices that would sort this label set: 
        labelinds = np.argsort(labels)

        # Get unpermuted results 
        # get mean average single model brier score and variance.
        score,var = compare_at_depth(wdata["preds"],wdata["label"])
        # get average single model brier score and variance for all samples.
        fullscore,fullvar = compare_at_depth(wdata["preds"],wdata["label"],reduce_mean=False)
        # ------------------------------------------------------
        # Control:
        # decorrelating the errors  by randomizing them
        # while keeping the same error rate per class.

        # Permute by samples:
        sample_perm_preds = []
        for preds in wdata["preds"]:
            perm_pred,perm_labels = permute_samples_inclass(preds,labels)
            sample_perm_preds.append(perm_pred)
        sample_permscore,sample_permvar = compare_at_depth(sample_perm_preds,perm_labels)

        # decorrelating the errors by randomizing
        # the class the model predicts for each error sample.

        # Permute by class    
        class_perm_preds = []
        for preds in wdata["preds"]:
            classes = np.argmax(preds,axis = 1)
            incorrect = np.where(classes != labels)[0]
            permuted_indices = generate_class_permute_indices(incorrect)
            preds = preds[np.arange(nb_examples),permuted_indices].T
            class_perm_preds.append(preds)
        class_permscore,class_permvar = compare_at_depth(class_perm_preds,wdata["label"])    
        # ----------------------------------------------

        # log(p) rescaling.
        softmaxed_preds = []
        for preds in wdata["preds"]:
            logpreds = np.log(preds)+100
            preds = softmax(500*logpreds,axis = 1)
            softmaxed_preds.append(preds)
        # compute avg single model and variance on rescaled predictions
        # both for mean and individual samples.
        softmaxed_permscore,softmaxed_permvar = compare_at_depth(softmaxed_preds,wdata["label"])    
        softmaxed_permscore_all,softmaxed_permvar_all = compare_at_depth(softmaxed_preds,wdata["label"],reduce_mean = False)    

        #    print(incorrect)
        #    print(len(incorrect))
        # Log plotting results
        to_plot[w] = {"base":[np.mean(score),np.mean(var)],"sample_shuffle":[np.mean(sample_permscore),np.mean(sample_permvar)],"class_error_shuffle":[np.mean(class_permscore),np.mean(class_permvar)],"all":[np.mean(np.array(fullscore),axis =0),fullvar],"softmax":[np.mean(softmaxed_permscore),np.mean(softmaxed_permvar)],"softmax_all":[np.mean(np.array(softmaxed_permscore_all),axis = 0),softmaxed_permvar_all]}

    fig,ax = plt.subplots(1,2,figsize = (16,8))

    # panel 1
    # plot means 
    for w,d in to_plot.items():
        if w == 1:
            kwargs = {"label":"means"}
        else:    
            kwargs = {}
        ax[0].plot(d["base"][1],d["base"][0],"o",color = coolwarm(w),markersize = 5,markeredgecolor = "black",**kwargs)

    # panel 2 (renormalized softmax)
    # plot means 
    for w,d in to_plot.items():
        if w == 1:
            kwargs = {"label":"new means"}
        else:    
            kwargs = {}
        ax[1].plot(d["softmax"][1],d["softmax"][0],"v",color = coolwarm(w),markersize = 8,markeredgecolor = "black",**kwargs)
        if w == 1:
            kwargs = {"label":"orig. means"}
        else:    
            kwargs = {}
        ax[1].plot(d["base"][1],d["base"][0],"o",color = coolwarm(w),markersize = 5,markeredgecolor = "black",alpha = 0.5,**kwargs)

    for i in range(40):
        offset = i*0.1-1
        line = np.linspace(0,1,100)
        [ax[i].plot(line,offset+line,alpha = 0.3,color = "black") for i in range(ax.shape[0])]
    [ax[i].legend() for i in range(ax.shape[0])]
    ax[0].set_title("RFF variance vs. single model loss:\n Empirical Frontier")    
    ax[1].set_title("RFF variance vs. single model loss (renormalized softmax):\n Empirical Frontier")    
    ax[0].set_xlim(0,0.75)
    ax[0].set_ylim(0.2,1.85)
    ax[1].set_xlim(0,0.75)
    ax[1].set_ylim(0.2,1.85)
    ax[0].set_xlabel("Variance (Pred. Diversity)")
    ax[1].set_xlabel("Variance (Pred. Diversity)")
    ax[0].set_ylabel("Avg. Single Model Brier Score")
    ax[1].set_ylabel("Avg. Single Model Brier Score")
    plt.savefig("figs/pareto_frontier_manip_{}.pdf".format(size_ensemble))
    fig,ax = plt.subplots(1,2,figsize = (16,8))

    # panel 1
    # plot samples 
    for w,d in to_plot.items():
        if w == 1:
            kwargs = {"label":"indiv. samples"}
        else:    
            kwargs = {}
        ax[0].scatter(d["all"][1],d["all"][0],s=0.5,marker ="o",alpha = 0.5,color = coolwarm(w),**kwargs)
    # plot means 
    for w,d in to_plot.items():
        if w == 1:
            kwargs = {"label":"means"}
        else:    
            kwargs = {}
        ax[0].plot(d["base"][1],d["base"][0],"o",color = coolwarm(w),markersize = 5,markeredgecolor = "black",**kwargs)

    # plot predictions (unmodified)
    theory_curves = [analytic(i) for i in [size_ensemble]]
    for t,tcurve in enumerate(theory_curves):
        if t == 0:
            kwargs = {"label":"predicted max var curve:\n infinite ensemble"}
        else:    
            kwargs = {}
        ax[0].plot(tcurve[:,1],tcurve[:,0],"--",color = "black",alpha = 0.5,**kwargs)

    variance,avg = get_all_dirichlet_predict(size_ensemble,nb_classes) 
    #for i in range(len(variance)):
    #    if i == 0:
    #        kwargs = {"label":"predicted max confidence points: \n finite ensemble"}
    #    else:    
    #        kwargs = {}
    #    ax[0].plot(variance[i],avg[i],"x",color = "black",markersize = 5,**kwargs)

    # panel 2 (renormalized softmax)
    # plot samples 
    for w,d in to_plot.items():
        if w == 1:
            kwargs = {"label":"indiv. samples"}
        else:    
            kwargs = {}
        ax[1].scatter(d["softmax_all"][1],d["softmax_all"][0],s=0.5,marker ="o",alpha = 0.5,color = coolwarm(w),**kwargs)
    # plot means 
    for w,d in to_plot.items():
        if w == 1:
            kwargs = {"label":"new means"}
        else:    
            kwargs = {}
        ax[1].plot(d["softmax"][1],d["softmax"][0],"v",color = coolwarm(w),markersize = 8,markeredgecolor = "black",**kwargs)
        if w == 1:
            kwargs = {"label":"orig. means"}
        else:    
            kwargs = {}
        ax[1].plot(d["base"][1],d["base"][0],"o",color = coolwarm(w),markersize = 5,markeredgecolor = "black",alpha = 0.5,**kwargs)

    # plot predictions (unmodified)
    for t,tcurve in enumerate(theory_curves):
        if t == 0:
            kwargs = {"label":"predicted max var\n infinite ensemble"}
        else:    
            kwargs = {}
        ax[1].plot(tcurve[:,1],tcurve[:,0],"--",color = "black",alpha = 0.5,**kwargs)

    #for i in range(len(variance)):
    #    if i == 0:
    #        kwargs = {"label":"predicted max confidence for\n finite ensemble"}
    #    else:    
    #        kwargs = {}
    #    ax[1].plot(variance[i],avg[i],"x",color = "black",markersize = 5,**kwargs)

    for i in range(40):
        offset = i*0.1-1
        line = np.linspace(0,1,100)
        [ax[i].plot(line,offset+line,alpha = 0.3,color = "black") for i in range(ax.shape[0])]
    [ax[i].legend() for i in range(ax.shape[0])]
    ax[0].set_title("RFF variance vs. single model loss:\n Empirical frontier w/ full distribution")    
    ax[1].set_title("RFF variance vs. single model loss (renormalized softmax):\n Empirical frontier w/ full distribution")    
    ax[0].set_xlim(0,1)
    ax[0].set_ylim(0,2.1)
    ax[1].set_xlim(0,1)
    ax[1].set_ylim(0,2.1)
    ax[0].set_xlabel("Variance (Pred. Diversity)")
    ax[1].set_xlabel("Variance (Pred. Diversity)")
    ax[0].set_ylabel("Avg. Single Model Brier Score")
    ax[1].set_ylabel("Avg. Single Model Brier Score")
    plt.savefig("figs/pareto_frontier_distribution_{}.pdf".format(str(size_ensemble)))
    
    fig,ax = plt.subplots(1,2,figsize = (16,8))

    for w,d in to_plot.items():
        if w == 1:
            kwargs = {"label":"shuffle means"}
        else:    
            kwargs = {}
        ax[0].plot(d["class_error_shuffle"][1],d["class_error_shuffle"][0],"s",color = coolwarm(w),markersize = 5,markeredgecolor = "black",**kwargs)
        if w == 1:
            kwargs = {"label":"means"}
        else:    
            kwargs = {}
        ax[0].plot(d["base"][1],d["base"][0],"o",color = coolwarm(w),markersize = 5,markeredgecolor = "black",**kwargs)

        if w == 1:
            kwargs = {"label":"shuffle means"}
        else:    
            kwargs = {}
        ax[1].plot(d["sample_shuffle"][1],d["sample_shuffle"][0],"^",color = coolwarm(w),markersize = 5,markeredgecolor = "black",**kwargs)
        if w == 1:
            kwargs = {"label":"means"}
        else:    
            kwargs = {}
        ax[1].plot(d["base"][1],d["base"][0],"o",color = coolwarm(w),markersize = 5,markeredgecolor = "black",**kwargs)
    for i in range(40):
        offset = i*0.1-1
        line = np.linspace(0,1,100)
        [ax[i].plot(line,offset+line,alpha = 0.3,color = "black") for i in range(ax.shape[0])]
    [ax[i].legend() for i in range(ax.shape[0])]

    
    ax[0].set_title("RFF variance vs. single model loss control:\n classwise shuffle (error samples only)")    
    ax[1].set_title("RFF variance vs. single model loss control:\n samplewise shuffle")    
    ax[0].set_xlim(0,0.35)
    ax[0].set_ylim(0.2,1.1)
    ax[1].set_xlim(0,0.35)
    ax[1].set_ylim(0.2,1.1)
    ax[1].set_xlabel("Variance (Pred. Diversity)")
    ax[0].set_xlabel("Variance (Pred. Diversity)")
    ax[0].set_ylabel("Avg. Single Model Brier Score")
    ax[1].set_ylabel("Avg. Single Model Brier Score")
    plt.savefig("figs/pareto_frontier_shuffle_control_{}.pdf".format(str(size_ensemble)))


    #fig,ax = plt.subplots(3,2,figsize = (3,10))
    #for w,d in to_plot.items():
    #    if w == 1:
    #        ax[0,0].plot(d["base"][1],d["base"][0],"x",color = coolwarm(w),label = "no shuffle")
    #        ax[1,0].plot(d["sample_shuffle"][1],d["sample_shuffle"][0],"o",color = coolwarm(w),label = "sample shuffle")
    #        ax[1,0].plot(d["base"][1],d["base"][0],"x",color = coolwarm(w),label = "no shuffle")
    #        ax[0,1].plot(d["class_error_shuffle"][1],d["class_error_shuffle"][0],"*",color = coolwarm(w),label = "error class shuffle")
    #        ax[0,1].plot(d["base"][1],d["base"][0],"x",color = coolwarm(w),label = "no shuffle")
    #        ax[1,1].scatter(d["all"][1],d["all"][0],s=1,color = coolwarm(w),marker ="x",alpha=0.5)
    #        #theory_curves = [analytic(i) for i in range(2,6)]
    #        #[ax[1,1].plot(t[:,1],t[:,0],"--",color = "black") for t in theory_curves]
    #        [ax[1,1].plot(variance[i],avg[i],"bx",markersize = 5) for i in range(len(variance))]
    #        ax[1,1].plot(d["base"][1],d["base"][0],"x",color = coolwarm(w),label = "no shuffle")
    #        ax[2,0].plot(d["base"][1],d["base"][0],"x",color = coolwarm(w),label = "no shuffle")
    #        ax[2,0].plot(d["softmax"][1],d["softmax"][0],"v",color = coolwarm(w),label = "softmaxed")
    #        ax[2,1].scatter(d["softmax_all"][1],d["softmax_all"][0],s=1,color = coolwarm(w),marker ="x",alpha=0.5,label = "retransform")
    #    else:    
    #        ax[0,0].plot(d["base"][1],d["base"][0],"x",color = coolwarm(w))
    #        ax[1,0].plot(d["sample_shuffle"][1],d["sample_shuffle"][0],"o",color = coolwarm(w))
    #        ax[1,0].plot(d["base"][1],d["base"][0],"x",color = coolwarm(w))
    #        ax[0,1].plot(d["class_error_shuffle"][1],d["class_error_shuffle"][0],"*",color = coolwarm(w))
    #        ax[0,1].plot(d["base"][1],d["base"][0],"x",color = coolwarm(w))
    #        ax[1,1].scatter(d["all"][1],d["all"][0],s=1,color = coolwarm(w),marker ="x",alpha=0.5)
    #        ax[1,1].plot(d["base"][1],d["base"][0],"x",color = coolwarm(w))
    #        ax[2,0].plot(d["base"][1],d["base"][0],"x",color = coolwarm(w))
    #        ax[2,0].plot(d["softmax"][1],d["softmax"][0],"v",color = coolwarm(w))
    #        ax[2,1].scatter(d["softmax_all"][1],d["softmax_all"][0],s=1,color = coolwarm(w),marker ="x",alpha=0.5)
    #for i in range(20):
    #    offset = i*0.1
    #    line = np.linspace(0,1,100)
    #    [ax[i,j].plot(line,offset+line,alpha = 0.3,color = "black") for i in range(ax.shape[0]) for j in range(ax.shape[1])]
    #    [ax[i,j].legend() for i in range(ax.shape[0]) for j in range(ax.shape[1])]
    #ax[0,0].set_title("RFF variance and single model uncertainty w/ \n increasing width")    
    #ax[1,0].set_title("Shuffle control: within class, samplewise shuffle")    
    #ax[0,1].set_title("Shuffle control: within sample, classwise shuffle (errors only)")    
    #ax[1,1].set_title("Per datapoint distribution")    
    #ax[2,0].set_title("Softmaxed")    
    plt.show()    

