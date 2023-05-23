"""
Trains multiple RFF models on mnist and
saves the rsulting logits in results/
"""
## implement random feature regression based classification.
## compare three things: 1. a 
import json 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import log_loss,brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris,load_wine,load_digits,load_breast_cancer,fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.utils import check_random_state
from interpensembles.random_features import get_rff_pipelined_logistic_classification
import os

results_dir = os.path.join(os.path.dirname(__file__),"results")
identifier = "mnist_no_reg_w_logits"

def brier_multi(labels,probs): 
    labels_onehot = np.zeros(np.shape(probs))
    for i,li in enumerate(labels):
        labels_onehot[i,int(li)] = 1
    return np.mean(np.sum((probs-labels_onehot)**2,axis = 1))    

def var_diversity(probs,labels):
    """Given probs of shape (models,samples, classes) and labels of shape (samples), generate a corresponding diversity measure.  

    """
    variances = np.sum(np.var(probs,axis = 0),axis = -1)
    return np.mean(variances)

def compare_at_depth(xtest,ytest,ens_clf,depth,save_params=None):
    """Compute the bias variance decompositions and scores for individual models, bags, and random forest classifiers. 

    """

    ## fitting bags
    ## This is the average of probabilities output by each individual tree. 
    ens_score = brier_multi(ytest,ens_clf.predict_proba(xtest))
    ens_individual_probs = [e.predict_proba(xtest) for e in ens_clf.estimators_]
    ens_individual_logits = [e.decision_function(xtest) for e in ens_clf.estimators_]
    ens_individual_score = [brier_multi(ytest,e.predict_proba(xtest)) for e in ens_clf.estimators_]
    ens_diversity = var_diversity(np.stack(ens_individual_probs,axis =0),ytest)

    if save_params is not None:
        width_dir = os.path.join(results_dir,identifier,str(depth))
        os.mkdir(width_dir)
        np.save(os.path.join(width_dir,"labels.npy"),ytest)
        for e,eprobs in enumerate(ens_individual_probs):
            np.save(os.path.join(width_dir,"{}_probs.npy".format(e)),eprobs)
            np.save(os.path.join(width_dir,"{}_logits.npy".format(e)),ens_individual_logits[e])


    return ens_score,ens_individual_score,ens_diversity

def main():
    train_samples = 5000
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

    random_state = check_random_state(1)
    permutation = random_state.permutation(X.shape[0])
    X = X[permutation]
    y = y[permutation]
    X = X.reshape((X.shape[0], -1))

    xtrain, xtest, ytrain, ytest = train_test_split(
        X, y, train_size=train_samples, test_size=10000
    )
    in_shape = X.shape[1]
    max_iter = 1000000

    ## ensemble params
    ens_size = 10 
    width_maxrange = 400  # max width
    matrix_seed = 5  # seed for RFF init but not for bagging.
    repeat_iterates = 1  # number of times to repeat the experiment for each width (bagging is included)
    interval = 10  # grid spacing for width of RFF.
    # plotting threshold for interpolation. approx should be (N*C) here is fixed for all models.
    interp_thresh = 250

    fig,ax = plt.subplots(1,2,figsize=(10,3))
    coolwarm = cm.get_cmap("coolwarm",interp_thresh)
    # Arguments for logistic classifier in scikit learn
    logistic_args = {"max_iter":max_iter,"tol":1e-6,"penalty":"none"}
    ens_params_dict = {
            "ens_size":ens_size,
            "width_maxrange":width_maxrange,
            "matrix_seed":matrix_seed,
            "repeat_iterates":repeat_iterates,
            "interval":interval,
            "interp_threshold":interp_thresh,
            "logistic_args":logistic_args
            }

    save_dir = os.path.join(results_dir,identifier)
    os.mkdir(os.path.join(results_dir,identifier))
    # save the parameters used for the experiment
    file = os.path.join(save_dir,"experiment_params.json")
    with open(file,"w") as f:
        json.dump(ens_params_dict,f)

    for width in np.arange(width_maxrange,step = interval):
        print(width)
        if width == 0:
            width = 1
        for it in range(repeat_iterates):
            # run scikit learn LR of custom RFF class.
            base_randomf = get_rff_pipelined_logistic_classification(width,logistic_args = logistic_args,random_state = random_state).fit(xtrain,ytrain)
            # random initialization of RFF weigths and specific width.
            init_randomf = BaggingClassifier(get_rff_pipelined_logistic_classification(width,matrix_seed=None,logistic_args = logistic_args),bootstrap = False,n_estimators = ens_size,random_state=random_state).fit(xtrain,ytrain)
            # fixed RFF weights with bagging.
            bag_randomf = BaggingClassifier(get_rff_pipelined_logistic_classification(width,matrix_seed=matrix_seed,logistic_args = logistic_args),bootstrap = False,n_estimators = ens_size,random_state=random_state).fit(xtrain,ytrain)
            # bagging and random initialization of RFF weights.
            bag_init_randomf = BaggingClassifier(get_rff_pipelined_logistic_classification(width,matrix_seed = None,logistic_args = logistic_args),n_estimators = ens_size,random_state=random_state).fit(xtrain,ytrain)

            ## Take trained models and generate predictions.
            base_score = brier_multi(ytest,base_randomf.predict_proba(xtest))
            # pull the ensemble , average single model and diversity scores and save the resulting logits.
            init_score,init_indiv,init_div = compare_at_depth(xtest,ytest,init_randomf,width,ens_params_dict)
            bag_score,bag_indiv,bag_div = compare_at_depth(xtest,ytest,bag_randomf,width)
            bag_init_score,bag_init_indiv,bag_init_div = compare_at_depth(xtest,ytest,bag_init_randomf,width)


            if width == 1 and it == 0:
                ax[0].plot(0,base_score,"x",color = coolwarm(width),label = "single tree")    
                ax[0].plot(np.mean(init_div),np.mean(init_indiv),"o",color = coolwarm(width),label = "init")
                ax[0].plot(np.mean(bag_div),np.mean(bag_indiv),"+",color = coolwarm(width),label = "bag")
                ax[0].plot(np.mean(bag_init_div),np.mean(bag_init_indiv),"*",color = coolwarm(width),label = "bag+init")
                ax[1].plot(width,base_score,"x",color = coolwarm(width),label = "single tree")
                ax[1].plot(width,init_score,"o",color = coolwarm(width), label = "init")
                ax[1].plot(width,bag_score,"+",color = coolwarm(width), label = "bag")
                ax[1].plot(width,bag_init_score,"*",color = coolwarm(width), label = "bag+init")
            else:     
                ax[0].plot(0,base_score,"x",color = coolwarm(width))    
                ax[0].plot(np.mean(init_div),np.mean(init_indiv),"o",color = coolwarm(width))
                ax[0].plot(np.mean(bag_div),np.mean(bag_indiv),"+",color = coolwarm(width))
                ax[0].plot(np.mean(bag_init_div),np.mean(bag_init_indiv),"*",color = coolwarm(width))
                ax[1].plot(width,base_score,"x",color = coolwarm(width))
                ax[1].plot(width,init_score,"o",color = coolwarm(width))
                ax[1].plot(width,bag_score,"+",color = coolwarm(width))
                ax[1].plot(width,bag_init_score,"*",color = coolwarm(width))

    ### Plot impurity thresholds: 
    #impstyle = {"base":"solid","bag":"dotted","rf":"dashed"}
    #for imptype, imp in impurity_threshold.items():
    #    if imp is not None:
    #        ax[1].axvline(x = imp,linestyle = impstyle[imptype],label = "{} imp thresh".format(imptype))
    for i in range(20):
        offset = -1+i*0.1
        line = np.linspace(0,1,100)
        ax[0].plot(line,offset+line,alpha = 0.3,color = "black")
    ax[0].set_xlim([0,1])
    ax[0].set_ylim([0,1])
    ax[0].set_xlabel("Variance")
    ax[0].set_ylabel("Avg. single model Brier score")
    ax[0].set_title("Bias Variance Decomposition")
    ax[1].set_title("Performance vs. Width")
    ax[1].set_xlabel("Width")
    ax[1].set_ylabel("Brier score")
    plt.legend()
    plt.suptitle("MNIST dataset: Single RFF classifier vs. Random Init Ensemble vs. Bag with l2")
    plt.show()

if __name__ == "__main__":
    main()
