# Compare a decision tree of given depth to a bag and a random forest of the same depth. 
## A sklearn BaggingClassifier averages the probability output of individuals, making it a good choice for our evaluation. 
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import log_loss,brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris,load_wine,load_digits,load_breast_cancer,fetch_openml
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
here = os.path.dirname(__file__)
plt.style.use(os.path.join(here,"../../etc/config/stylesheet.mplstyle"))

def brier_multi(labels,probs): 
    labels_onehot = np.zeros(np.shape(probs))
    for i,li in enumerate(labels):
        labels_onehot[i,int(li)] = 1
    return np.mean(np.sum((probs-labels_onehot)**2,axis = 1))    


def kl_diversity(probs,labels):
    """Given probs of shape (models,samples, classes) and labels of shape (samples), generate a corresponding diversity measure.  

    """
    correct_class = probs[:,np.arange(len(labels)),labels.astype(int)]+1e-10 # models,samples with a bit of padding. 
    nb_models = len(probs[:,0,0])
    norm = np.sum(correct_class,axis = 0)
    diversity = np.sum(np.log(1/nb_models)-np.log(correct_class/norm),axis = 0)*(1/nb_models)
    return diversity

def var_diversity(probs,labels):
    """Given probs of shape (models,samples, classes) and labels of shape (samples), generate a corresponding diversity measure.  

    """
    variances = np.sum(np.var(probs,axis = 0),axis = -1)
    return np.mean(variances)

def compare_savefig(xtest,ytest,rf_clf,depth):
    """Compute the nll/ce diversity decompositions for individual models, bags and random forest classifiers. 

    """
    ## fitting bags
    ## This is the average of probabilities output by each individual tree. 
    bag_score = brier_multi(ytest,rf_clf.predict_proba(xtest))
    bag_individual_probs = [e.predict_proba(xtest) for e in rf_clf.estimators_]
    np.save("probs_depth_{}".format(depth),bag_individual_probs)
    np.save("ytest_depth_{}".format(depth),ytest)
    grouped_probs = [np.mean(np.array(bag_individual_probs[i*100:(i+1)*100]),axis = 0) for i in range(5)]
    bag_individual_nll = [-np.log(e[np.arange(len(ytest)),ytest.astype(int)]) for e in grouped_probs]
    bag_diversity = kl_diversity(np.stack(bag_individual_probs,axis =0),ytest)

    ## fitting random forests 
    #rf_score = brier_multi(ytest,rf_clf.predict_proba(xtest))
    #rf_individual_probs = [e.predict_proba(xtest) for e in rf_clf.estimators_]
    #rf_individual_nll = [-np.log(e.predict_proba(xtest)[np.arange(len(ytest)),ytest]) for e in rf_clf.estimators_]
    #rf_diversity = var_diversity(np.stack(rf_individual_probs,axis =0),ytest)

    return bag_individual_nll,bag_diversity#,rf_individual_nll,rf_diversity

def compare_at_depth(xtest,ytest,base_clf,bag_clf,rf_clf,depth):
    """Compute the bias variance decompositions and scores for individual models, bags, and random forest classifiers. 

    """

    ## fitting trees
    base_score = brier_multi(ytest,base_clf.predict_proba(xtest))

    ## fitting bags
    ## This is the average of probabilities output by each individual tree. 
    bag_score = brier_multi(ytest,bag_clf.predict_proba(xtest))
    bag_individual_probs = [e.predict_proba(xtest) for e in bag_clf.estimators_]
    bag_individual_score = [brier_multi(ytest,e.predict_proba(xtest)) for e in bag_clf.estimators_]
    bag_diversity = var_diversity(np.stack(bag_individual_probs,axis =0),ytest)

    ## fitting random forests 
    rf_score = brier_multi(ytest,rf_clf.predict_proba(xtest))
    rf_individual_probs = [e.predict_proba(xtest) for e in rf_clf.estimators_]
    rf_individual_score = [brier_multi(ytest,e.predict_proba(xtest)) for e in rf_clf.estimators_]
    rf_diversity = var_diversity(np.stack(rf_individual_probs,axis =0),ytest)

    return base_score,bag_score,bag_individual_score,bag_diversity,rf_score,rf_individual_score,rf_diversity

def is_leaf(clf):
    """Given a tree, get leaf nodes. from https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py

    """
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True
    return is_leaves        

def compute_impurity(clf):
    """Given a tree, computes the gini impurity across the leaf nodes. Uses code from https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py

    """
    # First, get the set of leaf nodes. 
    leaves = is_leaf(clf)

    # Then, calculate the impurity and number of samples across all leaf nodes 
    impurities = clf.tree_.impurity[leaves]
    nodecount = clf.tree_.n_node_samples[leaves]
    node_props = nodecount/np.sum(nodecount)

    # return the weighted average impurity. 

    total = np.sum(impurities*node_props) 
    return total

def main():
    ## get data and create train/test splits
    # Load data from https://www.openml.org/d/554
    train_samples = 5000
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

    random_state = check_random_state(0)
    permutation = random_state.permutation(X.shape[0])
    X = X[permutation]
    y = y[permutation]
    X = X.reshape((X.shape[0], -1))

    xtrain, xtest, ytrain, ytest = train_test_split(
        X, y, train_size=train_samples, test_size=10000
    )

    scaler = StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)
    
    ## ensemble params
    ens_size = 600 
    depth_maxrange = 13 
    repeat_iterates = 1
    interp_thresh = 13 

    #fig, ax = plt.subplots(1,2,figsize = (10,3))
    savefig,saveax = plt.subplots()
    coolwarm = cm.get_cmap("coolwarm",interp_thresh)
    impurity_threshold = {"base":None,"bag":None,"rf":None}


    for depth in np.arange(1,depth_maxrange):
        print(depth)
        for it in range(repeat_iterates):

            base_clf = DecisionTreeClassifier(max_depth = depth).fit(xtrain,ytrain)
            #bag_clf = BaggingClassifier(DecisionTreeClassifier(max_depth = depth),n_estimators = ens_size).fit(xtrain,ytrain)
            rf_clf = BaggingClassifier(DecisionTreeClassifier(max_depth = depth,max_features = 0.7),n_estimators = ens_size).fit(xtrain,ytrain)
            #rf_clf = RandomForestClassifier(min_samples_split=10,max_depth = depth,bootstrap=False).fit(xtrain,ytrain)
            #print(base_clf.tree_.max_depth)

            ## Calculate gini impurities. 
            #base_imp = compute_impurity(base_clf)
            #bag_imp = np.mean([compute_impurity(bc) for bc in bag_clf.estimators_])
            #rf_imp = np.mean([compute_impurity(rc) for rc in rf_clf.estimators_])
            #imps = [base_imp,bag_imp,rf_imp]
            #for imptype,imp in {"base":base_imp,"bag":bag_imp,"rf":rf_imp}.items():
            #    if imp == 0 and impurity_threshold[imptype] is None:
            #        impurity_threshold[imptype] = depth


            base_score = brier_multi(ytest,base_clf.predict_proba(xtest))
            base_individual_probs = base_clf.predict_proba(xtest)
            np.save("base_probs_depth_{}".format(depth),base_individual_probs)
            np.save("base_ytest_depth_{}".format(depth),ytest)

            #base,bag,bag_indiv,bag_div,rf,rf_indiv,rf_div = compare_at_depth(xtest,ytest,base_clf,bag_clf,rf_clf,depth)
            bag_indiv,bag_div = compare_savefig(xtest,ytest,rf_clf,depth)


            #if depth == 1 and it == 1:
            #    ax[0].plot(0,base,"x",color = coolwarm(depth),label = "single tree")    
            #    ax[0].plot(np.mean(bag_div),np.mean(bag_indiv),"o",color = coolwarm(depth),label = "bag")
            #    ax[0].plot(np.mean(rf_div),np.mean(rf_indiv),"+",color = coolwarm(depth),label = "random forest")
            #    ax[1].plot(depth,base,"x",color = coolwarm(depth),label = "single tree")
            #    ax[1].plot(depth,bag,"o",color = coolwarm(depth), label = "bag")
            #    ax[1].plot(depth,rf,"+",color = coolwarm(depth), label = "random forest")
            #else:     
            #    ax[0].plot(0,base,"x",color = coolwarm(depth))    
            #    ax[0].plot(np.mean(bag_div),np.mean(bag_indiv),"o",color = coolwarm(depth))
            #    ax[0].plot(np.mean(rf_div),np.mean(rf_indiv),"+",color = coolwarm(depth))
            #    ax[1].plot(depth,base,"x",color = coolwarm(depth))
            #    ax[1].plot(depth,bag,"o",color = coolwarm(depth))
            #    ax[1].plot(depth,rf,"+",color = coolwarm(depth))
            
            saveax.plot(np.mean(bag_div),np.mean(bag_indiv),"o",color = coolwarm(depth))

    ## Plot impurity thresholds: 
    #impstyle = {"base":"solid","bag":"dotted","rf":"dashed"}
    #for imptype, imp in impurity_threshold.items():
    #    if imp is not None:
    #        ax[1].axvline(x = imp,linestyle = impstyle[imptype],label = "{} imp thresh".format(imptype))
    #for i in range(20):
    #    offset = -1+i*0.1
    #    line = np.linspace(0,1,100)
    #    ax[0].plot(line,offset+line,alpha = 0.3,color = "black")
    #ax[0].set_xlim([0,1])
    #ax[0].set_ylim([0,1])
    #ax[0].set_xlabel("Variance")
    #ax[0].set_ylabel("Avg. single model")
    #ax[0].set_title("Bias Variance Decomposition")
    #ax[1].set_xlabel("Max depth")
    #ax[1].set_ylabel("Performance")
    #plt.legend()
    #plt.suptitle("MNIST dataset: Single Trees vs. Bags vs. Random Forest")
    #plt.show()
    plt.show()

if __name__ == "__main__":
    main()

