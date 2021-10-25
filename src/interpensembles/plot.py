## plotting functions to plot results against the relative robustness curve. 
import numpy as np
import datetime
import matplotlib.pyplot as plt


def logit_transform(acc,dataset):
    """given an array of in distribution accuracies, returns the out of distribution accuracy predicted by the model from Andreassen et al., returns the OOD accuracies predicted by the logit model. 

    :param acc: one dimensional numpy array giving the accuracies at which we want to evaluate. Given as percentages. 
    :param dataset: specify which dataset we are considering: cifar 10 or imagenet. 
    """
    if dataset not in ["cifar10","imagenet"]: 
        raise ValueError("{} is not a valid option for computing this function".format(dataset))
    elif dataset == "cifar10":
        A = 0.8318
        B = -0.4736
    elif dataset == "imagenet":    
        A = 0.9225
        B = -0.4896

    prop = acc/100 ## assume we need to normalize accuracies. 
    ## linear fit in logit space
    logit_transformed = A*np.log(prop/(1-prop))+B
    ## transform back:
    return np.exp(logit_transformed)/(np.exp(logit_transformed)+1)*100

def plot_model_performance(modeldata,dataset):
    """given a model's in and out of distribution accuracy, plot it against the curve. 

    :param modeldata: a dictionary with fields "in_dist_acc" and "out_dist_acc"
    """
    accs = np.arange(1,100)
    plt.plot(accs,logit_transform(accs,dataset),label = "ER = 0")
    plt.plot(accs,accs,label = "Perfect Robustness")
    plt.plot(modeldata["in_dist_acc"],modeldata["out_dist_acc"],"x",label = "model performance")
    plt.title("Model Performance: {}".format(dataset))
    plt.legend()
    plt.savefig("example_model_{}.png".format(datetime.datetime.now().strftime("%m-%d-%y_%H:%M.%S")))

    


    
