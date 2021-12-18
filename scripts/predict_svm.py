## Generate accuracy and probabilistic metrics for linear SVM model. 
from argparse import ArgumentParser
import os
import numpy as np
from cifar10_ood.data import CIFAR10,CIFAR10_1,CINIC10,CIFAR10_C
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

here = os.path.dirname(os.path.abspath(__file__))
results = os.path.join(here,"../results/")

def get_datasets(args):
    """Get the training and test datasets. 

    :param args: args, with field args.data_dir = dataset directory. 
    :returns: a bunch of torchvision.dataset objects in a dictionary with keys "train", "test", "cifar10.1", "cinic10", "cifar10c{fog,brightness,contrast,gaussian_noise}_{1,5}". The dictionary looks like:  
    {"ood_dataset":{"data":numpy array,"labels":list}}
    """
    datasets = {}
    data_dir = args.data_dir
    data_objs = {"train":CIFAR10(os.path.join(data_dir,"cinic-10"),train = True),
            "test":CIFAR10(os.path.join(data_dir,"cinic-10"),train = False),
            "cifar10.1":CIFAR10_1(os.path.join(data_dir,"cinic-10"),version="v4",transform =None),
            "cinic10":CINIC10(os.path.join(data_dir,"cinic-10"),split="test",preload = True),
            "cifar10cfog_1":CIFAR10_C(os.path.join(data_dir,"cifar10-c"),"fog",1),
            "cifar50cfog_5":CIFAR10_C(os.path.join(data_dir,"cifar10-c"),"fog",5),
            "cifar10cbrightness_1":CIFAR10_C(os.path.join(data_dir,"cifar10-c"),"brightness",1),
            "cifar50cbrightness_5":CIFAR10_C(os.path.join(data_dir,"cifar10-c"),"brightness",5),
            "cifar10ccontrast_1":CIFAR10_C(os.path.join(data_dir,"cifar10-c"),"contrast",1),
            "cifar50ccontrast_5":CIFAR10_C(os.path.join(data_dir,"cifar10-c"),"contrast",5),
            "cifar10cgaussian_noise_1":CIFAR10_C(os.path.join(data_dir,"cifar10-c"),"gaussian_noise",1),
            "cifar50cgaussian_noise_5":CIFAR10_C(os.path.join(data_dir,"cifar10-c"),"gaussian_noise",5),
            }
    for dname,dobj in data_objs.items():
        datasets[dname] = {"data":dobj.data,"labels":dobj.targets}
    return datasets    

def create_svm_pipeline():
    """Creates a function that first mean subtracts and normalizes the variance of constituent data. 

    :returns: return pipeline that can be fit and predict. 
    """
    clf = make_pipeline(StandardScaler(),
            LinearSVC(random_state=0, tol=1e-5,verbose = 1))
    return clf

def fit_svm(clf,traindata,trainlabels):
    """Fits svm model to training data and training labels

    :returns: return pipeline after it has been fit on data. 
    """
    len_data = len(traindata)
    clf.fit(traindata.reshape(len_data,-1),trainlabels)
    return clf

def output_svm(clf,testdata):
    """Output the targets and the logits given test data. 

    :returns: returns a tuple (predictions,logits), where predictions is of shape (5000,) and logits is of shape (5000,10)
    """
    len_data = len(testdata)
    reshaped = testdata.reshape(len_data,-1)
    predicts = clf.predict(reshaped)
    logits = clf.decision_function(reshaped)
    return (predicts,logits)

def main(args):

    datasets= get_datasets(args)
    svm_model = create_svm_pipeline()
    traindata,trainlabels = datasets["train"]["data"],datasets["train"]["labels"]
    fit_svm(svm_model,traindata,trainlabels)
    dataoutputs = {}
    for dataname,dataset in datasets.items():
        if dataname == "train":
            continue
            
        predicts,logits = output_svm(svm_model,dataset["data"])
        dataoutputs[dataname] = (predicts,logits,dataset["labels"])
        np.save(os.path.join(results,"svm_{}_labels_ind".format(dataname)),dataoutputs["test"][-1])
        np.save(os.path.join(results,"svm_{}_preds_ind".format(dataname)),dataoutputs["test"][1])
        np.save(os.path.join(results,"svm_{}_labels_ood".format(dataname)),dataset["labels"])
        np.save(os.path.join(results,"svm_{}_preds_ood".format(dataname)),logits)


if __name__ == "__main__":
    parser = ArgumentParser()

    # PROGRAM level args
    parser.add_argument("--data_dir", type=str, default="/home/ubuntu/data/cifar10")
    args = parser.parse_args()
    main(args)
