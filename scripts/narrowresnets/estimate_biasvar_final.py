# Generate polished figures.  
from interpensembles import predictions,metrics
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import yaml
import os

this_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plt.style.use(os.path.join(this_dir,"../etc/config/geoff_stylesheet.mplstyle"))


ad = metrics.AccuracyData()
all_arch_names = [
                "narrowresnet10_16",
                "narrowresnet12_16",
                "narrowresnet14_16",
                "narrowresnet16_16",
                "narrowresnet18_16",
                "narrowresnet10_8",
                "narrowresnet12_8",
                "narrowresnet14_8",
                "narrowresnet16_8",
                "narrowresnet18_8",
                "narrowresnet10_4",
                "narrowresnet12_4",
                "narrowresnet14_4",
                "narrowresnet16_4",
                "narrowresnet18_4",
                "narrowresnet10_2",
                "narrowresnet12_2",
                "narrowresnet14_2",
                "narrowresnet16_2",
                "narrowresnet18_2",
                "resnet10",
                "resnet12",
                "resnet14",
                "resnet16",
                "resnet18",
                "wideresnet18",
                "wideresnet18_4",
                "resnet26",
                "resnet34",
                "resnet42",
                "resnet52"
                ]

fixed_width_arch_names = [
                "resnet10",
                "resnet12",
                "resnet14",
                "resnet16",
                "resnet18",
                "resnet26",
                "resnet34",
                "resnet42",
                "resnet52"
                ]

fixed_width_arch_cifar_names = [
                "resnet8_cifar",
                "resnet14_cifar",
                "resnet20_cifar",
                "resnet26_cifar",
                "resnet32_cifar",
                "resnet38_cifar",
                "resnet44_cifar",
                ]

fixed_depth_arch_names = [
                #"narrowresnet18_32",
                "narrowresnet18_16",
                "narrowresnet18_8",
                "narrowresnet18_4",
                "narrowresnet18_2",
                "resnet18",
                "wideresnet18",
                "wideresnet18_4",
                ]

variable_width_depth_arch_names = [
                "narrowresnet18_32",
                "narrowresnet10_16",
                "narrowresnet12_8",
                "narrowresnet14_4",
                "narrowresnet16_2",
                "resnet18",
                ]


arch_name_options = {
        "all":all_arch_names,
        "fixed width":fixed_width_arch_names,
        "fixed depth":fixed_depth_arch_names,
        "variable width depth":variable_width_depth_arch_names,
        "fixed width cifar": fixed_width_arch_cifar_names
        }

dir_paths_no_wd = [
        os.path.join(this_dir,"multirun","2022-10-15","20-07-13"), 
        os.path.join(this_dir,"multirun","2022-10-17","17-37-10"), 
        os.path.join(this_dir,"multirun","2022-10-17","22-21-17"), 
        os.path.join(this_dir,"multirun","2022-10-18","03-09-46"), ## "initial narrow architecturees"
        os.path.join(this_dir,"multirun","2022-10-18","20-52-12"), 
        os.path.join(this_dir,"multirun","2022-10-19","00-22-18"), 
        os.path.join(this_dir,"multirun","2022-10-19","03-53-18"), 
        os.path.join(this_dir,"multirun","2022-10-19","07-24-25"), ## "wideresnets, resnet 26 and 34"
        os.path.join(this_dir,"multirun","2022-10-19","15-42-14"), 
        os.path.join(this_dir,"multirun","2022-10-19","16-31-10"), 
        os.path.join(this_dir,"multirun","2022-10-19","17-20-14"), 
        os.path.join(this_dir,"multirun","2022-10-19","18-09-17"), ## resnet 42 and 52
        #os.path.join(this_dir,"multirun","2022-10-19","20-16-25"), ## 300 epochs for all width 18 architectures.  
        ]
dir_paths_wd = [        
        os.path.join(this_dir,"multirun","2022-10-22","01-59-15"), 
        os.path.join(this_dir,"multirun","2022-10-22","05-34-09"), ## remainder: 1 wrn18_4, 2 each of rn26 and 34
        os.path.join(this_dir,"multirun","2022-10-20","17-32-20"),
        os.path.join(this_dir,"multirun","2022-10-20","23-08-22"), ## missing wrn18_4,, rn26, rn34
        os.path.join(this_dir,"multirun","2022-10-21","14-29-53"), ## 
        os.path.join(this_dir,"multirun","2022-10-21","20-12-19"), ## missing rn26,rn34 
        os.path.join(this_dir,"multirun","2022-10-22","14-09-26"), ## 
        os.path.join(this_dir,"multirun","2022-10-22","14-59-33"), ## 
        os.path.join(this_dir,"multirun","2022-10-22","15-49-54"), ## 
        os.path.join(this_dir,"multirun","2022-10-22","16-40-02"), ## four instances each of rn42,52 
        ]
dir_paths_no_wd_fixed_width = [
        os.path.join(this_dir,"multirun","2022-10-25","19-50-48"), 
        os.path.join(this_dir,"multirun","2022-10-26","03-44-36"), 
        os.path.join(this_dir,"multirun","2022-10-26","11-41-30"), 
        os.path.join(this_dir,"multirun","2022-10-26","19-44-28"), ## "initial narrow architecturees"
        ]
dir_paths_wd_fixed_width = [
        os.path.join(this_dir,"multirun","2022-10-31","22-35-41"), 
        os.path.join(this_dir,"multirun","2022-11-01","06-20-28"), 
        os.path.join(this_dir,"multirun","2022-11-01","14-06-52"), 
        os.path.join(this_dir,"multirun","2022-11-01","21-58-45"), ## "initial narrow architecturees"
        ]

dir_paths_cifar = [
        os.path.join(this_dir,"multirun","2022-11-03","19-49-43"), 
        os.path.join(this_dir,"multirun","2022-11-03","23-51-10"), 
        os.path.join(this_dir,"multirun","2022-11-04","04-35-32"), 
        os.path.join(this_dir,"multirun","2022-11-04","08-36-28"), 
        ]

def compare_at_depth(ensemble,depth): 

    #return ensemble.get_brier(),ensemble.get_bias_bs(),ensemble.get_variance()
    return ensemble.get_nll(),ensemble.get_avg_nll(),ensemble.get_nll_div()

def main():
    datatype = "ind"
    both_paths = [dir_paths_cifar,dir_paths_wd]
    for ci,condition in enumerate(["fixed width cifar","fixed depth"]):
        fig,ax = plt.subplots(1,1,figsize =(5.5,5.5))
        dir_paths = both_paths[ci]#dir_paths_cifar#dir_paths_wd_fixed_width#dir_paths_no_wd
        arch_names = arch_name_options[condition]

        ensemble_dict = {name:predictions.EnsembleModel(name,datatype) for name in arch_names}

        for di,dir_path in enumerate(dir_paths):
            print(dir_path)
            run_paths = get_run_paths(dir_path)
            run_parsed = [get_parsed(run_path,datatype) for run_path in run_paths]
            models = {}
            for parsed in run_parsed:
                try:
                    ensemble_dict[parsed["prefix"]].register(parsed["filename"],di,labelpath = parsed["labelname"],logits =
                            False)
                except KeyError:    
                    print("not registered, continueing")
                    continue

                #model = predictions.Model(parsed["prefix"],parsed["datatype"])
                #model.register(parsed["filename"],labelpath = parsed["labelname"])
                #models[parsed["prefix"]] = model 


        line = np.linspace(0,1,100)
        coolwarm = cm.get_cmap("coolwarm",len(arch_names))
        for index,(ens_name,ensemble) in enumerate(ensemble_dict.items()):
            score,bias,variance = compare_at_depth(ensemble,ens_name)
            acc = ensemble.get_accuracy() 
            mean_acc = np.mean([ad.accuracy(modelvalues["preds"],modelvalues["labels"]) for modelvalues in ensemble.models.values()])
            print(ens_name,bias,variance)
            #ax[ci].plot(variance,bias,"o",color = coolwarm(index))
            ax.plot(variance,bias,"o",color = "C0", markersize =  index/len(ensemble_dict)*10+5)

        if ci == 0:
            ax.set_title("CIFAR10\n Deep Ensembles (Depth)")
            #ax.tick_params(axis='x', labelrotation = 45)
            ax.plot(line,line+0.1934,"--",color= "black")
            ax.set_xlabel("CE Jensen Gap")
            ax.set_ylabel("Avg. single model loss CE")
            ax.set_xlim([0.15,0.19])
            ax.set_ylim([0.36,0.39])
            [ax.plot(line,line+i*0.01,color="black",alpha =0.2) for i in range(100)]
        elif ci == 1:    
            ax.set_title("CIFAR10\n Deep Ensembles (Width)")
            ax.plot(line,line+0.174,"--",color="black")
            ax.set_xlabel("CE Jensen Gap")
            ax.set_ylabel("Avg. single model loss CE")
            ax.set_xlim([0.03,0.09])
            ax.set_ylim([0.19,0.53])
            [ax.plot(line,line+i*0.1,color="black",alpha =0.2) for i in range(100)]
            #ax[1].set_xlim([0.03,1])
        #ax[1].set_ylim([0.0,1])
        #plt.legend()    
        plt.tight_layout()
        plt.savefig("{ci}_cifarmodel_arch_search_ensembles.pdf".format(ci=ci))    

def get_run_paths(dir_path):
    """get paths to individual runs.

    """
    runs = [os.path.join(dir_path,path) for path in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path,path))]
    return runs
    

def get_parsed(run_path,data = "train"):
    """ given a run path, get the following info from it: 

    :param prefix: model prefix that indicates type 
    :param datatype: ind vs. ood
    :param filename: actual path to the logits 
    :param labelname: name of labels for logits
    """
    prefix = read_hydra_config(os.path.join(run_path,".hydra","config.yaml"))["classifier"]
    datatype = data
    filename = "{}_preds.npy".format(data)
    labelname = "{}_labels.npy".format(data)
    return {
            "prefix":prefix,
            "datatype":data,
            "filename":os.path.join(run_path,filename),
            "labelname":os.path.join(run_path,labelname)
            }

def read_hydra_config(run_path):
    with open(run_path) as f:
        config=yaml.safe_load(f)
    return config    

if __name__ == "__main__":
    main()
