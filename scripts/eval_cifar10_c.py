# Evaluate pretrained models on cinic10 dataset. 
import os 
import json
import subprocess
from plot_metrics import all_dataindices

here = os.path.dirname(os.path.abspath(__file__))
resultpath = os.path.join(here,"../results") 

def get_args(orig_stub):
    metadata = os.path.join(resultpath,orig_stub+"_meta.json")
    with open(metadata,"r") as f:
        metadict = json.load(f)
        pretrained = metadict["pretrained"]
        if bool(pretrained):
            weight_path = metadict["pretrained_path"]
        else:    
            weight_path = metadict["save_path"]
            checkpoint = os.listdir(weight_path)
        classifier = metadict["classifier"]
        softmax = metadict.get("softmax","1")
    if bool(pretrained):
        command_data = {"weight_path":os.path.join(here,weight_path),"pretrained":pretrained,"classifier":classifier,"softmax":str(softmax)}   
    else:     
        command_data = {"weight_path":os.path.join(weight_path,checkpoint[0]),"pretrained":pretrained,"classifier":classifier,"softmax":str(softmax)}   
    return command_data     

def run_eval(command_data,level,corruption):
    """Assume not checkpoint 

    """

    if bool(command_data["pretrained"]):
        command = ["export MKL_THREADING_LAYER=GNU;","python", os.path.join(here,"train.py"),"--classifier",command_data["classifier"],"--softmax",command_data["softmax"],"--pretrained",str(command_data["pretrained"]),"--pretrained-path",command_data["weight_path"],"--test_phase","1","--ood_dataset","cifar10_c","--level",level,"--corruption",corruption,"--module","base","--data_dir","/home/ubuntu/cifar10_ood/data","--deterministic","1"]
    else:     
        command = ["export MKL_THREADING_LAYER=GNU;","python", os.path.join(here,"train.py"),"--classifier",command_data["classifier"],"--softmax",command_data["softmax"],"--checkpoint",str(command_data["weight_path"]),"--test_phase","1","--ood_dataset","cifar10_c","--level",level,"--corruption",corruption,"--module","base","--data_dir","/home/ubuntu/cifar10_ood/data","--deterministic","1"]
    try:
        subprocess.run(" ".join(command),shell = True,check = True)
    except subprocess.CalledProcessError as e:    
        print(e.output)
        

if __name__ == "__main__":
    ## dictionary: {"modelcode":{"checkpoint":"checkpointpath","orig_stub":,"cinic10_path":}
    # 1. get the metadata files containing model checkpoints: -> checkpointpath
    # 2. run train.py for each of these model checkpoints. -> cinic10_path
    dataindex = all_dataindices["cifar10.1"]
    for d,dstub in dataindex.items():
        try:
            if not d.startswith("Ensemble"):
                if not d.startswith("Conv WideResNet-28"): 
                    continue

                command_data = get_args(dstub)
                for level in [1,5]:
                    for corruption in ["fog", "brightness", "gaussian_noise", "contrast"]:
                            run_eval(command_data,str(level),corruption)
        except Exception as e:        
            raise
            print("Encountered issue for {}: {}".format(dstub,e))
            
