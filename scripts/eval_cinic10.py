# Evaluate pretrained models on cinic10 dataset. 
import os 
import json
import subprocess
from plot_metrics import dataindex

here = os.path.dirname(os.path.abspath(__file__))
resultpath = os.path.join(here,"../results") 

def get_ckptpath(orig_stub):
    metadata = os.path.join(resultpath,orig_stub+"_meta.json")
    with open(metadata,"r") as f:
        metadict = json.load(f)
        checkpointpath = metadict["save_path"]
        checkpoint = os.listdir(checkpointpath)
        classifier = metadict["classifier"]
        softmax = metadict.get("softmax","1")
    command_data = {"checkpointpath":os.path.join(checkpointpath,checkpoint[0]),"classifier":classifier,"softmax":str(softmax)}   
    return command_data     

def run_eval(command_data):
    """Assume not checkpoint 

    """

    command = ["export MKL_THREADING_LAYER=GNU;","python", os.path.join(here,"train.py"),"--classifier",command_data["classifier"],"--softmax",command_data["softmax"],"--checkpoint",command_data["checkpointpath"],"--test_phase","1","--ood_dataset","cinic10","--module","base"]
    try:
        subprocess.run(" ".join(command),shell = True,check = True)
    except subprocess.CalledProcessError as e:    
        print(e.output)
        

if __name__ == "__main__":
    ## dictionary: {"modelcode":{"checkpoint":"checkpointpath","orig_stub":,"cinic10_path":}
    # 1. get the metadata files containing model checkpoints: -> checkpointpath
    # 2. run train.py for each of these model checkpoints. -> cinic10_path
    for d,dstub in dataindex.items():
        try:
            if not d.startswith("Ensemble"):
                command_data = get_ckptpath(dstub)
                run_eval(command_data)
        except Exception as e:        
            print("Encountered issue for {}: {}".format(dstub,e))
            
