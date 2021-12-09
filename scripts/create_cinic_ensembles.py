## use cinic-10 data created by `eval_cinic10.py` and synthesize ensembles from them.  
import os 
import json
import subprocess

here = os.path.dirname(os.path.abspath(__file__))
resultpath = os.path.join(here,"../results") 

def find_suffix(suffix):
    """Finds all files with given suffix, and returns the segment before it. 

    """
    all_results = os.listdir(resultpath)
    target_stubs = list(set([a.split(suffix)[0] for a in all_results if a.endswith(suffix)]))
    return target_stubs

def get_metadata(target_stubs):
    """Loads the metadata files for all stubs. 

    """
    dicts = {}
    for t in target_stubs:
        metadata = os.path.join(resultpath,t+"_meta.json")
        with open(metadata,"r") as f:
            metadict = json.load(f)
        dicts[t] = {"metadata":metadict}
    return dicts    

def sort_classifier(metadata):
    """Loads in a dictionary of metadata and sorts based on classifier 

    :param metadata: commands passed to `train.py` script. :w
    :returns: dictionary with keys giving identity of classifiers, and values lists of dictionaries giving the result stub and corresponding metadata. 
    """
    classifier_dict = {}
    for stub,data in metadata.items():
        classifier_id = data["metadata"]["classifier"] 
        if not classifier_dict.get(classifier_id,False):
            ## initialize as empty list if not exists
            classifier_dict[classifier_id] = []
        classifier_dict[classifier_id].append({"stub":stub,"metadata":data["metadata"]})    
    return classifier_dict    

def create_synth_ensembles(classifier_stubs,ood_set):
    for classifier,stubdata in classifier_stubs.items():
        module = stubdata[0]["metadata"]["module"]
        #command = ["export MKL_THREADING_LAYER=GNU;","python", os.path.join(here,"train.py"),"--classifier",command_data["classifier"],"--softmax",command_data["softmax"],"--checkpoint",command_data["checkpointpath"],"--test_phase","1","--ood_dataset","cinic10","--module","base"]
        command = ["python",os.path.join(here,"create_synth_ensembles.py"),"-e","4","-s","--nameprefix","{}_{}_{}".format(module,classifier,"e4_{}".format(ood_set)),"-os","ood_{}_".format(ood_set)]
        for s in stubdata:
            command.append("-r {}".format(s["stub"])) 
        try:
            subprocess.run(" ".join(command),shell = True,check = True)
        except subprocess.CalledProcessError as e:    
            print(e.output)

if __name__ == "__main__":
    ood_set = "cinic" 
    #1. Look for all datasets that have the suffix ood_cinic_preds.npy
    #2. generate the correct stubs from these datasets. 
    output = find_suffix("ood_{}_preds.npy".format(ood_set))

    #3. get corresponding metadata files. 
    metadata = get_metadata(output)

    #4. group similar stubs based on this.  
    classifier_stubs = sort_classifier(metadata)
    for ci in classifier_stubs.values():
        assert len(ci) == 5
    #5. generate ensemble prefix. 
    create_synth_ensembles(classifier_stubs,ood_set)


