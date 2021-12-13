import create_cinic_ensembles
import plot_metrics
import json
import tqdm


indexes = {0:"",1:".1",2:".2",3:".3",4:".4",5:".5"}

if __name__ == "__main__":
    ood_set_f = "cifar10_c_{}_{}" 
    all_ood = {}
    for corruption in ["fog","brightness","gaussian_noise","contrast"]:
        for level in [1,5]:
            ood_set = ood_set_f.format(corruption,level)
            #1. Look for all datasets that have the the given suffix. 
            #2. generate the correct stubs from these datasets. 
            output = create_cinic_ensembles.find_suffix("ood_{}_preds.npy".format(ood_set))

            #3. get corresponding metadata files. 
            metadata = create_cinic_ensembles.get_metadata(output)
            sample = metadata[list(metadata.keys())[0]]["metadata"]

            #4. group similar stubs based on this.  
            classifier_stubs = create_cinic_ensembles.sort_classifier(metadata)
            for ci in classifier_stubs.values():
                assert len(ci) == 5
            #5. generate ensemble prefix. 
            create_cinic_ensembles.create_synth_ensembles(classifier_stubs,ood_set)

            #6. save out a dictionary of grouped results: 
            ood_dict = {}
            for classifier,data in tqdm.tqdm(classifier_stubs.items()):
                for mi,model in enumerate(data):
                    ensembleprefix = "{}_{}_{}".format(sample["module"],classifier,"e4_{}".format(ood_set))
                    name = plot_metrics.commonname_map[classifier]+"{}".format(indexes[mi])
                    ood_dict[name] = model["stub"]
                    ood_dict["Ensemble-4 Synth "+name] = "synth_ensemble_{}_{}_".format(mi,ensembleprefix)
            all_ood[ood_set] = ood_dict    
    with open("cifar10c_all_stubs.json","w") as f:
        json.dump(all_ood,f,indent = 4)        

