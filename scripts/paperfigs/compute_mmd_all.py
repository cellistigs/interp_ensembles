"""
Works off of outputs of compare_performance: 

"""    
from interpensembles import mmd
import joblib
import os
import numpy as np

alpha = 0.05

def main():
    for ood_score,ood_pre in {"13-08-58":"ood_cinic_","13-11-09":"ood_cinic_","13-13-11":"ood_","13-13-24":"ood_"}.items():
        agg_results = os.path.join(os.path.dirname(os.path.abspath(__file__)),"multirun/2022-01-25/{}".format(ood_score))
        avg_avg = joblib.load(os.path.join(agg_results,"2/avg_avg"))
        ens_avg = joblib.load(os.path.join(agg_results,"0/avg_ens"))
        ens_ens = joblib.load(os.path.join(agg_results,"1/ens_ens"))

        accept_thresh_ub = lambda mi: 4/(np.sqrt(mi))*np.sqrt(np.log(1/alpha))
        accept_thresh_b = lambda mi: np.sqrt(2/mi)*(1+np.sqrt(2*np.log(1/alpha)))


        ## mmd between in and out of distribution: 
        mmd_first = mmd.MMDModule.compute_mmd2_rbf(ens_avg["ind_"].T,ens_avg[ood_pre].T)
        m = np.max([len(ens_avg["ind_"].T),len(ens_avg[ood_pre].T)])
        print("InD vs. OOD: MMD value: {}. Accept threshold: < {}".format(mmd_first,accept_thresh_ub(m)))

        ## mmd between ensemble/single model and single model/single model comparisons:
        mmd_ind_ens_avg = mmd.MMDModule.compute_mmd2_rbf(ens_avg["ind_"].T,avg_avg["ind_"].T)
        m = np.max([len(ens_avg["ind_"].T),len(avg_avg["ind_"].T)])
        print("InD (ensemble/single vs single/single):  MMD value: {}. Accept threshold: < {}".format(mmd_ind_ens_avg,accept_thresh_ub(m)))
        ## mmd 
        mmd_ood_ens_avg = mmd.MMDModule.compute_mmd2_rbf(ens_avg[ood_pre].T,avg_avg[ood_pre].T)
        m = np.max([len(ens_avg[ood_pre].T),len(avg_avg[ood_pre].T)])
        print("OOD (ensemble/single vs single/single):  MMD value: {}. Accept threshold: < {}".format(mmd_ood_ens_avg,accept_thresh_ub(m)))
        ## mmd between ensemble/single model and ensemble/ensemble model comparisons:
        mmd_ind_ens_ens = mmd.MMDModule.compute_mmd2_rbf(ens_avg["ind_"].T,ens_ens["ind_"].T)
        m = np.max([len(ens_avg["ind_"].T),len(ens_ens["ind_"].T)])
        print("InD (ensemble/single vs ensemble/ensemble):  MMD value: {}. Accept threshold: < {}".format(mmd_ind_ens_ens,accept_thresh_ub(m)))
        ## mmd 
        mmd_ood_ens_ens = mmd.MMDModule.compute_mmd2_rbf(ens_avg[ood_pre].T,ens_ens[ood_pre].T)
        m = np.max([len(ens_avg[ood_pre].T),len(ens_ens[ood_pre].T)])
        print("OOD (ensemble/single vs ensemble/ensemble):  MMD value: {}. Accept threshold: < {}".format(mmd_ood_ens_ens,accept_thresh_ub(m)))
        

if __name__ == "__main__":
    main()

