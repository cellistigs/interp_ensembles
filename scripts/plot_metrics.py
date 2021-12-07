## Given the models we have generated outputs for, plot interesting combinations of accuracy, calibration and nll. 
import os
import numpy as np
import joblib
import json
from scipy.special import softmax
import matplotlib.pyplot as plt 
from interpensembles.metrics import AccuracyData,NLLData,CalibrationData,VarianceData,BrierScoreData
from argparse import ArgumentParser

resultsfolder = os.path.join(os.path.abspath(os.path.dirname(__file__)),"../results")
imagesfolder = os.path.join(os.path.abspath(os.path.dirname(__file__)),"../images")

markers = {"ResNet":"rx",
        "Ensemble-2 Synth ResNet":"r*",
        "Ensemble-4 ResNet":"ro",
        "Ensemble-4 Synth ResNet":"ro",
        "WideResNet 18-2":"bx",
        "Ensemble-2 Synth WideResNet 18-2":"b*",
        "Ensemble-4 Synth WideResNet 18-2":"bo",
        "WideResNet 18-4":"gx",
        "Ensemble-2 Synth WideResNet 18-4":"g*",
        "Ensemble-4 Synth WideResNet 18-4":"go",
        "Ensemble-5 Synth ResNet":"r+",
        "VGG-11":"yx",
        "Ensemble-4 Synth VGG-11":"yo",
        "VGG-19":"mx",
        "Ensemble-4 Synth VGG-19":"mo",
        "GoogleNet":"cx",
        "Ensemble-4 Synth GoogleNet":"co",
        "Inception-v3":"kx",
        "Ensemble-4 Synth Inception-v3":"ko",
        "DenseNet-121":"C0x",
        "Ensemble-4 Synth DenseNet-121":"C0o",
        "DenseNet-169":"C2x",
        "Ensemble-4 Synth DenseNet-169":"C2o",
        "WideResNet-28-10":"C1x",
        "Ensemble-4 Synth WideResNet-28-10":"C1o"
        }

variancecalc = {}
for modelprefix in markers:
    variancecalc[modelprefix] = {"ind":VarianceData(modelprefix,"ind"),"ood":VarianceData(modelprefix,"ood"),"ind_cal":[],"ood_cal":[],"ind_ens_cal":[],"ood_ens_cal":[]}


### Now we define the common names for the data, and their prefixes: 
all_dataindices = {"cifar10.1":{
            "WideResNet-28-10":"cifar10_wrn28_s1_",
            "WideResNet-28-10.1":"cifar10_wrn28_s2_",
            "WideResNet-28-10.2":"cifar10_wrn28_s3_",
            "WideResNet-28-10.3":"cifar10_wrn28_s4_",
            "WideResNet-28-10.4":"cifar10_wrn28_s5_",
            "Ensemble-4 Synth WideResNet-28-10":"synth_ensemble_0_wideresnet_28_10_11_17_",
            "Ensemble-4 Synth WideResNet-28-10.1":"synth_ensemble_1_wideresnet_28_10_11_17_",
            "Ensemble-4 Synth WideResNet-28-10.2":"synth_ensemble_2_wideresnet_28_10_11_17_",
            "Ensemble-4 Synth WideResNet-28-10.3":"synth_ensemble_3_wideresnet_28_10_11_17_",
            "Ensemble-4 Synth WideResNet-28-10.4":"synth_ensemble_4_wideresnet_28_10_11_17_",
            "DenseNet-121":"robust_results11-17-21_00:55.03_base_densenet121",
            "DenseNet-121.1":"robust_results11-17-21_01:36.53_base_densenet121",
            "DenseNet-121.2":"robust_results11-17-21_02:19.01_base_densenet121",
            "DenseNet-121.3":"robust_results11-17-21_03:01.03_base_densenet121",
            "DenseNet-121.4":"robust_results11-17-21_03:43.39_base_densenet121",
            "Ensemble-4 Synth DenseNet-121":"synth_ensemble_0_densenet121_11_17_",
            "Ensemble-4 Synth DenseNet-121.1":"synth_ensemble_1_densenet121_11_17_",
            "Ensemble-4 Synth DenseNet-121.2":"synth_ensemble_2_densenet121_11_17_",
            "Ensemble-4 Synth DenseNet-121.3":"synth_ensemble_3_densenet121_11_17_",
            "Ensemble-4 Synth DenseNet-121.4":"synth_ensemble_4_densenet121_11_17_",
            "DenseNet-169":"robust_results11-17-21_20:28.21_base_densenet169",
            "DenseNet-169.1":"robust_results11-17-21_21:23.18_base_densenet169",
            "DenseNet-169.2":"robust_results11-17-21_22:20.10_base_densenet169",
            "DenseNet-169.3":"robust_results11-17-21_23:17.03_base_densenet169",
            "DenseNet-169.4":"robust_results11-18-21_00:13.12_base_densenet169",
            "Ensemble-4 Synth DenseNet-169":"synth_ensemble_0_densenet169_11_17_",
            "Ensemble-4 Synth DenseNet-169.1":"synth_ensemble_1_densenet169_11_17_",
            "Ensemble-4 Synth DenseNet-169.2":"synth_ensemble_2_densenet169_11_17_",
            "Ensemble-4 Synth DenseNet-169.3":"synth_ensemble_3_densenet169_11_17_",
            "Ensemble-4 Synth DenseNet-169.4":"synth_ensemble_4_densenet169_11_17_",
            "Inception-v3":"robust_results11-16-21_18:58.44_base_inception_v3",
            "Inception-v3.1":"robust_results11-16-21_22:29.39_base_inception_v3",
            "Inception-v3.2":"robust_results11-17-21_02:00.37_base_inception_v3",
            "Inception-v3.3":"robust_results11-17-21_05:31.50_base_inception_v3",
            "Inception-v3.4":"robust_results11-17-21_09:02.26_base_inception_v3",
            "Ensemble-4 Synth Inception-v3":"synth_ensemble_0_inception_11_17_",
            "Ensemble-4 Synth Inception-v3.1":"synth_ensemble_1_inception_11_17_",
            "Ensemble-4 Synth Inception-v3.2":"synth_ensemble_2_inception_11_17_",
            "Ensemble-4 Synth Inception-v3.3":"synth_ensemble_3_inception_11_17_",
            "Ensemble-4 Synth Inception-v3.4":"synth_ensemble_4_inception_11_17_",
            "GoogleNet":"robust_results11-17-21_01:19.42_base_googlenet",
            "GoogleNet.1":"robust_results11-17-21_02:53.46_base_googlenet",
            "GoogleNet.2":"robust_results11-17-21_04:27.55_base_googlenet",
            "GoogleNet.3":"robust_results11-17-21_06:02.07_base_googlenet",
            "GoogleNet.4":"robust_results11-17-21_07:36.23_base_googlenet",
            "Ensemble-4 Synth GoogleNet":"synth_ensemble_0_googlenet_11_17_",
            "Ensemble-4 Synth GoogleNet.1":"synth_ensemble_1_googlenet_11_17_",
            "Ensemble-4 Synth GoogleNet.2":"synth_ensemble_2_googlenet_11_17_",
            "Ensemble-4 Synth GoogleNet.3":"synth_ensemble_3_googlenet_11_17_",
            "Ensemble-4 Synth GoogleNet.4":"synth_ensemble_4_googlenet_11_17_",
            "VGG-11":"robust_results11-15-21_20:42.38_base_vgg11_bn",
            "VGG-11.1":"robust_results11-15-21_20:55.18_base_vgg11_bn",
            "VGG-11.2":"robust_results11-15-21_21:08.03_base_vgg11_bn",
            "VGG-11.3":"robust_results11-15-21_21:20.48_base_vgg11_bn",
            "VGG-11.4":"robust_results11-15-21_20:55.18_base_vgg11_bn",
            "Ensemble-4 Synth VGG-11":"synth_ensemble_0_vgg11_bn_11_16_",
            "Ensemble-4 Synth VGG-11.1":"synth_ensemble_1_vgg11_bn_11_16_",
            "Ensemble-4 Synth VGG-11.2":"synth_ensemble_2_vgg11_bn_11_16_",
            "Ensemble-4 Synth VGG-11.3":"synth_ensemble_3_vgg11_bn_11_16_",
            "Ensemble-4 Synth VGG-11.4":"synth_ensemble_4_vgg11_bn_11_16_",
            "VGG-19":"robust_results11-15-21_22:29.03_base_vgg19_bn",
            "VGG-19.1":"robust_results11-15-21_22:49.10_base_vgg19_bn",
            "VGG-19.2":"robust_results11-15-21_23:09.19_base_vgg19_bn",
            "VGG-19.3":"robust_results11-15-21_23:29.30_base_vgg19_bn",
            "VGG-19.4":"robust_results11-15-21_23:49.39_base_vgg19_bn",
            "Ensemble-4 Synth VGG-19":"synth_ensemble_0_vgg19_bn_11_16_",
            "Ensemble-4 Synth VGG-19.1":"synth_ensemble_1_vgg19_bn_11_16_",
            "Ensemble-4 Synth VGG-19.2":"synth_ensemble_2_vgg19_bn_11_16_",
            "Ensemble-4 Synth VGG-19.3":"synth_ensemble_3_vgg19_bn_11_16_",
            "Ensemble-4 Synth VGG-19.4":"synth_ensemble_4_vgg19_bn_11_16_",
            #"ResNet 18":"robust_results11-10-21_23:33.14_base_resnet18",
            #"ResNet 18.1":"robust_results11-10-21_23:34.02_base_resnet18",
            #"ResNet 18.2":"robust_results11-10-21_23:34.24_base_resnet18",
            #"ResNet 18.3":"robust_results11-10-21_23:34.43_base_resnet18",
            #"ResNet 18.4":"robust_results11-10-21_23:35.02_base_resnet18",
            #"ResNet 18.5":"robust_results11-10-21_23:35.21_base_resnet18",
            #"Ensemble-4 Synth ResNet 18":"synth_ensemble_0_resnet18_",
            #"Ensemble-4 Synth ResNet 18.1":"synth_ensemble_1_resnet18_",
            #"Ensemble-4 Synth ResNet 18.2":"synth_ensemble_2_resnet18_",
            #"Ensemble-4 Synth ResNet 18.3":"synth_ensemble_3_resnet18_",
            "ResNet 18":"robust_results11-15-21_02:46.11_base_resnet18",
            "ResNet 18.7":"robust_results11-15-21_03:01.22_base_resnet18",
            "ResNet 18.8":"robust_results11-15-21_03:16.31_base_resnet18",
            "ResNet 18.9":"robust_results11-15-21_03:31.40_base_resnet18",
            "ResNet 18.10":"robust_results11-15-21_03:46.53_base_resnet18",
            "Ensemble-2 Synth ResNet 18":"synth_ensemble_0_base_resnet18_sub2_11_15_",
            "Ensemble-2 Synth ResNet 18.1":"synth_ensemble_1_base_resnet18_sub2_11_15_",
            "Ensemble-2 Synth ResNet 18.2":"synth_ensemble_2_base_resnet18_sub2_11_15_",
            "Ensemble-2 Synth ResNet 18.3":"synth_ensemble_3_base_resnet18_sub2_11_15_",
            "Ensemble-2 Synth ResNet 18.4":"synth_ensemble_4_base_resnet18_sub2_11_15_",
            "Ensemble-2 Synth ResNet 18.5":"synth_ensemble_5_base_resnet18_sub2_11_15_",
            "Ensemble-2 Synth ResNet 18.6":"synth_ensemble_6_base_resnet18_sub2_11_15_",
            "Ensemble-2 Synth ResNet 18.7":"synth_ensemble_7_base_resnet18_sub2_11_15_",
            "Ensemble-2 Synth ResNet 18.8":"synth_ensemble_8_base_resnet18_sub2_11_15_",
            "Ensemble-2 Synth ResNet 18.9":"synth_ensemble_9_base_resnet18_sub2_11_15_",
            "Ensemble-4 Synth ResNet 18":"synth_ensemble_0_base_resnet18_11_15_",
            "Ensemble-4 Synth ResNet 18.6":"synth_ensemble_1_base_resnet18_11_15_",
            "Ensemble-4 Synth ResNet 18.7":"synth_ensemble_2_base_resnet18_11_15_",
            "Ensemble-4 Synth ResNet 18.8":"synth_ensemble_3_base_resnet18_11_15_",
            "Ensemble-4 Synth ResNet 18.9":"synth_ensemble_4_base_resnet18_11_15_",
            #"Ensemble-5 Synth ResNet 18":"synth_ensemble_0_base_resnet18_super5_11_15_",
            #"Ensemble-4 WideResNet 18-2 (mean loss)" :"robust_results11-11-21_20:46.40_ensemble_wideresnet18",
            #"WideResNet 18-2":"robust_results11-11-21_00:22.44_base_resnet18",
            #"Ensemble-4 WideResNet 18-2 (sum loss)":"robust_results11-11-21_20:44.39_ensemble_wideresnet18",
            #"Ensemble-4 ResNet 18":"robust_results11-10-21_23:39.36_ensemble_resnet18",
            #"Ensemble-4 Resnet 18.1":"robust_results11-11-21_00:20.25_ensemble_resnet18",
            #"Ensemble-4 Resnet 18.2":"robust_results11-11-21_00:20.56_ensemble_resnet18",
            #"Ensemble-4 Resnet 18.3":"robust_results11-11-21_00:21.33_ensemble_resnet18", 
            #"InterpEnsemble-4 WideResNet 18-2 $\lambda = 1$":"robust_results11-11-21_17:02.02_interpensemble_wideresnet_18",
            #"InterpEnsemble-4 WideResNet 18-2 $\lambda = 0.5$":"robust_results11-11-21_17:16.27_interpensemble_wideresnet_18",
            #"InterpEnsemble-4 WideResNet 18-2 $\lambda = 0$":"robust_results11-11-21_17:17.19_interpensemble_wideresnet_18"
            "WideResNet 18-2":"robust_results11-14-21_23:49.57_base_wideresnet18",
            "WideResNet 18-2.1":"robust_results11-15-21_00:30.20_base_wideresnet18",
            "WideResNet 18-2.2":"robust_results11-15-21_01:10.29_base_wideresnet18",
            "WideResNet 18-2.3":"robust_results11-15-21_01:50.49_base_wideresnet18",
            "WideResNet 18-2.4":"robust_results11-15-21_02:31.03_base_wideresnet18",
            "Ensemble-2 Synth WideResNet 18-2":"synth_ensemble_0_base_wideresnet18_sub2_11_15_",
            "Ensemble-2 Synth WideResNet 18-2.1":"synth_ensemble_1_base_wideresnet18_sub2_11_15_",
            "Ensemble-2 Synth WideResNet 18-2.2":"synth_ensemble_2_base_wideresnet18_sub2_11_15_",
            "Ensemble-2 Synth WideResNet 18-2.3":"synth_ensemble_3_base_wideresnet18_sub2_11_15_",
            "Ensemble-2 Synth WideResNet 18-2.4":"synth_ensemble_4_base_wideresnet18_sub2_11_15_",
            "Ensemble-2 Synth WideResNet 18-2.5":"synth_ensemble_5_base_wideresnet18_sub2_11_15_",
            "Ensemble-2 Synth WideResNet 18-2.6":"synth_ensemble_6_base_wideresnet18_sub2_11_15_",
            "Ensemble-2 Synth WideResNet 18-2.7":"synth_ensemble_7_base_wideresnet18_sub2_11_15_",
            "Ensemble-2 Synth WideResNet 18-2.8":"synth_ensemble_8_base_wideresnet18_sub2_11_15_",
            "Ensemble-2 Synth WideResNet 18-2.9":"synth_ensemble_9_base_wideresnet18_sub2_11_15_",
            "Ensemble-4 Synth WideResNet 18-2":"synth_ensemble_0_base_wideresnet18_11_15_",
            "Ensemble-4 Synth WideResNet 18-2.1":"synth_ensemble_1_base_wideresnet18_11_15_",
            "Ensemble-4 Synth WideResNet 18-2.2":"synth_ensemble_2_base_wideresnet18_11_15_",
            "Ensemble-4 Synth WideResNet 18-2.3":"synth_ensemble_3_base_wideresnet18_11_15_",
            "Ensemble-4 Synth WideResNet 18-2.4":"synth_ensemble_4_base_wideresnet18_11_15_",
            "WideResNet 18-4":"robust_results11-13-21_01:28.26_base_wideresnet18_4",
            "WideResNet 18-4.1":"robust_results11-13-21_03:38.44_base_wideresnet18_4",
            "WideResNet 18-4.2":"robust_results11-13-21_05:49.19_base_wideresnet18_4",
            "WideResNet 18-4.3":"robust_results11-13-21_07:59.30_base_wideresnet18_4",
            "WideResNet 18-4.4":"robust_results11-13-21_10:10.17_base_wideresnet18_4",
            "Ensemble-2 Synth WideResNet 18-4":"synth_ensemble_0_base_wideresnet18-4_sub2_11_15_",
            "Ensemble-2 Synth WideResNet 18-4.1":"synth_ensemble_1_base_wideresnet18-4_sub2_11_15_",
            "Ensemble-2 Synth WideResNet 18-4.2":"synth_ensemble_2_base_wideresnet18-4_sub2_11_15_",
            "Ensemble-2 Synth WideResNet 18-4.3":"synth_ensemble_3_base_wideresnet18-4_sub2_11_15_",
            "Ensemble-2 Synth WideResNet 18-4.4":"synth_ensemble_4_base_wideresnet18-4_sub2_11_15_",
            "Ensemble-2 Synth WideResNet 18-4.5":"synth_ensemble_5_base_wideresnet18-4_sub2_11_15_",
            "Ensemble-2 Synth WideResNet 18-4.6":"synth_ensemble_6_base_wideresnet18-4_sub2_11_15_",
            "Ensemble-2 Synth WideResNet 18-4.7":"synth_ensemble_7_base_wideresnet18-4_sub2_11_15_",
            "Ensemble-2 Synth WideResNet 18-4.8":"synth_ensemble_8_base_wideresnet18-4_sub2_11_15_",
            "Ensemble-2 Synth WideResNet 18-4.9":"synth_ensemble_9_base_wideresnet18-4_sub2_11_15_",
            "Ensemble-4 Synth WideResNet 18-4":"synth_ensemble_0_base_wideresnet18_4_",
            "Ensemble-4 Synth WideResNet 18-4.1":"synth_ensemble_1_base_wideresnet18_4_",
            "Ensemble-4 Synth WideResNet 18-4.2":"synth_ensemble_2_base_wideresnet18_4_",
            "Ensemble-4 Synth WideResNet 18-4.3":"synth_ensemble_3_base_wideresnet18_4_",
            "Ensemble-4 Synth WideResNet 18-4.4":"synth_ensemble_4_base_wideresnet18_4_"
            },
            "cinic10":{
                "DenseNet-121":"robust_results12-02-21_05:17.33_base_densenet121",
                "DenseNet-121.1":"robust_results12-02-21_05:18.02_base_densenet121",
                "DenseNet-121.2":"robust_results12-02-21_05:18.30_base_densenet121",
                "DenseNet-121.3":"robust_results12-02-21_05:18.59_base_densenet121",
                "DenseNet-121.4":"robust_results12-02-21_05:19.27_base_densenet121",
                "Ensemble-4 Synth DenseNet-121":"synth_ensemble_0_base_densenet121_e4_cinic_",
                "Ensemble-4 Synth DenseNet-121.1":"synth_ensemble_1_base_densenet121_e4_cinic_",
                "Ensemble-4 Synth DenseNet-121.2":"synth_ensemble_2_base_densenet121_e4_cinic_",
                "Ensemble-4 Synth DenseNet-121.3":"synth_ensemble_3_base_densenet121_e4_cinic_",
                "Ensemble-4 Synth DenseNet-121.4":"synth_ensemble_4_base_densenet121_e4_cinic_",
                "DenseNet-169":"robust_results12-02-21_05:20.06_base_densenet169",
                "DenseNet-169.1":"robust_results12-02-21_05:20.44_base_densenet169",
                "DenseNet-169.2":"robust_results12-02-21_05:21.22_base_densenet169",
                "DenseNet-169.3":"robust_results12-02-21_05:22.00_base_densenet169",
                "DenseNet-169.4":"robust_results12-02-21_05:22.39_base_densenet169",
                "Ensemble-4 Synth DenseNet-169":"synth_ensemble_0_base_densenet169_e4_cinic_",
                "Ensemble-4 Synth DenseNet-169.1":"synth_ensemble_1_base_densenet169_e4_cinic_",
                "Ensemble-4 Synth DenseNet-169.2":"synth_ensemble_2_base_densenet169_e4_cinic_",
                "Ensemble-4 Synth DenseNet-169.3":"synth_ensemble_3_base_densenet169_e4_cinic_",
                "Ensemble-4 Synth DenseNet-169.4":"synth_ensemble_4_base_densenet169_e4_cinic_",
                "Inception-v3":"robust_results12-02-21_05:24.11_base_inception_v3",
                "Inception-v3.1":"robust_results12-02-21_05:25.44_base_inception_v3",
                "Inception-v3.2":"robust_results12-02-21_05:27.17_base_inception_v3",
                "Inception-v3.3":"robust_results12-02-21_05:28.50_base_inception_v3",
                "Inception-v3.4":"robust_results12-02-21_05:30.22_base_inception_v3",
                "Ensemble-4 Synth Inception-v3":"synth_ensemble_0_base_inception_v3_e4_cinic_",
                "Ensemble-4 Synth Inception-v3.1":"synth_ensemble_1_base_inception_v3_e4_cinic_",
                "Ensemble-4 Synth Inception-v3.2":"synth_ensemble_2_base_inception_v3_e4_cinic_",
                "Ensemble-4 Synth Inception-v3.3":"synth_ensemble_3_base_inception_v3_e4_cinic_",
                "Ensemble-4 Synth Inception-v3.4":"synth_ensemble_4_base_inception_v3_e4_cinic_",
                "GoogleNet":"robust_results12-02-21_05:31.07_base_googlenet",
                "GoogleNet.1":"robust_results12-02-21_05:31.52_base_googlenet",
                "GoogleNet.2":"robust_results12-02-21_05:32.36_base_googlenet",
                "GoogleNet.3":"robust_results12-02-21_05:33.21_base_googlenet",
                "GoogleNet.4":"robust_results12-02-21_05:34.05_base_googlenet",
                "Ensemble-4 Synth GoogleNet":"synth_ensemble_0_base_googlenet_e4_cinic_",
                "Ensemble-4 Synth GoogleNet.1":"synth_ensemble_1_base_googlenet_e4_cinic_",
                "Ensemble-4 Synth GoogleNet.2":"synth_ensemble_2_base_googlenet_e4_cinic_",
                "Ensemble-4 Synth GoogleNet.3":"synth_ensemble_3_base_googlenet_e4_cinic_",
                "Ensemble-4 Synth GoogleNet.4":"synth_ensemble_4_base_googlenet_e4_cinic_",
                "VGG-11":"robust_results12-02-21_05:34.26_base_vgg11_bn",
                "VGG-11.1":"robust_results12-02-21_05:34.47_base_vgg11_bn",
                "VGG-11.2":"robust_results12-02-21_05:35.08_base_vgg11_bn",
                "VGG-11.3":"robust_results12-02-21_05:35.28_base_vgg11_bn",
                "VGG-11.4":"robust_results12-02-21_05:35.49_base_vgg11_bn",
                "Ensemble-4 Synth VGG-11":"synth_ensemble_0_base_vgg11_bn_e4_cinic_",
                "Ensemble-4 Synth VGG-11.1":"synth_ensemble_1_base_vgg11_bn_e4_cinic_",
                "Ensemble-4 Synth VGG-11.2":"synth_ensemble_2_base_vgg11_bn_e4_cinic_",
                "Ensemble-4 Synth VGG-11.3":"synth_ensemble_3_base_vgg11_bn_e4_cinic_",
                "Ensemble-4 Synth VGG-11.4":"synth_ensemble_4_base_vgg11_bn_e4_cinic_",
                "VGG-19":"robust_results12-02-21_05:36.12_base_vgg19_bn",
                "VGG-19.1":"robust_results12-02-21_05:36.35_base_vgg19_bn",
                "VGG-19.2":"robust_results12-02-21_05:36.57_base_vgg19_bn",
                "VGG-19.3":"robust_results12-02-21_05:37.20_base_vgg19_bn",
                "VGG-19.4":"robust_results12-02-21_05:37.43_base_vgg19_bn",
                "Ensemble-4 Synth VGG-19":"synth_ensemble_0_base_vgg19_bn_e4_cinic_",
                "Ensemble-4 Synth VGG-19.1":"synth_ensemble_1_base_vgg19_bn_e4_cinic_",
                "Ensemble-4 Synth VGG-19.2":"synth_ensemble_2_base_vgg19_bn_e4_cinic_",
                "Ensemble-4 Synth VGG-19.3":"synth_ensemble_3_base_vgg19_bn_e4_cinic_",
                "Ensemble-4 Synth VGG-19.4":"synth_ensemble_4_base_vgg19_bn_e4_cinic_",
                "ResNet 18":"robust_results12-02-21_05:38.04_base_resnet18",
                "ResNet 18.7":"robust_results12-02-21_05:38.24_base_resnet18",
                "ResNet 18.8":"robust_results12-02-21_05:38.45_base_resnet18",
                "ResNet 18.9":"robust_results12-02-21_05:39.05_base_resnet18",
                "ResNet 18.10":"robust_results12-02-21_05:39.26_base_resnet18",
                "Ensemble-4 Synth ResNet 18":"synth_ensemble_0_base_resnet18_e4_cinic_",
                "Ensemble-4 Synth ResNet 18.6":"synth_ensemble_1_base_resnet18_e4_cinic_",
                "Ensemble-4 Synth ResNet 18.7":"synth_ensemble_2_base_resnet18_e4_cinic_",
                "Ensemble-4 Synth ResNet 18.8":"synth_ensemble_3_base_resnet18_e4_cinic_",
                "Ensemble-4 Synth ResNet 18.9":"synth_ensemble_4_base_resnet18_e4_cinic_",
                "WideResNet 18-2":"robust_results12-02-21_05:39.49_base_wideresnet18",
                "WideResNet 18-2.1":"robust_results12-02-21_05:40.12_base_wideresnet18",
                "WideResNet 18-2.2":"robust_results12-02-21_05:40.35_base_wideresnet18",
                "WideResNet 18-2.3":"robust_results12-02-21_05:40.58_base_wideresnet18",
                "WideResNet 18-2.4":"robust_results12-02-21_05:41.21_base_wideresnet18",
                "Ensemble-4 Synth WideResNet 18-2":"synth_ensemble_0_base_wideresnet18_e4_cinic_",
                "Ensemble-4 Synth WideResNet 18-2.1":"synth_ensemble_1_base_wideresnet18_e4_cinic_",
                "Ensemble-4 Synth WideResNet 18-2.2":"synth_ensemble_2_base_wideresnet18_e4_cinic_",
                "Ensemble-4 Synth WideResNet 18-2.3":"synth_ensemble_3_base_wideresnet18_e4_cinic_",
                "Ensemble-4 Synth WideResNet 18-2.4":"synth_ensemble_4_base_wideresnet18_e4_cinic_",
                "WideResNet 18-4":"robust_results12-02-21_05:42.05_base_wideresnet18_4",
                "WideResNet 18-4.1":"robust_results12-02-21_05:42.50_base_wideresnet18_4",
                "WideResNet 18-4.2":"robust_results12-02-21_05:43.35_base_wideresnet18_4",
                "WideResNet 18-4.3":"robust_results12-02-21_05:44.20_base_wideresnet18_4",
                "WideResNet 18-4.4":"robust_results12-02-21_05:45.04_base_wideresnet18_4",
                "Ensemble-4 Synth WideResNet 18-4":"synth_ensemble_0_base_wideresnet18_4_e4_cinic_",
                "Ensemble-4 Synth WideResNet 18-4.1":"synth_ensemble_1_base_wideresnet18_4_e4_cinic_",
                "Ensemble-4 Synth WideResNet 18-4.2":"synth_ensemble_2_base_wideresnet18_4_e4_cinic_",
                "Ensemble-4 Synth WideResNet 18-4.3":"synth_ensemble_3_base_wideresnet18_4_e4_cinic_",
                "Ensemble-4 Synth WideResNet 18-4.4":"synth_ensemble_4_base_wideresnet18_4_e4_cinic_"
                    },
            }

all_suffixes = {"cifar10.1":{
        "InD Labels":"ind_labels.npy",
        "InD Probs": "ind_preds.npy",
        "OOD Labels":"ood_labels.npy",
        "OOD Probs": "ood_preds.npy",
        "meta":"_meta.json"
        },
        "cinic10":{
        "InD Labels":"ind_labels.npy",
        "InD Probs": "ind_preds.npy",
        "OOD Labels":"ood_cinic_labels.npy",
        "OOD Probs": "ood_cinic_preds.npy",
        "meta":"_meta.json"
        }
        }

bins = list(np.linspace(0,1,17)[1:-1])

def main(args,dataindex,suffixes):
    predfig,predax = plt.subplots(figsize = (15,15))
    calfig,calax = plt.subplots(figsize = (15,15))
    bsfig,bsax = plt.subplots(figsize = (15,15))
    bsmfig,bsmax = plt.subplots(figsize = (15,15))
    nllfig,nllax = plt.subplots(figsize = (15,15))


    for model,path in dataindex.items():
        modelmetrics = {"ind":[],"ood":[]}
        relfig,relax = plt.subplots(1,2)
        try:
            with open(os.path.join(resultsfolder,path+suffixes["meta"]),"r") as f:
                metadata = json.load(f)
        except FileNotFoundError:
            metadata = {"softmax":True}

        for di,dist in enumerate([["InD Labels","InD Probs"],["OOD Labels","OOD Probs"]]):
            labels = np.load(os.path.join(resultsfolder,path+suffixes[dist[0]])).astype(int)
            probs = np.load(os.path.join(resultsfolder,path+suffixes[dist[1]]))
            if model.startswith("InterpEnsemble"):
                probs = probs*2
            if bool(metadata.get("softmax",True)) is False:    
                print("applying softmax to: {} for {}".format(model,dist[1]))
                probs = softmax(probs,axis = 1)

            ad = AccuracyData()
            nld = NLLData()
            cd = CalibrationData(bins)
            bsd = BrierScoreData()
                
            accuracy = ad.accuracy(probs,labels)
            nll = nld.nll(probs,labels,normalize = True)
            ece = cd.ece(probs,labels)
            bs = bsd.brierscore(probs,labels)
            bsm = bsd.brierscore_multi(probs,labels)
            rel = cd.bin(probs,labels)
            interval = cd.binedges[0][1]-cd.binedges[0][0]
            relax[di].bar([bi[0] for bi in cd.binedges],[bi[0] for bi in cd.binedges],width= interval,color = "red",alpha = 0.3)
            relax[di].bar([bi[0] for bi in cd.binedges],[rel[i]["bin_acc"] for i in cd.binedges],width = interval,color = "blue")
            relax[di].set_aspect("equal")

            if dist[0] == "InD Labels":
                modelmetrics["ind"] = [accuracy,nll,ece,bs,bsm]
            if dist[0] == "OOD Labels":
                modelmetrics["ood"] = [accuracy,nll,ece,bs,bsm]

            for modelpre,mcand in markers.items():
                if model.startswith(modelpre):
                    if dist[0] == "InD Labels":
                        ## track diversity metrics
                        variancecalc[modelpre]["ind"].register(probs,labels,model)
                        variancecalc[modelpre]["ind_cal"].append(ece) 
                    elif dist[0] == "OOD Labels":    
                        variancecalc[modelpre]["ood"].register(probs,labels,model)
                        variancecalc[modelpre]["ood_cal"].append(ece) 
                elif model.startswith("Ensemble-4 Synth "+modelpre):        
                    if dist[0] == "InD Labels":
                        variancecalc[modelpre]["ind_ens_cal"].append(ece) 
                    elif dist[0] == "OOD Labels":    
                        variancecalc[modelpre]["ood_ens_cal"].append(ece) 

        for modelpre,mcand in markers.items():
            if model.startswith(modelpre):
                ## set marker
                marker = mcand
            else:    
                pass
        if len(model.split(".")) == 1: 
            predax.plot(modelmetrics["ind"][0],modelmetrics["ood"][0],marker,label = model)
            nllax.plot(modelmetrics["ind"][1],modelmetrics["ood"][1],marker,label = model)
            calax.plot(modelmetrics["ind"][2],modelmetrics["ood"][2],marker,label = model)
            bsax.plot(modelmetrics["ind"][3],modelmetrics["ood"][3],marker,label = model)
            bsmax.plot(modelmetrics["ind"][4],modelmetrics["ood"][4],marker,label = model)
        else:    
            predax.plot(modelmetrics["ind"][0],modelmetrics["ood"][0],marker)
            nllax.plot(modelmetrics["ind"][1],modelmetrics["ood"][1],marker)
            calax.plot(modelmetrics["ind"][2],modelmetrics["ood"][2],marker)
            bsax.plot(modelmetrics["ind"][3],modelmetrics["ood"][3],marker)
            bsmax.plot(modelmetrics["ind"][4],modelmetrics["ood"][4],marker)
        relfig.suptitle("{}".format(model))
        relax[0].set_title("InD Calibration")
        relax[1].set_title("OOD Calibration")
        relax[0].set_xlabel("Confidence")
        relax[1].set_ylabel("Accuracy")
        relfig.savefig(os.path.join(imagesfolder,"reliability_diag_{}_{}.png".format(model,args.ood_dataset)))    
        plt.close(relfig)
    ## Variance data: 
    varfig,varax = plt.subplots(figsize = (10,10))
    joblib.dump(variancecalc,"ensembledata_{}".format(args.ood_dataset))
    for modelclass,modeldata in variancecalc.items():
        varconffig,varconfax = plt.subplots(2,2,figsize=(20,20))
        if not modelclass.startswith("Ensemble"):
            try:
                print(np.mean(modeldata["ind_ens_cal"]),np.mean(modeldata["ood_ens_cal"]))
                varax.plot(modeldata["ind"].expected_variance()/modeldata["ood"].expected_variance(),(np.mean(modeldata["ind_ens_cal"])-np.mean(modeldata["ind_cal"]))/(np.mean(modeldata["ood_ens_cal"])-np.mean(modeldata["ood_cal"])),markers[modelclass],label = modelclass)
                
                varconfax[0,0].scatter(*modeldata["ind"].variance_confidence().T,c = markers[modelclass][:-1],marker=markers[modelclass][-1],label = modelclass)
                varconfax[0,1].scatter(*modeldata["ood"].variance_confidence().T,c = markers[modelclass][:-1],marker=markers[modelclass][-1],label = modelclass)
                varconfax[1,0].hist(modeldata["ind"].variance_confidence()[:,0],bins = 100, density = True, log = True)
                varconfax[1,1].hist(modeldata["ood"].variance_confidence()[:,0],bins = 100, density = True, log = True)
            except IndexError:    
                pass
        for i,d in enumerate(["ind","ood"]):
            varconfax[0,i].legend()    
            varconfax[0,i].set_title("Ensemble Confidence/Variance: {} ({})".format(modelclass,d))
            varconfax[0,i].set_xlabel("Mean Confidence")
            varconfax[1,i].set_xlabel("Mean Confidence")
            varconfax[0,i].set_ylabel("Variance")
            varconfax[1,i].set_title("Sample Density per mean confidence: {} ({})".format(modelclass,d))
        varconffig.savefig(os.path.join(imagesfolder,"variance_confidence_metrics_{}_{}.png".format(modelclass,args.ood_dataset)))    
        plt.close(varconffig)
    varax.legend()    
    varax.set_title("Ensemble Variance-ECE Ratio")
    varax.set_xlabel("InD/OOD LL variance ratio")
    varax.set_ylabel("InD/OOD ECE ratio")
    varfig.savefig(os.path.join(imagesfolder,"variance_metrics_{}.png").format(args.ood_dataset))    

    predax.legend()
    nllax.legend()
    calax.legend()
    bsax.legend()
    bsmax.legend()

    predax.set_title("Predictive Accuracy")
    predax.set_xlabel("InD Test Accuracy")
    predax.set_ylabel("OOD Test Accuracy")
    nllax.set_title("Negative Log Likelihood")
    nllax.set_xlabel("InD")
    nllax.set_ylabel("OOD")
    calax.set_title("Calibration")
    calax.set_xlabel("InD ECE")
    calax.set_ylabel("OOD ECE")
    bsax.set_title("Brier Score (Binarized)")
    bsax.set_xlabel("InD BS")
    bsax.set_ylabel("OOD BS")
    bsmax.set_title("Brier Score (Multiclass)")
    bsmax.set_xlabel("InD BS")
    bsmax.set_ylabel("OOD BS")
    predfig.savefig(os.path.join(imagesfolder,"prediction_metrics_{}.png").format(args.ood_dataset))    
    nllfig.savefig(os.path.join(imagesfolder,"nll_metrics_{}.png").format(args.ood_dataset))    
    calfig.savefig(os.path.join(imagesfolder,"calibration_metrics_{}.png").format(args.ood_dataset))    
    bsfig.savefig(os.path.join(imagesfolder,"brierscore_metrics_{}.png").format(args.ood_dataset))
    bsmfig.savefig(os.path.join(imagesfolder,"brierscoremulti_metrics_{}.png").format(args.ood_dataset))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-od","--ood_dataset",help = "which ood dataset to analyze for",default = "cifar10.1",choices = ["cifar10.1","cinic10"])
    args = parser.parse_args()
    dataindex = all_dataindices[args.ood_dataset]
    suffixes = all_suffixes[args.ood_dataset]
    main(args,dataindex,suffixes)

