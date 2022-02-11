#!/bin/bash
#

# creates a csv with metrics values for linear trends figs
#for all of the models that Taiga sent over

# Need to run results for
# cifar 10.1
# single model
# hom ensemble
# het ensemble

# cinic10
# single model
# hom ensemble
# het ensemble

# imagenetv2
# single model
# hom ensemble
# het ensemble

# imagenetc


declare -A LOGITS_DIR=(
["brightness"]="/datahd2a/imagenet_testbest_outputs/logits"
["contrast"]="/datahd2a/imagenet_testbest_outputs/logits"
["gaussian_noise"]="/datahd3a/imagenet_testbest_outputs/logits"
["fog"]="/datahd3a/imagenet_testbest_outputs/logits"
)


RESULTSDIR="/data/Projects/linear_ensembles/interp_ensembles/results/metrics"
#-----------------------------------------------------------------------------
# to run on cifar 10.1
# reads single model and homogeneous ensemble
output_name="$RESULTSDIR/cifar10.1_metrics.csv"
logits_folder="/data/Projects/linear_ensembles/interp_ensembles/data/cifar10.1"
python metrics.py $logits_folder $output_name

# add heterogeneous ensemble
metrics_file="$RESULTSDIR/cifar10.1_metrics.csv"
#ensemble_name="$RESULTSDIR/cifar10.1_het_ensemble_metrics.csv"
#python metrics_het_ensemble.py $metrics_file $ensemble_name

ensemble_name="$RESULTSDIR/cifar10.1_het_ensemble2_metrics.csv"
python metrics_het_ensemble.py $metrics_file $ensemble_name

#-----------------------------------------------------------------------------
# to run on imagenetv2mf
# reads single models
output_name="$RESULTSDIR/imagenetv2mf_metrics.csv"
logits_folder="/data/Projects/linear_ensembles/interp_ensembles/results/imagenet/logits"
python metrics_imagenet.py $logits_folder $output_name

# Add ensemble to imagenetv2mf data
ensemble_name="$RESULTSDIR/imagenetv2mf_ensemble_metrics.csv"
python metrics_ensemble.py $output_name $ensemble_name

# Read testbed data
# first add ind
#output_name="$RESULTSDIR/testbed_imagenet_val_metrics.csv"
ind_logits_folder="/datahd3a/imagenet_testbest_outputs/logits/val"

# then read ood
output_name="$RESULTSDIR/testbed_imagenetv2mf_metrics.csv"
logits_folder="/datahd3a/imagenet_testbest_outputs/logits/imagenetv2-matched-frequency-format-val"
python metrics_testbed.py $logits_folder $output_name $ind_logits_folder

# add heterogeneous ensemble, including testbed
metrics_file="$RESULTSDIR/imagenetv2mf_metrics.csv"
ood_file="$RESULTSDIR/testbed_imagenetv2mf_metrics.csv"
#ensemble_name="$RESULTSDIR/imagenetv2mf_het_ensemble_metrics.csv"
#python metrics_het_ensemble.py $metrics_file $ensemble_name $ood_file
ensemble_name="$RESULTSDIR/imagenetv2mf_het_ensemble2_metrics.csv"
python metrics_het_ensemble.py $metrics_file $ensemble_name $ood_file

#-----------------------------------------------------------------------------
# to run on cinic10
# reads single models
output_name="$RESULTSDIR/cinic10_metrics.csv"
logits_folder="/data/Projects/linear_ensembles/interp_ensembles/data/logits_miller"
python metrics_miller.py $logits_folder $output_name

# add ensemble to miller's data
# using cinic10
ensemble_name="$RESULTSDIR/cinic10_ensemble_metrics.csv"
python metrics_ensemble.py $output_name $ensemble_name

# add heterogeneous ensemble
output_name="$RESULTSDIR/cinic10_metrics.csv"
#ensemble_name="$RESULTSDIR/cinic10_het_ensemble_metrics.csv"
#python metrics_het_ensemble.py $output_name $ensemble_name
ensemble_name="$RESULTSDIR/cinic10_het_ensemble2_metrics.csv"
python metrics_het_ensemble.py $output_name $ensemble_name

#------------------------------------------------------------------------

# imagenetc
dist_type="brightness"
dist_level=1
for dist_type in "brightness" "contrast" "gaussian_noise" "fog"
do
  for dist_level in 5 3 1
  do
    echo ""
  # read imagenet-c metrics for resnet/alexnet
  distortion_name="${dist_type}--${dist_level}"
  output_name="$RESULTSDIR/imagenetc--${distortion_name}_metrics.csv"
  logits_folder="/data/Projects/linear_ensembles/interp_ensembles/results/imagenet/logits"
  logits_ood_folder="/data/Projects/linear_ensembles/interp_ensembles/results/imagenetc/logits"
  python metrics_imagenet_c.py $logits_folder $output_name $distortion_name $logits_ood_folder

  # add imagenet_c homogeneous ensemble
  ensemble_name="$RESULTSDIR/imagenetc--${distortion_name}_ensemble_metrics.csv"
  python metrics_ensemble.py $output_name $ensemble_name

  # add imagenet_c tested metrics
  # there are 2 types of imagenet v2 tesbed
  output_name="$RESULTSDIR/testbed_imagenetc--${distortion_name}_metrics.csv"
  ind_logits_folder="/datahd3a/imagenet_testbest_outputs/logits/val"
  logits_folder="${LOGITS_DIR[$dist_type]}/imagenet-c.${dist_type}.${dist_level}_on-disk"
  python metrics_testbed.py $logits_folder $output_name $ind_logits_folder $dist_type $dist_level

  # add heterogeneous ensemble, including testbed
  metrics_file="$RESULTSDIR/imagenetc--${distortion_name}_metrics.csv"
  ood_file="$RESULTSDIR/testbed_imagenetc--${distortion_name}_metrics.csv"
  #ensemble_name="$RESULTSDIR/imagenetc--${distortion_name}_het_ensemble_metrics.csv"
  #python metrics_het_ensemble.py $metrics_file $ensemble_name $ood_file
  ensemble_name="$RESULTSDIR/imagenetc--${distortion_name}_het_ensemble2_metrics.csv"
  python metrics_het_ensemble.py $metrics_file $ensemble_name $ood_file

  # Add plot
  ood_dataset="imagenetc--${distortion_name}"
  python plot_metrics.py $ood_dataset
done
done

#---- Plot metrics -----
for ensemble_type in   "ensemble_all" "ensemble_homog" "ensemble_heter"
do
for ood_dataset in "cinic10" "cifar10.1" "imagenetv2mf"
do
  echo ""
  python plot_metrics.py $ood_dataset $ensemble_type
done
done

for ensemble_type in "ensemble_all" "ensemble_homog" "ensemble_heter"
do
  echo ""
  python plot_ece.py $ensemble_type
done

for ensemble_type in "ensemble_all" # "ensemble_homog"  "ensemble_heter"
do
for dist_type in "brightness" "contrast" "gaussian_noise" "fog"
do
  for dist_level in 5 3 1
  do
    echo ""
    distortion_name="${dist_type}--${dist_level}"
    ood_dataset="imagenetc--${distortion_name}"
    #python plot_metrics.py $ood_dataset $ensemble_type
done
done
done