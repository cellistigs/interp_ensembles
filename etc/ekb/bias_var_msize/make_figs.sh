#!/bin/bash

set -ex

pushd ../../../

echo "$PWD"

echo "Calculate ensemble model performance"
for dataset in "imagenet" "imagenetv2mf"
do
python scripts/calculate_model_performance.py "--config-name=${dataset_name}"
echo "Build ensemble"
for binning in 3 6 8; do
python etc/ekb/bias_var_msize/metrics_het_ensemble_parallel.py --dataset=${dataset} --binning=${binning}
done
echo "Plot"
python etc/ekb/bias_var_msize/plot_bias_var.py --dataset=${dataset}
done


# deal with imagenet_c dataset
for dataset in "imagenet_c"; do
for corruption in gaussian_noise; do
for severity in 1 3 5; do
dataset_name="${dataset}_${corruption}_${severity}"
echo "Running ${dataset_name}"
python scripts/calculate_model_performance.py "--config-name=${dataset_name}"
for binning in 3 6 8; do
python etc/ekb/bias_var_msize/metrics_het_ensemble_parallel.py --dataset=${dataset_name} --binning=${binning}
done
python etc/ekb/bias_var_msize/plot_bias_var.py --dataset=${dataset_name}
done
done
done

popd

