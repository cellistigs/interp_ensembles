#!/bin/bash

# Evaluate pretrained models on imagenet and imagenetv2*
# and store the logits.

# imagenet directories
dataset_ood="imagenetc"
for distortion_name in 'brightness' 'contrast' 'fog'  'gaussian_noise'
do
for severity in 1 3 5
do
  data_type_ood="--${distortion_name}--${severity}"
  echo "$dataset_ood $data_type_ood"
  python plot_metrics_imagenet.py $dataset_ood  $data_type_ood
done
done
