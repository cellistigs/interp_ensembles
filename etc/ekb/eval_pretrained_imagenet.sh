#!/bin/bash

# Evaluate pretrained models on imagenet and imagenetv2*
# and store the logits.

# imagenet directories

# ------ set outputdirectory ------
OUTPUTDIR="/data/Projects/linear_ensembles/interp_ensembles/results/imagenet/logits"
mkdir -p $OUTPUTDIR

# ----- set dataset directories -----
BASEDATA_DIR="/home/ekellbuch/pytorch_datasets"

IMAGENET_DIR="$BASEDATA_DIR/imagenet"
IMAGENETV2_TH="$BASEDATA_DIR/imagenetv2-a-44"
IMAGENEV2_TOP="$BASEDATA_DIR/imagenetv2-c-12"
IMAGENETV2_MF="$BASEDATA_DIR/imagenetv2-b-33"

declare -A DATASETDIR=(
  ["imagenet"]=$IMAGENET_DIR
  ["imagenetv2th"]=$IMAGENETV2_TH
  ["imagenetv2top"]=$IMAGENEV2_TOP
  ["imagenetv2mf"]=$IMAGENETV2_MF
  )

# Pick a model
model='resnet50'
gpu=0
batch_size=250
store_logits=1 # store outputs
workers=1
#seed=2
#epochs=90

#dataset_name="imagenet"
#for model in "resnet50" "resnet101" "efficientnet_b0" "wide_resnet50_2" "wide_resnet101_2" "efficientnet_b1" "efficientnet_b2"
for model in "alexnet" "densenet121" "googlenet" "resnet18" "vgg11" "vgg13"

do
for dataset_name in "imagenet" "imagenetv2mf"
do

dataset_dir="${DATASETDIR[$dataset_name]}"
#echo "${dataset_name}:${dataset_dir}"
store_logits_fname="$OUTPUTDIR/${model}--${dataset_name}.hdf5"
echo "Output will be stored in ${store_logits_fname}"

python /data/Projects/linear_ensembles/interp_ensembles/scripts/train_imagenet.py ${dataset_dir} --arch=${model} \
  --workers=${workers} \
  --pretrained \
  --gpu ${gpu} \
  --evaluate \
  --batch-size ${batch_size} \
  --store_logits ${store_logits} \
  --store_logits_fname ${store_logits_fname} \
  #--epochs=${epochs} \
  #--seed=${seed} \
done
done
