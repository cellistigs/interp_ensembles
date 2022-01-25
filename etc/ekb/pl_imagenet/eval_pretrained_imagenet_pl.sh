#!/bin/bash

# Evaluate pretrained models on imagenet

# TODO: store logits: _pl appears to be slower for test set
# TODO: add eval imagenet_c
# Note: This appears to be slower than using train_imagenet.py


# ------ set outputdirectory ------
OUTPUTDIR="/data/Projects/linear_ensembles/interp_ensembles/results/imagenet/logits"
mkdir -p $OUTPUTDIR

# ----- set dataset directories -----
BASEDATA_DIR="/home/ekellbuch/pytorch_datasets"

IMAGENET_DIR="$BASEDATA_DIR/imagenet"
IMAGENETV2_TH="$BASEDATA_DIR/imagenetv2-a-44"
IMAGENEV2_TOP="$BASEDATA_DIR/imagenetv2-c-12"
IMAGENETV2_MF="$BASEDATA_DIR/imagenetv2-b-33"

TINYIMAGENET_DIR="$BASEDATA_DIR/tiny-imagenet-200"

declare -A DATASETDIR=(
  ["imagenet"]=$IMAGENET_DIR
  ["imagenetv2th"]=$IMAGENETV2_TH
  ["imagenetv2top"]=$IMAGENEV2_TOP
  ["imagenetv2mf"]=$IMAGENETV2_MF
  ["tinyimagenet"]=$TINYIMAGENET_DIR

  )

# Pick a model
gpus=1
batch_size=256
store_logits=0 # store outputs
workers=16 # number of cpus
seed=0
#epochs=90

#auto_select_gpus=1
resume_from_checkpoint='/data/Projects/linear_ensembles/interp_ensembles/etc/ekb/pl_imagenet/lightning_logs/version_1/checkpoints/epoch=0-step=399.ckpt'

profiler='simple'
deterministic=0
#dataset_name="imagenet"
#for model in "resnet50" "resnet101" "efficientnet_b0" "wide_resnet50_2" "wide_resnet101_2" "efficientnet_b1" "efficientnet_b2"
for model in "alexnet" #"densenet121" "googlenet" "resnet18" "vgg11" "vgg13"

do
for dataset_name in "tinyimagenet" #"imagenetv2mf"
do

dataset_dir="${DATASETDIR[$dataset_name]}"
#echo "${dataset_name}:${dataset_dir}"
store_logits_fname="$OUTPUTDIR/${model}--${dataset_name}.hdf5"
#echo "Output will be stored in ${store_logits_fname}"

python /data/Projects/linear_ensembles/interp_ensembles/scripts/train_imagenet_pl.py --data-path ${dataset_dir}  \
  --gpus=${gpus} \
  --arch=${model} \
  --seed=${seed} \
  --evaluate \
  --profiler=${profiler} \
  --deterministic=${deterministic} \
  --batch-size=${batch_size} \
  --workers=${workers} \
  --resume_from_checkpoint=${resume_from_checkpoint} \
#  --pretrained \

done
done
