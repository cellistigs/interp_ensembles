#!/bin/bash

# Train models on imagenet and imagenetv2*
# TODO: update checkpointing

# imagenet directories

now=$(date +"%m_%d_%Y_%H_%M_%S")
echo "$now"

# ------ set packahe directory ------
PKGDIR="/data/Projects/linear_ensembles/interp_ensembles"

# ------ set output directory ------
OUTPUTDIR="/data/Projects/linear_ensembles/interp_ensembles/results/imagenet/checkpoints"
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
model='resnet50'
gpu=0
batch_size=256
store_logits=1 # store outputs
workers=1
seed=0
epochs=1

dataset_name="tinyimagenet"

for model in "resnet18" # "resnet50" "resnet101" "efficientnet_b0" "wide_resnet50_2" "wide_resnet101_2" "efficientnet_b1" "efficientnet_b2"
do
dataset_dir="${DATASETDIR[$dataset_name]}"
OUTDIR_CPKT="${OUTPUTDIR}/${model}--${dataset_name}--${now}"
mkdir -p $OUTDIR_CPKT
save_checkpoint_fname="${OUTDIR_CPKT}/checkpoint.pth.tar"
run_log="${OUTDIR_CPKT}/run_log.out"
echo "Checkpoints will be stored in ${save_checkpoint_fname}"

python train_imagenet.py ${dataset_dir} --arch=${model} \
  --workers=${workers} \
  --batch-size ${batch_size} \
  --epochs=${epochs} \
  --save_checkpoint_fname=${save_checkpoint_fname} \
  > ${run_log}

done
