#!/bin/bash

# given  resnet 50  models trained from scratch on imagenet
# evalulate them on  imagenet and imagenetv2* and store the logits

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

ENSEMBLE_BASEDIR="/home/ekellbuch/data_axon/libraries/interp_ensembles/results/pl_imagenet/alexnet--imagenet"

declare -A ENSEMBLEDIR1=(
  ["deepens1"]="12_27_2021/20_27_26/lightning_logs/version_222492/checkpoints/epoch=89-step=450449.ckpt"
  ["deepens2"]="12_27_2021/20_34_08/lightning_logs/version_222493/checkpoints/epoch=89-step=450449.ckpt"
  ["deepens3"]="12_27_2021/20_34_08/lightning_logs/version_222494/checkpoints/epoch=89-step=450449.ckpt"
  ["deepens4"]="12_27_2021/20_34_08/lightning_logs/version_222495/checkpoints/epoch=89-step=450449.ckpt"
  ["deepens5"]="12_27_2021/20_34_08/lightning_logs/version_222496/checkpoints/epoch=89-step=450449.ckpt"

  )

#dataset_name="imagenet"
from_pl=1
#"resnet50" "resnet101" "efficientnet_b0" "wide_resnet50_2" "wide_resnet101_2" "efficientnet_b1" "efficientnet_b2"
for deepensn in "deepens1" "deepens2" "deepens3" "deepens4" "deepens5"
do
for model in "alexnet"
do
for dataset_name in "imagenet" "imagenetv2mf"
do
dataset_dir="${DATASETDIR[$dataset_name]}"
#echo "${dataset_name}:${dataset_dir}"
store_logits_fname="$OUTPUTDIR/${model}--${dataset_name}--${deepensn}.hdf5"
echo "Output will be stored in ${store_logits_fname}"
resume="${ENSEMBLE_BASEDIR}/${ENSEMBLEDIR1[$deepensn]}"
echo "$resume"
python /data/Projects/linear_ensembles/interp_ensembles/scripts/train_imagenet.py ${dataset_dir} --arch=${model} \
  --workers=${workers} \
  --resume=${resume} \
  --evaluate \
  --batch-size ${batch_size} \
  --store_logits ${store_logits} \
  --store_logits_fname ${store_logits_fname} \
  --from_pl=${from_pl}
done
done
done