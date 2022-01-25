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

ENSEMBLE_BASEDIR="/datahd3a/imagenet_networks"

declare -A ENSEMBLEDIR1=(
  ["deepens1"]="resnet101--imagenet/01_03_2022/22_05_35/lightning_logs/version_0/checkpoints/epoch=89-step=464939.ckpt"
  ["deepens2"]="resnet101--imagenet/01_04_2022/23_57_31/lightning_logs/version_0/checkpoints/epoch=89-step=464939.ckpt"
  ["deepens3"]="resnet101--imagenet/01_05_2022/00_40_34/lightning_logs/version_0/checkpoints/epoch=89-step=464939.ckpt"
  ["deepens4"]="resnet101--imagenet/01_06_2022/15_05_06/lightning_logs/version_0/checkpoints/epoch=89-step=470105.ckpt"
  ["deepens5"]="resnet101--imagenet--12_27_2021_23_12_23/checkpoint.pth.tar"
  )

#dataset_name="imagenet"
from_pl=1
#"resnet50" "resnet101" "efficientnet_b0" "wide_resnet50_2" "wide_resnet101_2" "efficientnet_b1" "efficientnet_b2"
for deepensn in "deepens5" # "deepens2" "deepens3" "deepens4" "deepens1"
do
for model in "resnet101"
do
for dataset_name in "imagenet" "imagenetv2mf"
do
dataset_dir="${DATASETDIR[$dataset_name]}"
#echo "${dataset_name}:${dataset_dir}"
store_logits_fname="$OUTPUTDIR/${model}--${dataset_name}--${deepensn}.hdf5"
echo "Output will be stored in ${store_logits_fname}"
resume="${ENSEMBLE_BASEDIR}/${ENSEMBLEDIR1[$deepensn]}"
echo "$resume"
python /data/Projects/linear_ensembles/interp_ensembles/scripts/pred_imagenet.py ${dataset_dir} --arch=${model} \
  --workers=${workers} \
  --resume=${resume} \
  --evaluate \
  --batch-size ${batch_size} \
  --store_logits ${store_logits} \
  --store_logits_fname ${store_logits_fname} \
#  --from_pl=${from_pl}
done
done
done