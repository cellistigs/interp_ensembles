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

ENSEMBLE_BASEDIR="/datahd3a/pytorch_ensembles/deepens_imagenet"

declare -A ENSEMBLEDIR1=(
  ["deepens1"]="ImageNet-ResNet50-052e7f78e4db--1564492444-1.pth.tar"
  ["deepens2"]="ImageNet-ResNet50-1132c260ef75--1564493784-1.pth.tar"
  ["deepens3"]="ImageNet-ResNet50-2f817072e8da--1564493734-1.pth.tar"
  ["deepens4"]="ImageNet-ResNet50-3177c697fbf4--1564495013-1.pth.tar"
  ["deepens5"]="ImageNet-ResNet50-628e11f9fd67--1564481099-1.pth.tar"
  ["deepens6"]="ImageNet-ResNet50-743e10f26a38--1564493675-1.pth.tar"
  ["deepens7"]="ImageNet-ResNet50-7ded66ec9900--1564481097-1.pth.tar"
  ["deepens8"]="ImageNet-ResNet50-8fc5076a66c9--1564481079-1.pth.tar"
  ["deepens9"]="ImageNet-ResNet50-a58ab8dd26fc--1564492521-1.pth.tar"
  ["deepens10"]="ImageNet-ResNet50-a80e40d84db2--1564492573-1.pth.tar"
  ["deepens11"]="ImageNet-ResNet50-be11903315ee--1564481101-1.pth.tar"
  )

declare -A ENSEMBLEDIR2=(
  ["deepens1"]="ImageNet-ResNet50-d08f2c418c5e--1564481091-1.pth.tar"
  ["deepens2"]="ImageNet-ResNet50-d8e55e99f790--1564481165-1.pth.tar"
  ["deepens3"]="ImageNet-ResNet50-d96fb844ee86--1564481152-1.pth.tar"
  ["deepens4"]="ImageNet-ResNet50-d9e083bd799e--1564492477-1.pth.tar"
  ["deepens5"]="ImageNet-ResNet50-f40d314de7e4--1564481095-1.pth.tar"
  ["deepens6"]="ImageNet-ResNet50-fea7f0f2b737--1564493491-1.pth.tar"
  )
#dataset_name="imagenet"

declare -A ENSEMBLE_ID=(
["deepensa"]=ENSEMBLEDIR1
["deepensb"]=ENSEMBLEDIR2
)

for deepensn in "deepens1" "deepens2" "deepens3" "deepens4" "deepens5"
do
for model in "resnet50" #"resnet101" "efficientnet_b0" "wide_resnet50_2" "wide_resnet101_2" "efficientnet_b1" "efficientnet_b2"
do
for dataset_name in "imagenet" "imagenetv2mf"
do
dataset_dir="${DATASETDIR[$dataset_name]}"
#echo "${dataset_name}:${dataset_dir}"
store_logits_fname="$OUTPUTDIR/${model}--${dataset_name}--${deepensn}.hdf5"
echo "Output will be stored in ${store_logits_fname}"
resume="${ENSEMBLE_BASEDIR}/${ENSEMBLEDIR1[$deepensn]}"
echo "$resume"
python train_imagenet.py ${dataset_dir} --arch=${model} \
  --workers=${workers} \
  --resume=${resume} \
  --evaluate \
  --batch-size ${batch_size} \
  --store_logits ${store_logits} \
  --store_logits_fname ${store_logits_fname} \
  #--gpu ${gpu} \

done
done
done