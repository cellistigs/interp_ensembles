#!/bin/bash

# Train a model on tiny imagenet using pytorch lightning.
# on local server (lion)



# ------ set outputdirectory ------
OUTPUTDIR="/home/ubuntu/interp_ensembles/results/imagenet/checkpoints"
#OUTPUTDIR="/data/Projects/linear_ensembles/interp_ensembles/etc/ekb/pl_imagenet/lightning_logs"
mkdir -p $OUTPUTDIR

# ----- set dataset directories -----
#BASEDATA_DIR="/home/ekellbuch/pytorch_datasets"
BASEDATA_DIR="/home/ubuntu/data"


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


now=$(date +"%m_%d_%Y/%H_%M_%S")
echo "$now"

# Pick a model
gpus=4
batch_size=250
workers=24 # number of cpus
#auto_select_gpus=True #it doesn't appear to be working

profiler='simple' # can't tell if this slows things down and by how much
deterministic=1
accelerator='ddp' ## choose the ddp accelerator to allow for effective use of multiple workers. We also add functionality to train_imagenet_pl.py to optimize ddp accelerator. 
#log_every_n_steps=50 # tensorboard logging checkpoint is harcoded to at least 10
max_epochs=90 #93
#save_top_k=-1

dataset_name="imagenet"
for seed in 2
do
#for model in "resnet50" "resnet101" "efficientnet_b0" "wide_resnet50_2" "wide_resnet101_2" "efficientnet_b1" "efficientnet_b2"
for model in "resnet101" #"densenet121" "googlenet" "resnet18" "vgg11" "vgg13"

do
dataset_dir="${DATASETDIR[$dataset_name]}"
echo "${dataset_name}:${dataset_dir}"

default_root_dir="$OUTPUTDIR/${model}--${dataset_name}/${now}"
#echo "Output will be stored in ${store_logits_fname}"
python /home/ubuntu/interp_ensembles/scripts/train_imagenet_pl.py --data-path ${dataset_dir}  \
  --gpus=${gpus} \
  --arch=${model} \
  --seed=${seed} \
  --deterministic=${deterministic} \
  --batch-size=${batch_size} \
  --workers=${workers} \
  --max_epochs=${max_epochs} \
  --default_root_dir=${default_root_dir} \
  --accelerator=${accelerator}  
#  --save_top_k=${save_top_k} \
#  --profiler=${profiler} \
#  --auto_select_gpus=${auto_select_gpus} \
#  --store_logits ${store_logits} \
#  --store_logits_fname ${store_logits_fname} \
#  --log_every_n_steps=${log_every_n_steps} \

done
done
