#!/usr/bin/env bash

# call code to train imagenet
call_train() {
    echo "$1"
    sbatch -p ctn --qos=high-priority ./train_imagenet.sh "$1"
    #python /home/ekb2154/data/libraries/interp_ensembles/scripts/train_imagenet_pl.py $1
}

# ------ set outputdirectory ------
OUTPUTDIR="/share/ctn/users/ekb2154/data/libraries/interp_ensembles/results/pl_imagenet"
mkdir -p $OUTPUTDIR

# ----- set dataset directories -----
BASEDATA_DIR="${HOME}/pytorch_datasets"

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
batch_size=256
workers=16 # number of cpus
#auto_select_gpus=True #it doesn't appear to be working

profiler="simple" # can't tell if this slows things down and by how much
deterministic=0
#log_every_n_steps=50 # tensorboard logging checkpoint is harcoded to at least 10
max_epochs=90 #93
#save_top_k=-1

accelerator="ddp"
find_unused_parameters=0

dataset_name='imagenet'
for seed in 1 2 3 4
do
#for model in "resnet50" "resnet101" "efficientnet_b0" "wide_resnet50_2" "wide_resnet101_2" "efficientnet_b1" "efficientnet_b2"
for model in "alexnet" #"densenet121" "googlenet" "resnet18" "vgg11" "vgg13"

do
dataset_dir="${DATASETDIR[$dataset_name]}"
echo "${dataset_name}:${dataset_dir}"
default_root_dir="$OUTPUTDIR/${model}--${dataset_name}/${now}"
call_train "--data-path ${dataset_dir}  \
  --gpus=${gpus} \
  --arch=${model} \
  --seed=${seed} \
  --deterministic=${deterministic} \
  --batch-size=${batch_size} \
  --workers=${workers} \
  --max_epochs=${max_epochs} \
  --default_root_dir=${default_root_dir} \
  --accelerator=${accelerator}
  --profiler=${profiler} "
#  --find_unused_parameters=${find_unused_parameters}

done
done

