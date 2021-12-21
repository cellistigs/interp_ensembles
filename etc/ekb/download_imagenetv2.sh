#!/bin/bash


DATADIR="/datahd3a/datasets/pytorch_datasets/imagenetv2"

pushd $DATADIR
# wget from http://imagenetv2public.s3-website-us-west-2.amazonaws.com/
# tar -xf each
#  mv /datahd3a/datasets/IMAGENET_v2/imagenetv2-*-format-val .

# run imagenetv2.py

mkdir -p imagenetv2-b-33
mkdir -p imagenetv2-a-44
mkdir -p imagenetv2-c-12

mv imagenetv2-matched-frequency-format-val imagenetv2-b-33/val
mv imagenetv2-threshold0.7-format-val imagenetv2-a-44/val
mv imagenetv2-top-images-format-val imagenetv2-c-12/val

# remove tar files
popd
