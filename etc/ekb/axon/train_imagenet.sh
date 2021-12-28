#!/usr/bin/env bash

#name the job pybench33 and place it's output in a file named slurm-<jobid>.out
# allow 40 minutes to run (it should not take 40 minutes however)
# set partition to 'all' so it runs on any available node on the cluster

#SBATCH -J 'slurm_ekb'
#SBATCH -o slurm_ekb-%j.out
#SBATCH -t 8-00:00:00
#SBATCH --mem 32gb
#SBATCH --gres=gpu:3
#SBATCH -c 5
#SBATCH --mail-type=ALL            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=ekb2154@columbia.edu         # Where to send mail (e.g. uni123@columbia.edu)
. activate interp


echo "train_imagenet_pl.py $1"
python  /home/ekb2154/data/Projects/linear_ensembles/interp_ensembles/scripts/train_imagenet_pl.py $1
echo "Ran train_imagenet_pl.py $1"
