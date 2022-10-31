
Model to train imagenet is from [pytorch_examples](https://github.com/pytorch/examples/blob/master/imagenet/main.py)

Changes to main.py
[x] store logits


Main steps:

[x] install interp library on lion (debugging) and axon (run experiments). <br>
[x] map imagenet in lion for debugging.  <br>
[x] map imagenetv2* in lion for debugging, use download_imagenetv2.sh  <br>
[x] to train/eval  models on imagenet use scripts/train_imagenet.py  <br>


[x] to eval architectures on imagenet: eval_pretrained_imagenet.sh  <br>
[x] to eval architectures on imagenet_c: eval_imagenet_c.sh  <br>
[x] to train architectures use: train_imagenet_lion.sh  <br>
[x] merge code with main package  <br>
[x] download deep ensemble checkpoints from "https://github.com/SamsungLabs/pytorch-ensembles"  <br>
[x] evaluate resnet 50 models trained on different seeds on imagenet: eval_scratch_imagenet.sh  <br>
[x] evaluate resnet 50 models trained on different seeds on imagenet_c: eval_scratch_imagenet_c.sh  <br>
[x] evaluate alexnet models trained on different seeds on imagenet using pytorch lightning: eval_scratch_imagenet_alexnet.sh  <br>

[x] plot metrics for each corruptions plot_metrics_imagenet.sh  <br>

[x] add code train imagenet w pytorch lightning: train_imagenet_pl.py  <br>
[x] add code call imagenet trainer in pl: train_imagenet_pl.sh  <br>
[x] add code call imagenet eval in pl: eval_pretrained_imagenet_pl.sh  <br>


Remote server steps:

[x] eval models on imagenet on remote server: axon/eval_pretrained_imagenet.sh  <br>
[x] debug training code tiny-imagenet: train_imagenet_axon_debug.sh  <br>
[x] slurm script to train model on in remote server: train_imagenet.sh  <br>
[x] caller to slurm script to train model on (tiny)imagenet in remote server: train_imagenet_sub.sh  <br>

Plotting code:

Follow plot/store_.sh to make pandas df which are used for plotting. Then run:
[] plot/plot_metrics.py to make linear trend figs.  <br>