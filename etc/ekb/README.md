
Model to train imagenet is from [pytorch_examples](https://github.com/pytorch/examples/blob/master/imagenet/main.py)

Changes to main.py
[x] store logits


Main steps 
[x] install interp library on lion (debugging) and axon (run experiments). 
[x] map imagenet in lion for debugging 
[x] map imagenetv2* in lion for debugging, use download_imagenetv2.sh
[x] to train/eval  models on imagenet use scripts/train_imagenet.py


[x] to eval architectures on imagenet: eval_pretrained_imagenet.sh
[x] to eval architectures on imagenet_c: eval_imagenet_c.sh
[x] to train architectures use: train_imagenet_lion.sh
[x] merge code with main package
[x] download deep ensemble checkpoints from "https://github.com/SamsungLabs/pytorch-ensembles"
[x] evaluate resnet 50 models trained on different seeds on imagenet : eval_scratch_imagenet.sh
[x] evaluate resnet 50 models trained on different seeds on imagenet_c : eval_scratch_imagenet_c.sh

[x] plot metrics for each corruptions plot_metrics_imagenet.sh