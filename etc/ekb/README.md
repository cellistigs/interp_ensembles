

# Train/eval models on imagenet

We use three approaches to get logits from models trained on imagenet.




1. [x] Train/evaluate models on scratch using [pytorch_examples](https://github.com/pytorch/examples/blob/master/imagenet/main.py)
```
# Bash scripts to train models locally (wo pytorch lightning)
train_imagenet_lion.sh
eval_scratch_imagenet.sh
eval_scratch_imagenet_c.sh

# Bash scripts to train models locally (w pytorch lightning)
train_imagenet_pl.sh
eval_pretrained_imagenet_pl.sh
eval_scratch_imagenet_alexnet.sh 

# Bash scripts to train models using slurm (in a remote server)
axon/train_imagenet_axon_debug.sh  # debug
axon/train_imagenet.sh 
axon/train_imagenet_sub.sh
axon/eval_pretrained_imagenet.sh

```

2. [x] Evaluate models pretrained on imagenet from pytorch:
```
eval_pretrained_imagenet.sh
eval_pretrained_imagenet_c.sh
```

3. [x] Evaluaye models trained on imagenet from different sources (follow the independent instructions): <br>
    1.[x] [Pitfalls of In-Domain Uncertainty Estimation and Ensembling in Deep Learning](https://github.com/SamsungLabs/pytorch-ensembles)
    2. [x] [Imagenet-testbed](https://github.com/modestyachts/imagenet-testbed)

    Then move logits into the desired format:
    ```
    download_imagenetv2.sh
    ```

# Plotting code

1. [x] plot metrics for each corruption 
```
plot_metrics_imagenet.sh
```
