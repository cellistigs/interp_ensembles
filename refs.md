# References 

## How do we understand improvements to OOD? 

- Andreassen et al. 2021; "The Evolution of Out-of-Distribution Robustness Throughout Fine-Tuning:" arxiv 2021. Investigation of Effective Robustness and its relationship to pre-training. 
    - ER is a measurement of qualitative improvements in robustness. 
        - Shape of the ER curve is itself an interesting empirical measurement. Why is it convex? Is this explained by theory (Mania and Sra 2020), or is it rather a feature of datasets with long tails? (Feldman et al.)
            - Related- what happens to a dataset without a long tail?  
    - Pretraining: under the view of a pretrained model, InD and OoD data as discussed here should look basically the same. The effective peak that we see in plots is more likely a monotonic regression to the baseline ER relative from a good initialization given by the pretraining data. 
        - Compare to Ref [41]: are there pretrained models here? 
    - Just finetuning the last layers/using a memory buffer to maintain performance on old data doesnt work. You can use the same feature extractors, or preserve performance on an OOD dataset that's further away and still do bad in terms of ER. 
        - How about distillation using the softmax probabilities from your original training set?  
    - This gives us a great evaluation protocol.

### Refs from Andreassen et al. 2021

- TOREAD H. Mania et al. 2019; "Model similarity mitigates test set overuse" arxiv 2019: empirical investigation of linear trend. 
- TOREAD **H. Mania and S. Sra 2020; "Why do classifier accuracies show linear trends under distribution shift?" arxiv 2020: Theoretical analysis of distribution shift. Important for framework understanding of model similarity- could be useful for ensembles** 
- TOREAD **Taori et al. 2020; "When robustness doesn't promote robustness: Synthetic vs. Natural distribution shifts on imagenet" 2020: def of effective robustness- evaluation of 204 imagenet models for effective robustness. Important bc it analyzes a great number of models.** It doesn't look like this includes pretrained models or ensembles. 
- TOREAD Radford et al. 30; Learning transferable visual models from natural language supervision. arxiv 2021: Introduction and analysis of CLIP models. CLIP effective ER during fine tuning. 
- TOREAD Kolesnikov et al. 2020- Big transfer (bit): General visual representation learning, 2020: the BiT model is a generic benchmark we should study too.  

## When do ensembles do "better" than single large models? 

- Kondratyuk et al. 2020; "When Ensembling Smaller Models is More Efficient than Single Large Models", arxiv 2020. Demonstration of the fact that in Imagenet and Cifar-10, you can perform better with an ensemble of smaller models with fewer flops than a larger model with more flops. The best thing to do seems to be to take N of the same network and ensemble those, at least for classification on benchmark datasets. 

## Training ensembles/Ensemble and single model hybrids 

- Lee et al. 2016; "Stochastic Multiple Choice Learning for Training Diverse Deep Ensembles", arxiv 2016. If you have an oracle that can correct multiple competing hypotheses downstream, it can be a good idea to learn multiple likely outcomes instead of a single one. They introduce a loss, stochastic Multiple Choice Learning (sMCL) in which one considers an ensemble of models, and trains them together, but only propagates the error to the model that currently has the lowest loss on any given example. Does better than classical ensembles with oracle evaluation. 
- Wen et al. 2020; "BatchEnsemble: An Alternative Approach to Efficient Ensemble and Lifelong Learning", arxiv 2020. Ensembles are expensive, and it's hard to decouple their effectiveness (probably diversity) from their effectiveness. This paper suggests an approach in which one constructs an ensemble by hadamard-multiplying a base set of weights with a set of N different rank 1 matrices, and training the result with gradient descent. This is a different combination of a base network with a structured perturbation to what we propose.  
- TOREAD Huang et al. 2016; "Deep Networks with Stochastic Depth", ECCV 2016. An alternative to wide networks is to use blockwise dropout- this is like implicitly training an ennsemble of networks too. 

## Partitioning convolutional filters along the channel dimension: 

- Krizhevsky et al 2012 (at least this is the common citation) propose group convolution- if your filters are of dimension Height,Width,Channel, and your data is of dimension Batch,Height,Width,Channel, group convolution with N groups means that the first 1/N channels in your filters see the first 1/N channels in your data, and so on for the other groups. This already led to interesting results in this paper, where one group learned black and white filters and the other learned color filters. 
- Xie et al. 2017 bring this idea back to the forefront with their Res-NeXt architecture- group convolutions that split up the ResNet filters into small self contained modules, which the authors identify as a "cardinality" domain. In order to compare networks of similar cardinality, we increase the channel dimension when we perform group convolution. However, not channels are duplicated (the output layer in each unit has the same size as the corresponding ResNet layer).   
 
## Wide networks: 

- Zagoruyko et al. 2017 propose wide-resnet: increasing the channel dimension of resnets, and showing that this can improve performance. However, when they go to deeper networks for ImageNet experiments, they only widen the middle layer of their bottleneck blocks, not the outer ones. They also emphasize that dropout is an important training aid for their situation.  
 
