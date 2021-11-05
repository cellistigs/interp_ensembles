# References 
 
## Foundational work on ensembling

- TOREAD Dietterich, 2000; "Ensemble Methods in Machine Learning" https://link.springer.com/chapter/10.1007/3-540-45014-9_1 
- TOREAD Dietterich, 2000; "An Experimental Comparison of Three Methods for Constructing Ensembles of Decision Trees: Bagging, Boosting, and Randomization" https://link.springer.com/article/10.1023/A:1007607513941

- TOREAD Schapire, 1990: "The Strength of Weak Learnability" https://link.springer.com/content/pdf/10.1007/BF00116037.pdf
- TOREAD Freund, 1995: "Boosting a Weak Learning Algorithm by Majority" https://www.sciencedirect.com/science/article/pii/S0890540185711364
- TOREAD Breiman, 1996: "Bagging Predictors" https://www.stat.berkeley.edu/~breiman/bagging.pdf
- TOREAD Bartlett, 1998: "Boosting the margin: a new explanation for the effectiveness of voting methods" https://projecteuclid.org/journals/annals-of-statistics/volume-26/issue-5/Boosting-the-margin--a-new-explanation-for-the-effectiveness/10.1214/aos/1024691352.full
- TOREAD Schapire, 1999: "Improved Boosting Algorithms Using Confidence-rated Predictions" https://link.springer.com/article/10.1023/A:1007614523901

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

- H. Mania and S. Sra 2020; "Why do classifier accuracies show linear trends under distribution shift?" arxiv 2020: Theoretical analysis of distribution shift. Important for framework understanding of model similarity the authors propose two metrics- one of dataset similarity, and one of dominance probability (probability that a worse model gets samples right that a better model fails on). From this perspective, it's not at all surprising that ensembles do better. 
- 
- TOREAD H. Mania et al. 2019; "Model similarity mitigates test set overuse" arxiv 2019: empirical investigation of linear trend. 
- TOREAD Radford et al. 30; Learning transferable visual models from natural language supervision. arxiv 2021: Introduction and analysis of CLIP models. CLIP effective ER during fine tuning. 
- TOREAD Kolesnikov et al. 2020- Big transfer (bit): General visual representation learning, 2020: the BiT model is a generic benchmark we should study too.

## "Robustness" (as measured by accuracy on a shifted/OOD dataset)

- Taori et al., 2020; "Measuring Robustness to Natural Distribution Shifts in Image Classification"
  - Re-introduces the linear trend relating InD accuracy and shifted/OOD accuracy
  - Introduces the notion of "effective robustness," which is when a model's OOD accuracy lies above the linear trend
  - Of the 204 ImageNet models/robust methods they test, *no method is effectively robust* on ImageNet V2
  - **Note:** they **do not** test ensembles.
- (Follow up) Miller et al., 2021; "Accuracy on the Line: On the Strong Correlation Between OOD and InD Generalization"
  - Extends Taori et al. to more datasets and shifts (train on CIFAR-10 or ImageNet, test on synthetic/real world shifts)
  - They only test standard models (no robust methodology)
  - Linear trends exist for almost all InD/OOD pairs that they test.
     - There are 2 pairs where a linear trend doesn't exist, and they offer explanations for this
  - **Note:** they **do not** test ensembles.

## UQ and ensembles

- TOREAD Ovadia et al., 2019; "Can You Trust Your Model's Uncertainty?"
  - Ensembles obtain the best accuracy and calibration on corrupted datasets (CIFAR10-C)
  - Ensembles work best for tasks like selective classification (e.g. measuring accuracy if we restrict the model to only make predictions when p(y|x) > threshold)
- TOREAD Minderer et al., 2021; "Revisiting the Calibration of Modern Neural Networks"

## When do ensembles do "better" than single large models? 

- Kondratyuk et al. 2020; "When Ensembling Smaller Models is More Efficient than Single Large Models", arxiv 2020. Demonstration of the fact that in Imagenet and Cifar-10, you can perform better with an ensemble of smaller models with fewer flops than a larger model with more flops. The best thing to do seems to be to take N of the same network and ensemble those, at least for classification on benchmark datasets. 

## Training ensembles/Ensemble and single model hybrids 

- Lee et al. 2016; "Stochastic Multiple Choice Learning for Training Diverse Deep Ensembles", arxiv 2016. If you have an oracle that can correct multiple competing hypotheses downstream, it can be a good idea to learn multiple likely outcomes instead of a single one. They introduce a loss, stochastic Multiple Choice Learning (sMCL) in which one considers an ensemble of models, and trains them together, but only propagates the error to the model that currently has the lowest loss on any given example. Does better than classical ensembles with oracle evaluation. 
- Wen et al. 2020; "BatchEnsemble: An Alternative Approach to Efficient Ensemble and Lifelong Learning", arxiv 2020. Ensembles are expensive, and it's hard to decouple their effectiveness (probably diversity) from their effectiveness. This paper suggests an approach in which one constructs an ensemble by hadamard-multiplying a base set of weights with a set of N different rank 1 matrices, and training the result with gradient descent. This is a different combination of a base network with a structured perturbation to what we propose.
- TOREAD Havasi et al. 2021; "Training independent subnetworks for robust prediction", ICLR 2021.
- TOREAD Huang et al. 2016; "Deep Networks with Stochastic Depth", ECCV 2016. An alternative to wide networks is to use blockwise dropout- this is like implicitly training an ennsemble of networks too. 

## Partitioning convolutional filters along the channel dimension: 

- Krizhevsky et al 2012 (at least this is the common citation) propose group convolution- if your filters are of dimension Height,Width,Channel, and your data is of dimension Batch,Height,Width,Channel, group convolution with N groups means that the first 1/N channels in your filters see the first 1/N channels in your data, and so on for the other groups. This already led to interesting results in this paper, where one group learned black and white filters and the other learned color filters. 
- Xie et al. 2017 bring this idea back to the forefront with their Res-NeXt architecture- group convolutions that split up the ResNet filters into small self contained modules, which the authors identify as a "cardinality" domain. In order to compare networks of similar cardinality, we increase the channel dimension when we perform group convolution. However, not channels are duplicated (the output layer in each unit has the same size as the corresponding ResNet layer).   
 
## Wide networks: 

- Zagoruyko et al. 2017 propose wide-resnet: increasing the channel dimension of resnets, and showing that this can improve performance. However, when they go to deeper networks for ImageNet experiments, they only widen the middle layer of their bottleneck blocks, not the outer ones. They also emphasize that dropout is an important training aid for their situation.  
