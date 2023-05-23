# References 
 
## Foundational work on ensembling

- T. P. Minka, 2000; "Bayesian model averaging is not model combination". (argues that ensembling is different than Bayesian inference)
- Dietterich, 2000; "Ensemble Methods in Machine Learning" https://link.springer.com/chapter/10.1007/3-540-45014-9_1 : A survey of different kinds of ensembling methods (Bayesian Model Averaging, AdaBoost, Bagging, Feature Randomization) and how they address the shortcomings of constituent classifiers. Identifies three different perspectives from which ensembles can make a difference: 
    - *Statistical*: In any context where the amount of training data you have is not sufficient to identify a single prediction function, you have a statistical problem on your hands: how do you consider all of the possible valid hypotheses? Ensembles can help because averaging different classifiers reduces the risk of choosing a classifier that does not generalize well. 
    - *Computational*: Even without a statistical problem to consider, there is a computational one: often times, classifiers may do local search on a cost that has many local optima.How do you know that you can find the best solution in this case? (You can't). By considering many different classifiers, we can hope to avoid this issue by visiting many local optima.
    - *Representational*: This is the most subtle point. It may be the case that the true prediction function you want to consider is not inside your hypothesis space. This could be because of algorithm design- perhaps the true hypothesis is a function on the reals but your classifier only outputs integers. It could also be practical- even if we know that certain algorithms have hypothesis classes that are the space of all classifiers (e.g. from universal function approximation theorems), with a finite data size we are limited in the space of hypotheses we can actually consider.    
    - Next the paper gives an overview of the different ways in which you might construct an ensemble. Something I've not seen before is error-correcting output coding- construct an ensemble of binary classifiers, each of which will output "1" on a subset of the class labels, and "0" on all the rest . We then sum across the predictions of all classifiers, and the class with the highest sum wins. There is a literature on the choice of these partitions such that we can minimize probability of error.   
    - Finally, Dietterich discusses how different ensembling methods might target different perspectives:  
        - When there is label noise, AdaBoost overfits. This is because unlike other ensembling methods, it is directly optimizing for the consensus output, and thereby "making a direct assault on the representational problem". This can lead to overfitting issues when there's label noise because you're optimizing over a potentially much larger set of hypotheses (is this true for deep ensembles?) 
        - When datasets are very large, Randomization (i.e. perturbing activations, dropout, random initialization) is expected to do better than bootstrap aggregation due to decreasing diversity in the training data- presumably at some point your sample captures enough of the data variance that it doesn't matter so much if you lose some samples.  
    - Stepping back from this final section, let's think about how much these points apply to our deep network setting. Interestingly, it looks like many of the empirical results and observations about deep ensembles are still in line with the results shown here (directly optimizing the ensemble is not as good; bagging does not work well), but the intutions are very different.  
        - One thing to consider is the parameter regime we are in. We are no longer thinking about more parameters as a bad thing for single models, but we still know that directly optimizing the ensemble cost doesn't work as well as training independent models. Is this more of a "computational" problem than a statistical one? 
        - We know that for large image datasets and deep nets, (CIFAR, Imagenets) bagging does not seem to work as well. However the popular interpretation of this result is not that we have enough data that we are no longer generating diversity, but rather that we suffer due to the loss of data that could help improve performance. 
- TOREAD Dietterich, 2000; "An Experimental Comparison of Three Methods for Constructing Ensembles of Decision Trees: Bagging, Boosting, and Randomization" https://link.springer.com/article/10.1023/A:1007607513941

- TOREAD Schapire, 1990: "The Strength of Weak Learnability" https://link.springer.com/content/pdf/10.1007/BF00116037.pdf
- TOREAD Freund, 1995: "Boosting a Weak Learning Algorithm by Majority" https://www.sciencedirect.com/science/article/pii/S0890540185711364
- TOREAD Breiman, 1996: "Bagging Predictors" https://www.stat.berkeley.edu/~breiman/bagging.pdf
- TOREAD Bartlett, 1998: "Boosting the margin: a new explanation for the effectiveness of voting methods" https://projecteuclid.org/journals/annals-of-statistics/volume-26/issue-5/Boosting-the-margin--a-new-explanation-for-the-effectiveness/10.1214/aos/1024691352.full
- TOREAD Schapire, 1999: "Improved Boosting Algorithms Using Confidence-rated Predictions" https://link.springer.com/article/10.1023/A:1007614523901
- L. Breiman. Random forests. Machine learning, 45(1):5–32, 2001. (Historic ensemble paper)
- T. P. Minka. Bayesian model averaging is not model combination. 2000. (argues that ensembling is different than Bayesian inference)
- R. Caruana et al. Ensemble Selection from Libraries of Models, ICML 2004. (Classic paper on how to construct ensembles.) 


## How do we understand improvements to OOD? 

- Sagawa et al. 2020: "An investigation into why overparametrization exacerbates spurious correlations" Looks at initialization-independent effects of memorized, bad examples. https://www.readcube.com/library/92a3b081-e383-4a2a-8633-10cd21a9b02e:30071f71-3a16-4949-88a7-264d5898f837k
- Andreassen et al. 2021; "The Evolution of Out-of-Distribution Robustness Throughout Fine-Tuning:" arxiv 2021. Investigation of Effective Robustness and its relationship to pre-training. 
    - ER is a measurement of qualitative improvements in robustness. 
        - Shape of the ER curve is itself an interesting empirical measurement. Why is it convex? Is this explained by theory (Mania and Sra 2020), or is it rather a feature of datasets with long tails? (Feldman et al.)
            - Related- what happens to a dataset without a long tail?  
    - Pretraining: under the view of a pretrained model, InD and OoD data as discussed here should look basically the same. The effective peak that we see in plots is more likely a monotonic regression to the baseline ER relative from a good initialization given by the pretraining data. 
        - Compare to Ref [41]: are there pretrained models here? 
    - Just finetuning the last layers/using a memory buffer to maintain performance on old data doesnt work. You can use the same feature extractors, or preserve performance on an OOD dataset that's further away and still do bad in terms of ER. 
        - How about distillation using the softmax probabilities from your original training set?  
    - This gives us a great evaluation protocol.
- Miller et al. 2021: "Accuracy on the Line: On the Strong Correlation Between Out-of-Distribution and In-Distribution Generalization (https://arxiv.org/pdf/2107.04649.pdf)

### Refs from Andreassen et al. 2021

- H. Mania and S. Sra 2020; "Why do classifier accuracies show linear trends under distribution shift?" arxiv 2020: Theoretical analysis of distribution shift. Important for framework understanding of model similarity the authors propose two metrics- one of dataset similarity, and one of dominance probability (probability that a worse model gets samples right that a better model fails on). From this perspective, it's not at all surprising that ensembles do better. 
- 
- TOREAD H. Mania et al. 2019; "Model similarity mitigates test set overuse" arxiv 2019: empirical investigation of linear trend. 
- TOREAD Radford et al. 30; Learning transferable visual models from natural language supervision. arxiv 2021: Introduction and analysis of CLIP models. CLIP effective ER during fine tuning. 
- TOREAD Kolesnikov et al. 2020- Big transfer (bit): General visual representation learning, 2020: the BiT model is a generic benchmark we should study too.

## Metrics for Ensemble Quatification

- D'Amour et al. 2020 discuss "stress tests" to identify differences between models that have the same performance on in-distribution training and validation data- These could be useful for evaluating ensembles vs. single models as well. 
- Bröcker, Jochen. "Reliability, sufficiency, and the decomposition of proper scores." Quarterly Journal of the Royal Meteorological Society: A journal of the atmospheric sciences, applied meteorology and physical oceanography 135.643 (2009): 1512-1519: Discusses the fact that all proper scoring rules will induce a calibration metric. 
- Gneiting, T. and Raftery, A. E. Strictly proper scoring rules, prediction, and estimation. Journal of
the American Statistical Association, 102(477):359–378, 2007.: Each proper scoring rule comes with an entropy metric (shannon entropy for log probs). 

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
- Kumar et al., 2021; "Calibrated Ensembles: A Simple Way to Mitigate ID-OOD Accuracy Tradeoffs"
  - **Note: ICLR 2022 submission version has more details, intuition.** 
  - Background: if we use methods of robustifying models (additional pretraining, only training the last layer, zero shot clip), we suffer a decrease in in distribution accuracy in order to improve out of distribution accuracy. If we use standard model training, we are able to get good in distribution accuracy, but bad out of distribution accuracy. If we (first calibrate) and then ensemble these two different models together, we can get the best of both worlds. 
  - Claim: It's actually a LOT harder to get independent errors on OOD data than you might expect. This method is better than standard ensembling because standard ensembles make a class of systematic errors on OOD data that do not depend upon random initialization. This makes OOD performance not as good as it potentially could be, and aligns with resuts seen in Sagawa et al. 2020b. By ensembling models that somehow do not share this same failure mode, the authors claim that they are able to recover performance that is not seen in a standard ensembling setting.  
  - Claim: By calibrating both standard and robust models on in distribution data, we somehow normalize the uncertainty estimates (even f they are still miscalibrated) relative to one another. This sets up an understanding in terms of relative calibration- the standard model and the robust model can both be miscalibrated, but if the better model is in general better calibrated then there will still be an improvement relative to single model performance. 
  - Takeaway: In short, I think this is a natural next step from the Andresassen et al. 2020 paper. In response to the observation that it's impossible to maintain high effective robustness and accuracy in a single model, the natural thing to do would be to ensemble them (as is effectively done here). I don't find the claim that we need calibration to be all that convincing from looking at Table 4. It seems like the most important thing they've found in this work is that ensembling models that are trained differently gives a lot better performance than models that are trained with the same loss. In particular, I think considering the baseline of ensembling a fine tuned model with one that is solely pretrained in the final layer would be interesting in a representation learning framework.
  - How do we square this with the observation that bootstrapping does not work as well as simple model averaging?
  - *It would be interesting to try bootstrapping pretrained ensembles and to see if that makes any difference.*   

## UQ and ensembles

- Gal, Yarin, and Zoubin Ghahramani. "Dropout as a bayesian approximation: Representing model uncertainty in deep learning." international conference on machine learning. PMLR, 2016.
- Lakshminarayanan, Balaji, Alexander Pritzel, and Charles Blundell. "Simple and scalable predictive uncertainty estimation using deep ensembles." arXiv preprint arXiv:1612.01474 (2016).
- A.G. Wilson and P. Izmailov. Bayesian Deep Learning and a Probabilistic Perspective of Generalization. NeurIPS 2020. (Argues that deep ensembles approximate Bayesian model averaging).
- Amodei, Dario, et al. "Concrete problems in AI safety." arXiv preprint arXiv:1606.06565 (2016).
- Ovadia et al., 2019; "Can You Trust Your Model's Uncertainty?"
  - Ensembles obtain the best accuracy and calibration on corrupted datasets (CIFAR10-C)
  - Ensembles work best for tasks like selective classification (e.g. measuring accuracy if we restrict the model to only make predictions when p(y|x) > threshold)
- TOREAD Minderer et al., 2021; "Revisiting the Calibration of Modern Neural Networks"
- F. Gustafsson et al. Evaluating Scalable Bayesian Deep Learning Methods for Robust Computer Vision. CVPR Workshops, 2020. (Similar analysis to the Ovadia paper)

## Ensembles and diversity

- Fort et al., 2019; "Deep Ensembles: A Loss Landscape Perspective"
- Lee et al. 2016; "Stochastic Multiple Choice Learning for Training Diverse Deep Ensembles", arxiv 2016. If you have an oracle that can correct multiple competing hypotheses downstream, it can be a good idea to learn multiple likely outcomes instead of a single one. They introduce a loss, stochastic Multiple Choice Learning (sMCL) in which one considers an ensemble of models, and trains them together, but only propagates the error to the model that currently has the lowest loss on any given example. Does better than classical ensembles with oracle evaluation. 

## When do ensembles do "better" than single large models? 

- Kondratyuk et al. 2020; "When Ensembling Smaller Models is More Efficient than Single Large Models", arxiv 2020. Demonstration of the fact that in Imagenet and Cifar-10, you can perform better with an ensemble of smaller models with fewer flops than a larger model with more flops. The best thing to do seems to be to take N of the same network and ensemble those, at least for classification on benchmark datasets. 

## Training ensembles/Ensemble and single model hybrids 

- Wen et al. 2020; "BatchEnsemble: An Alternative Approach to Efficient Ensemble and Lifelong Learning", arxiv 2020. Ensembles are expensive, and it's hard to decouple their effectiveness (probably diversity) from their effectiveness. This paper suggests an approach in which one constructs an ensemble by hadamard-multiplying a base set of weights with a set of N different rank 1 matrices, and training the result with gradient descent. This is a different combination of a base network with a structured perturbation to what we propose.
- Warde-Farley et al. 2014: "An empirical analysis of dropout in piecewise linear networks". arxiv 2014. [link](https://arxiv.org/pdf/1312.6197.pdf). This paper analyzes the interpretation of dropout in ReLU networks as creating an exponentially large ensemble of networks that share parameters. The relevant component of their work for this section is an alternaltive loss that they introduce- "dropout boosting", wherein they update parameters only for the subnetwork that is active at the moment, but evaluate the entire ensemble loss instead of the subnet loss. This is "boosting" in the sense that we are forming "a direct assault on the representational problem" and asking the network to fit the ensemble cost. Definitely an interesting citation and one for us to consider in analyzing the interpolating ensemble/mean field ensemble. Note however, that in this context the authors saw that this cost did worse than dropout, and only as well as the standard network trained with SGD. We see that our models do no differently than ensembling.  
- TOREAD Havasi et al. 2021; "Training independent subnetworks for robust prediction", ICLR 2021.
- TOREAD Huang et al. 2016; "Deep Networks with Stochastic Depth", ECCV 2016. An alternative to wide networks is to use blockwise dropout- this is like implicitly training an ennsemble of networks too. 

## Partitioning convolutional filters along the channel dimension: 

- Krizhevsky et al 2012 (at least this is the common citation) propose group convolution- if your filters are of dimension Height,Width,Channel, and your data is of dimension Batch,Height,Width,Channel, group convolution with N groups means that the first 1/N channels in your filters see the first 1/N channels in your data, and so on for the other groups. This already led to interesting results in this paper, where one group learned black and white filters and the other learned color filters. 
- Xie et al. 2017 bring this idea back to the forefront with their Res-NeXt architecture- group convolutions that split up the ResNet filters into small self contained modules, which the authors identify as a "cardinality" domain. In order to compare networks of similar cardinality, we increase the channel dimension when we perform group convolution. However, not channels are duplicated (the output layer in each unit has the same size as the corresponding ResNet layer).   
 
## Wide networks: 

- Zagoruyko et al. 2017 propose wide-resnet: increasing the channel dimension of resnets, and showing that this can improve performance. However, when they go to deeper networks for ImageNet experiments, they only widen the middle layer of their bottleneck blocks, not the outer ones. They also emphasize that dropout is an important training aid for their situation.  
