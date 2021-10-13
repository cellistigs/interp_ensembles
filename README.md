# Project with Geoff on investigating the effectiveness of ensembles compared to bigger models. 

We all love deep ensembles because they are great for out of distribution detection and covering a diverse set of solutions. However, these benefits, that are supposedly selectively targeted by ensembles (due to diversity among ensemble members, independent errors, etc) are also usually accompanied by an increase in performance on in-distribution data. Do we actually do better in a qualitatively different way when we ensemble, as opposed to when we just properly train a much larger model? 

We can map this out quantitatively: consider a plot of in distribution accuracy vs. out of distribution accuracy. Usually, in the literature there is a linear relation between the two (cite). Are we just pushing further up this 1-d relation when we ensemble, or are are we exploring a different part of the in-distribution/out-of-distribution space?  

Likewise, there have been recent results showing that increasing capacity of CNNs can outperform an ensemble sometimes, and underperform other times (cite). What governs this tradeoff? 

## How do we probe this question? 

Idea: consider a framework in which you have a model that is twice as wide as a standard model. This model actually has 4x the number of parameters of the narrow model: we can imagine assigning identical parameters to model $A$ and $B$, and for any group of connected parameters ${1,2}$, there are connections $A1>A2,B1>B2,A1>B2,B1>A2$.
This model can simultaneously be considered as a single, big model of size 2x ($\math{phi}(x)$), and four small models $\math{varphi}_i(x)$(straight connections, zigzag connections).
It can also be *trained* in these two different ways. What happens if we write the prediction of this model as a convex combination of these, $\lambda\math{phi}(x)+(1-\lambda)\sum_i\math{varphi}_i(x)$? We can train on this model, either by choosing a deterministic lambda or by sampling one of the small models to recieve a gradient (in addition to? instead of?) the small model. This has a reparametrization-trick flavor. By doing so at different lambdas, we can determine examine how the in distribution and out of distribution accuracy changes a function of the number of parameters you are using. We can also repeat this at different model capacities: there should be different regimes for us to consider where ensembles do better or worse than a single model on any given dataset.    

Pros: we can probably use similar hyperparameters, we know both ends of this spectrum work, and it would be interesting to see what performance looks like as a function of lambda no matter what the results are. 

Write down this ensembling thought too. 

Todos: 
- [ ] read papers that geoff sent (Friday 10/15)
- [ ] choose datasets/models where we expect interesting behavioral transitions. (Tuesday 10/17)
- [ ] investigate implementation of wide models in pytorch. (Week of 10/17)
    - [ ] is it easier to build as n subnetworks that can be combined, or one big network that can be taken apart? 
- [ ] prototype infrastructure for training these things. 

## Bibiliography: 

### Ensembles vs. Big Models in the literature: 
- Kondratyuk, Dan, et al. "When ensembling smaller models is more efficient than single large models." arXiv preprint arXiv:2005.00570 (2020). [link](https://arxiv.org/abs/2005.00570V)
- Wasay, Abdul, and Stratos Idreos. "More or Less: When and How to Build Convolutional Neural Network Ensembles." International Conference on Learning Representations. 2020.[link](https://openreview.net/forum?id=z5Z023VBmDZ)
- Lobacheva, Ekaterina, et al. "On power laws in deep ensembles." arXiv preprint arXiv:2007.08483 (2020). [link](https://arxiv.org/abs/2007.08483)

### Model Capacity and unintuitive behavior:
- Nakkiran, Preetum, et al. "Deep double descent: Where bigger models and more data hurt." arXiv preprint arXiv:1912.02292 (2019). [link](https://arxiv.org/abs/1912.02292)

### Interpolating between different models:  
- Benton, Gregory W., et al. "Loss Surface Simplexes for Mode Connecting Volumes and Fast Ensembling." arXiv preprint arXiv:2102.13042 (2021). [link](https://arxiv.org/abs/2102.13042)


