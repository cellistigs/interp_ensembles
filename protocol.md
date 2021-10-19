
Implementation details:

Resnets: 
- torchvision.models v0.8.2 
- resnet50x1, R101x3, R152x4 
- torchvision models v0.8.2 

Benchmarks (Andreassen et al. 2021): 
- BiT models (S,M,L) 
- CLIP models (ViT-B-32)


Strategy for evaluation: 

1. Calculate/find the curve `$\beta(x)$`.: this is given on page 13 of Andreassen et al. 
2. Choose training dataset and both in and out of distribution test set. 
3. Train N models. Each epoch, get full responses of softmax. Evaluate on both the in and out of distribution test accuracy for individual models and the ensemble. 
4. Compare ood performance to benchmarks given. We can do full evaluation on each ood data point instead of just a scalar evaluation: what kind of ood patterns are we learning? 
5. see how ensembles stack up compared to Kondratyuk et al, 2020. 


