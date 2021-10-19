# References 

- Andreassen et al. 2021; "The Evolution of Out-of-Distribution Robustness Throughout Fine-Tuning:" arxiv 2021. Investigation of Effective Robustness and its relationship to pre-training. 
    - ER is a measurement of qualitative improvements in robustness. 
        - Shape of the ER curve is itself an interesting empirical measurement. Why is it convex? Is this explained by theory (Mania and Sra 2020), or is it rather a feature of datasets with long tails? (Feldman et al.)
            - Related- what happens to a dataset without a long tail?  
    - Pretraining: under the view of a pretrained model, InD and OoD data as discussed here should look basically the same. The effective peak that we see in plots is more likely a monotonic regression to the baseline ER relative from a good initialization given by the pretraining data. 
        - Compare to Ref [41]: are there pretrained models here? 
    - Just finetuning the last layers/using a memory buffer to maintain performance on old data doesnt work. You can use the same feature extractors, or preserve performance on an OOD dataset that's further away and still do bad in terms of ER. 
        - How about distillation using the softmax probabilities from your original training set?  
    - This gives us a great evaluation protocol.

## Refs from Andreassen et al. 2021

- H. Mania et al. 2019; "Model similarity mitigates test set overuse" arxiv 2019: empirical investigation of linear trend. 
- **H. Mania and S. Sra 2020; "Why do classifier accuracies show linear trends under distribution shift?" arxiv 2020: Theoretical analysis of distribution shift. Important for framework understanding of model similarity- could be useful for ensembles** 
- **Taori et al. 2020; "When robustness doesn't promote robustness: Synthetic vs. Natural distribution shifts on imagenet" 2020: def of effective robustness- evaluation of 204 imagenet models for effective robustness. Important bc it analyzes a great number of models.** 
- Radford et al. 30; Learning transferable visual models from natural language supervision. arxiv 2021: Introduction and analysis of CLIP models. CLIP effective ER during fine tuning. 
- Kolesnikov et al. 2020- Big transfer (bit): General visual representation learning, 2020: the BiT model is a generic benchmark we should study too.  

Comments on Andreassen et al. 2021
- ER due to pretraining is weird. Most of the time, I would expect that the in and out of distribution datasets look incredibly similar under pretraining compared to the actual pretraining dataset. In this context, you're probably seeing something of a memory effect. 

- Kondratyuk et al. 2020; "When Ensembling Smaller Models is More Efficient than Single Large Models", arxiv 2020. Demonstration of the fact that in Imagenet and Cifar-10, you can perform better with an ensemble of smaller models with fewer flops than a larger model with more flops. The best thing to do seems to be to take N of the same network and ensemble those, at least for classification on benchmark datasets. 


