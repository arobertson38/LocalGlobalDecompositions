# LocalGlobalDecompositions

THIS REPOSITORY IS CURRENTLY IMCOMPLETE. IT WILL BE COMPLETELY POPULATED BY MAY 19TH, 2023. 

When the paper is published, this repository will contain research code for generation and training of the generative models proposed in Robertson et al.'s "Local-Global Decompositions for Conditional Microstructure Generation".

However, this code is definitely "research code". It was developed during the research process and, as a result, is rather messy. As a result, we also direct the interested reader to "https://github.com/arobertson38/DiffusionGeneration" for a cleaner version which is under will be continually improved. 


The following are brief explanations of the contexts of each folder. 
If the following is insufficient, please contact us at:
Andreas E. Robertson: arobertson38@gatech.edu

LGDGeneration: This folder contains a script for generating microstructures like those reported in Case Studies 1 and 2 of the associated papers. You will find implementations of both volume fraction conditioning methods suggested in the paper and examples of both methods on the Case Study 1 microstructure generation.

**'A note on volume fraction conditioning'**: For some implementations, we have found slight systemic shifts in the generated volume fractions (for example, for 2-phase, we see a very tight distribution of volume fractions -- as discussed in Case Study 1 -- whose average is shifted slightly from the desired volume fraction -- generally much less than 1%.). For 2-phase, this can be easily corrected by introducing small fictious volume fractions into the conditioning. For 3-phase, as discussed in the paper, the distribution of generated volume fractions is much wider. Therefore, the correction is much less meaningful.

TrainingFramework: This folder -- perhaps unsurprisingly -- contains the training framework for the proposed models. 
    (1) configs: a folder containing the config files. The 'baseline' config files are stored in the BASELINES subfolder. 
                 These are config files that the configwriter draws from to sweep a large range of parameters. 
    (2) models: a folder containing various model forms. 


