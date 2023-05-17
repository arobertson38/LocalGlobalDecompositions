# Local Global Decompositions

THIS REPOSITORY IS CURRENTLY IMCOMPLETE. IT WILL BE COMPLETELY POPULATED BY MAY 19TH, 2023. 

This repository contains research code associated with the proposed generative models in Robertson et al's "Local-Global Decompositions for Conditional Microstructure Generation". First, it contains the developed training environment for training the score-based diffusion models utilized as the local state in the paper (in TrainingFramework). Second, the repository contains the trained local SBG models and constructed LGD models utilized in Case Studies 1 and 2 as well as code to perform sampling (in LGDGeneration). As a warning, this code is definitely "research code". It was developed during the research process and, as a result, is rather messy. The biggest source of this is the proliferation of utilities that proved unnecessary. For example, during the research process we tested the impact of varying the initialization scheme -- it had no impact, but the code to do it remains in the training environment. 

If you find the contents of this folder useful, we hope that you cite us in your future work via the following pertitent publications:

A.E. Robertson, C. Kelly, M. Buzzy, S.R. Kalidindi, *Local-Global Decompositions for Conditional Microstructure Generation*, Acta Mater. 253 (2023) 118966, https://doi.org/10.1016/j.actamat.2023.118966.

A.E. Robertson, S.R. Kalidindi, *Efficient Generation of N-Field Microstructures from 2-Point Statistics using Multi-Output Gaussian Random Fields*, Acta Mater. 232 (2022) 117927, http://dx.doi.org/10.1016/j.actamat.2022.117927.

Alternatively, please reach out, we are always open to collaboration!

## Contents

The following are brief explanations of the contexts of each folder. 
If the following is insufficient, please contact us at:
Andreas E. Robertson: arobertson38@gatech.edu

### **LGDGeneration**

This folder contains a script for generating microstructures like those reported in Case Studies 1 and 2 of the associated papers. You will find implementations of both volume fraction conditioning methods suggested in the paper and examples of both methods on the Case Study 1 microstructure generation. Inside this folder, you will find a subfolder -- "materials" -- corresponding to the two case studies. These contain the trained models for each case study as well as the original reference images.

**A note on volume fraction conditioning**: For some implementations, we have found slight systemic shifts in the generated volume fractions (for example, for 2-phase, we see a very tight distribution of volume fractions -- as discussed in Case Study 1 -- whose average is shifted slightly from the desired volume fraction -- generally much less than 1%.). For 2-phase, this can be easily corrected by introducing small fictious volume fractions into the conditioning. For 3-phase, as discussed in the paper, the distribution of generated volume fractions is much wider. Therefore, such a correction would be largely meaningless and, as a result, is skipped.

To run the example generation, switch your working repository to the LGDGeneration folder -- this is necessary for all the relative imports -- and execute the *GeneratingExample.py* python file.

### **TrainingFramework**
This folder -- perhaps unsurprisingly -- contains the training framework for the proposed models. 

1. configs: a folder containing the config files. The 'cs1_NBSA_config.yml' and 'cs2_NBSA_config.yml' config files contain the training parameters utilized in the reported case studies. The BASELINES folder contains baseline config files. These are config files utilized by *config_writer.py* to generate wide range of config files if one is performing a sweep of many training settings. 
2. datasets: When utilizing this training framework, the local patch dataset should be placed in this folder. If you are interested in the datasets we used to perform training, please contact the email above. We don't include the datasets here due to size. Otherwise, datasets can be generated from a large experimental image using the *GeneratingDatasets.py* script provided. Similar datasets to the ones used can be generated using the reference images provided in the LGDGeneration folder. 
3. losses: contains loss functions.
4. MKS: contains useful helper functions for generation, analysis, and testing.
5. models: a folder containing various lightweight model implementations following the strategy described in the LGD paper.
6. runners: contains the primary workhorse code for executing training.
7. *config_cleaner.py*: a script for cleaning the config file spit out by the training framework. 
8. *config_writer.py*: a script for editing the baseline config files and generating new config files. This script is especially useful when running a batch of training processes with different training parameters in parallel on a computing cluster. The code takes two types of flags: --mode (currently supports NBSA and TI) which indicates which baseline config file to begin with and --input which indicates which input in the config file to edit. After the --input flag, you must identify the section, parameter, and value you want changed. For example, the following code will edit the number of channels, image size, and model in the NBSA baseline config. The code will output a config file named by appending the changed values. Periods are turned into 'd's. The name is ordered by section and order in which the section was called in the command line.  
        python config_writer.py --mode NBSA --input data channels 3 --input model model vnetd4 --input data image_size 70.20  
For example, this command will save the file *refNBSA_channels3_image_size70d20_modelvnetd4.yml*  
9. *error.py*: a script for running some of the error testing described in the paper. This analysis requires reference statistics to be collected which can be collected via the code in the *GeneratingDatasets.py* file. Additionally, the reference statistics need to be appended to the *settings* method in the *TestingSuite.py* file in MKS/utils. Generally, feel free to contact us if you are having trouble setting this up, there are a lot of moving parts.
10. *GeneratingDataset.py*: code for generating patch datasets (and reference statistics for the error computation) from a reference image. Datasets for the two examples decribed in the paper can be generated using the reference images provided in LGDGeneration/materials. 
11. *jobs.bat* a bat file for running the training regime on a windows computer. For Linux machines, this can be translated easily into an equivalent bash file. 

Good luck!

Andreas
