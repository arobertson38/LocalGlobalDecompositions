@echo off
title Running the LGD Training Code Locally

Rem This file contains example commands for running the provided code.
Rem It will perform the following:
Rem     (1) Training the local neighborhood model
Rem     (2) Generate 1000 patches from the generative model
Rem     (3) Run testing and extract microstructure statistics from the 1000 patches. 

Rem starting the running loop
python main.py --runner AnnealRunner --config cs1_NBSA_config.yml --run Results --doc NBSA
python config_cleaner.py --folder Results/logs/NBSA

Rem Testing
python main.py --runner AnnealRunner --config cs1_NBSA_config.yml --run Results --doc NBSA --image_folder Results/logs/NBSA/out --test
python error.py --image_folder Results/logs/NBSA/out --micro uncond_samples.pth --mode NBSA --stdout Results/logs/NBSA