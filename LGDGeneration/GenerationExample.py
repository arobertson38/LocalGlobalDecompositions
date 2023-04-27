'''
This script contains some example code for running
the complete generation described in this paper. 

In general, generation has three steps. 
(1) identify the 2-point statistics you want to generate with
(2) identify the local diffusion model you want to use
(3) 

Created by: Andreas E. Robertson
Contact: arobertson38@gatech.edu
'''
import matplotlib.pyplot as plt
import torch
import numpy as np
from MKS.DMKSGeneration import HelperFunctions_StochasticGeneration as helpers
from MKS.utils import HelperFunctions as hfuncs
import DiffusionGeneration as diffusion

def CaseStudy1_TrueVFImplementation():
    """
    This method performs sampling as is described in Case Study 1. 
    Volume fraction conditioning is performed using the proposed 
    "estimate the true volume fraction via segmentation" framework. 

    Visualizes the generated microstructures at the end. 
    """
    # loading the reference microstructure
    reference = torch.load('./materials/NBSA/reference.pth').numpy().astype(float)
    reference = reference.squeeze()[..., None]
    reference = np.concatenate(
        [
            reference,
            1-reference,
        ], axis=-1
    )
    
    # extract 2-point statistics. 
    # the provided 2-point statistics function takes microstructures
    # in the form: SPATIAL_DIMINENSIONS x NUMBER_OF_STATES
    stats = helpers.twopointstats(reference)

    # Define the Global Generator -- the GRF. 
    grf = diffusion.EigenGenerator(
        statistics = stats,
    )
    
    # Define the local neighborhood approximation:
    sbg = diffusion.TrueVFConditioned_CDGenerator(
        config = './materials/NBSA/config.yml',
        model_location = './materials/NBSA/swa_checkpoint.yml',
    )

    # Performing Sampling:
    # Layer 1
    grf_output, _, _ = grf.generate()
    grf_output = np.concatenate(
        [output[None, None, ..., 0] for output in grf_output],
        axis=0
    )
    grf_output = torch.from_numpy(grf_output).float()
    # Layer 2:
    framework_output = sbg.generate(
        structures = grf_output,
        starting_index = 430,
        langevin_parameters = {
            'starting_index' : 430,
            'lower_clamp' : -3,
            'upper_clamp' : 3,
        },
        shift = 0.0073,
    )
    # Segmentation
    framework_output = torch.cat(
        [
            framework_output,
            1.0 - framework_output,
        ], dim = 1
    )
    framework_output = hfuncs.threshold_torch(framework_output).numpy().astype(int)

    # plotting
    f, axes = plt.subplots(1, 3, figsize=[12, 4])
    labels = ['Reference', 'Hyb. 1', 'Hyb. 2']
    micros = [reference[..., 0], framework_output[0, 0], framework_output[1, 0]]

    for ax, micro, label in zip(axes, micros, labels):
        ax.imshow(micro.transpose(), cmap='gray')
        ax.invert_yaxis()
        ax.set_title(label)
    
    f.tight_layout()
    plt.show()


def CaseStudy1_DirectVFImplementation():
    """
    This method performs sampling on the system described in Case Study 1. 
    Volume fraction conditioning is performed using the simpler
    volume fraction estimate framework:
        => directly estimate the volume fraction by taking the mean. 
        
    To account for this difference, segmentation is performed using a 
    volume fraction perserving assignment. (as described in Robertson
    and Kalidindi, 2022) 

    Visualizes the generated microstructures at the end. 
    """
    # loading the reference microstructure
    reference = torch.load('./materials/NBSA/reference.pth').numpy().astype(float)
    reference = reference.squeeze()[..., None]
    reference = np.concatenate(
        [
            reference,
            1-reference,
        ], axis=-1
    )
    
    # extract 2-point statistics. 
    # the provided 2-point statistics function takes microstructures
    # in the form: SPATIAL_DIMINENSIONS x NUMBER_OF_STATES
    stats = helpers.twopointstats(reference)

    # Define the Global Generator -- the GRF. 
    grf = diffusion.EigenGenerator(
        statistics = stats,
    )
    
    # Define the local neighborhood approximation:
    sbg = diffusion.DirectAverageVFConditioned_CDGenerator(
        config = './materials/NBSA/config.yml',
        model_location = './materials/NBSA/swa_checkpoint.yml',
    )

    # Performing Sampling:
    # Layer 1
    _, _, grf_output = grf.generate()
    grf_output = np.concatenate(
        [output[None, None, ..., 0] for output in grf_output],
        axis=0
    )
    grf_output = torch.from_numpy(grf_output).float()
    # Layer 2:
    framework_output = sbg.generate(
        structures = grf_output,
        starting_index = 430,
        langevin_parameters = {
            'starting_index' : 430,
            'lower_clamp' : -3,
            'upper_clamp' : 3,
        },
        shift = 0.0073,
    )
    # Segmentation
    framework_output = torch.cat(
        [
            framework_output,
            1.0 - framework_output,
        ], dim = 1
    ).numpy().float()
    framework_output = hfuncs.threshold_vfenforced(framework_output).astype(int)

    # plotting
    f, axes = plt.subplots(1, 3, figsize=[12, 4])
    labels = ['Reference', 'Hyb. 1', 'Hyb. 2']
    micros = [reference[..., 0], framework_output[0, 0], framework_output[1, 0]]

    for ax, micro, label in zip(axes, micros, labels):
        ax.imshow(micro.transpose(), cmap='gray')
        ax.invert_yaxis()
        ax.set_title(label)
    
    f.tight_layout()
    plt.show()
    
    
def CaseStudy2_TrueVFImplementation():
    """
    This method performs sampling as is described in Case Study 2.
    Volume fraction conditioning is performed using the proposed 
    "estimate the true volume fraction via segmentation" framework. 

    Visualizes the generated microstructures at the end. 
    """
    # loading the reference microstructure
    reference = torch.load('./materials/TI/reference.pth').numpy().astype(float)
    reference = np.moveaxis(reference.squeeze(), 0, -1)
    
    # extract 2-point statistics. 
    # the provided 2-point statistics function takes microstructures
    # in the form: SPATIAL_DIMINENSIONS x NUMBER_OF_STATES
    stats = helpers.twopointstats(reference)

    # Define the Global Generator -- the GRF. 
    grf = diffusion.EigenGenerator(
        statistics = stats,
    )
    
    # Define the local neighborhood approximation:
    sbg = diffusion.TrueVFConditioned_CDGenerator(
        config = './materials/NBSA/config.yml',
        model_location = './materials/NBSA/swa_checkpoint.yml',
    )

    # Performing Sampling:
    # Layer 1
    grf_output, _, _ = grf.generate()
    grf_output = np.concatenate(
        [np.moveaxis(output, -1, 0)[None, ...] for output in grf_output],
        axis=0
    )
    grf_output = torch.from_numpy(grf_output).float()
    # Layer 2:
    framework_output = sbg.generate(
        structures = grf_output,
        starting_index = 492,
        langevin_parameters = {
            'starting_index' : 492,
            'lower_clamp' : -3,
            'upper_clamp' : 3,
        },
        shift = 0.0,
    )
    # Segmentation
    framework_output = hfuncs.threshold_torch(framework_output).numpy().astype(int)

    # plotting
    f, axes = plt.subplots(1, 3, figsize=[12, 4])
    labels = ['Reference', 'Hyb. 1', 'Hyb. 2']
    micros = [
        reference, 
        np.moveaxis(framework_output[0], 0, -1), 
        np.moveaxis(framework_output[1], 0, -1),
    ]

    from matplotlib import cm
    trans = np.array([
                        cm.viridis(1.0)[:3], 
                        cm.viridis(0.6)[:3],
                        cm.viridis(0.0)[:3],
                    ])

    for ax, micro, label in zip(axes, micros, labels):
        ax.imshow(micro @ trans)
        ax.invert_yaxis()
        ax.set_title(label)
    
    f.tight_layout()
    plt.show()

if __name__ == "__main__":
    CaseStudy1_TrueVFImplementation()
    CaseStudy1_DirectVFImplementation()
    CaseStudy2_TrueVFImplementation()