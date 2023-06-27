'''
This file contains some scripts for generating datasets. 
These methods can be used on the reference images to generate
the datasets necessary for training. 

The reference images can be found in the LGDGeneration folder
under the corresponding 'materials' subfolder (NBSA and TI)

Additionally, this method creates the corresponding
reference files necessary for the 'error.py' error 
computing codes to run properly. 

Created by: Andreas E. Robertson
Contact: arobertson38@gatech.edu
'''
import torch
import itertools
import MKS.utils.TestingSuite as ts
import numpy as np

# --------------------------------------------------
# Utility functions for generating datasets
# --------------------------------------------------

def get_cut_locations(number=1, setting='random'):
    """ returns N coordinate pairs in the unit cube """
    if setting.lower() == 'random':
        return np.random.rand(number, 2)
    elif setting.lower() == 'latin':
        from scipy.stats.qmc import LatinHypercube
        sampler = LatinHypercube(d=2)
        return sampler.random(n=number)
    else:
        raise NotImplementedError(f"{setting} is not supported.")

def cut_patches(image, patch_size=40, max_size=370, total_patches=1200, \
        sampling_strategy='latin'):
    """ this methods cuts patches out of large images """

    cut_locations = get_cut_locations(number=\
            int(min(*image.shape[-2:]) / max_size * total_patches), \
            setting=sampling_strategy)

    rescale_x, rescale_y = torch.tensor(image.shape[-2:]) - patch_size
    patches = []
    for cut_location in cut_locations:
        cut_center_x = int(patch_size / 2 + rescale_x * cut_location[0])
        cut_center_y = int(patch_size / 2 + rescale_y * cut_location[1])

        patches.append(image[..., \
                cut_center_x-int(patch_size / 2): \
                cut_center_x + int(patch_size / 2), \
                cut_center_y-int(patch_size / 2): \
                cut_center_y + int(patch_size / 2), \
                ])
    patches = torch.cat(patches, dim=0)

    return patches

# --------------------------------------------------
# Generating dataset files
# --------------------------------------------------

def make_NBSA_datasets(
    micros,
    patch_size,
    num_patches,
    save_location,
    error_file_tag=None,
):
    """
    Makes a nickel based superalloy dataset out of the
    corresponding reference microstructure. 

    :param micros: (torch.Tensor) a [1,1,SPATIAL_DIMS] torch tensor containing
                           the reference microstructure. 
    :param patch_size: (int) the patch size (assumes a square)
    :param num_patches: (int) the desired number of patches in the dataset.
    :param save_location: (str) the location and file name to save the dataset as. 
    :param error_file_tag: (str) the file tag for the error file. Default is None.
    """
    assert len(micros.shape) == 4
    assert micros.shape[0] == micros.shape[1]
    assert micros.shape[0] == 1
    
    # computing statistics and plotting
    patches = cut_patches(
            micros,
            patch_size = patch_size,
            max_size = min(*micros.shape[2:]),
            total_patches = num_patches)

    torch.save(patches.float(), save_location)

    # check if reference folder has been instantiated. 
    if not os.path.exists('./MKS/utils/ReferenceDataTestingSuite'):
        os.mkdir('./MKS/utils/ReferenceDataTestingSuite/')

    # save error methods reference files.
    if error_file_tag is not None:
        ts.create_reference_files(
                save_location,
                error_file_tag)

def make_TI_datasets(
    micros,
    patch_size,
    num_patches,
    save_location,
    error_file_tag=None,
):
    """
    Makes a TI dataset out of the
    corresponding reference microstructure. 

    :param micros: (torch.Tensor) a [1,3,SPATIAL_DIMS] torch tensor containing
                           the reference microstructure. 
    :param patch_size: (int) the patch size (assumes a square)
    :param num_patches: (int) the desired number of patches in the dataset.
    :param save_location: (str) the location and file name to save the dataset as. 
    :param error_file_tag: (str) the file tag for the error file. Default is None.
    """
    assert len(micros.shape) == 4
    assert micros.shape[1] == 3
    assert micros.shape[0] == 1
    
    # computing statistics and plotting
    patches = cut_patches(
            micros,
            patch_size = patch_size,
            max_size = min(*micros.shape[2:]),
            total_patches = num_patches)

    torch.save(patches.float(), save_location)

    # check if reference folder has been instantiated. 
    if not os.path.exists('./MKS/utils/ReferenceDataTestingSuite'):
        os.mkdir('./MKS/utils/ReferenceDataTestingSuite/')

    # save error methods reference files.
    if error_file_tag is not None:
        ts.create_reference_files_Nphase(
                save_location,
                error_file_tag)

if __name__ == "__main__":
    nbsa_flag = True
    
    if nbsa_flag:
        # hyperparameters for running this method. 
        micros = torch.load('../LGDGeneration/materials/NBSA/reference.pth')
        patch_size = 40
        num_patches = 4000
        save_location = './datasets/NBSA_index14_ps40_256.pth'
        error_file_tag = 'NBSA_index14_ps40_256'

        # running the method
        make_NBSA_datasets(
            micros = micros,
            patch_size = patch_size,
            num_patches = num_patches,
            save_location = save_location,
            error_file_tag = error_file_tag,
        )
    else:
        # Generating the TI dataset
        # hyperparameters for running this method. 
        micros = torch.load('../LGDGeneration/materials/TI/reference.pth')
        patch_size = 64
        num_patches = 8000
        save_location = './datasets/TI_index2_ps64_0d25_8k_UNFILTERED.pth'
        error_file_tag = 'TI_index2_ps64_0d25_8k_UNFILTERED.pth'

        # running the method
        make_TI_datasets(
            micros = micros,
            patch_size = patch_size,
            num_patches = num_patches,
            save_location = save_location,
            error_file_tag = error_file_tag,
        )
