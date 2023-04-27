"""
I am adapting the code in this file. It will contain a
mixture of PCA related helper functions for other files
as well as the original methods which computed PCA
and performed their regular tasks. 

In this file, I am writing code to load the datasets,
concatenate them, compute non-periodic two-point statistics
and discriminate them using PCA. 

Created by: Andreas Robertson
Contact: arobertson38@gatech.edu
"""
import torch
import numpy as np
import os
import pickle
from sklearn.decomposition import PCA
from .HelperFunctions import *
#from GeneratingDataset import extract_cuts
import torchvision.transforms as tf
import matplotlib.pyplot as plt

# ---------------------------------------------
# PCA Related Helper Methods
# ---------------------------------------------

def extract_NP2PS(data, crop=0.5):
    """
    A method for extracting 2PS from a 
    collection of microstructures. 
    """
    assert len(data.shape)==3, "Data tensor must be BxNxN."
    stats = []
    for i in range(data.shape[0]):
        stats.append(twopoint_np(data[i, ...], \
                           data[i, ...], \
                           crop=crop, \
                           centered=True, \
                          ))

    return torch.stack(stats)

def extract_invariantNP2PS(data, crop=0.3, return_rho_theta=False):
    """
    A method for extracting invariant 2PS from a 
    collection of microstructures. 
    """
    assert len(data.shape)==3, "Data tensor must be BxNxN."
    stats = []
    return_flag = True if return_rho_theta else False

    for i in range(data.shape[0]):
        temp = twopoint_np(data[i, ...], \
                           data[i, ...], \
                           crop=1.0, \
                           centered=True, \
                          ).detach().numpy()
        
        if return_flag:
            temp, rho, theta = rotational_invariance_2D(temp, \
                                            centered=True, \
                                            crop_rho=crop, \
                                            return_rho_theta=True, \
                                    )
            temp = torch.from_numpy(temp)
            rho = torch.from_numpy(rho)
            theta = torch.from_numpy(theta)
            return_flag = False
        else:
            temp = torch.from_numpy(rotational_invariance_2D(temp, \
                                            centered=True,
                                            crop_rho=crop))
        stats.append(temp)

    if return_rho_theta:
        return torch.stack(stats), rho, theta
    else:
        return torch.stack(stats)

def computePCA(stats, save_location=None, number_of_components=30):
    """
    A method for computing PCA and saving the results
    """
    if len(stats.shape) > 2:
        stats = stats.view(stats.shape[0], -1)

    stats = stats.detach().numpy()

    pca = PCA(n_components = number_of_components)
    trans = pca.fit_transform(stats)

    if save_location is not None:
        # saving
        np.save(save_location + 'trans.npy', trans)
        np.save(save_location + 'cumev.npy', pca.explained_variance_ratio_)

        with open(save_location + 'pca.pkl', 'wb') as f:
            pickle.dump(pca, f)

    return trans, pca




# ---------------------------------------------
# Original Methods for Performing PCA
# ---------------------------------------------
def train_PCA(data, save_loc, components=50):
    """ 
    This method runs and trains PCA. nonperiodic 2PS are used. 

    1 autocorrelation and every crosscorrelation is included. 
    """
    stats = [[] for _ in range(data.shape[1])]
    for i in range(len(data)):
        for j in range(data.shape[1]):
            stats[j].append(list(\
                twopoint_np(data[i, 0, ...].squeeze(),
                            data[i, j, ...].squeeze(),
                            crop=0.5).detach().numpy().flatten(order='F')))

    # rescaling
    stds = []
    for i in range(data.shape[1]):
        stats[i] = np.array(stats[i])
        std_ = stats[i].std()
        stats[i] = stats[i] / std_
        stds.append(std_)

    stats = np.concatenate(stats, axis=1)

    # training PCA. 

    pca = PCA(n_components=min(components, *stats.shape))
    transformed = pca.fit_transform(stats)
    
    with open(save_loc, 'wb') as f:
        pickle.dump(pca, f)

    # saving the standard deviations
    std_loc = save_loc.rindex('.')
    std_loc = save_loc[:std_loc] + '_stds' + save_loc[std_loc:]
    with open(std_loc, 'wb') as f:
        pickle.dump(stds, f)
    
    return transformed

def run_pca():
    """
    A method for performing pca
    """

    # parameters
    number_of_components=50
    real_data = "./data/76XX_V3_40x40_CUTDOWN.pth"
    fake_data = "./data/76XX_Synthetic_V1_40x40_CUTDOWN.pth"
    crop=0.5

    # variables
    data = []
    realfakelabel = []

    # code
    real_data = torch.load(real_data)
    fake_data = torch.load(fake_data)

    # Documentation
    with open('./results/DESCRIPTION.txt', 'w') as f:
        f.write("This PCA run was done on both the real and fake data.\n")
        f.write("The data is stacked with the real first then the fake.\n")
        f.write(f"Number of Components: {number_of_components}.\n")
        f.write(f"Number of Real: {real_data.shape[0]}.\n")
        f.write(f"Number of Fake: {fake_data.shape[0]}.\n")

    for i in range(real_data.shape[0]):
        stats = twopoint_np(real_data[i, ...], \
                            real_data[i, ...], \
                            crop=crop,
                           ).detach().numpy().flatten(order='F')
        data.append(stats)
        realfakelabel.append(1)

    for i in range(fake_data.shape[0]):
        stats = twopoint_np(fake_data[i, ...], \
                            fake_data[i, ...], \
                            crop=crop,
                           ).detach().numpy().flatten(order='F')
        data.append(stats)
        realfakelabel.append(0)

    data = np.array(data)
    realfakelabel = np.array(realfakelabel)

    # performing PCA

    pca = PCA(n_components = number_of_components)
    trans = pca.fit_transform(data)

    # saving
    np.save('./results/trans.npy', trans)
    np.save('./results/cumev.npy', pca.explained_variance_ratio_)
    np.save('./results/labels.npy', realfakelabel)

    with open('./results/pca.pkl', 'wb') as f:
        pickle.dump(pca, f)






def lengthscale_experiments():
    """
    These experiments are to define a necessary length
    scale for the microstructure. The experiments
    will be performed on the synthetic microstructures produced
    by the NGRF generator (since thats where patterns need to
    be observed). The goal of the experiment is
    to find the lengthscale at which the structures can
    be discrimintated. 
    """

    length_scale = [26, 28, 32, 34, 36, 38, 40, 42]
    base = torch.load('./data/76XX_SYN_CUTDOWN.pth')
    num_crops = 50
    number_of_components=20

    for leng_scale in length_scale:
        print('<><' * 20)
        # we really want double the length scale so that we
        # can cutdown the nonperiodic two-point statistics
        len_scale = leng_scale * 2

        data = extract_cuts(base[0, ...].unsqueeze(0), \
                            image_size=len_scale, \
                            num_random_crops=num_crops, \
                            resize = tf.Resize(369)
                            )
        labels = torch.zeros([num_crops])

        for image_indx in range(1, len(base)):
            data = torch.cat([data, \
                    extract_cuts(base[image_indx, ...].unsqueeze(0), \
                            image_size=len_scale, \
                            num_random_crops=50, \
                            resize = tf.Resize(369)
                            )], dim=0)
            labels = torch.cat([labels, image_indx * \
                                torch.ones([num_crops])])
        
        # now we have the data, compute the statistics and PCA
        stats_data = []
        for i in range(data.shape[0]):
            stats = twopoint_np(data[i, ...].double(), \
                                data[i, ...].double(), \
                                0.5).numpy().ravel(order='F')
            stats_data.append(stats)

        stats_data = np.array(stats_data)

        # performing PCA
        pca = PCA(n_components = number_of_components)
        trans = pca.fit_transform(stats_data)

        print('completed PCA')

        # saving
        os.mkdir('./LengthScaleExperiments/' + str(leng_scale))
        np.save(f'./LengthScaleExperiments/{leng_scale}/trans.npy', trans)
        np.save(f'./LengthScaleExperiments/{leng_scale}/cumev.npy', pca.explained_variance_ratio_)
        np.save(f'./LengthScaleExperiments/{leng_scale}/labels.npy', labels)

        with open(f'./LengthScaleExperiments/{leng_scale}/pca.pkl', 'wb') as f:
            pickle.dump(pca, f)

        print('saved')


if __name__ == "__main__":
    lengthscale_experiments()
    #run_pca()


