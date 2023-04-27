"""
The intentation of this code is to accelerate testing
of the SBG and SDE models. 

So far, the testing modules I want to include are:

1) vol fraction comparison and printing
2) chord_length comparison and printing
3) PCA distribution comparison

Created by: Andreas E. Robertson
Contact: arobertson38@gatech.edu
"""
import torch
import matplotlib.pyplot as plt
import h5py
import numpy as np
from . import MMD_functions as mmd
import os
from ..DMKSGeneration import HelperFunctions_StochasticGeneration as cld
import argparse
import pickle
from . import HelperFunctions as helpers
import functools
from . import PCA as PCA
from . import figures as fig

# Plotting

def save_plot_examples(data, save_location):
    """ displays 4 samples from a dataset and saves in the save location """
    assert type(data) == torch.Tensor
    f, ax = plt.subplots(1, 4, figsize=[16, 4])
    indxs = torch.randperm(len(data))[:4]
    for n, indx in enumerate(indxs):
        fig.plot(data[indx], ax[n])
    f.tight_layout()
    f.savefig(save_location)


def histogram_plotting(arr_test, arr_nbsa, xlabel, f, ax):
    """ a method for plotting histograms """
    # plotting
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')

    ax.hist(arr_nbsa.flatten(), \
        bins=60, \
        alpha=0.4, \
        label='NBSA', \
        density=True)
    
    ax.hist(arr_test.detach().numpy(), \
        bins=60, \
        alpha=0.4, \
        label='Test', \
        density=True)

    f.tight_layout()
    return f, ax

def pca_plotting(arr_test, arr_nbsa, xlabel, f, ax, pc1, pc2):
    """ a method for plotting histograms """
    # plotting
    ax.set_title(xlabel)
    ax.set_xlabel(f"PC {pc1+1}")
    ax.set_ylabel(f"PC {pc2+1}")

    ax.plot(arr_nbsa[:, pc1], arr_nbsa[:, pc2], \
        'b.', alpha=0.05, \
        label='NBSA', \
        )
    
    ax.plot(arr_test[:, pc1], arr_test[:, pc2], \
        '.', color='orange', alpha=0.2, \
        label='Test', \
        )

    f.tight_layout()
    return f, ax

# filtering settings

def settings(mode):
    """ Returns the location of the comparison files for the Testing Suite """
    if mode.lower() == 'exp':
        comp_loc='./MKS/utils/ReferenceDataTestingSuite/76XX_EXP_DataV2NoRotComparison.h5'
        pca_comp='./MKS/utils/ReferenceDataTestingSuite/pcaModel_76XX_EXP_DataV2NoRotComparison.pkl'

    elif mode.lower() == 'pymks':
        comp_loc='./MKS/utils/ReferenceDataTestingSuite/76XX_NBSA_pymksapprox.h5'
        pca_comp='./MKS/utils/ReferenceDataTestingSuite/pcaModel_76XX_NBSA_pymksapprox.pkl'

    elif mode.lower() == 'exp1116':
        comp_loc='./MKS/utils/ReferenceDataTestingSuite/NBSA_index11_ps16.h5'
        pca_comp='./MKS/utils/ReferenceDataTestingSuite/pcaModel_NBSA_index11_ps16.pkl'

    elif mode.lower() == 'exp1416':
        comp_loc='./MKS/utils/ReferenceDataTestingSuite/NBSA_index14_ps16.h5'
        pca_comp='./MKS/utils/ReferenceDataTestingSuite/pcaModel_NBSA_index14_ps16.pkl'

    elif mode.lower() == 'exp1120':
        comp_loc='./MKS/utils/ReferenceDataTestingSuite/NBSA_index11_ps20.h5'
        pca_comp='./MKS/utils/ReferenceDataTestingSuite/pcaModel_NBSA_index11_ps20.pkl'

    elif mode.lower() == 'exp1420':
        comp_loc='./MKS/utils/ReferenceDataTestingSuite/NBSA_index14_ps20.h5'
        pca_comp='./MKS/utils/ReferenceDataTestingSuite/pcaModel_NBSA_index14_ps20.pkl'

    elif mode.lower() == 'exp1428':
        comp_loc='./MKS/utils/ReferenceDataTestingSuite/NBSA_index14_ps28_512.h5'
        pca_comp='./MKS/utils/ReferenceDataTestingSuite/pcaModel_NBSA_index14_ps28_512.pkl'

    elif mode.lower() == 'exp1440':
        comp_loc='./MKS/utils/ReferenceDataTestingSuite/NBSA_index14_ps40_512.h5'
        pca_comp='./MKS/utils/ReferenceDataTestingSuite/pcaModel_NBSA_index14_ps40_512.pkl'


    else:
        raise NotImplementedError(f"{mode} is not supported.")

    return comp_loc, pca_comp

def settings_nphase(mode):
    """ Returns the location of the comparison files for the Testing Suite """

    if mode.lower() == 'ti40':
        comp_loc='./MKS/utils/ReferenceDataTestingSuite/TI_index3_ps40_ux40.h5'
        pca_comp='./MKS/utils/ReferenceDataTestingSuite/pcaModel_TI_index3_ps40_ux40.pkl'
        pca_stds='./MKS/utils/ReferenceDataTestingSuite/pcaModel_TI_index3_ps40_ux40_stds.pkl'

    else:
        raise NotImplementedError(f"{mode} is not supported.")

    return comp_loc, pca_comp, pca_stds

##############################################################
# Useful Tools for actually doing testing
##############################################################

def train_PCA_TI(data, save_loc, components=50):
    """ 
    This method runs PCA for specifically the TI dataset. 
    This is because we need some specialized 2PS to do this.
    Because some of the phases are not consistently present. 
    """
    pairs = [(0, 0), (0, 1), (1, 1)]

    stats = [[] for _ in range(len(pairs))]
    for i in range(len(data)):
        for j in range(len(pairs)):
            stats[j].append(list(\
                helpers.twopoint_np(data[i, pairs[j][0], ...].squeeze(),
                            data[i, pairs[j][1], ...].squeeze(),
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

    pca = PCA.PCA(n_components=min(components, *stats.shape))
    transformed = pca.fit_transform(stats)

    with open(save_loc, 'wb') as f:
        pickle.dump(pca, f)

    # saving the standard deviations
    std_loc = save_loc.rindex('.')
    std_loc = save_loc[:std_loc] + '_stds' + save_loc[std_loc:]
    with open(std_loc, 'wb') as f:
        pickle.dump(stds, f)
    
    return transformed


##############################################################
# 3 Phase

def create_reference_files_Nphase(dataset_filename, output_filename, \
        save_folder = './MKS/utils/ReferenceDataTestingSuite/', \
        max_reference_size=20000,
        specialized_pca=True,
        ):
    """ 
    This method creates the output h5 and PCA datasets necessary
    for creating a new reference set for the TestingSuite object. 

    Only works for 2D microstructures (because 3D is not supported
    by the CLD code). 

    h5 contains:
    => Volume Fraction
    => PC Scores
    => Average Chord Length (x)
    => Chord Length (x)
    """
    # check if its a torch or h5 file:
    if dataset_filename.__contains__('.h5'):
        raise NotImplementedError('h5 files are not currently supported.')
        reference_dataset = torch.from_numpy(\
                h5py.File(dataset_filename, 'r')['micros'][:]).float()[:, 0, ...].unsqueeze(1)
    else:
        reference_dataset = torch.load(dataset_filename).double()

    # check if reference dataset is too large
    if len(reference_dataset) > max_reference_size:
        kept_indexes = torch.randperm(len(reference_dataset))[:max_reference_size]
        reference_dataset = reference_dataset[kept_indexes, ...]

    # computing the necessary information
    # num phases:
    num_phases = reference_dataset.shape[1]

    # means:
    means = reference_dataset.mean(dim=tuple( \
            range(2, len(reference_dataset.shape)))).detach().numpy()

    # CLD computation
    chords = [[] for _ in range(num_phases)]
    chords_avg = [[] for _ in range(num_phases)]
    for i in range(len(reference_dataset)):
        for j in range(num_phases):
            temp = cld.cld2D(
                    reference_dataset[i, j, ...].squeeze().detach().numpy(), 
                    bins=True,
                    periodic=False,
                    )
            if len(temp) != 0:
                chords[j].extend(temp)
                chords_avg[j].append(np.array(temp).mean())
    
    for i in range(num_phases):
        chords[i] = np.array(chords[i])
        chords_avg[i] = np.array(chords_avg[i])

    #f, ax = plt.subplots(1, 3)
    #[ax[i].hist(chords_avg[i], bins=30) for i in range(num_phases)]
    #plt.show()
    #exit()

    # PCA Computation
    if specialized_pca:
        # this line is here because in many of the structures,
        # some of the phases are all zero. So a specialized PCA
        # is necessary to tell them all apart. 
        transformed = train_PCA_TI(reference_dataset, \
                save_folder + 'pcaModel_' + \
                output_filename + '.pkl')
        
    else:
        transformed = PCA.train_PCA(reference_dataset, \
                save_folder + 'pcaModel_' + \
                output_filename + '.pkl')

    # saving
    save_name = save_folder + \
            output_filename + '.h5'
    h5file = h5py.File(save_name, 'w')
    dset1 = h5file.create_dataset(
        "Volume Fraction",
        data=means,
        compression="gzip",
        compression_opts=4,
        shuffle=True,
        dtype=float,
    )

    for n, (chord, chord_avg) in enumerate(zip(chords, chords_avg)):
        dset2 = h5file.create_dataset(
            f"Chord Length (x) - Phase {n}",
            data=chord,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            dtype=float,
        )

        dset3 = h5file.create_dataset(
            f"Average Chord Length (x) - Phase {n}",
            data=chord_avg,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            dtype=float,
        )

    dset4 = h5file.create_dataset(
        "PC Scores",
        data=transformed,
        compression="gzip",
        compression_opts=4,
        shuffle=True,
        dtype=float,
    )

class TestingSuite_3Phase(object):
    """ Everything is assumed to be entered as a torch tensor """
    def __init__(self, mode='exp', \
                       mmd_sigmas=(0.5, 1.0, 5.0, 15.0)):
        """
        :param comp_loc: a h5py file containing the VF comparison, the
                         chord length comparison and the average chord
                         length comparison. 
        :param mmd_sigmas: the list of sigmas for the mmd loss
        """
        comp_loc, pca_comp, pca_stds = settings_nphase(mode)

        self.comparisons = h5py.File(comp_loc, 'r')
        self.num_phases = self.comparisons['Volume Fraction'].shape[-1]
        self.mmd_sigmas = mmd_sigmas

        if pca_comp is not None:
            with open(pca_comp, 'rb') as f:
                self.pca = pickle.load(f)
            with open(pca_stds, 'rb') as f:
                self.stds = pickle.load(f)
        else:
            self.pca = None   
            self.stds = None
        
    def project_PCA(self, micros,
                    pairs = [(0, 0), (0, 1), (1, 1)],
                    ):
        """ projects a provided set of microstructures into PC space """
        assert self.pca is not None, "PCA has not been loaded."
        
        stats = [[] for _ in range(len(pairs))]
        for i in range(len(micros)):
            for j in range(len(pairs)):
                stats[j].append(list(\
                    helpers.twopoint_np(micros[i, pairs[j][0], ...].squeeze(),
                                micros[i, pairs[j][1], ...].squeeze(),
                                crop=0.5).detach().numpy().flatten(order='F')))

        # rescaling
        if self.stds is not None:
            for n, std in enumerate(self.stds):
                stats[n] = stats[n] / std

        stats = np.concatenate(stats, axis=1)
        transformed = self.pca.transform(stats)
        return torch.from_numpy(transformed)

    def compute_mmd_distance(self, dist1, dist2, repeats=10, max_size=1000):
        """
        Computes the mmd distance between two distributions
        
        repeats are included because the lengths of the two
        arrays must be the same. 
        
        :param dist1: First distribution
        :param dist2: Second distribution
        :param repeats: number of times to repeat the MMD distance
        :param max_size: maximum number of samples to take
        """
        min_len = min(len(dist1), len(dist2), max_size)
        distance = []
        for _ in range(repeats):
            ordering_1 = torch.randperm(len(dist1))[:min_len]
            ordering_2 = torch.randperm(len(dist2))[:min_len]
            distance.append(
                mmd.mix_rbf_mmd2(
                    dist1.view(len(dist1), -1)[ordering_1, ...],
                    dist2.view(len(dist2), -1)[ordering_2, ...],
                    self.mmd_sigmas
                )
            )
        return np.array(distance).mean(), np.array(distance).std()

    def scalar_distribution_comparison(self, scalar_distribution, comparison_dist,
                                       data_key, f=None, ax=None, 
                                       plotter=histogram_plotting):
        """
        Compares a given scalar distribution against the internal
        standard
        """
        if ax is None:
            assert f is None
            f, ax = plt.subplots(1, 1)
        
        assert type(comparison_dist) is torch.Tensor, 'Comparison_dist should be a torch tensor.'
        
        # computing distributional distance
        dist_mu, dist_std = self.compute_mmd_distance(scalar_distribution, \
            comparison_dist)
        
        # plotting
        f, ax = plotter(scalar_distribution, comparison_dist, data_key, \
            f, ax)
        return dist_mu, dist_std, f, ax
    
    @staticmethod
    def write_error(file, label, result, maxlength=70):
        """ writes to the error file in a standard format """
        assert (len(label) + len(result)) < maxlength
        string = label + ' ' * (maxlength - len(label) - len(result)) + result
        file.write(string + '\n')
    
    def complete_suite(self, microstructures, save_location):
        """ 
        Runs the complete suite of testing possibilities 
        
        Unlike the other implementation, this
        implementation will produce two separate figures. 
        (1) With chord lengths and volume fractions
        (2) With PC plots (We will plot the first 3 PC scores)
        """
        with open(os.path.join(save_location, 'summary.txt'), 'w') as fil:
            fil.write('Testing Suite Results:\n')
            fil.write(f'MMD Distances {self.mmd_sigmas}: \n')

            # The first Figure. 
            f, axes = plt.subplots(self.num_phases, 3, \
                figsize=[9, 3 * self.num_phases])

            for phase_index in range(self.num_phases):
                micro = microstructures[:, phase_index, ...].unsqueeze(1)
                
                # volume fraction testing
                vf_micros = torch.mean(micro, \
                    dim=tuple(range(1, len(microstructures.shape))))
                mean, std, _, _ = self.scalar_distribution_comparison(\
                    vf_micros, \
                    torch.from_numpy(self.comparisons['Volume Fraction'][:, phase_index]),
                    f'Volume Fraction - Phase {phase_index}',\
                    f, axes[phase_index, 0])
                self.write_error(fil, f"Phase {phase_index} Volume Fraction MMD Distance: ", \
                    f"{mean:0.4e} +/- {std:0.4e}.")

                # chord length testing
                chords = []
                chords_avg = []
                for i in range(len(micro)):
                    temp = cld.cld2D(
                        micro[i, ...].squeeze().detach().numpy(), 
                        bins=True,
                        periodic=False,
                        )
                    if len(temp) > 0:
                        chords.extend(temp)
                        chords_avg.append(np.array(temp).mean())

                # chords
                mean, std, _, _ = self.scalar_distribution_comparison( \
                    torch.tensor(chords), \
                    torch.from_numpy(
                        self.comparisons[f'Chord Length (x) - Phase {phase_index}'][:]
                    ),
                    f'Chord Length (x) - Phase {phase_index}', f, axes[phase_index, 1])
                self.write_error(fil, "Chord Length (X) MMD Distance: ", \
                    f"{mean:0.4e} +/- {std:0.4e}.")
                
                # chords average
                mean, std, _, _ = self.scalar_distribution_comparison( \
                    torch.tensor(chords_avg), \
                    torch.from_numpy(
                        self.comparisons[f'Average Chord Length (x) - Phase {phase_index}'][:]
                    ),
                    f'Average Chord Length (x) - Phase {phase_index}', f, axes[phase_index, 2])
                self.write_error(fil, "Average Chord Length (X) MMD Distance: ", \
                    f"{mean:0.4e} +/- {std:0.4e}.")
            
            # saving the first image.
            axes[0, 0].legend()
            f.tight_layout()
            f.savefig(os.path.join(save_location, 'DistributionComparison.png'))
            
            # PCA
            if self.pca is not None:
                f, axes = plt.subplots(3, 3, figsize=[9, 9])

                projected = self.project_PCA(microstructures)
                comparisons = torch.from_numpy(self.comparisons['PC Scores'][:])
                
                mean, std, _, _ = self.scalar_distribution_comparison( \
                    projected, \
                    comparisons, \
                    'PC Scores', f, axes[0, 0], \
                    plotter=functools.partial(pca_plotting, pc1=0, pc2=1)
                    )
                self.write_error(fil, "PC MMD Distance: ", \
                    f"{mean:0.4e} +/- {std:0.4e}.")
                
                # plotting
                for i in range(0, 3):
                    for j in range(i+1, 4):
                        pca_plotting(
                            projected,
                            comparisons,
                            '',
                            f,
                            axes[i, j-1],
                            pc1 = i,
                            pc2 = j,
                        )

            # finishing plotting and saving
            axes[-1, 0].legend()
            f.tight_layout()
            f.savefig(os.path.join(save_location, 'PCADistributionComparison.png'))


##############################################################
# 2 Phase

def create_reference_files(dataset_filename, output_filename, \
        save_folder = './MKS/utils/ReferenceDataTestingSuite/', \
        max_reference_size=20000):
    """ 
    This method creates the output h5 and PCA datasets necessary
    for creating a new reference set for the TestingSuite object. 

    h5 contains:
    => Volume Fraction
    => PC Scores
    => Average Chord Length (x)
    => Chord Length (x)
    """
    # check if its a torch or h5 file:
    if dataset_filename.__contains__('.h5'):
        reference_dataset = torch.from_numpy(\
                h5py.File(dataset_filename, 'r')['micros'][:]).float()[:, 0, ...].unsqueeze(1)
    else:
        reference_dataset = torch.load(dataset_filename)

    # check if reference dataset is too large
    if len(reference_dataset) > max_reference_size:
        kept_indexes = torch.randperm(len(reference_dataset))[:max_reference_size]
        reference_dataset = reference_dataset[kept_indexes, ...]

    # computing the necessary information
    # means:
    means = reference_dataset.mean(dim=tuple( \
            range(1, len(reference_dataset.shape)))).detach().numpy()

    # CLD computation
    chords = []
    chords_avg = []
    for i in range(len(reference_dataset)):
        temp = cld.cld2D(reference_dataset[i, ...].squeeze().detach().numpy(), True)
        if len(temp) != 0:
            chords.extend(temp)
            chords_avg.append(np.array(temp).mean())
    
    chords_avg = np.array(chords_avg)
    chords = np.array(chords)

    # PCA Computation
    transformed = PCA.train_PCA(reference_dataset, \
            save_folder + 'pcaModel_' + \
            output_filename + '.pkl')

    # saving
    save_name = save_folder + \
            output_filename + '.h5'
    h5file = h5py.File(save_name, 'w')
    dset1 = h5file.create_dataset(
        "Volume Fraction",
        data=means,
        compression="gzip",
        compression_opts=4,
        shuffle=True,
        dtype=float,
    )

    dset2 = h5file.create_dataset(
        "Chord Length (x)",
        data=chords,
        compression="gzip",
        compression_opts=4,
        shuffle=True,
        dtype=float,
    )

    dset3 = h5file.create_dataset(
        "Average Chord Length (x)",
        data=chords_avg,
        compression="gzip",
        compression_opts=4,
        shuffle=True,
        dtype=float,
    )

    dset4 = h5file.create_dataset(
        "PC Scores",
        data=transformed,
        compression="gzip",
        compression_opts=4,
        shuffle=True,
        dtype=float,
    )


class TestingSuite(object):
    """ Everything is assumed to be entered as a torch tensor """
    def __init__(self, mode='exp', \
                       mmd_sigmas=(0.5, 1.0, 5.0, 15.0)):
        """
        :param comp_loc: a h5py file containing the VF comparison, the
                         chord length comparison and the average chord
                         length comparison. 
        :param mmd_sigmas: the list of sigmas for the mmd loss
        """
        comp_loc, pca_comp = settings(mode)

        self.comparisons = h5py.File(comp_loc, 'r')
        self.mmd_sigmas = mmd_sigmas

        if pca_comp is not None:
            with open(pca_comp, 'rb') as f:
                self.pca = pickle.load(f)
        else:
            self.pca = None   
        
    def project_PCA(self, micros):
        """ projects a provided set of microstructures into PC space """
        assert self.pca is not None, "PCA has not been loaded."
        stats = []
        for i in range(len(micros)):
            stats.append(helpers.twopoint_np(micros[i, ...].squeeze(),
                                micros[i, ...].squeeze(),
                                crop=0.5).detach().numpy().flatten(order='F'))
        stats = np.array(stats)
        transformed = self.pca.transform(stats)
        return torch.from_numpy(transformed)

    def compute_mmd_distance(self, dist1, dist2, repeats=10, max_size=1000):
        """
        Computes the mmd distance between two distributions
        
        repeats are included because the lengths of the two
        arrays must be the same. 
        
        :param dist1: First distribution
        :param dist2: Second distribution
        :param repeats: number of times to repeat the MMD distance
        :param max_size: maximum number of samples to take
        """
        min_len = min(len(dist1), len(dist2), max_size)
        distance = []
        for _ in range(repeats):
            ordering_1 = torch.randperm(len(dist1))[:min_len]
            ordering_2 = torch.randperm(len(dist2))[:min_len]
            distance.append(
                mmd.mix_rbf_mmd2(
                    dist1.view(len(dist1), -1)[ordering_1, ...],
                    dist2.view(len(dist2), -1)[ordering_2, ...],
                    self.mmd_sigmas
                )
            )
        return np.array(distance).mean(), np.array(distance).std()

    def scalar_distribution_comparison(self, scalar_distribution, data_key, \
        f=None, ax=None, plotter=histogram_plotting):
        """
        Compares a given scalar distribution against the internal
        standard
        """
        if ax is None:
            assert f is None
            f, ax = plt.subplots(1, 1)

        comparison_dist = self.comparisons[data_key][:]

        # computing distributional distance
        dist_mu, dist_std = self.compute_mmd_distance(scalar_distribution, \
            torch.from_numpy(comparison_dist))
        
        # plotting
        f, ax = plotter(scalar_distribution, comparison_dist, data_key, \
            f, ax)
        return dist_mu, dist_std, f, ax
    
    @staticmethod
    def write_error(file, label, result, maxlength=70):
        """ writes to the error file in a standard format """
        assert (len(label) + len(result)) < maxlength
        string = label + ' ' * (maxlength - len(label) - len(result)) + result
        file.write(string + '\n')
    
    def complete_suite(self, microstructures, save_location, axes=None, f=None):
        """ Runs the complete suite of testing possibilities """
        with open(os.path.join(save_location, 'summary.txt'), 'w') as fil:
            fil.write('Testing Suite Results:\n')
            fil.write(f'MMD Distances {self.mmd_sigmas}: \n')

            if axes is None:
                if self.pca is not None:
                    f, axes = plt.subplots(1, 4, figsize=[14, 4])
                else:
                    f, axes = plt.subplots(1, 3, figsize=[10, 4])
            else:
                assert f is not None
                if self.pca is not None:
                    assert len(axes) == 4
                else:
                    assert len(axes) == 3

            # volume fraction testing
            vf_micros = torch.mean(microstructures, \
                dim=tuple(range(1, len(microstructures.shape))))
            mean, std, _, _ = self.scalar_distribution_comparison(vf_micros, \
                'Volume Fraction', f, axes[0])
            self.write_error(fil, "Volume Fraction MMD Distance: ", \
                f"{mean:0.4e} +/- {std:0.4e}.")

            # chord length testing
            chords = []
            chords_avg = []
            for i in range(len(microstructures)):
                temp = cld.cld2D(microstructures[i, ...].squeeze().detach().numpy(), True)
                chords.extend(temp)
                chords_avg.append(np.array(temp).mean())

            # chords
            mean, std, _, _ = self.scalar_distribution_comparison( \
                torch.tensor(chords), \
                'Chord Length (x)', f, axes[1])
            self.write_error(fil, "Chord Length (X) MMD Distance: ", \
                f"{mean:0.4e} +/- {std:0.4e}.")
            
            # chords average
            mean, std, _, _ = self.scalar_distribution_comparison( \
                torch.tensor(chords_avg), \
                'Average Chord Length (x)', f, axes[2])
            self.write_error(fil, "Average Chord Length (X) MMD Distance: ", \
                f"{mean:0.4e} +/- {std:0.4e}.")

            if self.pca is not None:
                mean, std, _, _ = self.scalar_distribution_comparison( \
                    self.project_PCA(microstructures), \
                    'PC Scores', f, axes[3], \
                    plotter=functools.partial(pca_plotting, pc1=0, pc2=1)
                    )
                self.write_error(fil, "PC MMD Distance: ", \
                    f"{mean:0.4e} +/- {std:0.4e}.")

            # finishing plotting and saving
            axes[0].legend()
            f.tight_layout()
            f.savefig(os.path.join(save_location, 'DistributionComparison.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Automatically test output of SBG/SDE models.')
    parser.add_argument("--image", required=True, help="The save directory")
    parser.add_argument("--micro",
        type=str, 
        default='uncond_samples.pth',
        help="File name for the microstructures we are testing.")
    parser.add_argument("--mode",
            type=str,
            default='exp',
            help="The data to compare against (exp, pymks)."
            )

    
    args = parser.parse_args()
    data = torch.load(os.path.join(args.image, args.micro))
    tester = TestingSuite(mode=args.mode)
    tester.complete_suite(data, args.image)
