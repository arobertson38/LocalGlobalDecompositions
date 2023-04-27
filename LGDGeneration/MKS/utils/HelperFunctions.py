"""
This file contains various helper functions for this project.
For example: plotting functions, two-point statistics
computations, etc.

"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from queue import Queue

# --------------------------------------------------------
# Extracting loss information
# --------------------------------------------------------

def loss_SBG(filename, save_location=None, visualize=False, ybounds=None, \
        ax = None):
    """
    This method extracts the training v testing error
    from the files produced the Song's original ncsn
    code. 

    It assumes that the testing error is labeled by
    the phrase TEST. 
    """
    with open(filename, 'r') as f:
        # remove the original description
        line = f.readline().split(' ')
        while not line[2].lower().__contains__('runner'):
            line = f.readline().split(' ')

        count = 0
        train_loss = []
        train_step = []
        test_loss = []
        test_step = []

        while (len(line) > 1) and (count < 5e6):
            
            if line[7].lower().__contains__('step'):
                # its a training step
                train_step.append(int(line[8][:-1]))
                train_loss.append(float(line[10][:-1]))

            elif line[7].lower().__contains__('test'):
                # its a testing step
                test_step.append(int(line[9][:-1]))
                test_loss.append(float(line[11][:-1]))

            else:
                raise NotImplementedError(f"Unrecognized sequence: {line[7]}")

            line = f.readline().split(' ')
            count += 1

    # plotting
    ax_flag = False
    if ax is None:
        ax_flag = True
        f = plt.figure()
        ax = f.add_subplot(111)

    ax.plot(train_step, train_loss, 'k-', label='Training')
    ax.plot(test_step, test_loss, 'r-', label='Testing')

    ax.set_xlabel('Step')
    ax.set_ylabel('Denoising Loss')
    if ax_flag:
        ax.set_title(filename)

    if ybounds is None:
        mean_y = np.array(train_loss[int(len(train_loss) * 0.75):]).mean()
        ax.set_ylim((mean_y - 40, mean_y + 40))
    else:
        ax.set_ylim(ybounds)

    ax.legend()


    if visualize:
        plt.show()
    else:
        if ax_flag:
            f.tight_layout()
            assert save_location is not None, "Please give a save location for the loss curve image."
            f.savefig(save_location)

def loss_SDE(filename, save_location=None, visualize=False, ybounds=None):
    """
    This method extracts the training v testing error
    from the files produced the Song's original ncsn
    code. 

    It assumes that the testing error is labeled by
    the phrase TEST. 
    """
    with open(filename, 'r') as f:
        # remove the original description
        line = f.readline()
        while not line.__contains__('training_loss'):
            line = f.readline()

        count = 0
        train_loss = []
        train_step = []
        test_loss = []
        test_step = []

        while (len(line) > 1) and (count < 5e8):

            line = line[line.find('step: ')+6:]
            step = int(line[:line.find(',')])
            
            if line.__contains__('training_loss'):
                train_step.append(step)
                train_loss.append(float(line[line.find('training_loss: ')+15:-1]))
            elif line.__contains__('eval_loss'):
                test_step.append(step)
                test_loss.append(float(line[line.find('eval_loss: ')+11:-1]))
            else:
                raise AttributeError('Line contained neither training nor eval loss.')

            line = f.readline()
            count += 1

    # plotting
    f = plt.figure()
    ax = f.add_subplot(111)

    ax.plot(train_step, train_loss, 'k-', label='Training')
    ax.plot(test_step, test_loss, 'r-', label='Testing')

    ax.set_xlabel('Step')
    ax.set_ylabel('Denoising Loss')
    ax.set_title(filename)

    if ybounds is None:
        mean_y = train_loss[int(len(train_loss) * 0.75):].mean()
        ax.set_ylim((mean_y - 40, mean_y + 40))
    else:
        ax.set_ylim(ybounds)

    ax.legend()
    f.tight_layout()
    if visualize:
        plt.show()
    else:
        assert save_location is not None, "Please give a save location for the loss curve image."
        f.savefig(save_location)

# --------------------------------------------------------
# Computing Average feature size using Berryman's Method
# --------------------------------------------------------

def two_forward_derivative(short=1.0, long=np.sqrt(2), twoD=True):
    if twoD:
        return (-1 / 16) * np.array([[-1 / long, 0, -1 / short, 0, -1 / long],
                                    [0, 4 / long, 4 / short, 4 / long, 0],
                                    [-1 / short, 4 / short, -12 / short - 12 / long, 4 / short, -1 / short],
                                    [0, 4 / long, 4 / short, 4 / long, 0],
                                    [-1 / long, 0, -1 / short, 0, -1 / long]])
    else:
        # define the two forward 3D approximate matrix derivative
        #
        # Things left to do:
        # (1) update coefficient (x?)
        # (2) Update center value (x)
        # (3) update all layers (x)
        return (-1 / 36) * np.array([
                                    [[0, 0, -1 / long, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [-1 / long, 0, -1 / short, 0, -1 / long],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, -1 / long, 0, 0]],
                                    # z = 1
                                    [[0, 0, 0, 0, 0],
                                    [0, 0, 4 / long, 0, 0],
                                    [0, 4 / long, 4 / short, 4 / long, 0],
                                    [0, 0, 4 / long, 0, 0],
                                    [0, 0, 0, 0, 0]],
                                    # z = 2
                                    [[-1 / long, 0, -1 / short, 0, -1 / long],
                                    [0, 4 / long, 4 / short, 4 / long, 0],
                                    [-1 / short, 4 / short, -18 / short - 36 / long, 4 / short, -1 / short],
                                    [0, 4 / long, 4 / short, 4 / long, 0],
                                    [-1 / long, 0, -1 / short, 0, -1 / long]],
                                    # z = 3
                                    [[0, 0, 0, 0, 0],
                                    [0, 0, 4 / long, 0, 0],
                                    [0, 4 / long, 4 / short, 4 / long, 0],
                                    [0, 0, 4 / long, 0, 0],
                                    [0, 0, 0, 0, 0]],
                                    # z = 4
                                    [[0, 0, -1 / long, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [-1 / long, 0, -1 / short, 0, -1 / long],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, -1 / long, 0, 0]],
                                    ])


def approx_matrix_derivative(arr, matrix_deriv=two_forward_derivative, short_length=1.0, long_length=np.sqrt(2)):
    """
    assumes that the field enters uncentered
    """
    sh = arr.shape
    twoD = True if len(sh) == 2 else False
    if twoD:
        cent = np.fft.fftshift(arr)[int(sh[0] / 2 - 2):int(sh[0] / 2 + 3), \
                                    int(sh[1] / 2 - 2):int(sh[1] / 2 + 3)]
    else:
        cent = np.fft.fftshift(arr)[int(sh[0] / 2 - 2):int(sh[0] / 2 + 3), \
                                    int(sh[1] / 2 - 2):int(sh[1] / 2 + 3), \
                                    int(sh[2] / 2 - 2):int(sh[2] / 2 + 3)]
    return (cent * matrix_deriv(short_length, long_length, twoD)).sum()

def average_feature_size(arr, periodic=False, verbose=True):
    """
    Computes the average feature size via Berryman's method
    (i.e., by, first, computing the 2PS and the extracting the
    average size via a numerical derivative)

    Really only works for Single Phases from N-Phase structures. 
    
    Strictly speaking, Barryman's derivation assumes that the
    material is the lower volume fraction phase. 
    """
    if type(arr) is np.ndarray:
        arr = torch.from_numpy(arr)

    if arr.mean() >= 0.5:
        if verbose:
            print('Warning: Vf is above 0.5')
            print('Inverting Microstructure')
        arr = 1 - arr

    if arr.mean() <= 0.001:
        return arr.shape[-1] * 10
    else:
        if periodic:
            stats = twopoint(arr, arr).detach().numpy()
        else:
            stats = twopoint_np(arr, arr, centered=False).detach().numpy()

        deriv = approx_matrix_derivative(stats)
        return arr.mean().item() / deriv

# --------------------------------------------------------
# Preprocessing Methods - e.g., Discrete Segmentation
# --------------------------------------------------------

def expand(micros):
    """
    This method expands an eigenmicrostructure
    that is missing its last phase.

    Args:
        micros (torch.Tensor): microstructure array of the form
                               [#, Phases, SPATIALDIMENSIONS]
    """
    assert type(micros) is torch.Tensor, 'This method only supports torch tensors.'
    shape = list(micros.shape)
    shape[1] = 1
    last_phase = torch.ones(shape, dtype=micros.dtype).to(micros.device)
    last_phase = last_phase - micros.sum(dim=1).unsqueeze(1)
    micros = torch.cat([micros, last_phase], dim=1)
    return micros

def threshold(micros):
    """
    Accepts numpy arrays in the form:
            [#, C, SPATIALDIM]. 
    
    There is an assumption that each voxel is fully
    represented. Therefore, if we summed across the
    channels, we should get 1. (plus/minus some noise)

    This is a simple thresholding. Each voxel
    is assigned to its maximum.
    
    """
    assert type(micros) is np.ndarray, 'This method only supports numpy arrays.'
    N = np.product(micros.shape[2:])
    shape = micros.shape[2:]

    # creating the IJK indexing matrices
    if len(shape) == 2:
        twoD = True
        I, J = np.ogrid[:shape[0], :shape[1]]
    elif len(shape) == 3:
        twoD = False
        I, J, K = np.ogrid[:shape[0], :shape[1], :shape[2]]
    else:
        raise NotImplementedError("Only 2D and 3D structures are supported.")
    
    micros = np.moveaxis(micros, 1, -1)
    new_micros = np.zeros_like(micros)

    # code adapted from Base Class
    # Postprocessing Code in StochasticGeneration
    
    # iterator 1 by 1 and discretize
    for n, micro in enumerate(micros):
        # this is a flag to indicate if we have used a voxel
        ind = micro.argmax(axis=-1)
        if twoD:
            new_micros[n, I, J, ind] = 1.0
        else:
            new_micros[n, I, J, K, ind] = 1.0

    # move the axis back
    new_micros = np.moveaxis(new_micros, -1, 1)
    
    return new_micros

def threshold_vfenforced(micros):
    """
    Accepts numpy arrays in the form:
            [#, C, SPATIALDIM]. 
    
    There is an assumption that each voxel is fully
    represented. Therefore, if we summed across the
    channels, we should get 1. (plus/minus some noise)
    """
    assert type(micros) is np.ndarray, 'This method only supports numpy arrays.'
    N = np.product(micros.shape[2:])
    shape = micros.shape[2:]

    # creating the IJK indexing matrices
    if len(shape) == 2:
        twoD = True
        I, J = np.ogrid[:shape[0], :shape[1]]
    elif len(shape) == 3:
        twoD = False
        I, J, K = np.ogrid[:shape[0], :shape[1], :shape[2]]
    else:
        raise NotImplementedError("Only 2D and 3D structures are supported.")
    
    micros = np.moveaxis(micros, 1, -1)
    new_micros = np.zeros_like(micros)

    # code adapted from NMaximum Constrained 
    # Postprocessing Code in StochasticGeneration
    
    # iterator 1 by 1 and discretize
    for n, micro in enumerate(micros):
        # computing means
        means = micro.mean(axis=tuple(range(len(micro.shape)-1)))
        order = list(np.argsort(means))

        # this is a flag to indicate if we have used a voxel
        ind = np.ones(shape, dtype=np.int8) * order[-1]

        for iterator in range(0, len(order[:-1])):
            i = order[iterator]
            if (N * means[i]) >= 1:
                index = np.unravel_index(
                    np.argpartition(
                        micro[..., i], 
                        -int(N * means[i]), 
                        axis=None)[-int(N * means[i]):], shape)
                ind[index] = i
                for j in range(iterator+1, len(order[:-1])):
                    micro[..., order[j]][index] = -10
        
        if twoD:
            new_micros[n, I, J, ind] = 1.0
        else:
            new_micros[n, I, J, K, ind] = 1.0

    # move the axis back
    new_micros = np.moveaxis(new_micros, -1, 1)
    
    return new_micros

# --------------------------------------------------------
# Pytorch Reimplementations

def threshold_torch(micros):
    """
    Accepts numpy arrays in the form:
            [#, C, SPATIALDIM]. 
    
    There is an assumption that each voxel is fully
    represented. Therefore, if we summed across the
    channels, we should get 1. (plus/minus some noise)

    This is a simple thresholding. Each voxel
    is assigned to its maximum.
    
    """
    assert type(micros) is torch.Tensor, 'This method only supports torch tensors.'
    shape = micros.shape[2:]

    # creating the IJK indexing matrices
    if len(shape) == 2:
        twoD = True
        I, J = np.ogrid[:shape[0], :shape[1]]
    elif len(shape) == 3:
        twoD = False
        I, J, K = np.ogrid[:shape[0], :shape[1], :shape[2]]
    else:
        raise NotImplementedError("Only 2D and 3D structures are supported.")
    
    # after testing: this isn't necessary. We could just overwrite memory
    new_micros = torch.zeros_like(micros)

    # code adapted from Base Class
    # Postprocessing Code in StochasticGeneration
    
    # iterator 1 by 1 and discretize
    for n, micro in enumerate(micros):
        # this is a flag to indicate if we have used a voxel
        ind = micro.argmax(dim=0)
        if twoD:
            new_micros[n, ind, I, J] = 1.0
        else:
            new_micros[n, ind, I, J, K] = 1.0

    return new_micros

# estimate VF quickly

def true_volumefraction(micros):
    """
    method takes in a possibly noisy microstructure and returns
    the volume fraction after segmentation with the
    "take the maximum" approach. 

    It assumes that no expansion is necessary. 

    Args:
        micros (torch.Tensor): Tensor containing the microstructures.
                               [#, Channels, SPATIALDIM].
        incomplete (bool): Flag indicating whether a channel is missing. 
        
    Returns:
        torch.tensors: [#, Channels] containing the "true" vf.
    """
    assert type(micros) is torch.Tensor, 'This method only supports torch tensors.'

    max_locations = micros.argmax(dim=1)
    
    dimensions = tuple(range(1, len(micros.shape)-1))

    vfs = torch.cat(
        [(max_locations==i).float().mean(dim=dimensions).unsqueeze(-1) \
            for i in range(micros.shape[1])],
        dim=-1,
    )

    return vfs


# --------------------------------------------------------
# Analysis Methods
# --------------------------------------------------------

def train_test_split(data, fraction=0.8):
    """
    This method takes a data-set of the form Batchs x ...
    and splits it into 2 randomly ordered sets. 
    """
    indexes = torch.randperm(len(data))
    train = data[indexes[:int(fraction * len(data))], ...]
    test = data[indexes[int(fraction * len(data)):], ...]
    return train, test

# --------------------------------------------------------
# Plotting
# --------------------------------------------------------

def plotRI2PS(rho, theta, stats, ax=None):
    """
    This method plots rotationally invariant 2-point statistics.
    """
    if ax is not None:
        assert ax.name == 'polar', "Requires a 'polar' axis: add_subplot(ijk, projection='polar')."
    else:
        f = plt.figure()
        ax = f.add_subplot(111, projection='polar')

    ax.contourf(theta, rho, stats)
    return ax


def debug(flag=True):
    if flag:
        plt.figure(); plt.show()

def imshow(image):
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.imshow(image)
    plt.show()

# --------------------------------------------------------
# Clustering Measures
# --------------------------------------------------------

def J2(data, labels, unique_labels=(0, 1)):
    """
    This method computes J2 Cluster Metric for
    provided data.

    It is assumed that the probably of each class is equal. 

    data is assumed to be of the form Num Samples X 
    Num Features

    Written in numpy
    """
    assert len(data.shape)==2, "Data must be a MxN Matrix"

    S_b = Sb(data, labels, unique_labels)
    S_w = Sw(data, labels, unique_labels)
    
    return np.linalg.det(S_w + S_b) / np.linalg.det(S_w)


def Sw(data, labels, unique_labels=(0, 1), equal_prob=False):
    """
    This method computes the "Within Clusters" Scatter
    matrix, S_w. 

    It is assumed that the probably of each class is equal. 

    data is assumed to be of the form Num Samples X 
    Num Features

    Written in numpy
    """
    assert len(data.shape)==2, "Data must be a MxN Matrix"
    # Computing the probability of each class. 
    if equal_prob:
        p_i = [1 / len(unique_labels)] * len(unique_labels)
    else:
        p_i = []
        for label in unique_labels:
            p_i.append((labels == label).sum() / len(labels))

    Sw = np.zeros([data.shape[1], data.shape[1]])

    for n, label in enumerate(unique_labels):
        filt_data = data[labels == label, :]
        filt_data = filt_data - filt_data.mean(axis=0)

        S_temp = np.zeros_like(Sw)
        for i in range(filt_data.shape[0]):
            S_temp += np.outer(filt_data[i, :], filt_data[i, :])
        S_temp /= filt_data.shape[0]

        Sw += p_i[n] * S_temp

    return Sw
            
def Sb(data, labels, unique_labels=(0, 1), equal_prob=False):
    """
    This method computes the "Between Clusters" Scatter
    matrix, S_b. 

    It is assumed that the probably of each class is equal. 

    data is assumed to be of the form Num Samples X 
    Num Features

    Written in numpy
    """
    assert len(data.shape)==2, "Data must be a MxN Matrix"
    # Computing the probability of each class. 
    if equal_prob:
        p_i = [1 / len(unique_labels)] * len(unique_labels)
    else:
        p_i = []
        for label in unique_labels:
            p_i.append((labels == label).sum() / len(labels))

    Sb = np.zeros([data.shape[1], data.shape[1]])

    # computing the overall mean:
    mean_overall = np.zeros([data.shape[1]])
    for n, label in enumerate(unique_labels):
        mean_filt = data[labels == label, :].mean(axis=0)
        mean_overall += p_i[n] * mean_filt

    for n, label in enumerate(unique_labels):
        mean_filt = data[labels == label, :].mean(axis=0)
        Sb += p_i[n] * np.outer(mean_filt - mean_overall, \
                                mean_filt - mean_overall)

    return Sb



# --------------------------------------------------------
# Flooding 2D
# --------------------------------------------------------

def grain_statistics(image):
    """
    This method computes the grain statistics associated with
    the image passed in as "image"
    """
    area, perimeter = flood(image)
    return area, perimeter

def flood(image):
    """
    This method floods the passed image (starting at (0, 0)) and
    removes any feature that it runs into. 

    Assumes that the image is a torch tensor. 
    Also assumes that the image is square.
    """
    flags = torch.zeros_like(image)
    voxel_queue = Queue()
    voxel_queue.put((0,0))
    im = image.clone()
    size = image.shape[0]

    area = []
    perimeter = []

    while not voxel_queue.empty():
        x, y = voxel_queue.get()
        if (x >= 0) and (y >= 0) and (x < im.shape[0]) and \
                (y < im.shape[1]):
            if flags[x, y] == 0:
                # perform operations
                if im[x, y] > 0:
                    im, a_temp, p_temp = flood_feature(im, (x, y))
                    area.extend(a_temp)
                    perimeter.extend(p_temp)
                    
                [voxel_queue.put(item) for item in [(x+1, y), (x-1, y), \
                        (x, y+1), (x, y-1)]]
                flags[x, y] = 1.0

    return area, perimeter

def flood_feature(image, initial_index):
    """
    This method floods a single feature and removes it. 
    """
    flags = torch.zeros_like(image)
    voxel_queue = Queue()
    voxel_queue.put(initial_index)
    im = image.clone()

    no_stats_flag = False

    while not voxel_queue.empty():
        x, y = voxel_queue.get()
        if (x >= 0) and (y >= 0) and (x < im.shape[0]) and \
                (y < im.shape[1]):

            if (x == 0) or (y == 0) or (x == im.shape[0]-1) or \
                    (y == im.shape[1]-1):
                if flags[x, y] == 0.0:
                    if im[x, y] > 0:
                        # then the grain touches the boundary
                        no_stats_flag = True
                        flags[x, y] = 1
                        im[x, y] = 0.0
                        [voxel_queue.put(item) for item in [(x+1, y), (x-1, y), \
                                (x, y+1), (x, y-1)]]

            else:
                if flags[x, y] == 0.0:
                    # perform operations
                    if im[x, y] > 0:
                        flags[x, y] = 1
                        im[x, y] = 0.0
                        [voxel_queue.put(item) for item in [(x+1, y), (x-1, y), \
                                (x, y+1), (x, y-1)]]



    if no_stats_flag:
        return im, [], []
    else:
        boundary = np.linalg.norm(np.array(np.gradient(flags)), axis=0)
        boundary = boundary / boundary.max()
        boundary[boundary > 0.65] = 1.0
        boundary[boundary < 1] = 0
        return im, [flags.sum(),], [boundary.sum(),]

# --------------------------------------------------------
# 2-Point Statistics Computation
# --------------------------------------------------------

def cart2pol(x, y):
    """
    A method for converting from cartesian to polar. 
    Congruent to matlab's "cart2pol"
    """
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return rho, theta

def rotational_invariance_2D(statistics, centered=False, \
        rho_bins=40, angle_bin_width=7, return_rho_theta=False,
        crop_rho=1.0):
    """
    This method takes in a 2D array of 2-point statistics
    and makes them rotationally invariant through a series
    of awful tricks. 

    Converted from Ahmet's Rotationally Invariant Matlab
    Code

    The parameter "centered" is used to tell the code if
    the statistics you are passing are centered using
    np.fft.fftshift. 
    """
    from scipy.interpolate import griddata

    assert len(statistics.shape) == 2, "Only supports 2D."
    
    if not centered:
        stats = np.fft.fftshift(statistics)
    else:
        stats = statistics.copy()

    # Find Euclidean Center
    cx, cy = np.unravel_index(np.argmax(stats), stats.shape)

    # Compute Polar Coordinates
    XX, YY = np.meshgrid(np.arange(stats.shape[0]), \
                         np.arange(stats.shape[1]))

    rho, theta = cart2pol(XX.flatten() - cx, YY.flatten() - cy)
    flat_stats = stats.flatten()

    # extending slightly to facilitate interpolation
    indexes_to_pad = theta > np.pi / 2
    rho = np.concatenate([rho, rho[indexes_to_pad]], axis=0)
    theta = np.concatenate([theta, \
            theta[indexes_to_pad] - 2*np.pi], \
            axis=0)
    flat_stats = np.concatenate([flat_stats, \
            flat_stats[indexes_to_pad]], axis=0)

    # Accomodare (i.e., fix) volume fraction info
    rho = np.concatenate([rho, np.zeros([200])], axis=0)
    theta = np.concatenate([theta, np.linspace(-3 * np.pi / 2, np.pi, \
                    200)], axis=0)
    flat_stats = np.concatenate([flat_stats, 
                    np.ones([200]) * stats[int(cx), int(cx)], \
                    ], axis=0)

    # make a polar grid
    THETA, RHO = np.meshgrid(np.arange(-180, 180, angle_bin_width) /\
                             180 * np.pi, \
                             np.linspace(0, min(cx, cy), rho_bins))

    new = griddata((rho, theta), flat_stats, (RHO, THETA), 'cubic')
    
    invariant = np.fft.ifft(np.abs(np.fft.fft(new, axis=1))).real
    invariant = invariant[:int(invariant.shape[0]*crop_rho), :]
    RHO = RHO[:int(RHO.shape[0]*crop_rho), :]
    THETA = THETA[:int(THETA.shape[0]*crop_rho), :]
    
    if return_rho_theta:
        return invariant, RHO, THETA
    else:
        return invariant


def twopoint(micro1, micro2):
    """
    A method for computing the periodic correlation between
    two microstructure instances (micro1, micro2)
    """
    assert micro1.shape == micro2.shape
    assert len(micro1.shape) == 2


    two_point = torch.fft.ifftn(torch.fft.fftn(micro1).conj() * \
            torch.fft.fftn(micro2)).real / np.product(micro1.shape)

    return two_point



def twopoint_np(micro1, micro2, crop=0.5, centered=False):
    """
    A method for computing the nonperiodic correlation between
    two microstructure instances (micro1, micro2)

    returns the statistics uncentered. 

    Caution, this should be used with 64-bit floats (i.e., doubles)
    """
    assert crop >= 0 and crop <= 1, "crop must be between 0 and 1"
    assert micro1.shape == micro2.shape
    assert len(micro1.shape) == 2

    # check if its a numpy array
    is_numpy = False
    if type(micro1) is np.ndarray:
        micro1 = torch.from_numpy(micro1)
        micro2 = torch.from_numpy(micro2)
        is_numpy = True

    crop = 0.5 * crop
    x, y = micro1.shape

    embed_micro1 = torch.zeros([2*x, 2*y]).double()
    embed_micro1[:x, :y] = micro1

    embed_micro2 = torch.zeros([2*x, 2*y]).double()
    embed_micro2[:x, :y] = micro2

    normal = torch.zeros([2*x, 2*y]).double()
    normal[:x, :y] = torch.ones_like(micro1)

    two_point = torch.fft.fftshift( \
            torch.fft.ifftn( \
                torch.fft.fftn(embed_micro1).conj() * \
                torch.fft.fftn(embed_micro2)).real /\
            torch.fft.ifftn( \
                torch.fft.fftn(normal).conj() * \
                torch.fft.fftn(normal)).real \
            )

    # segment out the center
    x_start = x-int(crop*x)+1
    x_end = x+int(crop*x)
    y_start = y-int(crop*y)+1
    y_end = y+int(crop*y)

    two_point = two_point[x_start:x_end, \
                          y_start:y_end]

    # change back:
    if is_numpy:
        two_point = two_point.detach().numpy()

    if centered:
        return two_point
    else:
        if is_numpy:
            return np.fft.ifftshift(two_point)
        else:
            return torch.fft.ifftshift(two_point)


# ----------------------------------------------------------------
# Higher Order Statistical methods. 
# These are only going to be made for PERIODIC microstructures. 
# ----------------------------------------------------------------

def threepoint_np(micro1, micro2, crop=0.5, shift_tuple=(0, 0)):
    """
    SAME AS BELOW, but nonperiodic

    A simple piece of code for computing
    expanded local state three point statistics. 
    
    To facilitate development, the expanded local
    state is applied to 'micro1'

    ---------------------------------------------
    Examples for shift tuple:
    (1, 1) -> 
    
        0 1              3 2
        2 3     >>       1 0
        
    (0, 1)          

        0 1              1 0
        2 3     >>       3 2
    
    ---------------------------------------------

    Args:
        micro1 (torch.Tensor): just a spatial array
        micro2 (_type_): _description_
        shift_tuple (tuple): The indexes to shift
                             micro1. 
    """
    assert max(shift_tuple) < 11, "Only shifts less than 10 are allowed."
    assert min(shift_tuple) > -11
    
    micro1 = micro1[11:-11, 11:-11] * \
            micro1[(11 + shift_tuple[0]):(-11 + shift_tuple[0]),
                    (11 + shift_tuple[1]):(-11 + shift_tuple[1])
                    ]
    micro2 = micro2[11:-11, 11:-11]

    return twopoint_np(micro1, micro2, crop=crop)

def threepoint(micro1, micro2, shift_tuple=(0, 0)):
    """
    A simple piece of code for computing
    expanded local state three point statistics. 
    
    To facilitate development, the expanded local
    state is applied to 'micro1'

    ---------------------------------------------
    Examples for shift tuple:
    (1, 1) -> 
    
        0 1              3 2
        2 3     >>       1 0
        
    (0, 1)          

        0 1              1 0
        2 3     >>       3 2
    
    ---------------------------------------------

    Args:
        micro1 (torch.Tensor): just a spatial array
        micro2 (_type_): _description_
        shift_tuple (tuple): The indexes to shift
                             micro1. 
    """
    micro1 = micro1 * torch.roll(
        micro1,
        shifts = shift_tuple,
        dims = tuple(range(len(micro1.shape)))
    )
    
    return twopoint(micro1, micro2)

def threepoint_set(
            micro1, 
            micro2, 
            shift_tuple_set = [
                (0, 0),
                (1, 0),
                (1, 1),
                (0, 1),
                (-1, 1),
                (-1, 0),
                (-1, -1),
                (0, -1),
                (1, -1),
            ]
    ):
    """
    Expanded version of the previous function. 
    """
    stats = []
    for shift_tuple in shift_tuple_set:
        stats.append(threepoint(micro1, micro2, shift_tuple).unsqueeze(-1))
    
    stats = torch.cat(stats, dim=-1)
    return stats

# ----------------------------------------------------------------
# Class Methods
# ----------------------------------------------------------------

def pick_resolution():
    data = torch.load('./data/76XX_V1.pth')
    im1 = data[30, ...].unsqueeze(0)

    f = plt.figure(figsize=[12, 12])
    f.suptitle('Deciding on Resolution')
    for n, size in enumerate([32, 38, 40, 42, 48, 52, 56, 62]):
        ax = f.add_subplot(3, 3, n+1)
        ax.set_title(f"Size: {size}")
        ax.imshow(Resize(size)(im1)[0, ...])

    ax = f.add_subplot(3, 3, 9)
    ax.imshow(im1[0, ...])
    ax.set_title('Original')

    plt.show()
