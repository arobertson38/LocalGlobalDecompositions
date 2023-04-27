import numpy as np
import matplotlib.pyplot as plt
# from pymks.datasets.microstructure_generator import MicrostructureGenerator
from itertools import product
from scipy.stats import norm
import random
from queue import Queue
"""
As a word of warning. Most of these functions are generated for 2D structures. 

I am going to change this to be a helper function holding script. Just because I don't really ever use this for anything. 

Contact: arobertson38@gatech.edu
"""
# ------------------------------------------------------
# Plotting
# ------------------------------------------------------

def plot_4_phase(image, weights=[1.0, 0.75, 0.25, 0.5]):
    plt.imshow(image[..., 0] * weights[0] + image[..., 1] * weights[1] + \
               image[..., 2] * weights[2] + image[..., 3] * weights[3], cmap='viridis')
    plt.show()

# ------------------------------------------------------
# Resizing
# ------------------------------------------------------

def crop_in_two(arr, dim=2):
    """
    This method crops an image by 2. 
    """
    if dim == 2:
        dim1_filter = np.array([1-n%2 for n in range(arr.shape[0])], dtype=np.bool)
        dim2_filter = np.array([1-n%2 for n in range(arr.shape[1])], dtype=np.bool)

        arr2 = arr[dim1_filter, ...]
        return arr2[:, dim2_filter, ...]

    elif dim == 3:
        dim1_filter = np.array([1-n%2 for n in range(arr.shape[0])], dtype=np.bool)
        dim2_filter = np.array([1-n%2 for n in range(arr.shape[1])], dtype=np.bool)
        dim3_filter = np.array([1-n%2 for n in range(arr.shape[2])], dtype=np.bool)

        arr2 = arr[dim1_filter, ...]
        arr2 = arr2[:, dim2_filter, ...]
        return arr2[:, :, dim3_filter, ...]
    else:
        raise AttributeError("Only 2D and 3D are supported")

def zoomNphase(arr, factor, dim=2):
    '''
    This is a method that increases the resolution of a N-phase image by subdividing each voxel into factor**dim number
    of voxels.

    :param arr: The array that we are zooming in on. The last dimension must be number of phases
    :param factor: The number of voxels we are subdividing by
    :param dim: the number of dimensions of the image
    :return:
    '''
    assert len(arr.shape) == dim+1
    assert type(factor) == int
    assert factor >= 1

    shape = [int(factor*i) for i in arr.shape[:-1]]
    shape.append(arr.shape[-1])
    new = np.zeros(shape)

    for i in range(arr.shape[-1]):
        new[..., i] = zoom(arr[..., i], factor, dim)

    return new

def zoom(arr, factor, dim=2):
    '''
    This is a method that increases the resolution of a 1-phase image by subdividing each voxel into factor**dim number
    of voxels.

    This will be written to work on only an single phase image

    :param arr: The array that we are zooming in on
    :param factor: The number of voxels we are subdividing by
    :param dim: the number of dimensions of the image
    :return:
    '''
    assert len(arr.shape) == dim
    assert factor >= 1
    assert type(factor) == int

    if dim == 2:
        new = np.zeros([arr.shape[0] * factor, arr.shape[1] * factor])
        for i in range(0, arr.shape[0]):
            for j in range(0, arr.shape[1]):
                if arr[i, j] != 0:
                    new[i*factor:(i+1)*factor, j*factor:(j+1)*factor] = arr[i, j]
    elif dim == 3:
        new = np.zeros([arr.shape[0] * factor, arr.shape[1] * factor], arr.shape[2] * factor)
        for i in range(0, arr.shape[0]):
            for j in range(0, arr.shape[1]):
                for k in range(0, arr.shape[2]):
                    if arr[i, j, k] != 0:
                        new[i*factor:(i+1)*factor, j*factor:(j+1)*factor, k*factor:(k+1)*factor] = arr[i, j, k]
    else:
        raise AttributeError

    return new


def squared_exponential_autocorrelation(N, l, mean):
    """
    This method returns the autocorrelation matrix (of size NxN) for 
    an exponential kernel with lengthscale parameter 0 < l < N
    """
    assert l > 0
    assert l < N

    l = l / N

    side = np.linspace(-0.5, 0.5, N)
    XX, YY = np.meshgrid(side, side)

    auto = np.exp((-1 / l) * (np.square(XX) + np.square(YY)))

    return np.fft.ifftshift(auto + mean ** 2)



def generate_random_structure(N,M):
    """
    This method generates a random structure and computes its
    autocorrelation. 
    """
    struct = np.random.uniform(size = [N, M])
    struct[struct >= 0.5] = 1.0
    struct[struct < 0.9] = 0.0

    autos = autocorrelations(struct[..., np.newaxis])
    return struct, autos[..., 0]

def generate_one_phase(uncentered_autocorrelation):
    """
    Assumes that its just the autocorrelation (so [N x N] or [N] or [NxNxN])

    I want to see whats going on with the sampling code. So, this method is
    a direct copying of the generate method from the StatisticsGenerator class.
    
    The intention is that this will allow me to test the code in a closed
    environment without added noise from the remainder of the code. 
    """
    index = tuple([0] * len(uncentered_autocorrelation.shape))
    eigs = np.fft.fftn(uncentered_autocorrelation)
    mean = np.sqrt(eigs[index].real / np.product(eigs.shape))
    eigs[index] = 0
    #  print(mean)
    if eigs.min() < -1e-12:
        raise ValueError('The autocovariance contains at least one negative eigenvalue (' + \
                str(eigs.min()) + ').')

    eigs = np.sqrt(np.abs(eigs) /
                   np.product(eigs.shape))
    eps = np.random.normal(loc=0.0, scale=1.0, size=eigs.shape) + \
          1j * np.random.normal(loc=0.0, scale=1.0, size=eigs.shape)
    new = np.fft.fftn(eigs * eps)

    return new.real + mean, new.imag + mean


def _covariance_matrix(kern):
    """
    This method returns the covariance matrix associated with the
    passed covariance kernel. 
    """
    if len(kern.shape) == 1:
        # 1D case
        cov = np.zeros([kern.shape[0], kern.shape[0]])
        for i in range(kern.shape[0]):
            cov[i, :] = kern
            kern = np.roll(kern, 1)
        return cov

    elif len(kern.shape) == 2:
        # 2D case
        cov = np.zeros([np.product(kern.shape), np.product(kern.shape)])
        for i in range(kern.shape[0]):
            for j in range(kern.shape[1]):
                cov[i * kern.shape[0] + j, :] = kern.flatten()
                kern = np.roll(kern, 1, axis=1)
            kern = np.roll(kern, 1, axis=0)

        return cov
                
    else:
        raise AttributeError("Unsupported size")

def generate_one_phase_with_multivariate(uncentered_autocorrelation):
    """
    This method generates sampled instances using numpy's multivariate gaussian
    method. This is to test if something is going on as a result of the implementation
    as the code gets more complicated. 

    Assumes that its just the autocorrelation (so [N x N] or [N] or [NxNxN])

    I want to see whats going on with the sampling code. So, this method is
    a direct copying of the generate method from the StatisticsGenerator class.
    
    The intention is that this will allow me to test the code in a closed
    environment without added noise from the remainder of the code. 
    """
    # transfer to covariance kernel and compute means
    index = tuple([0] * len(uncentered_autocorrelation.shape))
    eigs = np.fft.fftn(uncentered_autocorrelation)
    mean = np.sqrt(eigs[index].real / np.product(eigs.shape))
    eigs[index] = 0
    cov = np.fft.ifftn(eigs).real

    # produce covariance matrix
    cov_matrix = _covariance_matrix(cov)

    # sampling

    return np.random.multivariate_normal(mean * np.ones([ \
                        np.product(uncentered_autocorrelation.shape)]), \
                        cov_matrix).reshape(uncentered_autocorrelation.shape, 
                                order='C')

def generator(cov, mean):
    """
    A method for generating new microstructures given a covariance matrix and a mean.
    for 2D
    :param cov:
    :param mean:
    :return:
    """
    eigs = np.fft.fft2(cov).real
    print(eigs.min())
    eigs = np.sqrt(np.abs(eigs) / (cov.shape[0] * cov.shape[1]))
    eps = np.random.normal(loc=0.0, scale=1.0, size=cov.shape) + 1j * np.random.normal(loc=0.0, scale=1.0, size=
                                                                                       cov.shape)
    new = np.fft.fft2(eigs * eps)

    return new.real+mean, new.imag+mean

def p2_crosscorrelation(arr1, arr2):
    """
    defines the crosscorrelation between arr1 and arr2:
    :param arr1:
    :param arr2:
    :return:
    """
    ax = list(range(0, len(arr1.shape)))
    arr1_FFT = np.fft.rfftn(arr1, axes=ax)
    arr2_FFT = np.fft.rfftn(arr2, axes=ax)
    return np.fft.irfftn(arr1_FFT.conjugate() * arr2_FFT, s=arr1.shape, axes=ax).real / np.product(
        arr1.shape)

def twopointstats(str):
    """
    THis method computes and returns the full two point statistics: ONLY FOR 2D
    :param str:
    :return:
    """
    stats = np.zeros_like(str, dtype=np.float64)
    for i in range(0, str.shape[-1]):
        stats[..., i] = p2_crosscorrelation(str[..., 0], str[..., i])
    return stats

def autocorrelations(str):
    """
    THis method computes and returns the complete set of autocorrelations: ONLY FOR 2D
    :param str:
    :return:
    """
    assert str.shape[-1] < 15
    stats = np.zeros_like(str)
    for i in range(0, str.shape[-1]):
        stats[..., i] = p2_crosscorrelation(str[..., i], str[..., i])
    return stats

#def pymks_generator(size=(201,201), grain_size=(20, 100), rand=None):
#    """
#    This is a method for calling pymks's built in generator so that I don't
#    forget how to do it myself.
#    :param size: This is the length of each domain axis in voxels.
#    :param grain_size: This is the average grain size widths in voxels (x width, y width, z width)
#    :param rand: the random seed
#    :return:
#    """
#    return MicrostructureGenerator(n_samples=1, size=size, n_phases=2, grain_size=grain_size, seed=rand).generate()[0, ...]

def gaussian_kernel(inverse_covariance, size=(201, 201)):
    """
        Produces the frequency domain of an quassian filter with integral of 1.
        It returns a 'real' fft transformation.
        :param size: the NxNxN dimension N
        :param sigma: the standard deviation, 1.165 is used to approximate a 7x7x7 gaussian blur
        :return:
        """
    assert inverse_covariance.shape[0] == 2
    xx, yy = np.meshgrid(np.linspace(-(size[0] - 1) / 2., (size[0] - 1) / 2., size[0]),
                         np.linspace(-(size[1] - 1) / 2., (size[1] - 1) / 2., size[1]))
    arr = np.concatenate([xx[..., np.newaxis], yy[..., np.newaxis]], axis=-1)

    kernel = np.squeeze(np.exp(-0.5 * arr[..., np.newaxis, :] @ inverse_covariance @ arr[..., np.newaxis]))

    return np.fft.fftn(np.fft.ifftshift(kernel / np.sum(kernel)))

def translate_cdf(arr):
    """
    This method translates the cdf by first computing the corresponding cdf using the standard normal and then
    inverting it with the desired cdf.

    :param arr:
    :return:
    """
    mean = arr.mean()
    std = arr.std()
    arr = norm.cdf((arr - mean) / std)
    return np.heaviside(arr - (1-mean), 1)

def project_from_labels(labels, dimensions=2):
    shapes = list(labels.shape)
    shapes.append(len(np.unique(labels)))
    arr = np.zeros(shapes)
    for i in range(arr.shape[-1]):
        arr[labels==i, i] = 1
    
    # ordering them by volume fraction
    arr2 = np.zeros_like(arr)
    vol_frac = (-1 * np.array([arr[..., i].mean() for i in range(arr.shape[-1])])).argsort()
    for j in range(len(vol_frac)):
        arr2[..., j] = arr[..., vol_frac[j]]
    return arr2

# -------------------------------------------------------------------------------
# This section includes methods for computing derivatives
# -------------------------------------------------------------------------------

def tf_x_derivative(short=1.0, long=np.sqrt(2)):
    """
    I think, because matplotlib displays images weirdly, the x and y axes are inverted. 
    """
    return (-1/4) * np.array([[0, 0, -1/short, 0, 0],
                              [0, 0, 4/short, 0, 0],
                              [0, 0, -6/short, 0, 0],
                              [0, 0, 4/short, 0, 0],
                              [0, 0, -1/short, 0, 0]])

def tf_y_derivative(short=1.0, long=np.sqrt(2)):
    """
    I think, because matplotlib displays images weirdly, the x and y axes are inverted. 
    """
    return (-1/4) * np.array([[0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [-1/short, 4/short, -6/short, 4/short, -1/short],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0]])

def two_forward_derivative(short=1.0, long=np.sqrt(2)):
    return (-1/16) * np.array([[-1/long, 0, -1/short, 0, -1/long],
                              [0, 4/long, 4/short, 4/long, 0],
                              [-1/short, 4/short, -12/short - 12/long, 4/short, -1/short],
                              [0, 4/long, 4/short, 4/long, 0],
                              [-1/long, 0, -1/short, 0, -1/long]])

def approx_matrix_derivative(arr, matrix_deriv=two_forward_derivative, short_length=1.0, long_length=np.sqrt(2)):
    """
    assumes that the field enters uncentered
    """
    sh = arr.shape
    cent = np.fft.fftshift(arr)[int(sh[0] / 2 - 2):int(sh[0] / 2 + 3), int(sh[1] / 2 - 2):int(sh[1] / 2 + 3)]
    return (cent * matrix_deriv(short_length, long_length)).sum()

# -------------------------------------------------------------------------------
# Counting Feature occurance
# -------------------------------------------------------------------------------

def _featuresNP2D(image):
    """
    This code is the backend for the feature occurance code.

    It assumes the microstructure is nonperiodic. 
    
    It runs rays in the x direction and counts the number
    of features that it hits as it runs across. 
    """
    features = []
    x_max = image.shape[0]
    for y in range(image.shape[1]):
        x = 0
        count_y = 0
        while x < x_max:
            if image[x, y] > 0:
                x += 1
                while (x < x_max) and image[x, y] > 0:
                    x += 1

                count_y += 1
            else:
                x += 1
        features.append(count_y)

    return features

def _features2D(image):
    """
    This code is the backend for the feature occurance code.

    It assumes the microstructure is periodic. 
    
    It runs rays in the x direction and counts the number
    of features that it hits as it runs across. 
    """
    features = []
    x_max = image.shape[0]
    for y in range(image.shape[1]):
        x = 0
        count_y = 0
        flag = True
        while x < x_max:
            if image[x, y] > 0:
                if x==0:
                    if image[-1, y] > 0:
                        flag = False
                x_end = x+1
                while image[x_end%x_max, y] > 0:
                    if x_end%x_max == x:
                        # then we have found a while line of one phase
                        flag = True
                        break
                    x_end += 1

                if flag:
                    # add the chord length to the list.
                    count_y += 1
                x = x_end
                flag = True
            else:
                x += 1
        features.append(count_y)

    return features

def features2D(image, axis=0, periodic=True):
    """
    This code computes the number of features along a given direction. 

    axis == 0 => the x direction
    axis == 1 => the y direction
    axis == -1 => both returns [x_chords, y_chords]
    """
    assert len(image.shape) == 2

    feature_func = _features2D if periodic else _featuresNP2D

    if axis == 0:
        return feature_func(image)
    elif axis == 1:
        return feature_func(image.transpose())
    elif axis == -1:
        return [feature_func(image), feature_func(image.transpose())]
    else:
        raise AttributeError("Invalid axis. 0: numpy x, 1: numpy y, -1: both")

# -------------------------------------------------------------------------------
# 2D Chord Length Distribution Code
# -------------------------------------------------------------------------------

def debug(flag=True):
    if flag:
        plt.figure(); plt.show()

def _cldNP2D_colored(image, bins):
    """
    This code is the backend for the CLD code provided below

    However, its the version for handling nonperiodic shifts. 

    it passes only the x direction
    """
    mask = np.zeros_like(image, dtype=bool)


    chords = []
    x_max = image.shape[0]
    for y in range(image.shape[1]):
        x = 0
        flag = True
        while x < x_max:
            if image[x, y] > 0:
                if x==0:
                    # if the chord starts on the boundary:
                    # IGNORE IT
                    flag = False
                x_end = x+1

                if x_end == x_max:
                    flag = False
                else:
                    while image[x_end, y] > 0:
                        x_end += 1
                        if x_end == x_max:
                            # if the chord ends on the boundary:
                            # IGNORE IT
                            flag = False
                            break

                if flag:
                    # add the chord length to the list.
                    chords.append(x_end - x)

                    if x_end - x <= 4:
                        mask[x, y] = True

                x = x_end
                flag = True
            else:
                x += 1

    if type(bins) != type(True):
        if type(bins) == np.ndarray:
            chords = list(np.histogram(chords, bins=bins)[0])
        else:
            chords = list(np.histogram(chords, bins=np.linspace(1, image.shape[0], bins+1))[0])

    image = image.transpose()
    mask = mask.transpose()
    image = np.concatenate([image[..., np.newaxis], 1-image[..., np.newaxis]], axis=-1)
    image[mask, 0] = 0
    image[mask, 1] = 0
    image = np.concatenate([mask.astype(int)[..., np.newaxis], image], axis=-1)

    print(mask.sum())
    plt.imshow(image)
    plt.show()

    return chords

def _cldNP2D(image, bins):
    """
    This code is the backend for the CLD code provided below

    However, its the version for handling nonperiodic shifts. 

    it passes only the x direction
    """
    chords = []
    x_max = image.shape[0]
    for y in range(image.shape[1]):
        x = 0
        flag = True
        while x < x_max:
            if image[x, y] > 0:
                if x==0:
                    # if the chord starts on the boundary:
                    #if (image[:, y] > 0).all():
                        # check if the whole row is a chord.
                    #    flag = True
                    #else:
                        # IGNORE IT
                    #    flag = False
                    # IGNORE IT
                    flag = False

                x_end = x+1

                if x_end == x_max:
                    flag = False
                else:
                    while image[x_end, y] > 0:
                        x_end += 1
                        if x_end == x_max:
                            # if the chord ends on the boundary:
                            # IGNORE IT
                            flag = False
                            break

                if flag:
                    # add the chord length to the list.
                    chords.append(x_end - x)
                x = x_end
                flag = True
            else:
                x += 1

    if type(bins) != type(True):
        if type(bins) == np.ndarray:
            chords = list(np.histogram(chords, bins=bins)[0])
        else:
            chords = list(np.histogram(chords, bins=np.linspace(1, image.shape[0], bins+1))[0])
    return chords

def _cld2D(image, bins):
    """
    This code is the backend for the CLD code provided below

    it passes only the x direction
    """
    chords = []
    x_max = image.shape[0]
    for y in range(image.shape[1]):
        x = 0
        flag = True
        while x < x_max:
            if image[x, y] > 0:
                if x==0:
                    if image[-1, y] > 0:
                        flag = False
                x_end = x+1
                while image[x_end%x_max, y] > 0:
                    if x_end%x_max == x:
                        # then we have found a while line of one phase
                        flag = True
                        break
                    x_end += 1

                if flag:
                    # add the chord length to the list.
                    chords.append(x_end - x)
                x = x_end
                flag = True
            else:
                x += 1
    if type(bins) != type(True):
        if type(bins) == np.ndarray:
            chords = list(np.histogram(chords, bins=bins)[0])
        else:
            chords = list(np.histogram(chords, bins=np.linspace(1, image.shape[0], bins+1))[0])
    return chords


def cld2D(image, bins, axis=0, periodic=False):
    """
    this is code that computes the chord length distribution for 2D PERIODIC microstructures along
    either the X or Y directions. 

    This uses numpy convention. Therefore, x is the visual y

    axis == 0 => the x direction
    axis == 1 => the y direction
    axis == -1 => both returns [x_chords, y_chords]
    """
    assert len(image.shape) == 2

    _cld_func = _cld2D if periodic else _cldNP2D

    if axis == 0:
        return _cld_func(image, bins)
    elif axis == 1:
        return _cld_func(image.transpose(), bins)
    elif axis == -1:
        return [_cld_func(image, bins), _cld_func(image.transpose(), bins)]
    else:
        raise AttributeError("Invalid axis. 0: numpy x, 1: numpy y, -1: both")


def plot_cld(weights, length=201):
    """
    This method plots the Chord Length Distributions produced by cld2D (above)
    """
    f = plt.figure(); ax = f.add_subplot(111)
    bins = np.linspace(1, length, len(weights) + 1)
    ax.hist(bins[:-1], weights=weights, bins=bins)
    return f, ax



# -------------------------------------------------------------------------------
# Generating Shape Based Microstructures
# -------------------------------------------------------------------------------

def square(size):
    return np.ones([int(2*size+1), int(2*size+1)])

def diagonal(size):
    assert size > 0
    a = np.ones(int(size*2 + 1))
    b = np.ones(int(size*2))
    return np.diag(a,0) + np.diag(b, 1) + np.diag(b, -1)

def X(size):
    a = diagonal(size) 
    a = a + a[:, ::-1]
    a[a>0.0] = 1.0
    return a

def diamond(size):
    shape = np.zeros([int(2*size+1), int(2*size+1)])
    for i in range(2*size+1):
        for j in range(2*size+1):
            if np.abs(i-size) + np.abs(j-size) <= size:
                shape[i, j] = 1.0
    return shape

def circle(size):
    shape = np.zeros([int(2*size+1), int(2*size+1)])
    for i in range(2*size+1):
        for j in range(2*size+1):
            if np.square(i-size) + np.square(j-size) <= np.square(size):
                shape[i, j] = 1.0
    return shape

def shape_structure_generation(dim=100, choices=[1,2,3,4,5,6], number_of_cubes_placed=30, max_iter=40, shape=square):
    center_x = []
    center_y = []
    collected_sizes = []
    iter = 0
    placed = 0
    struct = np.zeros([dim, dim])
    loc_choice = list(range(0, dim))
    while placed < number_of_cubes_placed:
        while iter < max_iter:
            size = random.choice(choices)
            x_select = random.choice(loc_choice[size:(-1*size)])
            y_select = random.choice(loc_choice[size:(-1*size)])
            if np.all(np.logical_or(np.abs(np.array(center_x) - x_select) > (size + 1 + np.array(collected_sizes)), \
                    np.abs(np.array(center_y) - y_select) > (size + 1 + np.array(collected_sizes)))):
                struct[(x_select-(size)):(x_select+size+1), (y_select-(size)):(y_select+size+1)] = shape(size)
                center_x.append(x_select)
                center_y.append(y_select)
                collected_sizes.append(size)
                iter = max_iter
                placed += 1
            else:
                iter += 1
        iter = 0
    return struct, collected_sizes

# -------------------------------------------------------------------------------
# Methods for extracting the center of autocorrelations
# -------------------------------------------------------------------------------

def _disect_floodfill(arr, cutoff=0.1, radius_cutoff_fraction=0.33, return_mask=False):
    """
    The user *should* input the covariance function, not the autocorrelation

    This will remove the center of the covariance.

    We assume that the arr is input centered (no np.fft.fftshift is necessary)
    
    Array is output centered, along with the mask and the longest direction

    We can also use the space inversion symmetry of the covariance to half the 
    number of computations. 

    :param arr:
    :param cutoff:
    :return:
    """
    
    arr_max = arr.max() * cutoff
    size = min(arr.shape)
    radius_cutoff = radius_cutoff_fraction * size
    xmax = arr.shape[0]
    ymax = arr.shape[1]

    centx = int(xmax / 2)
    centy = int(ymax / 2)

    flags = np.zeros_like(arr)
    flags[centx, centy] = 1
    voxel_queue = Queue()
    [voxel_queue.put(item) for item in
     [(centx - 1, centy), (centx, centy + 1), (centx + 1, centy)]]

    maxup = 0
    maxdown = 0
    maxleft = 0
    maxupright = 0
    maxdownright = 0


    while not voxel_queue.empty():
        x, y = voxel_queue.get()
        direc = np.linalg.norm(np.array([x, y]) - np.array([centx, centy]))

        if arr[x, y] > arr_max and direc < radius_cutoff and flags[x, y] != 1 and x>-1 and y >= centy and x < xmax and y < ymax:
            if x == centx:
                maxleft = max(maxleft, direc)
            elif y == centy and x < centx:
                maxdown = max(maxdown, direc)
            elif y == centy and x > centx:
                maxup = max(maxup, direc)
            elif y == x:
                maxupright = max(maxupright, direc)
            elif abs(y - centy) == abs(x - centx):
                maxdownright = max(maxdownright, direc)
                
            flags[x, y] = 1.0
            voxel_queue.put((x+1, y))
            voxel_queue.put((x-1, y))
            voxel_queue.put((x, y+1))
            voxel_queue.put((x, y-1))
    
    flags[:, :centy] = np.flip(np.flip(flags, axis=0), axis=1)[:, :centy]
    if return_mask:
        return np.fft.ifftshift(flags), np.array([maxup, maxdown, maxleft, maxupright, maxdownright]).mean()
    else:
        return np.fft.ifftshift(arr * flags), np.array([maxup, maxdown, maxleft, maxupright, maxdownright]).mean()


def _disect_floodfill_3D(arr, cutoff=0.1, radius_cutoff_fraction=0.33, return_mask=False):
    """
    The user *should* input the covariance function, not the autocorrelation

    This will remove the center of the covariance.

    We assume that the arr is input centered (no np.fft.fftshift is necessary)
    
    Array is output centered, along with the mask and the longest direction

    We can also use the space inversion symmetry of the covariance to half the 
    number of computations. 

    :param arr:
    :param cutoff:
    :return:
    """
    
    arr_max = arr.max() * cutoff
    size = min(arr.shape)
    radius_cutoff = radius_cutoff_fraction * size
    xmax = arr.shape[0]
    ymax = arr.shape[1]
    zmax = arr.shape[1]

    centx = int(xmax / 2)
    centy = int(ymax / 2)
    centz = int(zmax / 2)

    flags = np.zeros_like(arr)
    flags[centx, centy, centz] = 1
    voxel_queue = Queue()
    [voxel_queue.put(item) for item in
     [(centx - 1, centy, centz), (centx, centy + 1, centz), (centx + 1, centy, centz), \
             (centx, centy, centz-1), (centx, centy, centz+1)]]

    maxup = 0
    maxdown = 0
    maxleft = 0
    maxupright = 0
    maxdownright = 0

    maxzup = 0
    maxzupright = 0
    maxzupup = 0
    maxzupdown = 0

    maxzdown = 0
    maxzdownright = 0
    maxzdownup = 0
    maxzdowndown = 0


    while not voxel_queue.empty():
        x, y, z = voxel_queue.get()
        direc = np.linalg.norm(np.array([x, y, z]) - np.array([centx, centy, centz]))

        if arr[x, y, z] > arr_max and direc < radius_cutoff and flags[x, y, z] != 1 and x>-1 and y >= centy and x < xmax and y < ymax and z >-1 and z < zmax:

            # These conditions are checking the width of the kernel
            if (x == centx) and (z == centz):
                maxleft = max(maxleft, direc)
            elif y == centy and x < centx and (z==centz):
                maxdown = max(maxdown, direc)
            elif y == centy and x > centx and z==centz:
                maxup = max(maxup, direc)
            elif y == x and z==centz:
                maxupright = max(maxupright, direc)
            elif abs(y - centy) == abs(x - centx) and z==centz:
                maxdownright = max(maxdownright, direc)
            # I started adding here: z>centz
            elif (x==centx) and y==centy and z>centz:
                maxzup = max(maxzup, direc)
            elif (x==centx) and z==y:
                maxzupright = max(maxzupright, direc)
            elif (y==centy) and z==x and z>centz:
                maxzupup = max(maxzupup, direc)
            elif (y==centy) and abs(z-centz)==abs(x-centx) and z>centz:
                maxzupdown = max(maxzupdown, direc)
            # I continued adding: z<centz
            elif (x==centx) and y==centy and z<centz:
                maxzdown = max(maxzdown, direc)
            elif (x==centx) and abs(z-centz)==abs(y-centy) and z<centz:
                maxzdownright = max(maxzdownright, direc)
            elif (y==centy) and abs(z-centz)==abs(x-centx) and z<centz and x>centx:
                maxzdownup = max(maxzdownup, direc)
            elif (y==centy) and z==x and z<centz:
                maxzdowndown = max(maxzdowndown, direc)
            
            flags[x, y, z] = 1.0
            voxel_queue.put((x+1, y, z))
            voxel_queue.put((x-1, y, z))
            voxel_queue.put((x, y+1, z))
            voxel_queue.put((x, y-1, z))
            voxel_queue.put((x, y, z-1))
            voxel_queue.put((x, y, z+1))
    
    flags[:, :centy, :] = np.flip(np.flip(np.flip(flags, axis=0), axis=2), axis=1)[:, :centy, :]
    if return_mask:
        return np.fft.ifftshift(flags), np.array([maxup, maxdown, maxleft, maxupright, maxdownright,\
                maxzup, maxzupright, maxzupup, maxzupdown, maxzdown, maxzdownright, \
                maxzdownup, maxzdowndown]).mean()
    else:
        return np.fft.ifftshift(arr * flags), np.array([maxup, maxdown, maxleft, maxupright, maxdownright, \
                maxzup, maxzupright, maxzupup, maxzupdown, maxzdown, maxzdownright, \
                maxzdownup, maxzdowndown]).mean()


def disect(arr1, cutoff=0.2, radius_cutoff_fraction=0.33, twoD=True):
    '''
    A method that wraps the disect function above to transfer from autocorrelation to covariance

    Assumes input autocorrelation is uncentered
    '''
    arr = np.fft.fftn(arr1)
    arr[0,0] = 0.0
    arr = np.fft.ifftn(arr).real
    if twoD:
        cent, max_dir = _disect_floodfill(np.fft.fftshift(arr), cutoff=cutoff, radius_cutoff_fraction=radius_cutoff_fraction, return_mask=True)
        return arr1 * cent, max_dir
    else:
        cent, max_dir = _disect_floodfill_3D(np.fft.fftshift(arr), cutoff=cutoff, radius_cutoff_fraction=radius_cutoff_fraction, return_mask=True)
        return arr1 * cent, max_dir


def rescale(arr, length, desired_length, twoD=True):
    """
    This method and all its called methods assume that the field arr is not centered. 
    """
    if twoD:
        if length > desired_length:
            return downsample(arr, length, desired_length)
        elif length < desired_length:
            return upsample(arr, length, desired_length)
        else:
            return arr
    else:
        if length > desired_length:
            return downsample_3D(arr, length, desired_length)
        elif length < desired_length:
            return upsample_3D(arr, length, desired_length)
        else:
            return arr


def upsample(arr, length, desired_length):
    # Need to be careful with this. They are incorrectly scaled because of how the FFT is
    # implemented in Numpy. For my application, it doesn't matter. But, for the future it may
    assert desired_length <= np.array(arr.shape).min() / 2
    old_scales = arr.shape
    new_scales = np.array(arr.shape) * length / desired_length
    new_arr = np.zeros_like(arr, dtype=complex)
    arr = np.fft.fftshift(arr)[int(old_scales[0] / 2 - new_scales[0] / 2):int(old_scales[0] / 2 + new_scales[0] / 2 + 1), int(old_scales[1] / 2 - new_scales[1] / 2):int(old_scales[1] / 2 + new_scales[1] / 2 + 1)]
    arr = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(arr)))
    new_arr[int(old_scales[0] / 2 - new_scales[0] / 2):int(old_scales[0] / 2 + new_scales[0] / 2 + 1), int(old_scales[1] / 2 - new_scales[1] / 2):int(old_scales[1] / 2 + new_scales[1] / 2 + 1)] = arr
    new_arr = np.fft.ifftn(np.fft.ifftshift(new_arr)).real
    return new_arr

def upsample_3D(arr, length, desired_length):
    # Need to be careful with this. They are incorrectly scaled because of how the FFT is
    # implemented in Numpy. For my application, it doesn't matter. But, for the future it may
    #
    # This is the 3D implementation of upsample
    assert desired_length <= np.array(arr.shape).min() / 2
    old_scales = arr.shape
    new_scales = np.array(arr.shape) * length / desired_length
    new_arr = np.zeros_like(arr, dtype=complex)
    arr = np.fft.fftshift(arr)[int(old_scales[0] / 2 - new_scales[0] / 2):int(old_scales[0] / 2 + new_scales[0] / 2 + 1), \
                            int(old_scales[1] / 2 - new_scales[1] / 2):int(old_scales[1] / 2 + new_scales[1] / 2 + 1), \
                            int(old_scales[2] / 2 - new_scales[2] / 2):int(old_scales[2] / 2 + new_scales[2] / 2 + 1)]
    arr = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(arr)))
    new_arr[int(old_scales[0] / 2 - new_scales[0] / 2):int(old_scales[0] / 2 + new_scales[0] / 2 + 1), \
            int(old_scales[1] / 2 - new_scales[1] / 2):int(old_scales[1] / 2 + new_scales[1] / 2 + 1), \
            int(old_scales[2] / 2 - new_scales[2] / 2):int(old_scales[2] / 2 + new_scales[2] / 2 + 1)] = arr
    new_arr = np.fft.ifftn(np.fft.ifftshift(new_arr)).real
    return new_arr

def downsample(arr, length, desired_length):
    # Need to be careful with this. They are incorrectly scaled because of how the FFT is
    # implemented in Numpy. For my application, it doesn't matter. But, for the future it may
    assert length < np.array(arr.shape).min() / 2
    old_scales = arr.shape
    new_scales = np.array(arr.shape) * desired_length / length
    new_arr = np.zeros_like(arr)
    arr = np.fft.ifftshift(np.fft.fftshift(np.fft.fftn(arr))[
          int(old_scales[0] / 2 - new_scales[0] / 2):int(old_scales[0] / 2 + new_scales[0] / 2 + 1),
          int(old_scales[1] / 2 - new_scales[1] / 2):int(old_scales[1] / 2 + new_scales[1] / 2 + 1)])
   
    arr = np.fft.fftshift(np.fft.ifftn(arr).real)
    new_arr[int(old_scales[0] / 2 - new_scales[0] / 2):int(old_scales[0] / 2 + new_scales[0] / 2 + 1), int(old_scales[1] / 2 - new_scales[1] / 2):int(old_scales[1] / 2 + new_scales[1] / 2 + 1)] = arr
    return np.fft.ifftshift(new_arr)

def downsample_3D(arr, length, desired_length):
    # Need to be careful with this. They are incorrectly scaled because of how the FFT is
    # implemented in Numpy. For my application, it doesn't matter. But, for the future it may
    #
    # This is the 3D code. 
    assert length < np.array(arr.shape).min() / 2
    old_scales = arr.shape
    new_scales = np.array(arr.shape) * desired_length / length
    new_arr = np.zeros_like(arr)
    arr = np.fft.ifftshift(np.fft.fftshift(np.fft.fftn(arr))[
          int(old_scales[0] / 2 - new_scales[0] / 2):int(old_scales[0] / 2 + new_scales[0] / 2 + 1),
          int(old_scales[1] / 2 - new_scales[1] / 2):int(old_scales[1] / 2 + new_scales[1] / 2 + 1),
          int(old_scales[2] / 2 - new_scales[2] / 2):int(old_scales[2] / 2 + new_scales[2] / 2 + 1)])
   
    arr = np.fft.fftshift(np.fft.ifftn(arr).real)
    new_arr[int(old_scales[0] / 2 - new_scales[0] / 2):int(old_scales[0] / 2 + new_scales[0] / 2 + 1), \
            int(old_scales[1] / 2 - new_scales[1] / 2):int(old_scales[1] / 2 + new_scales[1] / 2 + 1), \
            int(old_scales[2] / 2 - new_scales[2] / 2):int(old_scales[2] / 2 + new_scales[2] / 2 + 1)] = arr
    return np.fft.ifftshift(new_arr)

# ------------------------------------------------------------------------------
# New code that I am working on
# ------------------------------------------------------------------------------

def cross():
    """
    This is a method that returns first the shifts and second the axis for the local
    mean method that is defined below. 
    """
    return [1, -1, 1, -1], [0, 0, 1, 1]

def local_mean(arr, selector=cross):
    """
    This is a method that sums the vector values across all of the surrounding neighbors. Which 
    neighbors is defined by the selector method. 

    We assume that the center voxel is always desired. 
    """
    shifts, axes = selector()
    final_arr = arr.copy()
    for i in range(len(shifts)):
        final_arr += np.roll(arr, shift=shifts[i], axis=axes[i])
    return final_arr

def local_square_mean(arr):
    """
    The above mean method wasn't as general as I had originally hoped. 
    """
    shifts = [(0,1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (-1, 1), (1, -1)]
    final_arr = arr.copy()
    for i in range(len(shifts)):
        final_arr += np.roll(np.roll(arr, shift=shifts[i][0], axis=0), shift=shifts[i][1], axis=1)
    return final_arr

# -------------------------------------------------------------------------------
# HERE LIES OLD CODE WHICH SHOULD PROBABLY GET DELETED
# Mostly on the fence because I haven't thoroughly tested the code above.
# -------------------------------------------------------------------------------


def _disect(arr, cutoff=0.1, radius_cutoff_fraction=0.33, return_mask=False):
    """
    The user *should* input the covariance function, not the autocorrelation

    This will remove the center of the covariance.

    We assume that the arr is input centered (no np.fft.fftshift is necessary)
    
    Array is output centered, along with the mask and the longest direction

    We can also use the space inversion symmetry of the covariance to half the 
    number of computations. 

    :param arr:
    :param cutoff:
    :return:
    """
    assert arr.shape[0] == arr.shape[1]
    size = arr.shape[0]
    radius_cutoff = radius_cutoff_fraction * size
    
    bot = int(size / 2)
    flags = np.zeros_like(arr)
    flags[bot, bot] = 1
    voxel_queue = Queue()
    [voxel_queue.put(item) for item in
     [[bot - 1, bot], [bot - 1, bot + 1], [bot, bot + 1], [bot + 1, bot], [bot + 1, bot + 1]]]

    max_direc = 0
    while not voxel_queue.empty():
        vox = voxel_queue.get()
        direc = np.array(vox) - np.array([bot, bot])
        if arr[tuple(vox)] > arr.max() * cutoff and np.linalg.norm(direc) < radius_cutoff:
            flags[tuple(vox)] = 1.0
            if np.linalg.norm(direc) > max_direc:
                max_direc = np.linalg.norm(direc)
            if vox[0] == vox[1]:
                [voxel_queue.put(item) for item in
                 [[vox[0] + 1, vox[1] + 1], [vox[0] + 1, vox[1]], [vox[0], vox[1] + 1]]]
            elif abs(vox[0] - bot) == abs(vox[1] - bot):
                [voxel_queue.put(item) for item in
                 [[vox[0] - 1, vox[1] + 1], [vox[0] - 1, vox[1]], [vox[0], vox[1] + 1]]]
            else:
                if direc[1] > 0 and direc[1] > abs(direc[0]):
                    voxel_queue.put([vox[0], vox[1] + 1])
                elif direc[0] > 0 and direc[0] > abs(direc[1]):
                    voxel_queue.put([vox[0] + 1, vox[1]])
                else:
                    voxel_queue.put([vox[0] - 1, vox[1]])

    flags[:, bot] *= 0.5
    flags += np.flip(np.flip(flags, axis=0), axis=1)
    if return_mask:
        return np.fft.ifftshift(flags), max_direc
    else:
        return np.fft.ifftshift(arr * flags), max_direc


def disect_old(arr, cutoff=0.2, radius_cutoff_fraction=0.33):
    '''
    A method that wraps the disect function above to transfer from autocorrelation to covariance

    Assumes input autocorrelation is uncentered
    '''
    arr = np.fft.fftn(arr)
    arr[0,0] = 0.0
    arr = np.fft.ifftn(arr).real
    cent, max_dir = _disect(np.fft.fftshift(arr), cutoff=cutoff, radius_cutoff_fraction=radius_cutoff_fraction)
    return cent, max_dir

def rescaling_indicator(arr, cutoff=0.1, radius_cutoff_fraction=0.33, downshift=5):
    """
    Assume that it does not come in centered
    """
    arr = np.fft.fftn(arr)
    arr[0,0] = 0.0
    arr = np.fft.ifftn(arr).real

    rescaled_arr = np.fft.fftshift(downsample(arr, 1, 1/downshift))

    rescaled_arr = rescaled_arr[int(arr.shape[0] * (1 / 2 - 1 / (2*downshift))):int(arr.shape[0] * (1 / 2 + 1 / (2*downshift))+1), int(arr.shape[1] * (1 / 2 - 1 / (2*downshift))):int(arr.shape[0] * (1 / 2 + 1 / (2*downshift)) + 1)]

    cent, max_length = _disect(rescaled_arr, cutoff=cutoff, radius_cutoff_fraction=radius_cutoff_fraction, return_mask=True)

    rescaled_arr = np.zeros_like(arr)
    rescaled_arr[int(arr.shape[0] * (1 / 2 - 1 / (2 * downshift))):int(arr.shape[0] * (1 / 2 + 1 / (2*downshift))+1), int(arr.shape[1] * (1 / 2 - 1 / (2*downshift))):int(arr.shape[0] * (1 / 2 + 1 / (2*downshift)) + 1)] = np.fft.fftshift(cent)

    rescale_arr = upsample(np.fft.ifftshift(rescaled_arr), 1/downshift, 1)
    
    rescale_arr = np.fft.fftshift(rescale_arr)
    rescale_arr += np.flip(np.flip(rescale_arr, axis=0), axis=1)
    
    rescale_arr[rescale_arr < 0.07] = 0.0
    rescale_arr[rescale_arr >= 0.07] = 1.0

    return arr * np.fft.ifftshift(rescale_arr)

def gradient_disect(struct, cutoff=0.1):
    """
    THIS IS OLD CODE. I AM ON THE FENCE ABOUT DELETING IT
    This will remove the center of the autocorrelation.
    :param arr:
    :param cutoff:
    :return:
    """
    assert struct.shape[0] == struct.shape[1]
    arr = np.linalg.norm(np.array(np.gradient(np.fft.fftshift(struct))), axis=0)
    size = arr.shape[0]
    top = int(size / 2) + 1
    bot = int(size / 2)
    flags = np.zeros_like(arr)
    flags[bot, bot] = 1
    voxel_queue = Queue()
    [voxel_queue.put(item) for item in
     [[bot - 1, bot - 1], [bot - 1, bot], [bot - 1, bot + 1], [bot, bot - 1], [bot, bot + 1], [bot + 1, bot - 1],
      [bot + 1, bot], [bot + 1, bot + 1]]]
    max_direc = 0
    while not voxel_queue.empty():
        vox = voxel_queue.get()
        if arr[tuple(vox)] > arr.max() * cutoff:
            flags[tuple(vox)] = 1.0
            direc = np.array(vox) - np.array([bot, bot])
            if np.linalg.norm(direc) > max_direc:
                max_direc = np.linalg.norm(direc)
            if vox[0] == vox[1]:
                if vox[0] < bot:
                    [voxel_queue.put(item) for item in
                     [[vox[0] - 1, vox[1] - 1], [vox[0] - 1, vox[1]], [vox[0], vox[1] - 1]]]
                else:
                    [voxel_queue.put(item) for item in
                     [[vox[0] + 1, vox[1] + 1], [vox[0] + 1, vox[1]], [vox[0], vox[1] + 1]]]
            elif abs(vox[0] - bot) == abs(vox[1] - bot):
                if vox[0] > bot:
                    [voxel_queue.put(item) for item in
                     [[vox[0] + 1, vox[1] - 1], [vox[0] + 1, vox[1]], [vox[0], vox[1] - 1]]]
                else:
                    [voxel_queue.put(item) for item in
                     [[vox[0] - 1, vox[1] + 1], [vox[0] - 1, vox[1]], [vox[0], vox[1] + 1]]]
            else:
                if direc[1] > 0 and direc[1] > abs(direc[0]):
                    voxel_queue.put([vox[0], vox[1] + 1])
                elif direc[1] < 0 and abs(direc[1]) > abs(direc[0]):
                    voxel_queue.put([vox[0], vox[1] - 1])
                elif direc[0] > 0 and direc[0] > abs(direc[1]):
                    voxel_queue.put([vox[0] + 1, vox[1]])
                else:
                    voxel_queue.put([vox[0] - 1, vox[1]])
    return np.fft.ifftshift(np.fft.fftshift(struct) * flags), max_direc

