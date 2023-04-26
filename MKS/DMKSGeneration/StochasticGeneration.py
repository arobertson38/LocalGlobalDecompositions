"""
This is a file where I am rewriting my univerisal generation code to try and clean it up a little bit.

By: Andreas E. Robertson
"""
import numpy as np
from .HelperFunctions_StochasticGeneration import disect, rescale, local_square_mean

class StatisticsGenerator():
    def __init__(self, statistics, statistics_type='complete'):
        """
        Initializing the class to be able to generate new structures given a set of statistics.

        The second term indicates to the code whether the statistics passed in are a complete row (N statistics), or
        a reduced row (N-1).

        Statistics are assumed to be provided as a numpy array, where the first array, [..., 0], is the autocorrelation
        and the remaining arrays are cross-correlations

        The statistics are also assumed to be provided [0,0,0] being the t=0 point (so fftshift has not been applied)

        :param statistics:
        :param statistics_type: an indicator which explains to the code whether the statistics are complete. Can be
        "incomplete" or "complete".
        :return:
        """
        # Some useful parameters
        self.N = np.array(statistics.shape)[:-1].prod()
        self.shape = statistics.shape[:-1]
        self.twoD = True if (self.shape.__len__()<3) else False

        # First we will transform all the statistics into the frequency domain
        self.two_point_fft = np.fft.fftn(statistics, axes=tuple(range(0, self.shape.__len__())))


        # Compute the zero mean:
        self.means = self.two_point_fft[tuple([0] * (self.shape.__len__()) + [slice(None)])].real.copy()
        self.means[0] = (self.means[0]/self.N)**(0.5)

        # Computing the Final Phase, if we are interested in a conserved N-Phase Structure
        if statistics_type.lower() == 'incomplete':
            final_phase = np.zeros(self.shape, dtype=np.complex128)
            final_phase[tuple([0] * self.shape.__len__())] = self.N * self.means[0]
            final_phase -= self.two_point_fft.sum(axis=-1)
            self.two_point_fft = np.concatenate((self.two_point_fft, final_phase[..., np.newaxis]), axis=-1)
            self.means = np.hstack((self.means, np.array([self.two_point_fft[tuple([0] * self.shape.__len__() + [-1])].real
                                                          ])))

        self.means[1:] = (self.means[1:] / self.N) / self.means[0]

        # Using the computed crosscorrelations, compute the inter phase transformation kernels
        self.interfilter = np.ones_like(self.two_point_fft)
        self.calculate_interfilters()

        # Defining the derivatives
        self.deriv_matrix = []

    def reinitialize(self, statistics, statistics_type='complete'):
        """
        A method to reinitialize
        :param statistics: The N-point statistics
        :param statistics_type: A indicator for whether the complete set of statistics have been provided.
        :return:
        """
        self.__init__(statistics, statistics_type)

    def filter(self, filter='flood', filter_weight=0.35, cutoff=0.3, cutoff_radius=0.15, maximum_sigma=0.02):
        """
        A method for filtering sample structures using the gaussian filter that has been defined
        :param filter_weight: This is the power that controls the filter radius based on the volume fractions
        :param filter: This is a string that indicates to the code the type of filter that you would like to use.
        The options are: 'none' for no filtering. 'Gaussian_volumes' for a gaussian filter paramterized by the volume
        fraction. 'chords' for a gaussian filter parameterized by the mean chord length of each phase.
        :param maximum_sigma: This provides the standard deviation length of the largest gaussian as a percentage of the
        shortest side of the image.
        :return:
        """
        self.filters = np.ones_like(self.two_point_fft)
        if filter.lower() == 'gaussian_volumes':
            for i in range(self.two_point_fft.shape[-1]):
                if self.means[i] > 0.0:
                    cov = np.diag(np.square(np.ones([len(self.two_point_fft.shape[:-1])]) \
                            * (self.means[0] / self.means[i]) ** filter_weight / (maximum_sigma \
                            * self.two_point_fft.shape[0])))
                    self.filters[..., i] = self.gaussian_kernel(cov)
        elif filter.lower() == 'chords':
            # I do not recommend that people use this. 
            #
            # only works in 2D
            assert self.twoD
            self.two_forward_derivative()
            self.tfd_sides()
            self.tfd_ups()
            self.derivative_estimate()
            for i in range(self.two_point_fft.shape[-1]):
                if self.means[i] > 0.0:
                    cov = np.diag(np.square(np.array(self.derivs[i][1:]) / self.means[i] / filter_weight))
                    self.filters[..., i] = self.gaussian_kernel(cov)
        elif filter.lower() == 'flood':
            # Filters are extracted from the autocorrelations using a Breadth for Search Flood
            # filling algorithm
            #
            # I have made some changes here to allow this to work for 3D. They have yet to be
            # debugged because I need to change some of the base code to do so. 
            self.two_forward_derivative()
            autos = self.derivative_estimate(return_autos=True)
            for n in range(self.two_point_fft.shape[-1]):
                if self.means[n] > 0:
                    filters_temp = disect(autos[..., n], cutoff=cutoff, \
                            radius_cutoff_fraction=cutoff_radius, twoD=self.twoD)

                    # second, we check our desired length and make sure it isn't too big:
                    # (lets say, half the domain?)
                    desired_length = self.means[n] / self.derivs[n][0] * filter_weight
                    if desired_length > filters_temp[0].shape[0] / 2:
                        desired_length = filters_temp[0].shape[0] / 2
                        
                    self.filters[..., n] = rescale(filters_temp[0], filters_temp[1], \
                            desired_length, twoD=self.twoD)
                    self.filters[..., n] /= self.filters[..., n].sum()
            self.filters = np.fft.fftn(self.filters, axes=tuple(range(len(self.shape))))
        else:
            pass

    def generate(self, number_of_structures=2):
        """
        This is a function to generate just the highest phase
        :param number_of_structures: a parameter which indicates the number of stuctures to
        generate. The number must be either 1 or 2.
        :return:
        """
        self.generator()
        self.images = []

        if (number_of_structures > 2) or (number_of_structures < 1):
            raise ValueError('number_of_structures parameter must be either 1 or 2.')

        for gen_iterator in range(0, number_of_structures):
            self.images.append(np.ones_like(self.two_point_fft))
            self.images[gen_iterator] *= np.fft.fftn(self.new[gen_iterator])[..., np.newaxis]
            self.images[gen_iterator] = self.postprocess(np.fft.ifftn(self.images[gen_iterator] *
                                                     self.interfilter * self.filters,
                                                     axes=tuple(range(0, self.shape.__len__()))).real)

        if number_of_structures == 1:
            return self.images[0]
        else:
            return self.images[0], self.images[1]

    def generator(self):
        """
        A method for generating new microstructures given a covariance matrix and a mean.
        for 2D
        """
        eigs = self.two_point_fft[...,0].real
        eigs[tuple([0] * self.shape.__len__())] = 0.0 # I changed this from 1e-10
        if eigs.min() < -1e-11:
            raise ValueError('The autocovariance contains at least one negative eigenvalue (' + \
                    str(eigs.min()) + ').')
        eigs = np.sqrt(np.abs(eigs) /
                       np.product(self.shape))
        eps = np.random.normal(loc=0.0, scale=1.0, size=self.shape) + \
              1j * np.random.normal(loc=0.0, scale=1.0, size=self.shape)
        new = np.fft.fftn(eigs * eps)

        self.new = []
        self.new.append(new.real + self.means[0])
        self.new.append(new.imag + self.means[0])

    def postprocess(self, arr):
        return arr

    def two_forward_derivative(self, short=1.0, long=np.sqrt(2)):
        if self.twoD:
            self.deriv_matrix.append((-1 / 16) * np.array([[-1 / long, 0, -1 / short, 0, -1 / long],
                                        [0, 4 / long, 4 / short, 4 / long, 0],
                                        [-1 / short, 4 / short, -12 / short - 12 / long, 4 / short, -1 / short],
                                        [0, 4 / long, 4 / short, 4 / long, 0],
                                        [-1 / long, 0, -1 / short, 0, -1 / long]]))
        else:
            # define the two forward 3D approximate matrix derivative
            #
            # Things left to do:
            # (1) update coefficient (x?)
            # (2) Update center value (x)
            # (3) update all layers (x)
            self.deriv_matrix.append((-1 / 36) * np.array([
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
                                        ]))

    def tfd_ups(self, short=1.0, long=np.sqrt(2)):
        self.deriv_matrix.append((-1 / 4) * np.array([[0, 0, -1 / short, 0, 0],
                                    [0, 0, 4 / short, 0, 0],
                                    [0, 0, -6 / short, 0, 0],
                                    [0, 0, 4 / short, 0, 0],
                                    [0, 0, -1 / short, 0, 0]]))

    def tfd_sides(self, short=1.0, long=np.sqrt(2)):
        self.deriv_matrix.append((-1 / 4) * np.array([[0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [-1 / short, 4 / short, -6 / short, 4 / short, -1 / short],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0]]))

    def derivative_estimate(self, return_autos=False):
        autos = np.fft.ifftn(self.two_point_fft * self.two_point_fft.conj() / self.two_point_fft[..., 0, np.newaxis],
                             axes=tuple(range(0, self.shape.__len__()))).real
        self.derivs = [self.derivative(autos[..., i]) for i in range(autos.shape[-1])]
        if return_autos:
            return autos

    def derivative(self, arr):
        if self.twoD:
            cent = np.fft.fftshift(arr)[int(self.shape[0] / 2 - 2):int(self.shape[0] / 2 + 3), int(self.shape[1] / 2 - 2):int(self.shape[1] / 2 + 3)]
            return [(cent * self.deriv_matrix[n]).sum() for n in range(len(self.deriv_matrix))]
        else:
            cent = np.fft.fftshift(arr)[int(self.shape[0] / 2 - 2):int(self.shape[0] / 2 + 3), \
                                        int(self.shape[1] / 2 - 2):int(self.shape[1] / 2 + 3), \
                                        int(self.shape[2] / 2 - 2):int(self.shape[2] / 2 + 3)]
            return [(cent * self.deriv_matrix[n]).sum() for n in range(len(self.deriv_matrix))]


    def calculate_interfilters(self):
        """
        A method for computing the interphase filters from the auto and cross correlations
        :return:
        """
        self.interfilter[..., 1:] = self.two_point_fft[..., 1:] / self.two_point_fft[..., 0, np.newaxis]

    def gaussian_kernel(self, inverse_covariance):
        """
            Produces the frequency domain of an quassian filter with integral of 1.
            It returns a 'real' fft transformation.
            :param size: the NxNxN dimension N
            :param sigma: the standard deviation, 1.165 is used to approximate a 7x7x7 gaussian blur
            :return:
            """

        if self.twoD:
            assert inverse_covariance.shape[0] == 2
            xx, yy = np.meshgrid(np.linspace(-(self.shape[0] - 1) / 2., (self.shape[0] - 1) / 2., self.shape[0]),
                                 np.linspace(-(self.shape[1] - 1) / 2., (self.shape[1] - 1) / 2., self.shape[1]))
            arr = np.concatenate([xx[..., np.newaxis], yy[..., np.newaxis]], axis=-1)

        else:
            assert inverse_covariance.shape[0] == 3
            xx, yy, zz = np.meshgrid(np.linspace(-(self.shape[0] - 1) / 2., (self.shape[0] - 1) / 2., self.shape[0]),
                                 np.linspace(-(self.shape[1] - 1) / 2., (self.shape[1] - 1) / 2., self.shape[1]),
                                 np.linspace(-(self.shape[2] - 1) / 2., (self.shape[2] - 1) / 2., self.shape[2]))

            arr = np.concatenate([xx[..., np.newaxis], yy[..., np.newaxis], zz[..., np.newaxis]], axis=-1)
        kernel = np.squeeze(np.exp(-0.5 * arr[..., np.newaxis, :] @ inverse_covariance @ arr[..., np.newaxis]))

        return np.fft.fftn(np.fft.ifftshift(kernel / np.sum(kernel)))



class StatisticsGenerator_PhaseRetrieval(StatisticsGenerator):
    def generate(self, iterations=50):
        """
        This is a function to generate just the highest phase
        :param number_of_structures: a parameter which indicates the number of stuctures to
        generate. The number must be either 1 or 2.
        :return:
        """
        number_of_structures = 1

        self.generator(iter=iterations)
        self.images = []

        if (number_of_structures > 2) or (number_of_structures < 1):
            raise ValueError('number_of_structures parameter must be either 1 or 2.')

        for gen_iterator in range(0, number_of_structures):
            self.images.append(np.ones_like(self.two_point_fft))
            self.images[gen_iterator] *= np.fft.fftn(self.new[gen_iterator])[..., np.newaxis]
            self.images[gen_iterator] = self.postprocess(np.fft.ifftn(self.images[gen_iterator] *
                                                     self.interfilter * self.filters,
                                                     axes=tuple(range(0, self.shape.__len__()))).real)

        if number_of_structures == 1:
            return self.images[0]
        else:
            return self.images[0], self.images[1]

    def generator(self, iter=40):
        """
        This is an updated method which generates using the simple phase recovery method.
        """
        mag = np.sqrt(self.N * self.two_point_fft[...,0].real)
        new = mag * np.exp(1j * np.random.uniform(0.0, 2*np.pi, self.two_point_fft[...,0].shape))

        for iterations in range(0, iter):
            new = self.phaseretrievelupdate(np.fft.ifftn(new).real)
            new = mag * np.exp(1j * np.angle(np.fft.fftn(new)))

        new = self.phaseretrievelupdate(np.fft.ifftn(new).real)
        self.new = [new]

    def phaseretrievelupdate(self, arr):
        """
        By updating this method, child classes can be constructed with different limits on the sample bounds.
        :param arr:
        :return:
        """
        arr[arr<0.0] = 0.0
        arr[arr>1.0] = 1.0
        return arr

class EigenGenerator_BaseClass(StatisticsGenerator):
    def __init__(self, statistics, statistics_type='complete'):
        super().__init__(statistics, statistics_type)

        # Initializing additional parameters necessary for eigen postprocessing:
        if self.twoD:
            self.I, self.J = np.ogrid[:statistics.shape[0], :statistics.shape[1]]
        else:
            self.I, self.J, self.K = np.ogrid[:statistics.shape[0], :statistics.shape[1], :statistics.shape[2]]

    def postprocess(self, arr):
        """
        This method takes in an N-phase image with multiple channels and returns the eigenmicrostructure using
        outright competition of the three phases.
        :param arr: an NxMx3 matrix which contains variables describing the three stochastically connected phases
        :return arr_returned: an NxMx3 matrix which contains 1s for the maximum phase
        """
        arr_returned = np.zeros_like(arr)
        ind = arr.argmax(axis=-1)
        if self.twoD:
            arr_returned[self.I, self.J, ind] = 1.0
        else:
            arr_returned[self.I, self.J, self.K, ind] = 1.0

        return arr_returned

class EigenGenerator_NMaximumConstrained(EigenGenerator_BaseClass):
    def postprocess(self, arr):
        """
        This method takes in an N-phase image with multiple channels and returns the eigenmicrostructure using
        a weighted competition of the three phases. The weighting is done so that the volume fractions remain
        correct.

        In this method, the following strategy is adopted:
        (1) The lowest volume fraction phase is assigned to the N voxels containing the highest values for that phase.
                    -> N is selected such that the volume fraction is correct
        (2) For the remaining phases, phases are assigned based on a similar strategy, but previously assigned
        voxels cannot be given away again.

        :param arr: an NxMx3 matrix which contains variables describing the three stochastically connected phases
        :return arr_returned: an NxMx3 matrix which contains 1s for the maximum phase
        """

        # This will be the "naive" implementation
        # Lets go smallest magnitude from 1/0 to highest

        arr_returned = np.zeros_like(arr)

        # updating the indexing so that we go smallest to largest
        ord = list(np.argsort(self.means))

        ind = np.ones(self.shape, dtype=np.int8) * (ord[-1])

        for iterator in range(0, ord[:-1].__len__()):
            i = ord[iterator]
            if (self.N * self.means[i]) >= 1:
                index = np.unravel_index(np.argpartition(arr[..., i], -int(self.N * self.means[i]),
                                                     axis=None)[-int(self.N * self.means[i]):], self.shape)
                ind[index] = i
                for j in range(iterator+1, ord[:-1].__len__()):
                    arr[..., ord[j]][index] = -10

        if self.twoD:
            arr_returned[self.I, self.J, ind] = 1.0
        else:
            arr_returned[self.I, self.J, self.K, ind] = 1.0
        return arr_returned

class EigenGenerator_NFurthestConstrained(EigenGenerator_BaseClass):
    def postprocess(self, arr1):
        """
        This method takes in an N-phase image with multiple channels and returns the eigenmicrostructure using
        a weighted competition of the three phases. The weighting is done so that the volume fractions remain
        correct.

        In this method, the following strategy is adopted:
        (1) The minimum distance between each phase and all other phases is computed.
        (1) The lowest volume fraction phase is assigned to the N voxels containing the highest minimum distances
         for that phase.
                    -> N is selected such that the volume fraction is correct
        (2) For the remaining phases, phases are assigned based on a similar strategy, but previously assigned
        voxels cannot be given away again.


        :param arr: an NxMx3 matrix which contains variables describing the three stochastically connected phases
        :return arr_returned: an NxMx3 matrix which contains 1s for the maximum phase
        """

        # This will be the "naive" implementation
        # Lets go smallest magnitude from 1/0 to highest

        arr = arr1.copy()
        arr_returned = np.zeros_like(arr)

        # updating the indexing so that we go smallest to largest
        ord = list(np.argsort(self.means))

        # Adjusting arr:
        for iterator in ord[:-1]:
            if (self.N * self.means[iterator]) >= 1:
                others = [i for i in ord if i != iterator]
                arr[..., iterator] = (arr[..., iterator, np.newaxis] - arr1[..., others]).min(axis=-1)


        ind = np.ones(self.shape, dtype=np.int8) * (ord[-1])

        for iterator in range(0, ord[:-1].__len__()):
            i = ord[iterator]
            if (self.N * self.means[i]) >= 1:
                index = np.unravel_index(np.argpartition(arr[..., i], -int(self.N * self.means[i]),
                                                         axis=None)[-int(self.N * self.means[i]):], self.shape)
                ind[index] = i
                for j in range(iterator+1, ord[:-1].__len__()):
                    arr[..., ord[j]][index] = -10

        if self.twoD:
            arr_returned[self.I, self.J, ind] = 1.0
        else:
            arr_returned[self.I, self.J, self.K, ind] = 1.0

        return arr_returned


# This next two methods are reimplementations of the previous two methods but using square local mean
# to try and grow the size of individual voxels. 

class EigenGenerator_SquareLocalNMaximumConstrained(EigenGenerator_BaseClass):
    def postprocess(self, arr):
        """
        This method takes in an N-phase image with multiple channels and returns the eigenmicrostructure using
        a weighted competition of the three phases. The weighting is done so that the volume fractions remain
        correct.

        In this method, the following strategy is adopted:
        (1) The lowest volume fraction phase is assigned to the N voxels containing the highest values for that phase.
                    -> N is selected such that the volume fraction is correct
        (2) For the remaining phases, phases are assigned based on a similar strategy, but previously assigned
        voxels cannot be given away again.

        :param arr: an NxMx3 matrix which contains variables describing the three stochastically connected phases
        :return arr_returned: an NxMx3 matrix which contains 1s for the maximum phase
        """

        # This will be the "naive" implementation
        # Lets go smallest magnitude from 1/0 to highest
        
        arr = local_square_mean(arr)
        arr_returned = np.zeros_like(arr)

        # updating the indexing so that we go smallest to largest
        ord = list(np.argsort(self.means))

        ind = np.ones(self.shape, dtype=np.int8) * (ord[-1])

        for iterator in range(0, ord[:-1].__len__()):
            i = ord[iterator]
            if (self.N * self.means[i]) >= 1:
                index = np.unravel_index(np.argpartition(arr[..., i], -int(self.N * self.means[i]),
                                                     axis=None)[-int(self.N * self.means[i]):], self.shape)
                ind[index] = i
                for j in range(iterator+1, ord[:-1].__len__()):
                    arr[..., ord[j]][index] = -10

        if self.twoD:
            arr_returned[self.I, self.J, ind] = 1.0
        else:
            arr_returned[self.I, self.J, self.K, ind] = 1.0
        return arr_returned

class EigenGenerator_SquareLocalNFurthestConstrained(EigenGenerator_BaseClass):
    def postprocess(self, arr1):
        """
        This method takes in an N-phase image with multiple channels and returns the eigenmicrostructure using
        a weighted competition of the three phases. The weighting is done so that the volume fractions remain
        correct.

        In contrast to the other NFurthestConstrained, this changes the value of each voxel to be the square average of the voxels around it. 

        In this method, the following strategy is adopted:
        (1) The minimum distance between each phase and all other phases is computed.
        (1) The lowest volume fraction phase is assigned to the N voxels containing the highest minimum distances
         for that phase.
                    -> N is selected such that the volume fraction is correct
        (2) For the remaining phases, phases are assigned based on a similar strategy, but previously assigned
        voxels cannot be given away again.


        :param arr: an NxMx3 matrix which contains variables describing the three stochastically connected phases
        :return arr_returned: an NxMx3 matrix which contains 1s for the maximum phase
        """

        # This will be the "naive" implementation
        # Lets go smallest magnitude from 1/0 to highest

        arr = local_square_mean(arr1.copy())
        arr_returned = np.zeros_like(arr)

        # updating the indexing so that we go smallest to largest
        ord = list(np.argsort(self.means))

        # Adjusting arr:
        for iterator in ord[:-1]:
            if (self.N * self.means[iterator]) >= 1:
                others = [i for i in ord if i != iterator]
                arr[..., iterator] = (arr[..., iterator, np.newaxis] - arr1[..., others]).min(axis=-1)


        ind = np.ones(self.shape, dtype=np.int8) * (ord[-1])

        for iterator in range(0, ord[:-1].__len__()):
            i = ord[iterator]
            if (self.N * self.means[i]) >= 1:
                index = np.unravel_index(np.argpartition(arr[..., i], -int(self.N * self.means[i]),
                                                         axis=None)[-int(self.N * self.means[i]):], self.shape)
                ind[index] = i
                for j in range(iterator+1, ord[:-1].__len__()):
                    arr[..., ord[j]][index] = -10

        if self.twoD:
            arr_returned[self.I, self.J, ind] = 1.0
        else:
            arr_returned[self.I, self.J, self.K, ind] = 1.0

        return arr_returned

class EigenGenerator_PhaseRetrieval(StatisticsGenerator_PhaseRetrieval, EigenGenerator_NFurthestConstrained):
    pass

class GaoGenerator():
    def __init__(self, statistics, volume_fractions=(0.3, 0.3, 0.4), unique_phase_labels=(0,1,2)):
        """
        This class can be used to generate structures using a strategy that is inspired by Gao's 2021 paper: "Ultra-eﬃcient
        reconstruction of 3D microstructure and distribution of properties of random heterogeneous materials
        containing multiple phases"

        Because their proposed method cannot match the expressiveness necessary to compete with my method, this is a
        mix of the two that has adopted the strategies defined by them.

        :param statistics: The autocorrelation corresponding to Gao's composite function.
        :param unique_phase_labels: A list containing the phase labels that are used in computing the 2PS given in statistics.
        :return:
        """
        # Some useful parameters
        statistics = np.squeeze(statistics)
        self.N = np.array(statistics.shape).prod()
        self.shape = statistics.shape
        self.twoD = True if (self.shape.__len__()<3) else False
        self.unique_phase_labels = unique_phase_labels
        self.volume_fraction = volume_fractions

        # First we will transform all the statistics into the frequency domain
        self.two_point_fft = np.fft.fftn(statistics)

        # Compute the zero mean:
        self.means = np.sqrt(self.two_point_fft[tuple([0] * self.shape.__len__())] / self.N)

        # Initializing additional parameters necessary for eigen postprocessing:
        if self.twoD:
            self.I, self.J = np.ogrid[:statistics.shape[0], :statistics.shape[1]]
        else:
            self.I, self.J, self.K = np.ogrid[:statistics.shape[0], :statistics.shape[1], :statistics.shape[2]]

    def reinitialize(self, statistics, unique_phase_labels=(0,1,2)):
        """
        A method to reinitialize
        :param statistics: The N-point statistics
        :param statistics_type: A indicator for whether the complete set of statistics have been provided.
        :return:
        """
        self.__init__(statistics, unique_phase_labels)

    def filter(self, maximum_sigma=0.02, filter='gaussian'):
        """
        A method for filtering sample structures using the gaussian filter that has been defined
        :param filter: This is a string that indicates to the code the type of filter that you would like to use.
        The options are: 'combined' - a pointwise combination of the 2-point statistics and a gaussian
        'gaussian' - a gaussian.
        :param maximum_sigma: This provides the standard deviation length of the largest gaussian as a percentage of the
        shortest side of the image.
        :return:
        """
        if filter.lower() == 'gaussian':
            self.filters = self.gaussian_kernel(sigma_percent=maximum_sigma)
        else:
            pass

    def generate(self, number_of_structures=2):
        """
        This is a function to generate just the highest phase
        :param number_of_structures: a parameter which indicates the number of stuctures to
        generate. The number must be either 1 or 2.
        :return:
        """
        self.generator()
        self.images = []

        if (number_of_structures > 2) or (number_of_structures < 1):
            raise ValueError('number_of_structures parameter must be either 1 or 2.')

        for gen_iterator in range(0, number_of_structures):
            new_image = np.fft.fftn(self.new[gen_iterator])
            self.images.append(self.postprocess(np.fft.ifftn(new_image * self.filters).real))

        if number_of_structures == 1:
            return self.images[0]
        else:
            return self.images[0], self.images[1]

    def postprocess(self, arr):
        """
        This processing method will reflect the process used in Gao's. Using the information stored in means, the cut_offs
        will be imposed.

        Starting from the 0 labeled phase, the number of voxels necessary to achieve the indicated 0 phase volume fraction
        will be labeled as 0.

        :param arr:
        :return:
        """
        returned_image = np.ones_like(arr)
        for i in range(0, len(self.volume_fraction)):
            index = np.unravel_index(np.argpartition(arr, int(self.N * self.volume_fraction[i]),
                                                     axis=None)[:int(self.N * self.volume_fraction[i])], self.shape)
            returned_image[index] = self.unique_phase_labels[i]
            arr[index] = 100.0

        return returned_image

    def gaussian_kernel(self, sigma_percent=0.02):
        """
            Produces the frequency domain of an quassian filter with integral of 1.
            It returns a 'real' fft transformation.
            :param size: the NxNxN dimension N
            :param sigma: the standard deviation, 1.165 is used to approximate a 7x7x7 gaussian blur
            :return:
            """
        sigma = np.array(self.shape).min() * sigma_percent

        if self.twoD:
            xx, yy = np.meshgrid(np.linspace(-(self.shape[0] - 1 / 2.), (self.shape[0] - 1) / 2., self.shape[0]),
                                 np.linspace(-(self.shape[1] - 1 / 2.), (self.shape[1] - 1) / 2., self.shape[1]))

            kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
            return np.fft.fftn(np.fft.ifftshift(kernel / np.sum(kernel)))

        else:
            xx, yy, zz = np.meshgrid(np.linspace(-(self.shape[0] - 1 / 2.), (self.shape[0] - 1) / 2., self.shape[0]),
                                 np.linspace(-(self.shape[1] - 1 / 2.), (self.shape[1] - 1) / 2., self.shape[1]),
                                 np.linspace(-(self.shape[2] - 1 / 2.), (self.shape[2] - 1) / 2., self.shape[2]))

            kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy) + np.square(zz)) / np.square(sigma))
            return np.fft.fftn(np.fft.ifftshift(kernel / np.sum(kernel)))

    def generator(self):
        """
        A method for generating new microstructures given a covariance matrix and a mean.
        for 2D
        """
        eigs = self.two_point_fft.real
        eigs[tuple([0] * self.shape.__len__())] = 1e-10
        print(eigs.min())
        eigs = np.sqrt(eigs /
                       np.product(self.shape))
        eps = np.random.normal(loc=0.0, scale=1.0, size=self.shape) + \
              1j * np.random.normal(loc=0.0, scale=1.0, size=self.shape)
        new = np.fft.fftn(eigs * eps)

        self.new = []
        self.new.append(new.real + self.means)
        self.new.append(new.imag + self.means)
