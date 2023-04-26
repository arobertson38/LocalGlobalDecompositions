"""
Code for a Diffusion Generator. 

Created by: Andreas Robertson
Contact: arobertson38@gatech.edu
"""
import numpy as np
import torch.nn as nn
import torch
import os
import torch.optim as optim
import models.small as small
import models.vnet as vnet
import models.deepvnet as deepvnet
import yaml
import argparse
import MKS.utils.HelperFunctions as hfuncs

# --------------------------------------------
# Useful Utility Methods
# --------------------------------------------

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def parse_config(config):
    """
    takes a string indicating the location of the config file
    and reads and parses it. 
    """
    # parsing the args file
    with open(config, 'r') as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    new_config.device = device
    return new_config

def __get_model(config):
    """
    Returns the specific model implementation of interest.
    """
    # checking which model I want to use:
    if config.model.model.lower() == 'inception':
        return small.Inception
    elif config.model.model.lower() == 'vnetci':
        return small.VNetC_Inception
    elif config.model.model.lower() == 'vnetciia':
        return small.VNetC_Inception_Attn
    elif config.model.model.lower() == 'vnetc':
        return small.VNetC
    elif config.model.model.lower() == 'vnetca':
        return small.VNetC_Attn
    elif config.model.model.lower() == 'vnet1':
        return vnet.VNet1
    elif config.model.model.lower() == 'vnet3':
        return vnet.VNet3
    elif config.model.model.lower() == 'vnetd2':
        return deepvnet.VNetD2
    elif config.model.model.lower() == 'vnetd3':
        return deepvnet.VNetD3
    elif config.model.model.lower() == 'vnetd4':
        return deepvnet.VNetD4
    else:
        raise NotImplementedError(f"{config.model.model} is not supported.")

def get_swa_model(model, config):
    """
    Returns the SWA (Stochastic Weighted Averaging)
    averaged model. 
    """
    if config.optim.swa.lower() == 'ema':
        # Exponential Moving Average (recommended by Song)
        def EMA(averaged_model_parameter, model_parameter, num_averaged):
            if num_averaged == 1:
                # the very first update. Just take the model.
                return model_parameter
            else:
                return config.optim.ema_param * averaged_model_parameter + \
                        (1 - config.optim.ema_param) * model_parameter
        averaged_model = optim.swa_utils.AveragedModel(model, \
                avg_fn = EMA
                )
        return averaged_model

    elif config.optim.swa.lower() == 'swa':
        # standard SWA using Equal Running Average
        return optim.swa_utils.AveragedModel(model)
    else:
        raise NotImplementedError(f"{config.optim.swa} is not supported.")


def get_model(model_location, config, periodic):
    """
    Reads a standard yml config file and returns the
    loaded diffusion model associated with it. 
    
    :param model_location: A string containing the model location.
    :param config: a namespace datatype containing the contents of 
                   a config file. 
    :config periodic: a boolean identifying whether we want
                      to convert the padding of the convolutions
                      to periodic. 
    """
    states = torch.load(model_location, 
            map_location=config.device)
    deep_model = __get_model(config)
    score = deep_model(config).to(config.device)

    if periodic:
        score = convert_to_periodic(score)

    score = torch.nn.DataParallel(score)

    # I am going to change this to mandate
    # that you need 'swa' in the model location
    # to use a swa model
    if not hasattr(config.optim, 'swa'):
        score.load_state_dict(states[0])
    elif config.optim.swa.lower() == 'none':
        score.load_state_dict(states[0])
    elif not model_location.lower().__contains__('swa'):
        score.load_state_dict(states[0])
    else:
        score = get_swa_model(score, config)
        score.load_state_dict(states)
    return score

def convert_to_periodic(model):
    """
    a method that converts all of the convolutions in a model to
    periodic convolutions. 
    """
    def recursion(model):
        for n, child in enumerate(model.children()):
            if len(list(child.children())) == 0:
                if (type(child) == torch.nn.Conv2d):
                    child.padding_mode = 'circular'
            else:
                recursion(child)

    recursion(model)
    return model


def periodize_embedding(x, width=9):
    """
    This method takes as input a structure that has already been 
    embedded into a slightly larger domain: [0, 0, STRUCTURE, 0, 0].
    The width of the embedding on each side is dictated by the parameter
    'width'. 
    
    This method replaces the padded regions with periodic
    extensions to the internal structure: [R, E, STRUCTURE, S, T]. 
    
    Obviously, meant to be used with periodic structures. 
    Only works in 2D!

    WARNING: This method does in-place transformations to the data.

    Args:
        x (torch.Tensor): of shape # x Channels x N x M
        width (int, optional): _description_. The width of the
                                              embedding on all sides.

    Returns:
        torch.Tensor: periodized embedding of X as described above.
    """
    assert len(x.shape) == 4, "Only 2D structures are currently supported."
    min_dim = min(*x.shape[2:])
    assert width < min_dim, "More than one periodic wrap is not supported."
    
    # replace top with bottom of domain
    x[..., :width, width:-width] = x[..., -2*width:-width, width:-width]
    x[..., -width:, width:-width] = x[..., width:2*width, width:-width]

    x[..., width:-width, :width] = x[..., width:-width, -2*width:-width]
    x[..., width:-width, -width:] = x[..., width:-width, width:2*width]

    # corners
    x[..., :width, :width] = x[..., -2*width:-width, -2*width:-width]
    x[..., -width:, :width] = x[..., width:2*width, -2*width:-width]

    x[..., :width, -width:] = x[..., -2*width:-width, width:2*width]
    x[..., -width:, -width:] = x[..., width:2*width, width:2*width]

    return x

# ------------------------------------------------------
# The Diffusion Generators
# ------------------------------------------------------

class ConditionalDiffusionGenerator(object):
    """
    Conditional Diffusion Generator. 

    This class is just a wrapper that accomodates diffusion. 
    """
    def __init__(self, config, model_location, periodic=False):
        """
        :param config: a string identifying the location of
                       the desired config file. 
        :param model_location: a string identifying the location
                               of the desired trained model
        :config periodic: a boolean identifying whether we want
                          to convert the padding of the convolutions
                          to periodic. 
        """
        if type(config) is str:
            config = parse_config(config)
        self.config = config
        self.score = get_model(model_location, self.config, periodic)
        self.score.eval()
        self.sigmas = np.exp(
                        np.linspace(\
                            np.log(config.model.sigma_begin), \
                            np.log(config.model.sigma_end),\
                            config.model.num_classes
                        )
                    )

    def get_score(self, vfs=None, 
                        padding=16, 
                        embed_shape=None,
                        shift=0.0):
        """
        Returns the score function. This method is added so 
        that - if I want to condition on volume fraction - 
        I can easily do exactly that. 

        :param vfs: a vector of the volume fraction for each phase
                    for each passed microstructure
        :param padding: the width of the padding on each side of the domain.
        :param embed_shape: the shape of the embedded domain
        :param shift: (float) a parameter for shifting the parameterizing volume
                              fraction to correct for persistent biases that arise
                              in some problems. 
        """
        return self.score

    def anneal_Langevin_dynamics(self, x_mod, score,\
                step_lr=0.00002, starting_index=2,
                lower_clamp=0.0,
                upper_clamp=1.0,
                repeats=1,
                padding=16):
        """
        This method is a annealed langevin dynamics solver where
        the starting point is not random noise, but an image. 
        The solver is not run from the max to min noise level,
        but rather from a starting_index to the minimum noise
        level. 

        Additionally, the whole microstructure is run through the model
        """
        # cropping the sigmas.
        final_sigma = self.sigmas[-1]
        sigmas = self.sigmas[starting_index:]

        # perterb the input based on the noise level it should be at. 
        #x_mod = x_mod + torch.randn_like(x_mod) * sigmas[0]

        with torch.no_grad():
            for c, sigma in enumerate(sigmas):
                # adjusting c:
                c = c + starting_index
                # creating the labels for the annealed sampling level
                labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
                labels = labels.long()
                step_size = step_lr * (sigma / final_sigma) ** 2

                for _ in range(repeats):
                    # iterations for noise level. 
                    noise = torch.randn_like(x_mod,
                            device=x_mod.device) * \
                            np.sqrt(step_size * 2)
                    grad = score(x_mod, labels)
                    x_mod = x_mod + step_size * grad + noise

                    # perform the periodic embedding
                    x_mod = periodize_embedding(x_mod, padding)

            return torch.clamp(x_mod, lower_clamp, upper_clamp).to('cpu')

    def generate(self, 
            structures,
            starting_index=430,
            langevin_parameters={},
            padding=16,
            shift=0.0,
            ):
        """
        Apply diffusion to the microstructures. 

        ONLY WORKS FOR 2D MICROSTRUCTURES

        :param structures: an array of structures of the form [#, Phases, SPATIALDIM]
        :param starting_index: Explicit input for the langevin dynamic
                               starting index parameter. 
        :param langevin_parameters: a dictionary of parameters to pass to the
                                    langevin dynamics method. 
        :param padding: a integer saying how much to pad on each side. Should
                        be a value compatible with the down sampling of the network. 
                        So, VNetD4: units of 8
        :param shift: (float) a parameter for shifting the parameterizing volume
                              fraction to correct for persistent biases that arise
                              in some problems. 
        """
        assert len(structures.shape) == 4, "Only 2D microstructures are supported."
        assert structures.shape[1] == self.config.data.channels, \
            f"The number of channels ({structures.shape[1]}) does not match the provided config file ({self.config.data.channels})."

        # add the starting index to the langevin dictionary
        langevin_parameters['starting_index'] = starting_index

        # handling numpy arrays
        if type(structures) is np.ndarray:
            structures = torch.from_numpy(structures)

        # embed the structure
        dim_x, dim_y = structures.shape[2:]
        struct = torch.zeros(*structures.shape[:2], dim_x + 2*padding, dim_y + 2*padding)
        struct[..., padding:-padding, padding:-padding] = structures
        struct = periodize_embedding(struct, padding)
        struct = struct.to(self.config.device)
                
        # ---------------------------------------------
        # initialize the score
        means = torch.mean(structures, dim=tuple(range(2, len(structures.shape))))
        del structures
        score = self.get_score(means, padding, struct.shape, shift=shift)

        # ---------------------------------------------
        struct = self.anneal_Langevin_dynamics(struct, score, **langevin_parameters, padding=padding)
        return struct[:, :, padding:-padding, padding:-padding]

class TrueVFConditioned_CDGenerator(ConditionalDiffusionGenerator):
    """
    This is an extension of the ConditionalDiffusionGenerator 
    class that specifically adds volume fraction conditioning
    to the diffusion process. 
    
    The conditioning is imposed via the "true" volume fraction
    of the structures. The "true" volume fraction is
    estimated by segmenting the structures and then computing
    the volume fraction. 
    
    Basically, this is based on the observation that 
    there is a difference between having the correct average 
    and having the correct volume fraction. This is because 
    volume fraction is counted only after segmentation. 

    The vf sigma is going to be defined during each call of
    generation. This is so that you could, theoretically, 
    use the generator for different spatial domains without
    needing to reinitialize it. 
    """

    def get_score(self, vfs=None, 
                        padding=16, 
                        embed_shape=None,
                        shift=0.0):
        """
        Returns the score function. This method is added so 
        that - if I want to condition on volume fraction - 
        I can easily do exactly that. 

        :param vfs: a vector of the volume fraction for each phase
                    for each passed microstructure
        :param padding: the width of the padding on each side of the domain.
        :param embed_shape: the shape of the embedded domain
        :param shift: (float) a parameter for shifting the parameterizing volume
                              fraction to correct for persistent biases that arise
                              in some problems. 
        """
        # defining the vector component for the likelihood
        # (this can't just be a vector of ones because of the padding)
        
        vfs = vfs.to(self.config.device) + shift #0.0075
        vf_grad = torch.zeros(embed_shape)
        vf_grad[..., padding:-padding, padding:-padding] = 1.0
        N = vf_grad[0, 0].sum().item()
        vf_grad = vf_grad.float().to(self.config.device)

        ones = torch.ones(embed_shape).float().to(self.config.device)

        # initializing the likelihood's sigma values
        correction = np.sqrt(256 ** 2 / N)
        vf_initial = 0.008 * correction
        vf_final = 0.0001 * correction

        vf_sigmas_base = torch.from_numpy(
                    np.exp(
                        np.linspace(\
                            np.log(vf_initial),\
                            np.log(vf_final),\
                            len(self.sigmas)
                        )
                    )
        ).float().to(self.config.device)

        # initializing the score
        def score(x, sigma_indexes):
            """
            The composite score function for the vf posterior. 

            :param x: the images we are generating. 
            :param sigmas_indexes: the indexes of the sigmas we are
                                   producing. 
            """
            grad = self.score(x, sigma_indexes)
            vf_sigmas = vf_sigmas_base[sigma_indexes].to(self.config.device)[..., None]

            # compute the "true" vf
            # TODO: THE TRUE VOLUME FRACTION IS ESTIMATED ON THE ENTIRE X.
            # IT SHOULD ONLY BE ESTIMATED ON THE CENTRAL SUBSECTION OF X. 
            # TODO: I already changed it. but, it needs to be debugged to show that
            # it still works. 
            true_vf = hfuncs.true_volumefraction(
                    hfuncs.expand(x[..., padding:-padding, padding:-padding])
            )[:, :-1]

            # the volume fraction condition:
            true_vf_condition = \
                -1 * \
                ((1 / (N * torch.square(vf_sigmas))) * \
                (true_vf - vfs) \
                )[..., None, None] * \
                vf_grad
                #ones

            return grad + true_vf_condition

        return score

class DirectAverageVFConditioned_CDGenerator(ConditionalDiffusionGenerator):
    """
    This is an extension of the ConditionalDiffusionGenerator 
    class that specifically adds volume fraction conditioning
    to the diffusion process. 
    
    The conditioning is imposed by direct averaging (i.e.,
    the volume fraction is estimated by directly taking the
    average of each of the phases). 

    The vf sigma is going to be defined during each call of
    generation. This is so that you could, theoretically, 
    use the generator for different spatial domains without
    needing to reinitialize it. 
    """

    def get_score(self, vfs=None, 
                        padding=16, 
                        embed_shape=None,
                        shift=0.0):
        """
        Returns the score function. This method is added so 
        that - if I want to condition on volume fraction - 
        I can easily do exactly that. 

        :param vfs: a vector of the volume fraction for each phase
                    for each passed microstructure
        :param padding: the width of the padding on each side of the domain.
        :param embed_shape: the shape of the embedded domain
        :param shift: (float) a parameter for shifting the parameterizing volume
                              fraction to correct for persistent biases that arise
                              in some problems. 
        """
        # defining the vector component for the likelihood
        # (this can't just be a vector of ones because of the padding)
        
        vfs = vfs.to(self.config.device) + shift
        vf_grad = torch.zeros(embed_shape)
        vf_grad[..., padding:-padding, padding:-padding] = 1.0
        N = vf_grad[0, 0].sum().item()
        vf_grad = vf_grad.float().to(self.config.device)

        # initializing the likelihood's sigma values
        correction = np.sqrt(256 ** 2 / N)
        vf_initial = 0.020 * correction # 0.010
        vf_final = 0.0003 * correction # 0.0001

        vf_sigmas_base = torch.from_numpy(
                    np.exp(
                        np.linspace(\
                            np.log(vf_initial),\
                            np.log(vf_final),\
                            len(self.sigmas)
                        )
                    )
        ).float().to(self.config.device)

        # initializing the score
        def score(x, sigma_indexes):
            """
            The composite score function for the vf posterior. 

            :param x: the images we are generating. 
            :param sigmas_indexes: the indexes of the sigmas we are
                                   producing. 
            """
            grad = self.score(x, sigma_indexes)
            vf_sigmas = vf_sigmas_base[sigma_indexes].to(self.config.device)[..., None]

            # compute the "true" vf
            vf = torch.mean(x[..., padding:-padding, padding:-padding], dim=(2, 3))

            # the volume fraction condition:
            vf_condition = \
                -1 * \
                ((1 / (N * torch.square(vf_sigmas))) * \
                (vf - vfs) \
                )[..., None, None] * \
                vf_grad

            return grad + vf_condition

        return score
