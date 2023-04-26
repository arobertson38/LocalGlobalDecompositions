"""
This file contains "small" networks. This are networks with - ideally - 
very few parameters that we can use effectively for materials informatics
tasks. 

But, just VNET models. The other file was getting much too cluttered. 

^^^^^ blurb from vnet.py ^^^^^

Additionally: these are deeper networks in the U-Net sense.
              (as in they drop deeper)
"""
try:
    import models.utils as utils
except ModuleNotFoundError:
    import utils
    print('You better be running from inside the models folder.')
import torch
import torch.nn as nn
import numpy as np

class VNetD2(nn.Module):
    """
    This model contain a VNet Structure. 
    All skip connections will be mediated
    using 1x1 convolutions to combine feature vectors. 

    This is a VNet with 2 across in the skip connection
    and 3 drops (so three layers) - hense the D3. Each
    connection has 2 across
    """
    def __init__(self, configs):
        """
        Configs should contain specific, model specific information. 
        (This defines things like number of inception blocks)

        Information necessary in the config file:

        model_structure:
            norm: the normalization method used. 
            activation: the activation used
            activation_all: Whether the activation should have channel dependent
                            parameters. 
            latent_features: The latent features in the model. 
            num_blocks: the number of initial blocks to include

        """
        super().__init__()

        # reading and storing sigmas
        self.register_buffer('sigmas', torch.tensor(utils.get_sigmas(configs)))

        # reading the model parameters
        latent_features = configs.model_structure.latent_features
        norm = configs.model_structure.norm
        in_features = out_features = configs.data.channels
        activation = configs.model_structure.activation
        activation_all = configs.model_structure.activation_all
        num_outer_incept_blocks = configs.model_structure.num_blocks # these are blocks on either end of the V. 

        # initial projector and final combiner - these will be symmetric
        self.initial = nn.ModuleList()
        self.final = nn.ModuleList()

        self.initial.append(utils.ConvLayer2D(
                in_features = in_features,
                out_features = latent_features,
                kernel = 3,
                stride = 1,
                activation = activation,
                activation_all = activation_all,
                norm = norm
                ))

        for i in range(num_outer_incept_blocks):
            self.initial.append(
                utils.ConvLayer2D(
                    in_features = latent_features,
                    out_features = latent_features,
                    kernel = 3,
                    stride = 1,
                    activation = activation,
                    activation_all = activation_all,
                    norm = norm,
                    ))

            self.final.append(
                utils.ConvLayer2D(
                    in_features = latent_features,
                    out_features = latent_features,
                    kernel = 3,
                    stride = 1,
                    activation=activation,
                    activation_all = activation_all,
                    norm = norm,
                    ))

        self.final.append(utils.ConvLayer2D(
                in_features = latent_features,
                out_features = out_features,
                kernel = 3,
                stride = 1,
                activation = None,
                activation_all = activation_all,
                norm = None,
                ))


        # defining the VNet Structure
        # must define up, down, combine, and skip units. 
        self.down_unit = nn.ModuleList(
                    [
                        utils.ConvLayer2D(
                            in_features = latent_features,
                            out_features = latent_features,
                            kernel = 2,
                            stride = 2,
                            activation = activation,
                            activation_all = activation_all,
                            norm = norm,
                            padding = None,
                            ),
                    ]
                        )

        self.up_unit = nn.ModuleList(
                    [
                        utils.ConvTransposeLayer2D(
                            in_features = latent_features,
                            out_features = latent_features,
                            kernel = 2,
                            stride = 2,
                            activation = activation,
                            activation_all = activation_all,
                            norm = norm,
                            padding = None,
                            ),
                    ]
                        )

        self.skip_unit = nn.ModuleList(
                            [
                            utils.Bucket(
                                utils.ConvLayer2D(
                                    in_features = latent_features,
                                    out_features = latent_features,
                                    kernel = 3,
                                    stride = 1,
                                    activation=activation,
                                    activation_all = activation_all,
                                    norm = norm,
                                    ),
                                nn.Dropout(0.1),
                                utils.ConvLayer2D(
                                    in_features = latent_features,
                                    out_features = latent_features,
                                    kernel = 3,
                                    stride = 1,
                                    activation=activation,
                                    activation_all = activation_all,
                                    norm = norm,
                                    ),
                                ),
                            utils.Bucket(
                                utils.ConvLayer2D(
                                    in_features = latent_features,
                                    out_features = latent_features,
                                    kernel = 3,
                                    stride = 1,
                                    activation=activation,
                                    activation_all = activation_all,
                                    norm = norm,
                                    ),
                                nn.Dropout(0.1),
                                utils.ConvLayer2D(
                                    in_features = latent_features,
                                    out_features = latent_features,
                                    kernel = 3,
                                    stride = 1,
                                    activation=activation,
                                    activation_all = activation_all,
                                    norm = norm,
                                    ),
                                ),
                            ]
                            )

        self.combiner_unit = nn.ModuleList(
                            [
                            utils.ConvLayer2D(
                                in_features = 2 * latent_features,
                                out_features = latent_features,
                                kernel = 1,
                                stride = 1,
                                activation = activation,
                                activation_all = activation_all,
                                norm = norm
                                ),
                            ]
                            )

    @staticmethod
    def _list(x, modulelist):
        """
        Directly runs through a module list. 
        """
        for module in modulelist:
            x = module(x)
        return x

    def forward(self, x, sigma):
        """
        The forward method of a V2 type. 
        """
        x = self._list(x, self.initial)
        x = utils.VNET_Constant(
                        x = x,
                        down_unit = self.down_unit,
                        up_unit = self.up_unit,
                        skip_unit = self.skip_unit,
                        combine_unit = self.combiner_unit,
                                )
        x = self._list(x, self.final)

        # normalizing by Sigma. 
        used_sigmas = self.sigmas[sigma].view(x.shape[0], *([1] * len(x.shape[1:]))).float()
        x = x / used_sigmas
        return x


class VNetD3(nn.Module):
    """
    This model contain a VNet Structure. 
    All skip connections will be mediated
    using 1x1 convolutions to combine feature vectors. 

    This is a VNet with 2 across in the skip connection
    and 3 drops (so three layers) - hense the D3. Each
    connection has 2 across
    """
    def __init__(self, configs):
        """
        Configs should contain specific, model specific information. 
        (This defines things like number of inception blocks)

        Information necessary in the config file:

        model_structure:
            norm: the normalization method used. 
            activation: the activation used
            activation_all: Whether the activation should have channel dependent
                            parameters. 
            latent_features: The latent features in the model. 
            num_blocks: the number of initial blocks to include

        """
        super().__init__()

        # reading and storing sigmas
        self.register_buffer('sigmas', torch.tensor(utils.get_sigmas(configs)))

        # reading the model parameters
        latent_features = configs.model_structure.latent_features
        norm = configs.model_structure.norm
        in_features = out_features = configs.data.channels
        activation = configs.model_structure.activation
        activation_all = configs.model_structure.activation_all
        num_outer_incept_blocks = configs.model_structure.num_blocks # these are blocks on either end of the V. 

        # initial projector and final combiner - these will be symmetric
        self.initial = nn.ModuleList()
        self.final = nn.ModuleList()

        self.initial.append(utils.ConvLayer2D(
                in_features = in_features,
                out_features = latent_features,
                kernel = 3,
                stride = 1,
                activation = activation,
                activation_all = activation_all,
                norm = norm
                ))

        for i in range(num_outer_incept_blocks):
            self.initial.append(
                utils.ConvLayer2D(
                    in_features = latent_features,
                    out_features = latent_features,
                    kernel = 3,
                    stride = 1,
                    activation = activation,
                    activation_all = activation_all,
                    norm = norm,
                    ))

            self.final.append(
                utils.ConvLayer2D(
                    in_features = latent_features,
                    out_features = latent_features,
                    kernel = 3,
                    stride = 1,
                    activation=activation,
                    activation_all = activation_all,
                    norm = norm,
                    ))

        self.final.append(utils.ConvLayer2D(
                in_features = latent_features,
                out_features = out_features,
                kernel = 3,
                stride = 1,
                activation = None,
                activation_all = activation_all,
                norm = None,
                ))


        # defining the VNet Structure
        # must define up, down, combine, and skip units. 
        self.down_unit = nn.ModuleList(
                    [
                        utils.ConvLayer2D(
                            in_features = latent_features,
                            out_features = latent_features,
                            kernel = 2,
                            stride = 2,
                            activation = activation,
                            activation_all = activation_all,
                            norm = norm,
                            padding = None,
                            ),
                        utils.ConvLayer2D(
                            in_features = latent_features,
                            out_features = latent_features,
                            kernel = 2,
                            stride = 2,
                            activation = activation,
                            activation_all = activation_all,
                            norm = norm,
                            padding = None,
                            ),
                    ]
                        )

        self.up_unit = nn.ModuleList(
                    [
                        utils.ConvTransposeLayer2D(
                            in_features = latent_features,
                            out_features = latent_features,
                            kernel = 2,
                            stride = 2,
                            activation = activation,
                            activation_all = activation_all,
                            norm = norm,
                            padding = None,
                            ),
                        utils.ConvTransposeLayer2D(
                            in_features = latent_features,
                            out_features = latent_features,
                            kernel = 2,
                            stride = 2,
                            activation = activation,
                            activation_all = activation_all,
                            norm = norm,
                            padding = None,
                            ),
                    ]
                        )

        self.skip_unit = nn.ModuleList(
                            [
                            utils.Bucket(
                                utils.ConvLayer2D(
                                    in_features = latent_features,
                                    out_features = latent_features,
                                    kernel = 3,
                                    stride = 1,
                                    activation=activation,
                                    activation_all = activation_all,
                                    norm = norm,
                                    ),
                                #nn.Dropout(0.1),
                                utils.ConvLayer2D(
                                    in_features = latent_features,
                                    out_features = latent_features,
                                    kernel = 3,
                                    stride = 1,
                                    activation=activation,
                                    activation_all = activation_all,
                                    norm = norm,
                                    ),
                                ),
                            utils.Bucket(
                                utils.ConvLayer2D(
                                    in_features = latent_features,
                                    out_features = latent_features,
                                    kernel = 3,
                                    stride = 1,
                                    activation=activation,
                                    activation_all = activation_all,
                                    norm = norm,
                                    ),
                                #nn.Dropout(0.1),
                                utils.ConvLayer2D(
                                    in_features = latent_features,
                                    out_features = latent_features,
                                    kernel = 3,
                                    stride = 1,
                                    activation=activation,
                                    activation_all = activation_all,
                                    norm = norm,
                                    ),
                                ),
                            utils.Bucket(
                                utils.ConvLayer2D(
                                    in_features = latent_features,
                                    out_features = latent_features,
                                    kernel = 3,
                                    stride = 1,
                                    activation=activation,
                                    activation_all = activation_all,
                                    norm = norm,
                                    ),
                                #nn.Dropout(0.1),
                                utils.ConvLayer2D(
                                    in_features = latent_features,
                                    out_features = latent_features,
                                    kernel = 3,
                                    stride = 1,
                                    activation=activation,
                                    activation_all = activation_all,
                                    norm = norm,
                                    ),
                                ),
                            ]
                            )

        self.combiner_unit = nn.ModuleList(
                            [
                            utils.ConvLayer2D(
                                in_features = 2 * latent_features,
                                out_features = latent_features,
                                kernel = 1,
                                stride = 1,
                                activation = activation,
                                activation_all = activation_all,
                                norm = norm
                                ),
                            utils.ConvLayer2D(
                                in_features = 2 * latent_features,
                                out_features = latent_features,
                                kernel = 1,
                                stride = 1,
                                activation = activation,
                                activation_all = activation_all,
                                norm = norm
                                ),
                            ]
                            )

    @staticmethod
    def _list(x, modulelist):
        """
        Directly runs through a module list. 
        """
        for module in modulelist:
            x = module(x)
        return x

    def forward(self, x, sigma):
        """
        The forward method of a V2 type. 
        """
        x = self._list(x, self.initial)
        x = utils.VNET_Constant(
                        x = x,
                        down_unit = self.down_unit,
                        up_unit = self.up_unit,
                        skip_unit = self.skip_unit,
                        combine_unit = self.combiner_unit,
                                )
        x = self._list(x, self.final)

        # normalizing by Sigma. 
        used_sigmas = self.sigmas[sigma].view(x.shape[0], *([1] * len(x.shape[1:]))).float()
        x = x / used_sigmas
        return x


class VNetD4(nn.Module):
    """
    This model contain a VNet Structure. 
    All skip connections will be mediated
    using 1x1 convolutions to combine feature vectors. 

    This is a VNet with 2 across in the skip connection
    and 3 drops (so three layers) - hense the D3. Each
    connection has 2 across
    """
    def __init__(self, configs):
        """
        Configs should contain specific, model specific information. 
        (This defines things like number of inception blocks)

        Information necessary in the config file:

        model_structure:
            norm: the normalization method used. 
            activation: the activation used
            activation_all: Whether the activation should have channel dependent
                            parameters. 
            latent_features: The latent features in the model. 
            num_blocks: the number of initial blocks to include

        """
        super().__init__()

        # reading and storing sigmas
        self.register_buffer('sigmas', torch.tensor(utils.get_sigmas(configs)))

        # reading the model parameters
        latent_features = configs.model_structure.latent_features
        norm = configs.model_structure.norm
        in_features = out_features = configs.data.channels
        activation = configs.model_structure.activation
        activation_all = configs.model_structure.activation_all
        num_outer_incept_blocks = configs.model_structure.num_blocks # these are blocks on either end of the V. 

        # initial projector and final combiner - these will be symmetric
        self.initial = nn.ModuleList()
        self.final = nn.ModuleList()

        self.initial.append(utils.ConvLayer2D(
                in_features = in_features,
                out_features = latent_features,
                kernel = 3,
                stride = 1,
                activation = activation,
                activation_all = activation_all,
                norm = norm
                ))

        for i in range(num_outer_incept_blocks):
            self.initial.append(
                utils.ConvLayer2D(
                    in_features = latent_features,
                    out_features = latent_features,
                    kernel = 3,
                    stride = 1,
                    activation = activation,
                    activation_all = activation_all,
                    norm = norm,
                    ))

            self.final.append(
                utils.ConvLayer2D(
                    in_features = latent_features,
                    out_features = latent_features,
                    kernel = 3,
                    stride = 1,
                    activation=activation,
                    activation_all = activation_all,
                    norm = norm,
                    ))

        self.final.append(utils.ConvLayer2D(
                in_features = latent_features,
                out_features = out_features,
                kernel = 3,
                stride = 1,
                activation = None,
                activation_all = activation_all,
                norm = None,
                ))


        # defining the VNet Structure
        # must define up, down, combine, and skip units. 
        self.down_unit = nn.ModuleList(
                    [
                        utils.ConvLayer2D(
                            in_features = latent_features,
                            out_features = latent_features,
                            kernel = 2,
                            stride = 2,
                            activation = activation,
                            activation_all = activation_all,
                            norm = norm,
                            padding = None,
                            ),
                        utils.ConvLayer2D(
                            in_features = latent_features,
                            out_features = latent_features,
                            kernel = 2,
                            stride = 2,
                            activation = activation,
                            activation_all = activation_all,
                            norm = norm,
                            padding = None,
                            ),
                        utils.ConvLayer2D(
                            in_features = latent_features,
                            out_features = latent_features,
                            kernel = 2,
                            stride = 2,
                            activation = activation,
                            activation_all = activation_all,
                            norm = norm,
                            padding = None,
                            ),
                    ]
                        )

        self.up_unit = nn.ModuleList(
                    [
                        utils.ConvTransposeLayer2D(
                            in_features = latent_features,
                            out_features = latent_features,
                            kernel = 2,
                            stride = 2,
                            activation = activation,
                            activation_all = activation_all,
                            norm = norm,
                            padding = None,
                            ),
                        utils.ConvTransposeLayer2D(
                            in_features = latent_features,
                            out_features = latent_features,
                            kernel = 2,
                            stride = 2,
                            activation = activation,
                            activation_all = activation_all,
                            norm = norm,
                            padding = None,
                            ),
                        utils.ConvTransposeLayer2D(
                            in_features = latent_features,
                            out_features = latent_features,
                            kernel = 2,
                            stride = 2,
                            activation = activation,
                            activation_all = activation_all,
                            norm = norm,
                            padding = None,
                            ),
                    ]
                        )

        self.skip_unit = nn.ModuleList(
                            [
                            utils.Bucket(
                                utils.ConvLayer2D(
                                    in_features = latent_features,
                                    out_features = latent_features,
                                    kernel = 3,
                                    stride = 1,
                                    activation=activation,
                                    activation_all = activation_all,
                                    norm = norm,
                                    ),
                                #nn.Dropout(0.1),
                                utils.ConvLayer2D(
                                    in_features = latent_features,
                                    out_features = latent_features,
                                    kernel = 3,
                                    stride = 1,
                                    activation=activation,
                                    activation_all = activation_all,
                                    norm = norm,
                                    ),
                                ),
                            utils.Bucket(
                                utils.ConvLayer2D(
                                    in_features = latent_features,
                                    out_features = latent_features,
                                    kernel = 3,
                                    stride = 1,
                                    activation=activation,
                                    activation_all = activation_all,
                                    norm = norm,
                                    ),
                                #nn.Dropout(0.1),
                                utils.ConvLayer2D(
                                    in_features = latent_features,
                                    out_features = latent_features,
                                    kernel = 3,
                                    stride = 1,
                                    activation=activation,
                                    activation_all = activation_all,
                                    norm = norm,
                                    ),
                                ),
                            utils.Bucket(
                                utils.ConvLayer2D(
                                    in_features = latent_features,
                                    out_features = latent_features,
                                    kernel = 3,
                                    stride = 1,
                                    activation=activation,
                                    activation_all = activation_all,
                                    norm = norm,
                                    ),
                                #nn.Dropout(0.1),
                                utils.ConvLayer2D(
                                    in_features = latent_features,
                                    out_features = latent_features,
                                    kernel = 3,
                                    stride = 1,
                                    activation=activation,
                                    activation_all = activation_all,
                                    norm = norm,
                                    ),
                                ),
                            utils.Bucket(
                                utils.ConvLayer2D(
                                    in_features = latent_features,
                                    out_features = latent_features,
                                    kernel = 3,
                                    stride = 1,
                                    activation=activation,
                                    activation_all = activation_all,
                                    norm = norm,
                                    ),
                                #nn.Dropout(0.1),
                                utils.ConvLayer2D(
                                    in_features = latent_features,
                                    out_features = latent_features,
                                    kernel = 3,
                                    stride = 1,
                                    activation=activation,
                                    activation_all = activation_all,
                                    norm = norm,
                                    ),
                                ),
                            ]
                            )

        self.combiner_unit = nn.ModuleList(
                            [
                            utils.ConvLayer2D(
                                in_features = 2 * latent_features,
                                out_features = latent_features,
                                kernel = 1,
                                stride = 1,
                                activation = activation,
                                activation_all = activation_all,
                                norm = norm
                                ),
                            utils.ConvLayer2D(
                                in_features = 2 * latent_features,
                                out_features = latent_features,
                                kernel = 1,
                                stride = 1,
                                activation = activation,
                                activation_all = activation_all,
                                norm = norm
                                ),
                            utils.ConvLayer2D(
                                in_features = 2 * latent_features,
                                out_features = latent_features,
                                kernel = 1,
                                stride = 1,
                                activation = activation,
                                activation_all = activation_all,
                                norm = norm
                                ),
                            ]
                            )

    @staticmethod
    def _list(x, modulelist):
        """
        Directly runs through a module list. 
        """
        for module in modulelist:
            x = module(x)
        return x

    def forward(self, x, sigma):
        """
        The forward method of a V2 type. 
        """
        x = self._list(x, self.initial)
        x = utils.VNET_Constant(
                        x = x,
                        down_unit = self.down_unit,
                        up_unit = self.up_unit,
                        skip_unit = self.skip_unit,
                        combine_unit = self.combiner_unit,
                                )
        x = self._list(x, self.final)

        # normalizing by Sigma. 
        used_sigmas = self.sigmas[sigma].view(x.shape[0], *([1] * len(x.shape[1:]))).float()
        x = x / used_sigmas
        return x

