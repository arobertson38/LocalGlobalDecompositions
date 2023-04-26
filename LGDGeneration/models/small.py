"""
This file contains "small" networks. This are networks with - ideally - 
very few parameters that we can use effectively for materials informatics
tasks. 

"""
try:
    import models.utils as utils
except ModuleNotFoundError:
    import utils
    print('You better be running from inside the models folder.')

import torch
import torch.nn as nn
import numpy as np

class Inception(nn.Module):
    """
    an Inception Model
    """
    def __init__(self, configs):
        """
        Configs should contain specific, model specific information. 
        (This defines things like number of inception blocks)
        """
        super().__init__()

        # reading and storing sigmas
        self.register_buffer('sigmas', torch.tensor(utils.get_sigmas(configs)))
    
        # reading the model parameters
        num_blocks = configs.model_structure.num_blocks
        latent_features = configs.model_structure.latent_features
        norm = configs.model_structure.norm
        in_features = out_features = configs.data.channels
        activation = configs.model_structure.activation
        activation_all = configs.model_structure.activation_all

        # projector
        self.models = nn.ModuleList()
        self.models.append(utils.ConvLayer2D(
                in_features = in_features,
                out_features = latent_features,
                kernel = 3,
                stride = 1,
                activation = activation,
                activation_all = activation_all,
                norm = norm
                ))

        self.models.extend(
                [utils.InceptionBlock_Combiner(
                    latent_features,
                    latent_features,
                    activation=activation,
                    norm=norm,
                    activation_all=activation_all
                    ) for n in range(num_blocks)]
                )

        self.models.append(utils.ConvLayer2D(
                in_features = latent_features,
                out_features = out_features,
                kernel = 3,
                stride = 1,
                activation = None,
                activation_all = activation_all,
                norm = norm
                ))

    def forward(self, x, sigma):
        """
        The forward operator. 
        """
        for model in self.models:
            x = model(x)

        used_sigmas = self.sigmas[sigma].view(x.shape[0], *([1] * len(x.shape[1:]))).float()
        x = x / used_sigmas
        return x



class VNetC_Inception(nn.Module):
    """
    This model contain a VNet Structure while using Inception blocks
    in some of its main connections. All skip connections will be mediated
    using 1x1 convolutions to combine feature vectors. 

    This uses conlains 1 down VNet. 
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
                utils.InceptionBlock_Combiner(
                    latent_features,
                    latent_features,
                    activation=activation,
                    norm=norm,
                    activation_all=activation_all
                    ))

            self.final.append(
                utils.InceptionBlock_Combiner(
                    latent_features,
                    latent_features,
                    activation=activation,
                    norm=norm,
                    activation_all=activation_all
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
                    utils.Bucket(
                        utils.InceptionBlock_Combiner(
                            latent_features,
                            latent_features,
                            activation=activation,
                            norm=norm,
                            activation_all=activation_all
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
                            )
                            ),
                    ]
                        )

        self.up_unit = nn.ModuleList(
                    [
                    utils.Bucket(
                        utils.InceptionBlock_Combiner(
                            latent_features,
                            latent_features,
                            activation=activation,
                            norm=norm,
                            activation_all=activation_all
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
                            )
                            ),
                    ]
                        )

        self.skip_unit = nn.ModuleList(
                            [
                            utils.DirectConnection(),
                            utils.Bucket(
                                utils.InceptionBlock_Combiner(
                                    latent_features,
                                    latent_features,
                                    activation=activation,
                                    norm=norm,
                                    activation_all=activation_all
                                    ),
                                utils.InceptionBlock_Combiner(
                                    latent_features,
                                    latent_features,
                                    activation=activation,
                                    norm=norm,
                                    activation_all=activation_all
                                    ),
                                )
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

class VNetC(nn.Module):
    """
    This model contain a VNet Structure. 
    All skip connections will be mediated
    using 1x1 convolutions to combine feature vectors. 

    This uses conlains 1 down VNet. 
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
                        utils.ConvLayer2D(
                            in_features = latent_features,
                            out_features = latent_features,
                            kernel = 2,
                            stride = 2,
                            activation = activation,
                            activation_all = activation_all,
                            norm = norm,
                            padding = None,
                            )
                            ),
                    ]
                        )

        self.up_unit = nn.ModuleList(
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
                        utils.ConvTransposeLayer2D(
                            in_features = latent_features,
                            out_features = latent_features,
                            kernel = 2,
                            stride = 2,
                            activation = activation,
                            activation_all = activation_all,
                            norm = norm,
                            padding = None,
                            )
                            ),
                    ]
                        )

        self.skip_unit = nn.ModuleList(
                            [
                            utils.DirectConnection(),
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
                                utils.ConvLayer2D(
                                    in_features = latent_features,
                                    out_features = latent_features,
                                    kernel = 3,
                                    stride = 1,
                                    activation=activation,
                                    activation_all = activation_all,
                                    norm = norm,
                                    ),
                                )
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


class Residual_VNetC_Inception(nn.Module):
    """
    This model contain a VNet Structure while using Inception blocks
    in some of its main connections. All skip connections will be mediated
    using 1x1 convolutions to combine feature vectors. 

    This uses conlains 1 down VNet. 

    Also uses the following:
        (1) residual inception blocks
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
                utils.ResidualInceptionBlock_Combiner(
                    latent_features,
                    latent_features,
                    activation=activation,
                    norm=norm,
                    activation_all=activation_all
                    ))

            self.final.append(
                utils.ResidualInceptionBlock_Combiner(
                    latent_features,
                    latent_features,
                    activation=activation \
                            if i < (num_outer_incept_blocks-1) else None,
                    norm=norm,
                    activation_all=activation_all
                    ))

        self.final.append(utils.ConvLayer2D(
                in_features = latent_features,
                out_features = out_features,
                kernel = 3,
                stride = 1,
                activation = activation,
                activation_all = activation_all,
                norm = norm
                ))


        # defining the VNet Structure
        # must define up, down, combine, and skip units. 
        self.down_unit = nn.ModuleList(
                    [
                    utils.Bucket(
                        utils.ResidualInceptionBlock_Combiner(
                            latent_features,
                            latent_features,
                            activation=activation,
                            norm=norm,
                            activation_all=activation_all
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
                            )
                            ),
                    ]
                        )

        self.up_unit = nn.ModuleList(
                    [
                    utils.Bucket(
                        utils.ResidualInceptionBlock_Combiner(
                            latent_features,
                            latent_features,
                            activation=activation,
                            norm=norm,
                            activation_all=activation_all
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
                            )
                            ),
                    ]
                        )

        self.skip_unit = nn.ModuleList(
                            [
                            utils.DirectConnection(),
                            utils.Bucket(
                                utils.ResidualInceptionBlock_Combiner(
                                    latent_features,
                                    latent_features,
                                    activation=activation,
                                    norm=norm,
                                    activation_all=activation_all
                                    ),
                                utils.ResidualInceptionBlock_Combiner(
                                    latent_features,
                                    latent_features,
                                    activation=activation,
                                    norm=norm,
                                    activation_all=activation_all
                                    ),
                                )
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

class VNet1(nn.Module):
    """
    This model contain a VNet Structure. 
    All skip connections will be mediated
    using 1x1 convolutions to combine feature vectors. 

    This uses conlains 1 down VNet. 
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
                        utils.ConvLayer2D(
                            in_features = latent_features,
                            out_features = latent_features,
                            kernel = 2,
                            stride = 2,
                            activation = activation,
                            activation_all = activation_all,
                            norm = norm,
                            padding = None,
                            )
                            ),
                    ]
                        )

        self.up_unit = nn.ModuleList(
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
                        utils.ConvTransposeLayer2D(
                            in_features = latent_features,
                            out_features = latent_features,
                            kernel = 2,
                            stride = 2,
                            activation = activation,
                            activation_all = activation_all,
                            norm = norm,
                            padding = None,
                            )
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
                                utils.ConvLayer2D(
                                    in_features = latent_features,
                                    out_features = latent_features,
                                    kernel = 3,
                                    stride = 1,
                                    activation=activation,
                                    activation_all = activation_all,
                                    norm = norm,
                                    ),
                                )
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



class VNet1_Inception(nn.Module):
    """
    This model contain a VNet Structure while using Inception blocks
    in some of its main connections. All skip connections will be mediated
    using 1x1 convolutions to combine feature vectors. 

    This includes 2 drops. (Hence the 2...)
    This is chosen so that the lowest one has a visual resolution of 10x10. 
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
                utils.InceptionBlock_Combiner(
                    latent_features,
                    latent_features,
                    activation=activation,
                    norm=norm,
                    activation_all=activation_all
                    ))

            self.final.append(
                utils.InceptionBlock_Combiner(
                    latent_features,
                    latent_features,
                    activation=activation \
                            if i < (num_outer_incept_blocks-1) else None,
                    norm=norm,
                    activation_all=activation_all
                    ))

        self.final.append(utils.ConvLayer2D(
                in_features = latent_features,
                out_features = out_features,
                kernel = 3,
                stride = 1,
                activation = activation,
                activation_all = activation_all,
                norm = norm
                ))


        # defining the VNet Structure
        # must define up, down, combine, and skip units. 
        self.down_unit = nn.ModuleList(
                    [
                    utils.Bucket(
                        utils.InceptionBlock_Combiner(
                            latent_features,
                            latent_features,
                            activation=activation,
                            norm=norm,
                            activation_all=activation_all
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
                            )
                            ),
                    ]
                        )

        self.up_unit = nn.ModuleList(
                    [
                    utils.Bucket(
                        utils.InceptionBlock_Combiner(
                            latent_features,
                            latent_features,
                            activation=activation,
                            norm=norm,
                            activation_all=activation_all
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
                            )
                            ),
                    ]
                        )

        self.skip_unit = nn.ModuleList(
                            [
                            utils.Bucket(
                                utils.InceptionBlock_Combiner(
                                    latent_features,
                                    latent_features,
                                    activation=activation,
                                    norm=norm,
                                    activation_all=activation_all
                                    ),
                                utils.InceptionBlock_Combiner(
                                    latent_features,
                                    latent_features,
                                    activation=activation,
                                    norm=norm,
                                    activation_all=activation_all
                                    ),
                                ),
                            utils.Bucket(
                                utils.InceptionBlock_Combiner(
                                    latent_features,
                                    latent_features,
                                    activation=activation,
                                    norm=norm,
                                    activation_all=activation_all
                                    ),
                                utils.InceptionBlock_Combiner(
                                    latent_features,
                                    latent_features,
                                    activation=activation,
                                    norm=norm,
                                    activation_all=activation_all
                                    ),
                                )
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

class VNet2_Inception(nn.Module):
    """
    This model contain a VNet Structure while using Inception blocks
    in some of its main connections. All skip connections will be mediated
    using 1x1 convolutions to combine feature vectors. 

    This includes 2 drops. (Hence the 2...)
    This is chosen so that the lowest one has a visual resolution of 10x10. 
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
                utils.InceptionBlock_Combiner(
                    latent_features,
                    latent_features,
                    activation=activation,
                    norm=norm,
                    activation_all=activation_all
                    ))

            self.final.append(
                utils.InceptionBlock_Combiner(
                    latent_features,
                    latent_features,
                    activation=activation \
                            if i < (num_outer_incept_blocks-1) else None,
                    norm=norm,
                    activation_all=activation_all
                    ))

        self.final.append(utils.ConvLayer2D(
                in_features = latent_features,
                out_features = out_features,
                kernel = 3,
                stride = 1,
                activation = activation,
                activation_all = activation_all,
                norm = norm
                ))


        # defining the VNet Structure
        # must define up, down, combine, and skip units. 
        self.down_unit = nn.ModuleList(
                    [
                    utils.Bucket(
                        utils.InceptionBlock_Combiner(
                            latent_features,
                            latent_features,
                            activation=activation,
                            norm=norm,
                            activation_all=activation_all
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
                            )
                            ),
                    utils.Bucket(
                        utils.InceptionBlock_Combiner(
                            latent_features,
                            latent_features,
                            activation=activation,
                            norm=norm,
                            activation_all=activation_all
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
                            )
                            ),
                    ]
                        )

        self.up_unit = nn.ModuleList(
                    [
                    utils.Bucket(
                        utils.InceptionBlock_Combiner(
                            latent_features,
                            latent_features,
                            activation=activation,
                            norm=norm,
                            activation_all=activation_all
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
                            )
                            ),
                    utils.Bucket(
                        utils.InceptionBlock_Combiner(
                            latent_features,
                            latent_features,
                            activation=activation,
                            norm=norm,
                            activation_all=activation_all
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
                            )
                            ),
                    ]
                        )

        self.skip_unit = nn.ModuleList(
                            [
                            utils.Bucket(
                                utils.InceptionBlock_Combiner(
                                    latent_features,
                                    latent_features,
                                    activation=activation,
                                    norm=norm,
                                    activation_all=activation_all
                                    ),
                                utils.InceptionBlock_Combiner(
                                    latent_features,
                                    latent_features,
                                    activation=activation,
                                    norm=norm,
                                    activation_all=activation_all
                                    ),
                                ),
                            utils.SkipConnection(),
                            utils.Bucket(
                                utils.InceptionBlock_Combiner(
                                    latent_features,
                                    latent_features,
                                    activation=activation,
                                    norm=norm,
                                    activation_all=activation_all
                                    ),
                                utils.InceptionBlock_Combiner(
                                    latent_features,
                                    latent_features,
                                    activation=activation,
                                    norm=norm,
                                    activation_all=activation_all
                                    ),
                                )
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
                            utils.DirectConnection(),
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

class VNet3_Inception(nn.Module):
    """
    This model contain a VNet Structure while using Inception blocks
    in some of its main connections. All skip connections will be mediated
    using 1x1 convolutions to combine feature vectors. 

    This includes 3 drops. (Hence the 3...)
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
                utils.InceptionBlock_Combiner(
                    latent_features,
                    latent_features,
                    activation=activation,
                    norm=norm,
                    activation_all=activation_all
                    ))

            self.final.append(
                utils.InceptionBlock_Combiner(
                    latent_features,
                    latent_features,
                    activation=activation \
                            if i < (num_outer_incept_blocks-1) else None,
                    norm=norm,
                    activation_all=activation_all
                    ))

        self.final.append(utils.ConvLayer2D(
                in_features = latent_features,
                out_features = out_features,
                kernel = 3,
                stride = 1,
                activation = activation,
                activation_all = activation_all,
                norm = norm
                ))


        # defining the VNet Structure
        # must define up, down, combine, and skip units. 
        self.down_unit = nn.ModuleList(
                    [
                    utils.Bucket(
                        utils.InceptionBlock_Combiner(
                            latent_features,
                            latent_features,
                            activation=activation,
                            norm=norm,
                            activation_all=activation_all
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
                            )
                            ),
                    utils.Bucket(
                        utils.InceptionBlock_Combiner(
                            latent_features,
                            latent_features,
                            activation=activation,
                            norm=norm,
                            activation_all=activation_all
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
                            )
                            ),
                    utils.Bucket(
                        utils.InceptionBlock_Combiner(
                            latent_features,
                            latent_features,
                            activation=activation,
                            norm=norm,
                            activation_all=activation_all
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
                            )
                            ),
                    ]
                        )

        self.up_unit = nn.ModuleList(
                    [
                    utils.Bucket(
                        utils.InceptionBlock_Combiner(
                            latent_features,
                            latent_features,
                            activation=activation,
                            norm=norm,
                            activation_all=activation_all
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
                            )
                            ),
                    utils.Bucket(
                        utils.InceptionBlock_Combiner(
                            latent_features,
                            latent_features,
                            activation=activation,
                            norm=norm,
                            activation_all=activation_all
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
                            )
                            ),
                    utils.Bucket(
                        utils.InceptionBlock_Combiner(
                            latent_features,
                            latent_features,
                            activation=activation,
                            norm=norm,
                            activation_all=activation_all
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
                            )
                            ),
                    ]
                        )

        self.skip_unit = nn.ModuleList(
                            [
                            utils.DirectConnection(),
                            utils.DirectConnection(),
                            utils.SkipConnection(),
                            utils.Bucket(
                                utils.InceptionBlock_Combiner(
                                    latent_features,
                                    latent_features,
                                    activation=activation,
                                    norm=norm,
                                    activation_all=activation_all
                                    ),
                                utils.InceptionBlock_Combiner(
                                    latent_features,
                                    latent_features,
                                    activation=activation,
                                    norm=norm,
                                    activation_all=activation_all
                                    ),
                                )
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
                            utils.DirectConnection(),
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
        
# ------------------------------------------------ 
# Models with Attention
# ------------------------------------------------

class VNetC_Inception_InitialAttn(nn.Module):
    """
    This model contain a VNet Structure while using Inception blocks
    in some of its main connections. All skip connections will be mediated
    using 1x1 convolutions to combine feature vectors. 

    This uses conlains 1 down VNet. 

    Also uses the following:
        (2) an initial attention block at the very beginning
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

        self.initial.append(utils.AttnBlockpp(
                channels = in_features,
                skip_rescale = True,
                ))

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
                utils.InceptionBlock_Combiner(
                    latent_features,
                    latent_features,
                    activation=activation,
                    norm=norm,
                    activation_all=activation_all
                    ))

            self.final.append(
                utils.InceptionBlock_Combiner(
                    latent_features,
                    latent_features,
                    activation=activation \
                            if i < (num_outer_incept_blocks-1) else None,
                    norm=norm,
                    activation_all=activation_all
                    ))

        self.final.append(utils.ConvLayer2D(
                in_features = latent_features,
                out_features = out_features,
                kernel = 3,
                stride = 1,
                activation = activation,
                activation_all = activation_all,
                norm = norm
                ))


        # defining the VNet Structure
        # must define up, down, combine, and skip units. 
        self.down_unit = nn.ModuleList(
                    [
                    utils.Bucket(
                        utils.InceptionBlock_Combiner(
                            latent_features,
                            latent_features,
                            activation=activation,
                            norm=norm,
                            activation_all=activation_all
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
                            )
                            ),
                    ]
                        )

        self.up_unit = nn.ModuleList(
                    [
                    utils.Bucket(
                        utils.InceptionBlock_Combiner(
                            latent_features,
                            latent_features,
                            activation=activation,
                            norm=norm,
                            activation_all=activation_all
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
                            )
                            ),
                    ]
                        )

        self.skip_unit = nn.ModuleList(
                            [
                            utils.DirectConnection(),
                            utils.Bucket(
                                utils.InceptionBlock_Combiner(
                                    latent_features,
                                    latent_features,
                                    activation=activation,
                                    norm=norm,
                                    activation_all=activation_all
                                    ),
                                utils.InceptionBlock_Combiner(
                                    latent_features,
                                    latent_features,
                                    activation=activation,
                                    norm=norm,
                                    activation_all=activation_all
                                    ),
                                )
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


class VNetC_Inception_Attn(nn.Module):
    """
    This model contain a VNet Structure while using Inception blocks
    in some of its main connections. All skip connections will be mediated
    using 1x1 convolutions to combine feature vectors. 

    This uses conlains 1 down VNet. 

    Also uses the following:
        (2) uses several attention blocks throughout the whole structure
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

        self.initial.append(utils.AttnBlockpp(
                channels = in_features,
                skip_rescale = True,
                ))

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
                utils.InceptionBlock_Combiner(
                    latent_features,
                    latent_features,
                    activation=activation,
                    norm=norm,
                    activation_all=activation_all
                    ))

            self.final.append(
                utils.InceptionBlock_Combiner(
                    latent_features,
                    latent_features,
                    activation=activation,
                    norm=norm,
                    activation_all=activation_all
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
                    utils.Bucket(
                        utils.AttnBlockpp(
                                channels = latent_features,
                                skip_rescale = True,
                                ),
                        utils.InceptionBlock_Combiner(
                            latent_features,
                            latent_features,
                            activation=activation,
                            norm=norm,
                            activation_all=activation_all
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
                            )
                            ),
                    ]
                        )

        self.up_unit = nn.ModuleList(
                    [
                    utils.Bucket(
                        utils.AttnBlockpp(
                                channels = latent_features,
                                skip_rescale = True,
                                ),
                        utils.InceptionBlock_Combiner(
                            latent_features,
                            latent_features,
                            activation=activation,
                            norm=norm,
                            activation_all=activation_all
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
                            )
                            ),
                    ]
                        )

        self.skip_unit = nn.ModuleList(
                            [
                            utils.DirectConnection(),
                            utils.Bucket(
                                utils.AttnBlockpp(
                                        channels = latent_features,
                                        skip_rescale = True,
                                        ),
                                utils.InceptionBlock_Combiner(
                                    latent_features,
                                    latent_features,
                                    activation=activation,
                                    norm=norm,
                                    activation_all=activation_all
                                    ),
                                utils.InceptionBlock_Combiner(
                                    latent_features,
                                    latent_features,
                                    activation=activation,
                                    norm=norm,
                                    activation_all=activation_all
                                    ),
                                )
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

class Residual_VNetC_Inception_InitialAttn(nn.Module):
    """
    This model contain a VNet Structure while using Inception blocks
    in some of its main connections. All skip connections will be mediated
    using 1x1 convolutions to combine feature vectors. 

    This uses conlains 1 down VNet. 

    Also uses the following:
        (1) residual inception blocks
        (2) an initial attention block at the very beginning
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

        self.initial.append(utils.AttnBlockpp(
                channels = in_features,
                skip_rescale = True,
                ))

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
                utils.ResidualInceptionBlock_Combiner(
                    latent_features,
                    latent_features,
                    activation=activation,
                    norm=norm,
                    activation_all=activation_all
                    ))

            self.final.append(
                utils.ResidualInceptionBlock_Combiner(
                    latent_features,
                    latent_features,
                    activation=activation \
                            if i < (num_outer_incept_blocks-1) else None,
                    norm=norm,
                    activation_all=activation_all
                    ))

        self.final.append(utils.ConvLayer2D(
                in_features = latent_features,
                out_features = out_features,
                kernel = 3,
                stride = 1,
                activation = activation,
                activation_all = activation_all,
                norm = norm
                ))


        # defining the VNet Structure
        # must define up, down, combine, and skip units. 
        self.down_unit = nn.ModuleList(
                    [
                    utils.Bucket(
                        utils.ResidualInceptionBlock_Combiner(
                            latent_features,
                            latent_features,
                            activation=activation,
                            norm=norm,
                            activation_all=activation_all
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
                            )
                            ),
                    ]
                        )

        self.up_unit = nn.ModuleList(
                    [
                    utils.Bucket(
                        utils.ResidualInceptionBlock_Combiner(
                            latent_features,
                            latent_features,
                            activation=activation,
                            norm=norm,
                            activation_all=activation_all
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
                            )
                            ),
                    ]
                        )

        self.skip_unit = nn.ModuleList(
                            [
                            utils.DirectConnection(),
                            utils.Bucket(
                                utils.ResidualInceptionBlock_Combiner(
                                    latent_features,
                                    latent_features,
                                    activation=activation,
                                    norm=norm,
                                    activation_all=activation_all
                                    ),
                                utils.ResidualInceptionBlock_Combiner(
                                    latent_features,
                                    latent_features,
                                    activation=activation,
                                    norm=norm,
                                    activation_all=activation_all
                                    ),
                                )
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


class Residual_VNetC_Inception_Attn(nn.Module):
    """
    This model contain a VNet Structure while using Inception blocks
    in some of its main connections. All skip connections will be mediated
    using 1x1 convolutions to combine feature vectors. 

    This uses conlains 1 down VNet. 

    Also uses the following:
        (1) residual inception blocks
        (2) uses several attention blocks throughout the whole structure
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

        self.initial.append(utils.AttnBlockpp(
                channels = in_features,
                skip_rescale = True,
                ))

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
                utils.ResidualInceptionBlock_Combiner(
                    latent_features,
                    latent_features,
                    activation=activation,
                    norm=norm,
                    activation_all=activation_all
                    ))

            self.final.append(
                utils.ResidualInceptionBlock_Combiner(
                    latent_features,
                    latent_features,
                    activation=activation \
                            if i < (num_outer_incept_blocks-1) else None,
                    norm=norm,
                    activation_all=activation_all
                    ))

        self.final.append(utils.ConvLayer2D(
                in_features = latent_features,
                out_features = out_features,
                kernel = 3,
                stride = 1,
                activation = activation,
                activation_all = activation_all,
                norm = norm
                ))


        # defining the VNet Structure
        # must define up, down, combine, and skip units. 
        self.down_unit = nn.ModuleList(
                    [
                    utils.Bucket(
                        utils.AttnBlockpp(
                                channels = latent_features,
                                skip_rescale = True,
                                ),
                        utils.ResidualInceptionBlock_Combiner(
                            latent_features,
                            latent_features,
                            activation=activation,
                            norm=norm,
                            activation_all=activation_all
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
                            )
                            ),
                    ]
                        )

        self.up_unit = nn.ModuleList(
                    [
                    utils.Bucket(
                        utils.AttnBlockpp(
                                channels = latent_features,
                                skip_rescale = True,
                                ),
                        utils.ResidualInceptionBlock_Combiner(
                            latent_features,
                            latent_features,
                            activation=activation,
                            norm=norm,
                            activation_all=activation_all
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
                            )
                            ),
                    ]
                        )

        self.skip_unit = nn.ModuleList(
                            [
                            utils.DirectConnection(),
                            utils.Bucket(
                                utils.AttnBlockpp(
                                        channels = latent_features,
                                        skip_rescale = True,
                                        ),
                                utils.ResidualInceptionBlock_Combiner(
                                    latent_features,
                                    latent_features,
                                    activation=activation,
                                    norm=norm,
                                    activation_all=activation_all
                                    ),
                                utils.ResidualInceptionBlock_Combiner(
                                    latent_features,
                                    latent_features,
                                    activation=activation,
                                    norm=norm,
                                    activation_all=activation_all
                                    ),
                                )
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

class VNetC_Attn(nn.Module):
    """
    This model contain a VNet Structure. 
    All skip connections will be mediated
    using 1x1 convolutions to combine feature vectors. 

    This uses conlains 1 down VNet. 
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
        attention = utils.get_attention(configs.model_structure.attention)

        # initial projector and final combiner - these will be symmetric
        self.initial = nn.ModuleList()
        self.final = nn.ModuleList()

        self.initial.append(
                attention(channels = in_features)
                )

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
                    utils.Bucket(
                        attention(channels = latent_features),
                        utils.ConvLayer2D(
                            in_features = latent_features,
                            out_features = latent_features,
                            kernel = 3,
                            stride = 1,
                            activation=activation,
                            activation_all = activation_all,
                            norm = norm,
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
                            )
                            ),
                    ]
                        )

        self.up_unit = nn.ModuleList(
                    [
                    utils.Bucket(
                        attention(channels = latent_features),
                        utils.ConvLayer2D(
                            in_features = latent_features,
                            out_features = latent_features,
                            kernel = 3,
                            stride = 1,
                            activation=activation,
                            activation_all = activation_all,
                            norm = norm,
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
                            )
                            ),
                    ]
                        )

        self.skip_unit = nn.ModuleList(
                            [
                            utils.DirectConnection(),
                            utils.Bucket(
                                attention(channels = latent_features),
                                utils.ConvLayer2D(
                                    in_features = latent_features,
                                    out_features = latent_features,
                                    kernel = 3,
                                    stride = 1,
                                    activation=activation,
                                    activation_all = activation_all,
                                    norm = norm,
                                    ),
                                utils.ConvLayer2D(
                                    in_features = latent_features,
                                    out_features = latent_features,
                                    kernel = 3,
                                    stride = 1,
                                    activation=activation,
                                    activation_all = activation_all,
                                    norm = norm,
                                    ),
                                )
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

class AttentionInception(nn.Module):
    """
    an Inception Model
    """
    def __init__(self, configs):
        """
        Configs should contain specific, model specific information. 
        (This defines things like number of inception blocks)
        """
        super().__init__()

        # reading and storing sigmas
        self.register_buffer('sigmas', torch.tensor(utils.get_sigmas(configs)))
    
        # reading the model parameters
        num_blocks = configs.model_structure.num_blocks
        latent_features = configs.model_structure.latent_features
        norm = configs.model_structure.norm
        in_features = out_features = configs.data.channels
        activation = configs.model_structure.activation
        activation_all = configs.model_structure.activation_all

        # projector
        self.models = nn.ModuleList()

        self.models.append(utils.AttnBlockpp(
                channels = in_features,
                skip_rescale = True,
                ))

        self.models.append(utils.ConvLayer2D(
                in_features = in_features,
                out_features = latent_features,
                kernel = 3,
                stride = 1,
                activation = activation,
                activation_all = activation_all,
                norm = norm
                ))

        for n in range(num_blocks):
            self.models.append(utils.AttnBlockpp(
                    channels = latent_features,
                    skip_rescale = True,
                    ))

            self.models.append( \
                utils.InceptionBlock_Combiner(
                    latent_features,
                    latent_features,
                    activation=activation,
                    norm=norm,
                    activation_all=activation_all
                    )
                )

        self.models.append(utils.ConvLayer2D(
                in_features = latent_features,
                out_features = out_features,
                kernel = 3,
                stride = 1,
                activation = None,
                activation_all = activation_all,
                norm = None,
                ))

    def forward(self, x, sigma):
        """
        The forward operator. 
        """
        for model in self.models:
            x = model(x)

        used_sigmas = self.sigmas[sigma].view(x.shape[0], *([1] * len(x.shape[1:]))).float()
        x = x / used_sigmas
        return x
