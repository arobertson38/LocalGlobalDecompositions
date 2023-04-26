"""
This file contains important utility and layer files for building networks
from othe projects. 
"""
import numpy as np
import torch
import torch.nn as nn
import functools
import torch.nn.functional as F

# -------------------------------------------------------
# Utilities
# -------------------------------------------------------

def dict2namespace(config):
    """
    Converts a dictionary to a namespace
    """
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def get_sigmas(config):
    """Get sigmas --- the set of noise levels for SMLD from config files.
    Args:
    config: A ConfigDict object parsed from the config file
    Returns:
    sigmas: a jax numpy arrary of noise levels
    """
    sigmas = np.exp( \
                    np.linspace(np.log(config.model.sigma_begin), \
                    np.log(config.model.sigma_end), \
                    config.model.num_classes))

    return sigmas

def get_activation(activation, num_features):
    """
    Returns the activation identified (along with parameters if necessary
    """
    if activation is None:
        return lambda x: x
    elif activation.lower() == 'prelu':
        return nn.PReLU(num_parameters=num_features)
    elif activation.lower() == 'relu':
        return nn.ReLU()
    elif activation.lower() == 'leaky':
        return nn.LeakyReLU()
    elif activation.lower() == 'elu':
        return nn.ELU()
    elif activation.lower() == 'swish':
        return nn.SiLU()
    else:
        raise NotImplementedError(f"Nonlinearity={activation} is not supported.")

def get_normalization(normalization, num_features):
    """
    Returns a normalization block. 
    """
    if normalization is None:
        return lambda x: x
    elif normalization.lower() == 'instancenorm2dplus':
        return InstanceNorm2dPlus(num_features)
    elif normalization.lower() == 'instancenorm2d':
        return nn.InstanceNorm2d(num_features)
    else:
        raise NotImplementedError(f"Norm={normalization} is not supported.")


# -------------------------------------------------------
# Attention Mechanisms
# -------------------------------------------------------

class ECAAttn(nn.Module):

    """
    Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channels, k_size=3):
        super(ECAAttn, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)
        
        return x * y.expand_as(x)

class SEAttn(nn.Module):
    """
    Constructs a SE Attention Module. 

    This is basically the oldest form of 
    channel based attention mechanisms. 
    """
    def __init__(self, channels, reduction=8):
        super(SEAttn, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.activation = nn.Sigmoid()
        self.reduction = reduction

        self.fc = nn.Sequential(
                nn.Linear(channels, max(channels // reduction, 2)),
                nn.ReLU(inplace=True),
                nn.Linear(max(channels // reduction, 2), channels),
            )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_y = self.avgpool(x).view(b, c)
        gate = self.fc(avg_y).view(b, c, 1, 1)
        gate = self.activation(gate)

        return x * gate

class AttnBlockpp(nn.Module):
  """Channel-wise self-attention block. Modified from DDPM."""

  def __init__(self, channels, skip_rescale=True, init_scale=0.):
    super().__init__()
    num_groups = min(channels // 4, 32)
    num_groups = channels if num_groups == 0 else num_groups
    self.GroupNorm_0 = nn.GroupNorm(num_groups=num_groups, num_channels=channels,
                                  eps=1e-6)

    self.NIN_0 = ConvLayer2D(
                    in_features = channels,
                    out_features = channels,
                    kernel = 1,
                    stride = 1,
                    bias = True,
                    activation=None,
                    norm=None,
                    activation_all=False,
                    )
    self.NIN_1 = ConvLayer2D(
                    in_features = channels,
                    out_features = channels,
                    kernel = 1,
                    stride = 1,
                    bias = True,
                    activation=None,
                    norm=None,
                    activation_all=False,
                    )
    self.NIN_2 = ConvLayer2D(
                    in_features = channels,
                    out_features = channels,
                    kernel = 1,
                    stride = 1,
                    bias = True,
                    activation=None,
                    norm=None,
                    activation_all=False,
                    )
    self.NIN_3 = ConvLayer2D(
                    in_features = channels,
                    out_features = channels,
                    kernel = 1,
                    stride = 1,
                    bias = True,
                    activation=None,
                    norm=None,
                    activation_all=False,
                    )

    #self.NIN_0 = NIN(channels, channels)
    #self.NIN_1 = NIN(channels, channels)
    #self.NIN_2 = NIN(channels, channels)
    #self.NIN_3 = NIN(channels, channels, init_scale=init_scale)
    self.skip_rescale = skip_rescale

  def forward(self, x):
    B, C, H, W = x.shape
    h = self.GroupNorm_0(x)
    q = self.NIN_0(h)
    k = self.NIN_1(h)
    v = self.NIN_2(h)

    w = torch.einsum('bchw,bcij->bhwij', q, k) * (int(C) ** (-0.5))
    w = torch.reshape(w, (B, H, W, H * W))
    w = F.softmax(w, dim=-1)
    w = torch.reshape(w, (B, H, W, H, W))
    h = torch.einsum('bhwij,bcij->bchw', w, v)
    h = self.NIN_3(h)
    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)

def get_attention(attention):
    """
    Returns an attention block
    """
    if attention.lower() == "eca":
        return ECAAttn
    elif attention.lower() == "se":
        return SEAttn
    elif attention.lower() == "attnblockpp":
        return AttnBlockpp

# -------------------------------------------------------
# Time Embedding
# -------------------------------------------------------

class GaussianFourierEmbedding(nn.Module):
    """
    Taken directly from Song's SDE implementation. 

    Notes:
    I think the way he assigns frequencies is rather
    haphazard... but, it works. So...

    Specifically, I am referring to the fact that he assigns
    frequencies using a normal distribution. 
    """
    def __init__(self, embedding_size=128, scale=16.0):
        """

        :param embedding_size: the number of individual frequencies
                               to use in the time embedding. 
        :param scale: the width of the normal to select freq from. 
        """
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, \
                requires_grad=False)

    def forward(self, t):
        t_proj = t[:, None] * self.W[None, :] * 2 * torch.pi
        return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)


# -------------------------------------------------------
# Normalization Objects
# -------------------------------------------------------

class InstanceNorm2dPlus(nn.Module):
    def __init__(self, num_features, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)
        self.alpha = nn.Parameter(torch.zeros(num_features))
        self.gamma = nn.Parameter(torch.zeros(num_features))
        self.alpha.data.normal_(1, 0.02)
        self.gamma.data.normal_(1, 0.02)
        if bias:
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        means = torch.mean(x, dim=(2, 3))
        m = torch.mean(means, dim=-1, keepdim=True)
        v = torch.var(means, dim=-1, keepdim=True)
        means = (means - m) / (torch.sqrt(v + 1e-5))
        h = self.instance_norm(x)

        if self.bias:
            h = h + means[..., None, None] * self.alpha[..., None, None]
            out = self.gamma.view(-1, self.num_features, 1, 1) * h + self.beta.view(-1, self.num_features, 1, 1)
        else:
            h = h + means[..., None, None] * self.alpha[..., None, None]
            out = self.gamma.view(-1, self.num_features, 1, 1) * h
        return out

class ConditionalInstanceNorm2dPlus(nn.Module):
    def __init__(self, num_features, num_classes, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)
        if bias:
            self.embed = nn.Embedding(num_classes, num_features * 3)
            self.embed.weight.data[:, :2 * num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
            self.embed.weight.data[:, 2 * num_features:].zero_()  # Initialise bias at 0
        else:
            self.embed = nn.Embedding(num_classes, 2 * num_features)
            self.embed.weight.data.normal_(1, 0.02)

    def forward(self, x, y):
        means = torch.mean(x, dim=(2, 3))
        m = torch.mean(means, dim=-1, keepdim=True)
        v = torch.var(means, dim=-1, keepdim=True)
        means = (means - m) / (torch.sqrt(v + 1e-5))
        h = self.instance_norm(x)

        if self.bias:
            gamma, alpha, beta = self.embed(y).chunk(3, dim=-1)
            h = h + means[..., None, None] * alpha[..., None, None]
            out = gamma.view(-1, self.num_features, 1, 1) * h + beta.view(-1, self.num_features, 1, 1)
        else:
            gamma, alpha = self.embed(y).chunk(2, dim=-1)
            h = h + means[..., None, None] * alpha[..., None, None]
            out = gamma.view(-1, self.num_features, 1, 1) * h
        return out



# -------------------------------------------------------
# Network Layers
# -------------------------------------------------------

class ConvLayer2D(nn.Module):
    """
    An individual Convolutional Layer. 
    """
    def __init__(self, in_features, out_features, kernel, 
            stride, bias=True, padding='zeros', activation='prelu',
            norm='InstanceNorm2d', activation_all=True):
        super().__init__()

        pad = kernel // 2 if padding is not None else 0
        if padding is None:
            padding = 'zeros'

        self.conv = nn.Conv2d(
                            in_channels = in_features,
                            out_channels = out_features,
                            kernel_size = kernel,
                            stride = stride,
                            padding = pad, 
                            padding_mode = padding,
                            bias=bias,
                        )
        self.activation = get_activation(activation, 
                num_features=out_features if activation_all else 1)
        self.norm = get_normalization(norm if out_features > 1 else None, num_features=out_features)

    def forward(self, x):
        """
        The forward call. 
        """
        return self.activation(self.norm(self.conv(x)))


class ConvTransposeLayer2D(nn.Module):
    """
    An individual Convolutional Layer. 
    """
    def __init__(self, in_features, out_features, kernel, 
            stride, bias=True, padding='zeros', activation='prelu', 
            norm='InstanceNorm2d', activation_all=True):
        super().__init__()

        pad = kernel // 2 if padding is not None else 0
        if padding is None:
            padding = 'zeros'

        self.conv = nn.ConvTranspose2d(
                            in_channels = in_features,
                            out_channels = out_features,
                            kernel_size = kernel,
                            stride = stride,
                            padding = pad, 
                            padding_mode = padding,
                            bias=bias,
                        )
        self.activation = get_activation(activation, 
                num_features=out_features if activation_all else 1)
        self.norm = get_normalization(norm, num_features=out_features)

    def forward(self, x):
        """
        The forward call. 
        """
        return self.activation(self.norm(self.conv(x)))

# --------------------------------------------------
# Flag Classes
# --------------------------------------------------

class DirectConnection(nn.Module):
    """
    This class just implements a direct connection layer in code. 
    """
    def forward(self, x):
        return x

class SkipConnection(nn.Module):
    """
    This class is just to indicate the function to skip the
    current connection. 
    """
    def forward(self, x):
        raise NotImplementedError("This is a skip, it should never be called")

class Bucket(nn.Module):
    """
    A bucket for combining multiple different layers into a 
    "single layer"
    """
    def __init__(self, *args):
        super().__init__()
        self.model = nn.Sequential(*args)

    def forward(self, x):
        return self.model(x)

def VNET_Constant(x, down_unit, up_unit, skip_unit, combine_unit):
    """
    This method just takes the elements that compose a VNet and operates them
    together in a suscicent fashion. 

    
    All of these module lists should be ordered such that the
    go from the highest level to the lowest level. 

    :down_unit: a module list containing all of the down connections
    :up_unit: a module list containing all of the upconnections


    """
    assert len(down_unit) == len(up_unit), f"down: {len(down_unit)} and up: {len(up_unit)}."
    assert len(skip_unit) == (len(up_unit) + 1), f"skip: {len(skip_unit)} and up: {len(up_unit)}."
    assert len(up_unit) == len(combine_unit), f"combine: {len(combine_unit)} and up: {len(up_unit)}."
    assert skip_unit[-1] is not SkipConnection


    skipped_variables = []

    for i in range(len(down_unit)):
        # apply the skip
        if type(skip_unit[i]) is not SkipConnection:
            skipped_variables.append(skip_unit[i](x))
        else:
            skipped_variables.append(None)

        # apply the down
        x = down_unit[i](x)

    # apply the last skip
    x = skip_unit[-1](x)

    # now for the ups
    for i in range(1, len(up_unit)+1):
        x = up_unit[-i](x)
        if skipped_variables[-i] is not None:
            x = torch.cat([x, skipped_variables[-i]], dim=1)
            x = combine_unit[-i](x)

    return x


# ----------------------------------------------------
# Inception Blocks
# ----------------------------------------------------

class InceptionBlock_Combiner(nn.Module):
    """
    A simple Inception block as described in 
    "Going Deep With Convolutions"

    Predictions from each inception component are
    combined using a 1x1 convolution across all weights. 

    Employs both activation and instancenorm2dplus normalization
    """
    def __init__(self, 
            in_features, 
            out_features, 
            activation='prelu',
            norm='InstanceNorm2d',
            activation_all=True,
            ):
        super().__init__()
        conv = functools.partial(ConvLayer2D, activation=activation, \
                norm=norm, activation_all=activation_all)
        self.conv1x1 = conv(in_features, in_features, 1, 1)
        self.conv3x3 = conv(in_features, in_features, 3, 1)
        self.conv5x5 = conv(in_features, in_features, 5, 1)
        self.combiner = conv(in_features * 3, out_features, 1, 1)

    def forward(self, x):
        """
        The forward operation for the Inception Block
        """
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.conv5x5(x)

        out = self.combiner(
                    torch.cat([x1, x2, x3], dim=1)
                             )
        return out


class ResidualInceptionBlock_Combiner(nn.Module):
    """
    Extension of the InceptionBlock from earlier to structure
    it like a ResNet Layer. I'm doing this because it seems like
    we are going to need some fairly deep networks. 

    A simple Inception block as described in 
    "Going Deep With Convolutions"

    Predictions from each inception component are
    combined using a 1x1 convolution across all weights. 

    Employs both activation and instancenorm2dplus normalization
    """
    def __init__(self, 
            in_features, 
            out_features, 
            activation='prelu',
            norm='InstanceNorm2d',
            activation_all=True,
            ):
        super().__init__()
        conv = functools.partial(ConvLayer2D, activation=activation, \
                norm=norm, activation_all=activation_all)
        self.conv1x1 = conv(in_features, in_features, 1, 1)
        self.conv3x3 = conv(in_features, in_features, 3, 1)
        self.conv5x5 = conv(in_features, in_features, 3, 1)
        self.combiner = conv(in_features * 3, out_features, 1, 1)

        # 1x1 conv to rescale the skip connection in the 
        # residual structure. 
        if in_features == out_features:
            self.residual = lambda x: x
        else:
            self.residual = conv(in_features, out_features, 1, 1)


    def forward(self, x):
        """
        The forward operation for the Inception Block
        """
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.conv5x5(x)

        out = self.combiner(
                    torch.cat([x1, x2, x3], dim=1)
                             )
        return self.residual(x) + out

