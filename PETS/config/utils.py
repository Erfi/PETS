import numpy as np
import torch
from torch import nn as nn


def swish(x):
    return x * torch.sigmoid(x)


def truncated_normal(size, std):
    """
    Pytorch does not have a truncated normal function to we manually make one
    in order to cut the dependancy on tensorflow
    Modified Version of: https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/20
    """
    mean = 0
    tensor = torch.zeros(size)
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


def get_affine_params(ensemble_size, in_features, out_features):

    w = truncated_normal(size=(ensemble_size, in_features, out_features),
                         std=1.0 / (2.0 * np.sqrt(in_features)))
    w = nn.Parameter(w)

    b = nn.Parameter(torch.zeros(ensemble_size, 1,
                                 out_features, dtype=torch.float32))

    return w, b
