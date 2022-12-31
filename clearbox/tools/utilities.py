import numpy as np

import torch

def logsumexp(array, axis, keepdims=False):
    """
    Numerically stable log-sum over
    :param array:
    :param axis:
    :return:
    """

    ma = array.max(axis=axis, keepdims=True)
    array = np.log(np.exp(array - ma).sum(axis=axis, keepdims=keepdims))
    array = array + (ma if keepdims else ma.squeeze(axis))

    return array

def kl_categorical(p, q, eps=1e-10):
    """
    Computes KL divergence between two categorical distributions.

    :param p: (..., n)
    :param q: (..., n)
    :return:
    """

    p = p + eps
    q = q + eps

    # entropy of p
    entp = - p * np.log2(p)
    entp = entp.sum(axis=-1)

    # cross-entropy of p to q
    xent = - p * np.log2(q)
    xent = xent.sum(axis=-1)

    return xent - entp

def flatten(tensors):
    """
    Flattens an iterable over tensors into a single vector
    :param tensors:
    :return:
    """
    return torch.cat([p.reshape(-1) for p in tensors], dim=0)

def prod(tuple):
    res = 1
    for v in tuple:
        res *= v
    return res

def set_parameters(parameters, model):
    """
    Take the given vector of parameters and set them as the parameters of the model, slicing and reshaping as necessary.

    :param params:
    :param model:
    :return:
    """
    if parameters is not None:

        cursor = 0
        for p in model.parameters():
            size = prod(p.size())
            slice = parameters[cursor:cursor + size]
            slice = slice.reshape(p.size())
            p.data = slice.contiguous()

            cursor = cursor + size