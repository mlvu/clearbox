import unittest

import clearbox as cb
import numpy as np

from clearbox.tools import logsumexp

import torch

class TestUtilities(unittest.TestCase):

    def test_logsumexp(self):
        # sample somematrix that sums to one
        x = np.random.rand(15, 4)
        x = x / x.sum(axis=1, keepdims=True)

        logx = np.log(x)

        summed = logsumexp(logx, axis=1)
        naive = np.log(np.exp(logx).sum(axis=1))

        print(np.log(x.sum(axis=1)))
        print(summed)
        print(naive)

