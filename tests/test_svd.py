import unittest

import numpy as np
from numpy.testing import assert_allclose

import clearbox as cb

from clearbox.linear import orth_svd, qr_svd

class TestEigen(unittest.TestCase):

    def test_orthogonal(self):

        # identity
        eye = np.eye(10)
        sig, u, v = orth_svd(eye)

        assert_allclose(sig, np.diag(eye), atol=1e-10)
        assert_allclose(u @ np.diag(sig) @ v.T, eye, atol=1e-10)

        # Generic random matrix
        m = np.random.standard_normal((100, 10))

        sig, u, v = orth_svd(m, maxit=10_000)
        assert_allclose(u @ np.diag(sig) @ v.T, m, atol=1e-5)

        unp, signp, vtnp = np.linalg.svd(m, full_matrices=False)
        assert_allclose(np.abs(sig), np.abs(signp), atol=1e-8)
