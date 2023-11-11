import unittest

import numpy as np
from numpy.testing import assert_allclose

import clearbox as cb

from clearbox.linear import orth_eigen, qr_eigen

class TestEigen(unittest.TestCase):

    def test_orthogonal(self):

        # identity
        eye = np.eye(10)
        v, x = orth_eigen(eye)

        self.assertAlmostEqual(v[3], 1., places=10)

        # Symmetric matrix
        m = np.random.standard_normal((100, 10))
        s = m.T @ m

        cbval, cbvect = orth_eigen(s, maxit=10_000)
        npval, npvect = np.linalg.eigh(s)

        assert_allclose(cbval, npval[::-1])

        rec = cbvect @ np.diag(cbval) @ cbvect.T
        assert_allclose(rec, s, rtol=1e-5, atol=1e-5)

        # Almost symmetric matrix
        m = np.random.standard_normal((100, 10))
        s = m.T @ m + np.random.standard_normal((10, 10)) * 1e-10


        cbval, cbvect = orth_eigen(s, maxit=10_000)
        npval, npvect = np.linalg.eigh(s)

        assert_allclose(cbval, npval[::-1])

        rec = cbvect @ np.diag(cbval) @ cbvect.T
        assert_allclose(rec, s, rtol=1e-5, atol=1e-5)

        # Non-symmetric matrix (almost certainly invertible)
        # m = np.random.standard_normal((10, 10))
        # cbval, cbvect = orth_eigen(m, maxit=1e4, check_sym=False)
        #
        # print(cbval)

        # Singular matrix
        # m[:, 7:] = 0
        # cbval, cbvect = orth_eigen(m, maxit=10_000)
        #
        # print(cbval)

    def test_qr(self):
        # identity
        # eye = np.eye(10)
        # v, x = qr_eigen(eye)
        #
        # self.assertAlmostEqual(v[3], 1., places=10)

        # Symmetric matrix
        m = np.random.standard_normal((100, 3))
        s = m.T @ m

        cbval, cbvect = qr_eigen(s, maxit=10_000)
        npval, npvect = np.linalg.eigh(s)

        assert_allclose(cbval, npval[::-1])

        rec = cbvect @ np.diag(cbval) @ cbvect.T
        assert_allclose(rec, s, rtol=1e-5, atol=1e-5)

        # Almost symmetric matrix
        # m = np.random.standard_normal((100, 10))
        # s = m.T @ m + np.random.standard_normal((10, 10)) * 1e-10
        #
        # cbval, cbvect = qr_eigen(s, maxit=100_000)
        # npval, npvect = np.linalg.eigh(s)
        #
        # assert_allclose(cbval, npval[::-1])
        #
        # rec = cbvect @ np.diag(cbval) @ cbvect.T
        # assert_allclose(rec, s, rtol=1e-5, atol=1e-5)

