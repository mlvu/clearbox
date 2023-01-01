import numpy as np

import numpy.typing as nt
import numpy.linalg as la

import warnings

"""
Various methods for computing the singular value decomposition of a matrix.

"""


def orthogonal(a : nt.ArrayLike, num='full', maxit=float('inf'), eps=1e-8):
    """
    Computes the first n singular vectors and the corresponding singular values of the matrix `a` using the
    orthogonal iteration algorithm.

    References:
     - https://peterbloem.nl/blog/pca-5

    :param a: A square matrix.
    :param num: The number of singular vectors and -values to return. The n eigenpairs with the largest eigenvalues by
    magnitude are returned.
    :param maxit: Max number of iterations. Default `inf` (i.e. no maximum).
    :param eps: Epsilon for the stop condition. If the norm of the candidate eigenvectors changes less than this between
    iterations, the algorithm considers the iteration to be converged and returns.
    :return: A triple (values, left vectors, right vector) of computed singular values and their left and right vectors
    """

    assert len(a.shape) == 2, f'Input a should be a matrix. Size was {a.shape}.'

    n, m = a.shape

    k = min(n, m) # max rank
    num = k if num == 'full' else num

    assert type(num) == int and num <= n, f'Number of singular vectors ({n}) should be an integer or "full".'

    p = np.random.standard_normal((m, k) )

    i = 0
    while i < maxit:

        p0 = p

        y = a @ p
        q, r = la.qr(y)

        x = a.T @ q
        p, l = la.qr(x)

        # -- Stop condition: q should be equal to x normalized to unit vectors.
        if (diff := la.norm(p - p0)) < eps:
            return np.diag(r), q, p

        i += 1

    warnings.warn(f'Algorithm did not converge (last diff={diff}). Returning best guess for singular triples.')

    return np.diag(r), q, p

def diag(diag, n, m):
    """
    Returns a matrix of arbitrary size with the given elements alon the diagonal.
    :param n:
    :param m:
    :param diag:
    :return:
    """
    out = np.zeros((n, m))
    diag = np.diag(diag)

    nd, md = diag.shape; assert nd == md

    out[:nd, :md] = diag

    return out

def qr(a : nt.ArrayLike, maxit=float('inf'), eps=1e-8, check_sym=True):
    """
    Computes the  singular value decomposition for a matrix `a` by the QR aliteration algorithm.

    References:
     - https://peterbloem.nl/blog/pca-5

    :param a: A matrix.
    :param maxit: Max number of iterations. Default `inf` (i.e. no maximum).
    :param eps: Epsilon for the stop condition. If the norm of the candidate eigenvectors changes less than this between
    iterations, the algorithm considers the itertion to be converged and returns.
    :return: A triple (values, left vectors, right vector) of computed singular values and their left and right vectors
    """

    assert len(a.shape) == 2, f'Input a should be a matrix. Size was {a.shape}.'

    n, m = a.shape

    x, y = a, a.T
    i = 0

    qprod, pprod = np.eye(n, n), np.eye(m, m)

    while i < maxit:
        x0 = x
        y0 = y

        q, r = la.qr(x, mode='complete')
        p, b = la.qr(y, mode='complete')

        x = r @ p
        y = b @ q

        qprod = qprod @ q
        pprod = pprod @ p

        # -- Stop condition: q should be equal to x normalized to unit vectors.
        if (diff := la.norm(x0 - x) + la.norm(y0 - y)) < eps:
            print(i)
            return np.diag(qprod.T @ a @ pprod), qprod, pprod

        i += 1

    warnings.warn(f'Algorithm did not converge (last diff={diff:.04}). Returning best guess for singular triples.')

    return np.diag(qprod.T @ a @ pprod), qprod, pprod
