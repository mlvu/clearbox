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

    TODO: Quantify behavior for non-square matrices.

    References:
     - https://peterbloem.nl/blog/pca-5

    :param a: A square matrix.
    :param num: The number of singular vectors and -values to return. The n eigenpairs with the largest eigenvalues by
    magnitude are returned.
    :param maxit: Max number of iterations. Default `inf` (i.e. no maximum).
    :param eps: Epsilon for the stop condition. If the norm of the candidate eigenvectors changes less than this between
    iterations, the algorithm considers the itertion to be converged and returns.
    :param check_sym: Asserts that the matrix is symmetric (up to small differences).
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
            # -- `q` contains unit eigenvectors, so the eigenvalues are the norms of the vectors after multiplication
            #     by `a`.

        i += 1

    warnings.warn(f'Algorithm did not converge (last diff={diff}). Returning best guess for eigenpairs.')

    return np.diag(r), q, p


def qr(a : nt.ArrayLike, maxit=float('inf'), eps=1e-8, check_sym=True):
    """
    Computes the first n eigenvectors and the corresponding eigenvalues of the square, symmetric matrix `a` using the QR
    iteration algorithm.

    TODO: Quantify behavior for non-square matrices.

    References:
     - https://peterbloem.nl/blog/pca-5

    :param a: A square matrix.
    :param maxit: Max number of iterations. Default `inf` (i.e. no maximum).
    :param eps: Epsilon for the stop condition. If the norm of the candidate eigenvectors changes less than this between
    iterations, the algorithm considers the itertion to be converged and returns.
    :param check_sym: Asserts that the matrix is symmetric (up to small differences).
    :return: A pair (values, vectors) of computed eigenvectors and their eigenvalues.
    """

    assert len(a.shape) == 2 and a.shape[0] == a.shape[1], f'Input a should be a square matrix. Size was {a.shape}.'

    if check_sym:
        assert np.allclose(a, a.T, rtol=1e-5, atol=1e-8)

    x = a
    i = 0

    while i < maxit:
        x0 = x

        q, r = la.qr(x)
        x = r @ q

        # xnorm = x / la.norm(x, axis=0, keepdims=True)
        # -- Stop condition: q should be equal to x normalized to unit vectors.
        if (diff := la.norm(x0 - x)) < eps:
            print(i)
            return np.diag(x), q
            # -- `q` contains unit eigenvectors, so the eigenvalues are the norms of the vectors after multiplication
            #     by `a`.

        # print(x)

        i += 1

    warnings.warn(f'Algorithm did not converge (last diff={diff:.04}). Returning best guess for eigenpairs.')

    return np.diag(x), q