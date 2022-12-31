import numpy as np

from clearbox.tools import logsumexp, kl_categorical

from warnings import warn

class EM():

    def __init__(self, k, diagonal=False):
        """

        :param k: Number of components
        """

        self.k = k
        self.diagonal = diagonal

        # Parameters of the components
        self.means = None # (k, d)
        self.covariances = None # (k, d, d)
        self.weights = None # (k,)

        # Responsibilities each component claims for each instance (n, k)
        self.responsibilities = None

    def assign_responsibilities(self, data):
        """
        Assign responsibilities to each component based for each instance, based on the current best guess for the
        component parameters.

        :return:
        """
        n, d = data.shape
        k = self.k

        if self.means is None:
            self.reset_components(data)

        if self.responsibilities is not None:
            old = self.responsibilities.copy()
        else:
            old = np.full(fill_value=1/k, shape=(n, k))

        # We start by computing the log probability density of each instance in the data under each component. We'll
        # end up with a matrix of (n, k) containing these.

        # - Invert the covariance matrices
        invs = np.linalg.inv(self.covariances)

        # - Compute the difference between the component mean and the instances over all dimensions
        diffs = data[:, None, :] - self.means[None, :, :]
        # -- Note the use of broadcasting: we add a component dimension to the data, and an instance dimension to the
        #    means, and they are automatically repeated along those dimensions to make the tensors match.

        # - Next, we need to compute d^T S^-1 d: the dot product of the difference vectors with the inverse covariance
        #   in the middle. We'll use einsum notation for this.
        #   Note that this is a dot product between two vectors, so, the dimensions will be matched up and summed out
        #   and we will end up with a matrix of (n, k) values to put into the exponent of the density function.
        exponent = np.einsum('kij, nkj -> nki', invs, diffs) # -- matrix/vector multiplication
        exponent = np.einsum('nki, nki -> nk', diffs, exponent) # -- dot product
        exponent *= - 0.5

        # - The second part of the log probability density is the normalization factor. We remove the sqrt(2 pi) because
        #   they will disappear after normalization anyway, and focus only on the factors depending on the parameters
        norm = (np.log(np.linalg.det(self.covariances)) * - 0.5)
        # -- This gives us one log-normalization factor per component

        logdensities = norm[None, :] + exponent
        assert logdensities.shape == (n, k)

        # Next, we need to normalize the densities, so that they sum to 1 over the components, which requires the
        # logarithm of the sum of the log probabilities.
        #
        # Naively, this would involve exponentiating, summing and then taking the logarithm again to sum the values
        # inside the logarithm. For numerical stability, we use the log-sum-exp trick instead.
        logsum = logsumexp(logdensities, axis=1, keepdims=True)

        self.responsibilities = np.exp(logdensities - logsum)
        # -- Note that dividing by the sum is subtracting when we take it out of the logarithm.
        # -- We don't keep the responsibilities in log space, as they sum to one, so there is little chance of numeric
        #    instability from this point.

        return kl_categorical(self.responsibilities, old)

    def fit_components(self, data):
        """
        Fit the components to the data, based on the current responsibilities.

        :param data:
        :return:
        """
        n, d = data.shape
        k = self.k

        if self.responsibilities is None:
            self.reset_responsibilities(data)

        # First, we compute the mean for each component. This is just a weighted sum, normalized by the total
        # responsibility for each component.

        # Duplicate the data for each component and multiply it by the responsibilities
        weighted = data[:, None, :] * self.responsibilities[:, :, None]
        assert weighted.shape == (n, k, d)

        resp_sums = self.responsibilities.sum(axis=0) # shape (k,)
        assert resp_sums.shape == (k,)
        self.means = weighted.sum(axis=0) / resp_sums[:, None]

        # For the covariance matrices we compute the outer products of the mean centered data, weighted by
        # responsibility and summed.
        centered = data[:, None, :] - self.means[None, :, :]
        assert centered.shape == (n, k, d)

        outer = np.einsum('nki, nkj -> nkij', centered, centered)
        covs = (outer * self.responsibilities[:, :, None, None]).sum(axis=0) # -- weight and sum
        assert covs.shape == (k, d, d), f'{covs.shape=} not {k, d, d}'
        self.covariances = covs / resp_sums[:, None, None] # -- normalize by total responsibility claimed

        # Weights are set proportional to the responsibilities claimed
        self.weights = resp_sums / resp_sums.sum(axis=0, keepdims=True)
        assert self.weights.shape == (k, )

        # TODO return the KL divergence averaged over the components (the actual KL divergence between two GMMs has no
        # analytical expression).

    def iterate(self, data):
        """
        Compute one iteration of the EM search algorithm on the given data.

        :param data:
        :return:
        """

        kl_resp = self.assign_responsibilities(data)
        self.fit_components(data)

        return kl_resp

    def fit(self, data, max_its=10_000, threshold=1e-10, verbose=False):

        kl = float('inf')
        it = 0

        while it < max_its and kl > threshold:

            kl = self.iterate(data)

            kl = kl.mean()
            it += 1

            if verbose and it % verbose == 0:
                print(f'{it} iterations: {kl=:.04}')
            if it == max_its:
                warn('Max iterations reached')


        # TODO check for singular covariance matrices

    def reset_responsibilities(self, data):
        """
        Assigns the clusters to the points. Intializes as equal responsibilities.

        :param data:
        :return:
        """
        n, d = data.shape

        self.responsibilities = np.full(fill_value=1/self.k, shape=(n, self.k))

    def reset_components(self, data):
        """
        Resets the component values to random values. The means ar chosen uniformly over the range of the data. The
        covariance matrices are set to the identity matrix.

        :param data:
        :return:
        """

        n, d = data.shape
        k = self.k

        self.means = np.random.uniform(0, 1, (self.k, d))

        mi, ma = np.min(data, axis=0), np.max(data, axis=0)
        rng = ma - mi

        self.means *= rng[None, :]
        self.means += mi[None, :]

        # Create a stack of identity matrices.
        self.covariances = np.eye(d)[None, :, :].repeat(k, axis=0)

        # Equal weights over the components
        self.weights = np.full(fill_value=1/k, shape=(k, ))