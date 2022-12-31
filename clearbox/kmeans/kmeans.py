import numpy as np

from warnings import warn

class KMeans():

    def __init__(self, k):
        self.k = k

        self.means = None
        self.clusters = None

    def fit_means(self, data):
        """
        Fit the means to the current cluster assignments.
        """
        n, d = data.shape

        if self.clusters is None:
            self.reset_clusters(data)

        if self.means is not None:
            oldmeans = self.means.copy()
        else:
            oldmeans = np.zeros_like(self.means)

        self.means = []
        for c in range(self.k):
            self.means.append(data[self.clusters == c].mean(axis=0, keepdims=True))

        self.means = np.concatenate(self.means, axis=0)

        # return the average root-mean-square distance between the old and new means
        return (((self.means - oldmeans) ** 2.0).sum() ** 0.5) / self.k

    def assign_clusters(self, data):
        """
        Assign the clusters based on the current means
        """
        if self.means is None:
            self._init_means(data)

        n, d = data.shape
        k = self.k

        if self.clusters is not None:
            oldclusters = self.clusters.copy()
            assert self.clusters.shape[0] == n
        else:
            oldclusters = np.full(fill_value=-1, shape=(1, ))

        # Compute the distance from each point to each mean along each axis
        distances = data[:, :, None] - self.means.transpose()[None, :, :]
        # -- Note the broadcasting: the data is replicated once for each mean, and the means are replicated once for
        #    each instance in the data.

        assert distances.shape == (n, d, k)
        # -- This tensor contains for each: instance, dimension and mean the difference between the value of the instance
        #    at that feature and the mean at that feature. Squaring and summing these values will give us the distances.

        # Square the differences and sum out over the dimensions.
        distances = (distances ** 2).sum(axis=1)
        assert distances.shape == (n, k)
        # -- We now have the a distance to each mean for every instance in our data.

        # For each instance, we assign the cluster that it is closest to.
        self.clusters = distances.argmin(axis=1)

        # Return a boolean vector indicating which points have changed cluster
        return (oldclusters != self.clusters)

    def iterate(self, data):
        """
        Perform one iteration of the kmeans algorithm
        :return: The number of points whose cluster assignment changed,
        """

        changed = self.assign_clusters(data)
        dist = self.fit_means(data)

        return changed.sum(), dist

    def fit(self, data, max_its=1e5):
        """
        Clusters the data using the k-means algorithm. Iterates until a stable state is reached.

        :param max_its:
        :return:
        """
        nchanged = 1
        its = 0

        while nchanged > 0 and its < max_its:
            nchanged, dist = self.iterate(data)
            its += 1

            if its >= max_its:
                warn('Maximum iterations reached without convergence.')

    def reset_clusters(self, data):
        """
        Assigns the clusters to the points randomly.

        :param data:
        :return:
        """
        n, d = data.shape

        self.clusters = np.random.randint(low=0, high=self.k, size=(n,))

    def reset_means(self, data):
        """
        Sets the means to random values. These are chosen uniformly over the range of the data.

        :param data:
        :return:
        """

        n, d = data.shape

        self.means = np.random.uniform(0, 1, (self.k, d))

        mi, ma = np.min(data, axis=0), np.max(data, axis=0)
        rng = ma - mi

        self.means *= rng[None, :]
        self.means += mi[None, :]




