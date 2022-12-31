import unittest

import clearbox as cb
import numpy as np

from clearbox import svm

import torch

from sklearn import SVC as svc

class TestSVM(unittest.TestCase):

    # def test_svmfit1(self):
    #
    #     # Create some simple example data
    #     features = np.asarray([
    #         [2.45, 1.31],
    #         [5.16, 1.31],
    #         [2.45, 3.35],
    #         [5.88, 3.72],
    #         [8.99, 3.35],
    #         [5.16, 5.73]
    #     ])
    #
    #     labels = np.asarray([0, 0, 0, 1, 1, 1])
    #
    #     # Fit the simplest, non-kernel SVM, with basic gradient descent
    #     model = svm.SVM()
    #     model.fit(features, labels, verbose=True, lr=3e-7, max_its=1_000_000, print_every=10_000)
    #
    #
    #     print('done')

    def test_svmfit2(self):

        # Create some simple example data
        features = np.asarray([
            [2.45, 1.31],
            [5.16, 1.31],
            [2.45, 3.35],
            [5.88, 3.72],
            [8.99, 3.35],
            [5.16, 5.73]
        ])

        labels = np.asarray([0, 0, 0, 1, 1, 1])

        # Fit the simplest, non-kernel SVM, with projected gradient descent
        model = svm.SVM(search=cb.svm.projected_gd_search)
        model.fit(features, labels, lr=0.001, verbose=True, max_its=100, print_every=10)

        w, b = model.compute_primal_params(features, labels)

        print(w, b)
        print('done')
