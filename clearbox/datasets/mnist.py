# Numpy-only MNIST loader. Courtesy of Hyeonseok Jung
# https://github.com/hsjeong5/MNIST-for-Numpy

import numpy as np

from urllib import request
import gzip
import pickle
import os

filename = [
["training_images","train-images-idx3-ubyte.gz"],
["test_images","t10k-images-idx3-ubyte.gz"],
["training_labels","train-labels-idx1-ubyte.gz"],
["test_labels","t10k-labels-idx1-ubyte.gz"]
]

def download_mnist():

    base_url = "http://yann.lecun.com/exdb/mnist/"

    for name in filename:
        print("Downloading "+name[1]+"...")
        request.urlretrieve(base_url+name[1], name[1])

    print("Download complete.")

def save_mnist(file):

    mnist = {}

    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)

    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)

    with open(file, 'wb') as f:
        pickle.dump(mnist,f)

    print("Save complete.")

def init(file):
    download_mnist()
    save_mnist(file)

def mnist(file='mnist.pkl', final=False, nvalid=10_000):
    """
    Returns the MNIST data in its standard test/train split

    :param file:
    :param final: Whether to return the final test data. If false, splits off a section of the training data to validate
      on instead
    :param nvalid: If final=False, how much of the training data to withhold for validation
    :return:
    """
    if not os.path.isfile(file):
        init(file)

    with open(file,'rb') as f:
        mnist = pickle.load(f)

    x_train = mnist["training_images"]
    y_train = mnist["training_labels"]
    x_test  = mnist["test_images"]
    y_test  = mnist["test_labels"]

    if final:
        return (x_train, y_train), (x_test, y_test)

    return (x_train[:-nvalid], y_train[:-nvalid]), (x_train[-nvalid:], y_train[-nvalid:])