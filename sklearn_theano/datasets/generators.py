# Authors: Michael Eickenberg
#          Kyle Kastner
# License: BSD 3 Clause

import os
import zipfile
import theano
import numpy as np
from sklearn.externals.joblib import load
from sklearn.utils import check_random_state

from ..base import (Feedforward, fuse)
from ..datasets import get_dataset_dir, download

GENERATORS_PATH = get_dataset_dir("adversarial_weights")

MNIST_NETWORK_FILE = os.path.join('mnist', 'generator_params.jb')


def _get_generator_weights():
    url = "https://dl.dropboxusercontent.com/u/15378192/adversarial_generator_weights.zip"
    if not os.path.exists(GENERATORS_PATH):
        os.makedirs(GENERATORS_PATH)
    full_path = os.path.join(GENERATORS_PATH,
                             "adversarial_generator_weights.zip")
    if not os.path.exists(full_path):
        download(url, full_path, progress_update_percentage=1)
        zip_obj = zipfile.ZipFile(full_path, 'r')
        zip_obj.extractall(GENERATORS_PATH)
        zip_obj.close()


def fetch_mnist_weights():
    fname = MNIST_NETWORK_FILE

    full_path = os.path.join(GENERATORS_PATH, fname)
    if not os.path.exists(full_path):
        _get_generator_weights()

    all_params = load(full_path)
    weights = all_params[::2]
    biases = all_params[1::2]
    return weights, biases


def _get_mnist_architecture():
    weights, biases = fetch_mnist_weights()

    architecture = [Feedforward(weights[0], biases[0], activation='relu'),
                    Feedforward(weights[1], biases[1], activation='relu'),
                    Feedforward(weights[2], biases[2], activation='sigmoid')]
    return architecture


def _get_mnist_fprop(output_layers=[-1]):
    arch = _get_mnist_architecture()
    expressions, input_var = fuse(arch, fuse_dim=2,
                                  output_expressions=output_layers)
    fprop = theano.function([input_var], expressions)
    return fprop


def fetch_mnist_generated(n_samples=1000, random_state=None):
    rng = check_random_state(random_state)
    generator_func = _get_mnist_fprop()
    X = 2 * np.sqrt(3) * rng.rand(n_samples, 100).astype('float32') - np.sqrt(3)
    return generator_func(X)[0]
