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

GENERATOR_WEIGHTS_PATH = get_dataset_dir("adversarial_weights")

MNIST_NETWORK_WEIGHT_FILE = 'mnist/generator_weights.pkl'
MNIST_NETWORK_FILTER_SHAPES = np.array([(96, 3, 11, 11),
                                        (256, 96, 5, 5),
                                        (512, 256, 3, 3),
                                        (1024, 512, 3, 3),
                                        (1024, 1024, 3, 3),
                                        (3072, 1024, 6, 6),
                                        (4096, 3072, 1, 1),
                                        (1000, 4096, 1, 1)])
MNIST_NETWORK_BIAS_SHAPES = MNIST_NETWORK_FILTER_SHAPES[:, 0]
MNIST_NETWORK = (MNIST_NETWORK_WEIGHT_FILE,
                 MNIST_NETWORK_FILTER_SHAPES,
                 MNIST_NETWORK_BIAS_SHAPES)


def _get_generator_weights():
    url = "https://dl.dropboxusercontent.com/u/15378192/adversarial_generator_weights.zip"
    if not os.path.exists(GENERATOR_WEIGHTS_PATH):
        os.makedirs(GENERATOR_WEIGHTS_PATH)
    full_path = os.path.join(GENERATOR_WEIGHTS_PATH,
                             "adversarial_generator_weights.zip")
    if not os.path.exists(full_path):
        download(url, full_path, progress_update_percentage=1)
        zip_obj = zipfile.ZipFile(full_path, 'r')
        zip_obj.extractall(GENERATOR_WEIGHTS_PATH)
        zip_obj.close()


def fetch_mnist_weights(weights_file=None):
    fname, weight_shapes, bias_shapes = MNIST_NETWORK

    if weights_file is None:
        weights_file = os.path.join(GENERATOR_WEIGHTS_PATH, fname)
        if not os.path.exists(weights_file):
            _get_generator_weights()

    weights = load(weights_file)
    return weights


def _get_mnist_architecture(weights=None):
    if weights is None:
        weights = fetch_mnist_weights()

    architecture = [Feedforward(weights[0], activation='relu'),
                    Feedforward(weights[1], activation='relu'),
                    Feedforward(weights[2], activation='sigmoid')]
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
    X = 2 * np.sqrt(3) * rng.randn(n_samples, 100).astype('float32') - np.sqrt(3)
    return generator_func(X)[0]
