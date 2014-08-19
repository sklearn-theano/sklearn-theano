# Authors: Michael Eickenberg
#          Kyle Kastner
# License: BSD 3 Clause

import os
import theano
# Required to avoid fuse errors... very strange
theano.config.floatX = 'float32'
import zipfile
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from ..datasets import get_dataset_dir, download
from ..base import (Convolution, MaxPool, PassThrough,
                    Standardize, fuse)


# better get it from a config file
NETWORK_WEIGHTS_PATH = get_dataset_dir("overfeat_weights")

SMALL_NETWORK_WEIGHT_FILE = 'net_weight_0'
SMALL_NETWORK_FILTER_SHAPES = np.array([(96, 3, 11, 11),
                                        (256, 96, 5, 5),
                                        (512, 256, 3, 3),
                                        (1024, 512, 3, 3),
                                        (1024, 1024, 3, 3),
                                        (3072, 1024, 6, 6),
                                        (4096, 3072, 1, 1),
                                        (1000, 4096, 1, 1)])
SMALL_NETWORK_BIAS_SHAPES = SMALL_NETWORK_FILTER_SHAPES[:, 0]
SMALL_NETWORK = (SMALL_NETWORK_WEIGHT_FILE,
                 SMALL_NETWORK_FILTER_SHAPES,
                 SMALL_NETWORK_BIAS_SHAPES)

LARGE_NETWORK_WEIGHT_FILE = 'net_weight_1'
LARGE_NETWORK_FILTER_SHAPES = np.array([(96, 3, 7, 7),
                                        (256, 96, 7, 7),
                                        (512, 256, 3, 3),
                                        (512, 512, 3, 3),
                                        (1024, 512, 3, 3),
                                        (1024, 1024, 3, 3),
                                        (4096, 1024, 5, 5),
                                        (4096, 4096, 1, 1),
                                        (1000, 4096, 1, 1)])
LARGE_NETWORK_BIAS_SHAPES = LARGE_NETWORK_FILTER_SHAPES[:, 0]
LARGE_NETWORK = (LARGE_NETWORK_WEIGHT_FILE,
                 LARGE_NETWORK_FILTER_SHAPES,
                 LARGE_NETWORK_BIAS_SHAPES)


def fetch_overfeat_weights_and_biases(large_network=False, weights_file=None):
    network = LARGE_NETWORK if large_network else SMALL_NETWORK
    fname, weight_shapes, bias_shapes = network

    if weights_file is None:
        weights_file = os.path.join(NETWORK_WEIGHTS_PATH, fname)
        if not os.path.exists(weights_file):
            url = "https://dl.dropboxusercontent.com/u/15378192/net_weights.zip"
            if not os.path.exists(NETWORK_WEIGHTS_PATH):
                os.makedirs(NETWORK_WEIGHTS_PATH)
            full_path = os.path.join(NETWORK_WEIGHTS_PATH, "net_weights.zip")
            if not os.path.exists(full_path):
                download(url, full_path, progress_update_percentage=1)
            zip_obj = zipfile.ZipFile(full_path, 'r')
            zip_obj.extractall(NETWORK_WEIGHTS_PATH)
            zip_obj.close()

    memmap = np.memmap(weights_file, dtype=np.float32)
    mempointer = 0

    weights = []
    biases = []
    for weight_shape, bias_shape in zip(weight_shapes, bias_shapes):
        filter_size = np.prod(weight_shape)
        weights.append(
            memmap[mempointer:mempointer + filter_size].reshape(weight_shape))
        mempointer += filter_size
        biases.append(memmap[mempointer:mempointer + bias_shape])
        mempointer += bias_shape

    return weights, biases


def _get_architecture(large_network=False, weights_and_biases=None):
    if weights_and_biases is None:
        weights_and_biases = fetch_overfeat_weights_and_biases(large_network)

    weights, biases = weights_and_biases

    # flip weights to make Xcorr
    ws = [w[:, :, ::-1, ::-1] for w in weights]
    bs = biases

    if large_network:
        architecture = [
            Standardize(118.380948, 61.896913),

            Convolution(ws[0], bs[0], subsample=(2, 2),
                        activation='relu'),
            MaxPool((3, 3)),

            Convolution(ws[1], bs[1], activation='relu'),
            MaxPool((2, 2)),

            PassThrough(),  # Torch does spatial padding here
            Convolution(ws[2], bs[2],
                        activation='relu',
                        cropping=[(1, -1), (1, -1)],
                        border_mode='full'),

            PassThrough(),  # Torch does spatial padding
            Convolution(ws[3], bs[3],
                        activation='relu',
                        cropping=[(1, -1), (1, -1)],
                        border_mode='full'),

            PassThrough(),  # Torch does spatial padding
            Convolution(ws[4], bs[4],
                        activation='relu',
                        cropping=[(1, -1), (1, -1)],
                        border_mode='full'),

            PassThrough(),  # Torch does spatial padding
            Convolution(ws[5], bs[5],
                        activation='relu',
                        cropping=[(1, -1), (1, -1)],
                        border_mode='full'),
            MaxPool((3, 3)),

            Convolution(ws[6], bs[6],
                        activation='relu'),

            Convolution(ws[7], bs[7],
                        activation='relu'),

            Convolution(ws[8], bs[8],
                        activation='identity')]
    else:
        architecture = [
            Standardize(118.380948, 61.896913),

            Convolution(ws[0], bs[0], subsample=(4, 4),
                        activation='relu'),
            MaxPool((2, 2)),

            Convolution(ws[1], bs[1], activation='relu'),
            MaxPool((2, 2)),

            PassThrough(),  # Torch does spatial padding here
            Convolution(ws[2], bs[2],
                        activation='relu',
                        cropping=[(1, -1), (1, -1)],
                        border_mode='full'),

            PassThrough(),  # Torch does spatial padding
            Convolution(ws[3], bs[3],
                        activation='relu',
                        cropping=[(1, -1), (1, -1)],
                        border_mode='full'),

            PassThrough(),  # Torch does spatial padding
            Convolution(ws[4], bs[4],
                        activation='relu',
                        cropping=[(1, -1), (1, -1)],
                        border_mode='full'),
            MaxPool((2, 2)),

            Convolution(ws[5], bs[5],
                        activation='relu'),

            Convolution(ws[6], bs[6],
                        activation='relu'),

            Convolution(ws[7], bs[7],
                        activation='identity')]
    return architecture


def _get_fprop(large_network=False, output_layers=[-1]):
    arch = _get_architecture(large_network)
    expressions, input_var = fuse(arch, output_expressions=output_layers,
                                  input_dtype='float32')
    fprop = theano.function([input_var], expressions)
    return fprop


class OverfeatTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, large_network=False, output_layers=[-1],
                 transpose_order=(0, 3, 1, 2)):
        self.large_network = large_network
        self.output_layers = output_layers
        self.transpose_order = transpose_order
        self.transform_function = _get_fprop(self.large_network, output_layers)

    def fit(self, X, y=None):
        """Passthrough."""
        return self

    def transform(self, X):
        if len(self.output_layers) == 1:
            return self.transform_function(X.transpose(
                *self.transpose_order))[0]
        else:
            return self.transform_function(X.transpose(*self.transpose_order))
