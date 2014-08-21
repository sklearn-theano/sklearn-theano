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

from .overfeat_class_labels import get_overfeat_class_label
from .overfeat_class_labels import get_all_overfeat_labels
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
                 force_reshape=True,
                 transpose_order=(0, 3, 1, 2)):
        self.large_network = large_network
        self.output_layers = output_layers
        self.force_reshape = force_reshape
        self.transpose_order = transpose_order
        self.transform_function = _get_fprop(self.large_network, output_layers)

    def fit(self, X, y=None):
        """Passthrough for scikit-learn pipeline compatibility."""
        return self

    def transform(self, X):
        if self.force_reshape:
            return self.transform_function(X.transpose(
                *self.transpose_order))[0].reshape((len(X), -1))
        else:
            return self.transform_function(X.transpose(*self.transpose_order))


class OverfeatClassifier(BaseEstimator):
    def __init__(self, top_n=5, large_network=False, output_strings=True,
                 transpose_order=(0, 3, 1, 2)):
        self.top_n = top_n
        self.large_network = large_network
        self.output_strings = output_strings
        self.transpose_order = transpose_order
        self.transform_function = _get_fprop(self.large_network, [-1])

    def fit(self, X, y=None):
        """Passthrough for scikit-learn pipeline compatibility."""
        return self

    def predict(self, X):
        res = self.transform_function(X.transpose(*self.transpose_order))[0]
        # Softmax activation
        res = 1. / (1 + np.exp(res))
        indices = np.argsort(res, axis=1)[:, :self.top_n, :, :]
        if self.output_strings:
            class_strings = np.empty_like(indices,
                                          dtype=object)
            for index, value in enumerate(indices.flat):
                class_strings.flat[index] = get_overfeat_class_label(value)
            return class_strings
        else:
            return indices


class OverfeatLocalizer(BaseEstimator):
    def __init__(self, match_strings, top_n=5, large_network=False,
                 transpose_order=(2, 0, 1)):
        self.top_n = top_n
        self.large_network = large_network
        if self.large_network:
            self.min_size = (227, 227)
        else:
            self.min_size = (231, 231)
        self.match_strings = match_strings
        self.transpose_order = transpose_order
        self.transform_function = _get_fprop(self.large_network, [-1])

    def fit(self, X, y=None):
        """Passthrough for scikit-learn pipeline compatibility."""
        return self

    def predict(self, X):
        if len(X.shape) != 3:
            raise ValueError("X must be a 3 dimensional array of "
                             "(width, height, color).")
        res = self.transform_function(X.transpose(
            *self.transpose_order)[None])[0]
        # Softmax activation
        res = 1. / (1 + np.exp(res))
        indices = np.argsort(res, axis=1)[:, :self.top_n, :, :]
        height = X.shape[0]
        width = X.shape[1]
        x_bound = width - self.min_size[0]
        y_bound = height - self.min_size[1]
        n_y = indices.shape[2]
        n_x = indices.shape[3]
        x_points = np.linspace(0,  x_bound, n_x).astype('int32')
        y_points = np.linspace(0,  y_bound, n_y).astype('int32')
        x_points = x_points + self.min_size[0] // 2
        y_points = y_points + self.min_size[1] // 2
        xx, yy = np.meshgrid(x_points, y_points)
        per_window_labels = indices[0]
        per_window_labels = per_window_labels.reshape(len(per_window_labels),
                                                      -1)
        all_matches = []
        for match_string in self.match_strings:
            match_index = get_all_overfeat_labels().index(match_string)
            matches = np.where(per_window_labels == match_index)[1]
            all_matches.append(np.vstack((xx.flat[matches],
                                          yy.flat[matches])).T)
        return all_matches
