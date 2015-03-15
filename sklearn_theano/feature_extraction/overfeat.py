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
from .overfeat_class_labels import get_all_overfeat_leaves
from ..datasets import get_dataset_dir, download
from ..base import (Convolution, MaxPool, PassThrough,
                    Standardize, ZeroPad, Relu, fuse)
from ..utils import check_tensor


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


def _get_architecture(large_network=False, weights_and_biases=None,
                      detailed=False):
    if weights_and_biases is None:
        weights_and_biases = fetch_overfeat_weights_and_biases(large_network)

    weights, biases = weights_and_biases

    # flip weights to make Xcorr
    ws = [w[:, :, ::-1, ::-1] for w in weights]
    bs = biases

    if large_network and not detailed:
        architecture = [
            Standardize(118.380948, 61.896913),

            Convolution(ws[0], bs[0], subsample=(2, 2),
                        activation='relu'),
            MaxPool((3, 3)),

            Convolution(ws[1], bs[1], activation='relu'),
            MaxPool((2, 2)),

            Convolution(ws[2], bs[2],
                        activation='relu',
                        cropping=[(1, -1), (1, -1)],
                        border_mode='full'),

            Convolution(ws[3], bs[3],
                        activation='relu',
                        cropping=[(1, -1), (1, -1)],
                        border_mode='full'),

            Convolution(ws[4], bs[4],
                        activation='relu',
                        cropping=[(1, -1), (1, -1)],
                        border_mode='full'),

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
    elif not large_network and not detailed:
        architecture = [
            Standardize(118.380948, 61.896913),

            Convolution(ws[0], bs[0], subsample=(4, 4),
                        activation='relu'),
            MaxPool((2, 2)),

            Convolution(ws[1], bs[1], activation='relu'),
            MaxPool((2, 2)),

            Convolution(ws[2], bs[2],
                        activation='relu',
                        cropping=[(1, -1), (1, -1)],
                        border_mode='full'),

            Convolution(ws[3], bs[3],
                        activation='relu',
                        cropping=[(1, -1), (1, -1)],
                        border_mode='full'),

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
    elif large_network and detailed:
        architecture = [
            Standardize(118.380948, 61.896913),

            Convolution(ws[0], bs[0], subsample=(2, 2),
                        activation=None),
            Relu(),
            MaxPool((3, 3)),

            Convolution(ws[1], bs[1], activation=None),
            Relu(),
            MaxPool((2, 2)),

            ZeroPad(1),
            Convolution(ws[2], bs[2], activation=None),
            Relu(),

            ZeroPad(1),
            Convolution(ws[3], bs[3], activation=None),
            Relu(),

            ZeroPad(1),
            Convolution(ws[4], bs[4], activation=None),
            Relu(),

            ZeroPad(1),
            Convolution(ws[5], bs[5], activation=None),
            Relu(),
            MaxPool((3, 3)),

            Convolution(ws[6], bs[6], activation=None),
            Relu(),

            Convolution(ws[7], bs[7], activation=None),
            Relu(),

            Convolution(ws[8], bs[8], activation=None)
            ]
    elif not large_network and detailed:
        architecture = [
            Standardize(118.380948, 61.896913),

            Convolution(ws[0], bs[0], subsample=(4, 4), activation=None),
            Relu(),
            MaxPool((2, 2)),

            Convolution(ws[1], bs[1], activation=None),
            Relu(),
            MaxPool((2, 2)),

            ZeroPad(1),
            Convolution(ws[2], bs[2], activation=None),
            Relu(),

            ZeroPad(1),
            Convolution(ws[3], bs[3], activation=None),
            Relu(),

            ZeroPad(1),
            Convolution(ws[4], bs[4], activation=None),
            Relu(),

            MaxPool((2, 2)),

            Convolution(ws[5], bs[5], activation=None),
            Relu(),

            Convolution(ws[6], bs[6], activation=None),
            Relu(),

            Convolution(ws[7], bs[7], activation=None)
            ]

    return architecture


def _get_fprop(large_network=False, output_layers=[-1], detailed=False):
    arch = _get_architecture(large_network, detailed=detailed)
    expressions, input_var = fuse(arch, output_expressions=output_layers,
                                  input_dtype='float32')
    fprop = theano.function([input_var], expressions)
    return fprop


class OverfeatTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer/feature extractor for images using the OverFeat neural network.

    Parameters
    ----------
    large_network : boolean, optional (default=False)
        Which network to use. If True, the transform will operate over X in
        windows of 221x221 pixels. Otherwise, these windows will be 231x231.

    output_layers : iterable, optional (default=[-1])
        Which layers to return. Can be used to retrieve multiple levels of
        output with a single call to transform.

    force_reshape : boolean, optional (default=True)
        Whether or not to force the output to be two dimensional. If true,
        this class can be used as part of a scikit-learn pipeline.
        force_reshape currently only supports len(output_layers) == 1!

    detailed_network : boolean, optional (default=True)
        If set to True, layers will be indexed and counted as in the binary
        version provided by the authors of OverFeat. I.e. convolution, relu,
        zero-padding, max-pooling are all separate layers. If False specified
        then convolution and relu are one unit and zero-padding layers are
        omitted.

    batch_size : int, optional (default=None)
        If set, input will be transformed in batches of size batch_size. This
        can save memory at intermediate processing steps.
    """
    def __init__(self, large_network=False, output_layers=[-1],
                 force_reshape=True,
                 transpose_order=(0, 3, 1, 2),
                 detailed_network=False,
                 batch_size=None):
        self.large_network = large_network
        self.output_layers = output_layers
        self.force_reshape = force_reshape
        self.transpose_order = transpose_order
        self.transform_function = _get_fprop(self.large_network,
                                             output_layers,
                                             detailed=detailed_network)
        self.batch_size = batch_size

    def fit(self, X, y=None):
        """Passthrough for scikit-learn pipeline compatibility."""
        return self

    def transform(self, X):
        """
        Transform a set of images.

        Returns the features from each layer.

        Parameters
        ----------
        X : array-like, shape = [n_images, height, width, color]
                        or
                        shape = [height, width, color]

        Returns
        -------
        T : array-like, shape = [n_images, n_features]

            If force_reshape = False,
            list of array-like, length output_layers,
                                each shape = [n_images, n_windows,
                                              n_window_features]

            Returns the features extracted for each of the n_images in X..
        """
        X = check_tensor(X, dtype=np.float32, n_dim=4)
        if self.batch_size is None:
            if self.force_reshape:
                return self.transform_function(X.transpose(
                        *self.transpose_order))[0].reshape((len(X), -1))
            else:
                return self.transform_function(
                    X.transpose(*self.transpose_order))
        else:
            XT = X.transpose(*self.transpose_order)
            n_samples = XT.shape[0]
            for i in range(0, n_samples, self.batch_size):
                transformed_batch = self.transform_function(
                    XT[i:i + self.batch_size])
                # at first iteration, initialize output arrays to correct size
                if i == 0:
                    shapes = [(n_samples,) + t.shape[1:] for t in
                              transformed_batch]
                    ravelled_shapes = [np.prod(shp[1:]) for shp in shapes]
                    if self.force_reshape:
                        output_width = np.sum(ravelled_shapes)
                        output = np.empty((n_samples, output_width),
                                          dtype=transformed_batch[0].dtype)
                        break_points = np.r_([0], np.cumsum(ravelled_shapes))
                        raw_output = [
                            output[:, start:stop] for start, stop in
                            zip(break_points[:-1], break_points[1:])]
                    else:
                        output = [np.empty(shape,
                                           dtype=transformed_batch.dtype)
                                  for shape in shapes]
                        raw_output = [arr.reshape(n_samples, -1)
                                      for arr in output]

                for transformed, out in zip(transformed_batch, raw_output):
                    out[i:i + batch_size] = transformed
        return output


class OverfeatClassifier(BaseEstimator):
    """
    A classifier for cropped images using the OverFeat neural network.

    If large_network=True, this X will be cropped to the center
    221x221 pixels. Otherwise, this cropped box will be 231x231.

    Parameters
    ----------
    large_network : boolean, optional (default=False)
        Which network to use. If large_network = True, input will be cropped
        to the center 221 x 221 pixels. Otherwise, input will be cropped to the
        center 231 x 231 pixels.

    top_n : integer, optional (default=5)
        How many classes to return, based on sorted class probabilities.

    output_strings : boolean, optional (default=True)
        Whether to return class strings or integer classes. Returns class
        strings by default.

    Attributes
    ----------
    crop_bounds_ : tuple, (x_left, x_right, y_lower, y_upper)
        The coordinate boundaries of the cropping box used.

    """
    def __init__(self, top_n=5, large_network=False, output_strings=True,
                 transpose_order=(0, 3, 1, 2)):

        self.top_n = top_n
        self.large_network = large_network
        if self.large_network:
            self.min_size = (221, 221)
        else:
            self.min_size = (231, 231)
        self.output_strings = output_strings
        self.transpose_order = transpose_order
        self.transform_function = _get_fprop(self.large_network, [-1])

    def fit(self, X, y=None):
        """Passthrough for scikit-learn pipeline compatibility."""
        return self

    def _predict_proba(self, X):
        x_midpoint = X.shape[2] // 2
        y_midpoint = X.shape[1] // 2

        x_lower_bound = x_midpoint - self.min_size[0] // 2
        if x_lower_bound <= 0:
            x_lower_bound = 0
        x_upper_bound = x_lower_bound + self.min_size[0]
        y_lower_bound = y_midpoint - self.min_size[1] // 2
        if y_lower_bound <= 0:
            y_lower_bound = 0
        y_upper_bound = y_lower_bound + self.min_size[1]
        self.crop_bounds_ = (x_lower_bound, x_upper_bound, y_lower_bound,
                             y_upper_bound)

        res = self.transform_function(
            X[:, y_lower_bound:y_upper_bound,
                x_lower_bound:x_upper_bound, :].transpose(
                    *self.transpose_order))[0]
        # Softmax activation
        exp_res = np.exp(res - res.max(axis=1))
        exp_res /= np.sum(exp_res, axis=1)
        return exp_res

    def predict(self, X):
        """
        Classify a set of cropped input images.

        Returns the top_n classes.

        Parameters
        ----------
        X : array-like, shape = [n_images, height, width, color]
                        or
                        shape = [height, width, color]

        Returns
        -------
        T : array-like, shape = [n_images, top_n]

            Returns the top_n classes for each of the n_images in X.
            If output_strings is True, then the result will be string
            description of the class label.

            Otherwise, the returned values will be the integer class label.
        """
        X = check_tensor(X, dtype=np.float32, n_dim=4)
        res = self._predict_proba(X)[:, :, 0, 0]
        indices = np.argsort(res, axis=1)
        indices = indices[:, -self.top_n:]
        if self.output_strings:
            class_strings = np.empty_like(indices,
                                          dtype=object)
            for index, value in enumerate(indices.flat):
                class_strings.flat[index] = get_overfeat_class_label(value)
            return class_strings
        else:
            return indices

    def predict_proba(self, X):
        """
        Prediction probability for a set of cropped input images.

        Returns the top_n probabilities.

        Parameters
        ----------
        X : array-like, shape = [n_images, height, width, color]
                        or
                        shape = [height, width, color]

        Returns
        -------
        T : array-like, shape = [n_images, top_n]

            Returns the top_n probabilities for each of the n_images in X.
        """
        X = check_tensor(X, dtype=np.float32, n_dim=4)
        res = self._predict_proba(X)[:, :, 0, 0]
        return np.sort(res, axis=1)[:, -self.top_n:]


class OverfeatLocalizer(BaseEstimator):
    """
    A localizer for single images using the OverFeat neural network.

    If large_network=True, this X will be cropped to the center
    221x221 pixels. Otherwise, this box will be 231x231.

    Parameters
    ----------
    match_strings : iterable of strings
        An iterable of class names to match with localizer. Can be a full
        ImageNet class string or a WordNet leaf such as 'dog.n.01'. If the
        pattern '.n.' is found in the match string, it will be treated as a
        WordNet leaf, otherwise the string is assumed to be a class label.

    large_network : boolean, optional (default=False)
        Which network to use. If True, the transform will operate over X in
        windows of 221x221 pixels. Otherwise, these windows will be 231x231.

    top_n : integer, optional (default=5)
        How many classes to return, based on sorted class probabilities.

    output_strings : boolean, optional (default=True)
        Whether to return class strings or integer classes. Returns class
        strings by default.
    """
    def __init__(self, match_strings, top_n=5, large_network=False,
                 transpose_order=(2, 0, 1)):
        self.top_n = top_n
        self.large_network = large_network
        if self.large_network:
            self.min_size = (221, 221)
        else:
            self.min_size = (231, 231)
        self.match_strings = match_strings
        self.transpose_order = transpose_order
        self.transform_function = _get_fprop(self.large_network, [-1])

    def fit(self, X, y=None):
        """Passthrough for scikit-learn pipeline compatibility."""
        return self

    def predict(self, X):
        """
        Localize an input image.

        Returns the points where the top_n classes contains any of the
        match_strings.

        Parameters
        ----------
        X : array-like, shape = [height, width, color]

        Returns
        -------
        T : list of array-likes, each of shape = [n_points, 2]

            For each string in match_strings, points where that string was
            in the top_n classes. len(T) will be equal to len(match_strings).

            Each array in T is of size n_points x 2, where column 0 is
            x point coordinate and column 1 is y point coordinate.

            This means that an entry in T can be plotted with
            plt.scatter(T[i][:, 0], T[i][:, 1])
        """
        X = check_tensor(X, dtype=np.float32, n_dim=3)
        res = self.transform_function(X.transpose(
            *self.transpose_order)[None])[0]
        # Softmax activation
        exp_res = np.exp(res - res.max(axis=1))
        exp_res /= np.sum(exp_res, axis=1)
        indices = np.argsort(exp_res, axis=1)[:, -self.top_n:, :, :]

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
        overfeat_leaves = get_all_overfeat_leaves()
        for match_string in self.match_strings:
            if '.n.' in match_string:
                # We were provided a wordnet category and must conglomerate
                # points
                all_match_labels = overfeat_leaves[match_string]
                overfeat_labels = get_all_overfeat_labels()
                match_indices = np.array(([overfeat_labels.index(s)
                                           for s in all_match_labels]))
                match_indices = np.unique(match_indices)
                matches = np.where(
                    np.in1d(per_window_labels, match_indices).reshape(
                        per_window_labels.shape) == True)[1]
                all_matches.append(np.vstack((xx.flat[matches],
                                              yy.flat[matches])).T)
            else:
                # Asssume this is an OverFeat class
                match_index = get_all_overfeat_labels().index(match_string)
                matches = np.where(per_window_labels == match_index)[1]
                all_matches.append(np.vstack((xx.flat[matches],
                                              yy.flat[matches])).T)
        return all_matches
