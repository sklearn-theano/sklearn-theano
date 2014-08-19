from numpy.testing import assert_array_almost_equal


import numpy as np
import theano
from scipy.signal import convolve2d
from ..base import Convolution, PassThrough, MaxPool, fuse


def test_convolution():
    convolution_filter = np.zeros([3, 3]).astype(np.float32)
    convolution_filter[2, 2] = 1.

    images = np.arange(60).reshape(2, 5, 6).astype(np.float32)

    for border_mode in ['full', 'valid']:
        for cropping in [None, [(1, -1), (1, -1)]]:
            conv = Convolution(convolution_filter[np.newaxis, np.newaxis],
                               border_mode=border_mode,
                               cropping=cropping)
            conv_func = theano.function([conv.input_],
                                        conv.expression_)

            convolved = conv_func(images[:, np.newaxis])

            if cropping is None:
                cropping = [(0, None)] * 2
            cropping = [slice(*c) for c in cropping]
            convolutions = np.array([
                    convolve2d(img, convolution_filter,
                               mode=border_mode)[cropping]
                    for img in images])

            assert_array_almost_equal(convolved, convolutions[:, np.newaxis])


def test_fuse():
    convolution_filter = np.zeros([3, 3], dtype=np.float32)
    convolution_filter[2, 2] = 1.

    images = np.arange(96).reshape(2, 8, 6).astype(np.float32)

    conv = Convolution(convolution_filter[np.newaxis, np.newaxis],
                       border_mode='valid',
                       cropping=None,
                       activation='relu')

    max_pool = MaxPool((2, 2))

    pipe = [PassThrough(), conv, PassThrough(), PassThrough(), max_pool]

    expressions, input_variable = fuse(pipe, output_expressions=[1, 3, 4])

    convolutions = np.array([convolve2d(img, convolution_filter,
                                        mode='valid')
                             for img in images])

    rectified = convolutions.copy()
    rectified[rectified < 0] = 0.

    n, h, w = rectified.shape
    max_pooled = rectified.reshape(
        n, h / 2, 2, w / 2, 2).transpose(0, 1, 3, 2, 4).reshape(
        n, h / 2, w / 2, 4).max(-1)

    function = theano.function([input_variable],
                               outputs=expressions)

    results = function(images[:, np.newaxis])

    assert_array_almost_equal(results[0], rectified[:, np.newaxis])
    assert_array_almost_equal(results[1], rectified[:, np.newaxis])
    assert_array_almost_equal(results[2], max_pooled[:, np.newaxis])
