# Authors: Michael Eickenberg
#          Kyle Kastner
# License: BSD 3 clause

from __future__ import print_function
import theano
import numbers
import numpy as np
import theano.tensor as T
from theano.tensor.signal.downsample import max_pool_2d


def _relu(x):
    return T.maximum(x, 0)


def _identity(x):
    return x

ACTIVATIONS = {'sigmoid': T.nnet.sigmoid,
               'identity': _identity,
               'linear': _identity,
               'relu': _relu,
               None: _identity}


class Feedforward(object):
    def __init__(self, weights, biases=None, activation='relu',
                 input_dtype='float32'):
        pass  # should maybe initialize the type of variable
        self.input_dtype = input_dtype
        self.weights = theano.shared(weights.astype(self.input_dtype))
        self.activation_function = ACTIVATIONS[activation]
        if biases is not None:
            self.biases = theano.shared(biases.astype(self.input_dtype))
        else:
            self.biases = None
        self._build_expression()  # maybe have a flag

    def _build_expression(self, input_tensor=None):
        if input_tensor is not None:
            self.input_ = input_tensor
        else:
            self.input_ = T.matrix(dtype=self.input_dtype)
        self.expression_ = T.dot(self.input_, self.weights)
        if self.biases is not None:
            self.expression_ += self.biases
        self.expression_ = self.activation_function(self.expression_)


class Convolution(object):
    """A wrapper for a 2D convolution building block"""
    def __init__(self,
                 convolution_filter,
                 biases=None,
                 activation='relu',
                 border_mode='valid',
                 subsample=None,
                 cropping=None,
                 input_dtype='float32'):
        self.convolution_filter = convolution_filter
        self.biases = biases
        self.activation_function = ACTIVATIONS[activation]
        self.subsample = subsample
        self.border_mode = border_mode
        self.cropping = cropping
        self.input_dtype = input_dtype

        self._build_expression()  # not sure whether this is legit here

    def _build_expression(self, input_tensor=None):
        if self.cropping is None:
            self.cropping_ = [(0, None), (0, None)]
        else:
            self.cropping_ = self.cropping

        if self.subsample is None:
            self.subsample_ = (1, 1)
        else:
            self.subsample_ = self.subsample

        cf = self.convolution_filter
        if not isinstance(cf, T.sharedvar.TensorSharedVariable):
            if isinstance(cf, np.ndarray):
                self.convolution_filter_ = theano.shared(cf)
            else:
                raise ValueError("Variable type not understood")
        else:
            self.convolution_filter_ = cf

        biases = self.biases
        if biases is not None:
            if not isinstance(biases, T.sharedvar.TensorSharedVariable):
                if isinstance(biases, np.ndarray):
                    self.biases_ = theano.shared(biases)
                else:
                    raise ValueError("Variable type not understood")
            else:
                self.biases_ = biases

        if input_tensor is None:
            self.input_ = T.tensor4(dtype=self.input_dtype)
        else:
            self.input_ = input_tensor

        c = self.cropping_
        self.expression_ = T.nnet.conv2d(self.input_,
            self.convolution_filter_,
            border_mode=self.border_mode,
            subsample=self.subsample_)[:, :, c[0][0]:c[0][1],
                                             c[1][0]:c[1][1]]
        if self.biases is not None:
            self.expression_ += self.biases_.dimshuffle('x', 0, 'x', 'x')
        self.expression_ = self.activation_function(self.expression_)


class MarginalConvolution(object):
    """Same as Convolution if there is only one input channel. If there are
    several input channels, then it convolves each with the same filter,
    instead of using a 3D filter as does Convolution.
    Used for scattering transform.
    And for global smoothing.
    """
    def __init__(self,
                 convolution_filter,
                 activation=None,
                 border_mode='valid',
                 subsample=None,
                 cropping=None,
                 input_dtype='float32'):
        self.convolution_filter = convolution_filter
        self.activation = activation
        self.border_mode = border_mode
        self.subsample = subsample
        self.input_dtype = input_dtype

        self._build_expression()

    def _build_expression(self):

        if self.subsample is None:
            self.subsample_ = (1, 1)
        else:
            self.subsample_ = self.subsample

        cf = self.convolution_filter
        if not isinstance(cf, T.sharedvar.TensorSharedVariable):
            if isinstance(cf, np.ndarray):
                self.convolution_filter_ = theano.shared(cf)
            else:
                raise ValueError("Variable type not understood")
        else:
            self.convolution_filter_ = cf

        self.input_ = T.tensor4(dtype=self.input_dtype)
        self.expression_ = T.nnet.conv2d(
            self.input_.reshape((-1, 1,
                                  self.input_.shape[-2],
                                  self.input_.shape[-1])),
            self.convolution_filter_.reshape((-1, 1,
                                  self.convolution_filter_.shape[-2],
                                  self.convolution_filter_.shape[-1])),
            border_mode=self.border_mode,
            subsample=self.subsample_)
        if self.border_mode == 'valid':
            output_shape = (
                self.input_.shape[-2] -
                self.convolution_filter_.shape[-2] + 1,
                self.input_.shape[-1] -
                self.convolution_filter_.shape[-1] + 1)
        elif self.border_mode == 'full':
            output_shape = (
                self.input_.shape[-2] +
                self.convolution_filter_.shape[-2] - 1,
                self.input_.shape[-1] +
                self.convolution_filter_.shape[-1] - 1)
        self.expression_ = self.expression_.reshape(
            (self.input_.shape[0], -1) + output_shape)
        activation_function = ACTIVATIONS[self.activation]
        self.expression_ = activation_function(self.expression_)


class PassThrough(object):
    def __init__(self, input_dtype='float32'):
        pass  # should maybe initialize the type of variable
        self.input_dtype = input_dtype
        self._build_expression()  # maybe have a flag

    def _build_expression(self):
        self.input_ = T.tensor4(dtype=self.input_dtype)
        self.expression_ = self.input_


class Standardize(object):
    def __init__(self, mean=None, std=None, input_dtype='float32'):
        self.mean = mean
        self.std = std
        self.input_dtype = input_dtype

        self._build_expression()

    def _build_expression(self):
        self.input_ = T.tensor4(dtype=self.input_dtype)
        if self.mean is not None:
            centered = self.input_ - self.mean
        else:
            raise NotImplementedError("One could default to mean per sample")
        if self.std is not None:
            scaled = centered / self.std
        else:
            raise NotImplementedError("One could default to std per sample")
        self.expression_ = scaled


class MaxPool(object):
    def __init__(self, max_pool_stride, input_dtype='float32'):
        self.max_pool_stride = max_pool_stride
        self.input_dtype = input_dtype

        self._build_expression()

    def _build_expression(self):
        self.input_ = T.tensor4(dtype=self.input_dtype)
        self.expression_ = max_pool_2d(self.input_, self.max_pool_stride,
                                       ignore_border=True)


def _gcd(num1, num2):
    """Calculate gcd(num1, num2), greatest common divisor, using euclid's
    algorithm"""
    while (num2 != 0):
        if num1 > num2:
            num1, num2 = num2, num1
        num2 -= (num2 // num1) * num1
    return num1


def _lcm(num1, num2):
    """Calculate least common multiple of num1 and num2"""
    return num1 * num2 // _gcd(num1, num2)


def fancy_max_pool(input_tensor, pool_shape, pool_stride,
                   ignore_border=False):
    """Using theano built-in maxpooling, create a more flexible version.

    Obviously suboptimal, but gets the work done."""

    if isinstance(pool_shape, numbers.Number):
        pool_shape = pool_shape,
    if isinstance(pool_stride, numbers.Number):
        pool_stride = pool_stride,

    if len(pool_shape) == 1:
        pool_shape = pool_shape * 2
    if len(pool_stride) == 1:
        pool_stride = pool_stride * 2

    lcmh, lcmw = [_lcm(p, s) for p, s in zip(pool_shape, pool_stride)]
    dsh, dsw = lcmh // pool_shape[0], lcmw // pool_shape[1]

    pre_shape = input_tensor.shape[:-2]
    length = T.prod(pre_shape)
    post_shape = input_tensor.shape[-2:]
    new_shape = T.concatenate([[length], post_shape])
    reshaped_input = input_tensor.reshape(new_shape, ndim=3)
    sub_pools = []
    for sh in range(0, lcmh, pool_stride[0]):
        sub_pool = []
        sub_pools.append(sub_pool)
        for sw in range(0, lcmw, pool_stride[1]):
            full_pool = max_pool_2d(reshaped_input[:, sh:, sw:],
                                    pool_shape, ignore_border=ignore_border)
            ds_pool = full_pool[:, ::dsh, ::dsw]
            concat_shape = T.concatenate([[length], ds_pool.shape[-2:]])
            sub_pool.append(ds_pool.reshape(concat_shape, ndim=3))
    output_shape = (length,
                    T.sum([l[0].shape[1] for l in sub_pools]),
                    T.sum([i.shape[2] for i in sub_pools[0]]))
    output = T.zeros(output_shape, dtype=input_tensor.dtype)
    for i, line in enumerate(sub_pools):
        for j, item in enumerate(line):
            output = T.set_subtensor(output[:, i::lcmh // pool_stride[0],
                                               j::lcmw // pool_stride[1]],
                                     item)
    return output.reshape(T.concatenate([pre_shape, output.shape[1:]]),
                          ndim=input_tensor.ndim)


class FancyMaxPool(object):
    """Extended pooling functionality. Allows independent specification
    of pooling region shape and stride shape.

    Parameters
    ==========

    pool_shape: tuple, (height, width)
        Specifies the shape of the pooling regions

    pool_stride: tuple, (vert, horiz)
        Specifies the step between each pooling region

    input_dtype: string, default 'float32'
        Specifies the dtype of the input
    """
    def __init__(self, pool_shape, pool_stride,
                 ignore_border=False, input_dtype='float32'):
        self.pool_shape = pool_shape
        self.pool_stride = pool_stride
        self.ignore_border = ignore_border
        self.input_dtype = input_dtype

        self._build_expression()

    def _build_expression(self, input_expression=None):
        if input_expression is None:
            self.input_ = T.tensor4(dtype=self.input_dtype)
        else:
            self.input_ = input_expression
        print("Pooling: shape %s stride %s" % (str(self.pool_shape),
                                               str(self.pool_stride)))
        if self.pool_stride == self.pool_shape:
            self.expression_ = T.signal.downsample.max_pool_2d(
                self.input_, self.pool_shape,
                ignore_border=self.ignore_border)
        else:
            self.expression_ = fancy_max_pool(self.input_, self.pool_shape,
                                              self.pool_stride,
                                              self.ignore_border)


class CaffePool(object):
    """Replicate the caffe pooling layer exactly.
    For the moment, explicit zero-padding will be used."""

    def __init__(self, pool_shape, pool_stride=1,
                 padding=0, pool_type='max',
                 input_dtype='float32'):
        """Caffe pooling layer with the known params."""

        self.pool_shape = pool_shape
        self.pool_stride = pool_stride
        self.padding = padding
        self.pool_type = pool_type
        self.input_dtype = input_dtype

        self._build_expression()

    def _build_expression(self, input_expression=None):
        if self.pool_type not in ['max', 'avg']:
            raise NotImplementedError(
                'Pooling only implemented for max and avg')

        if input_expression is None:
            self.input_ = T.tensor4(dtype=self.input_dtype)
        else:
            self.input_ = input_expression

        # Replicating caffe style pooling means zero padding
        # then strided pooling with ignore_border=True
        if self.padding in [0, (0, 0)]:
            padded_input = self.input_
        else:
            zero_padder = ZeroPad(padding=self.padding)
            zero_padder._build_expression(self.input_)
            padded_input = zero_padder.expression_
        if self.pool_type == 'max':
            pooled = fancy_max_pool(padded_input,
                                    self.pool_shape, self.pool_stride,
                                    ignore_border=False)
        elif self.pool_type == 'avg':
            # self.pool_shape needs to be a tuple
            avg_kernel = T.cast(T.ones((1, 1) + self.pool_shape,
                                dtype=self.input_.dtype
                                ) / np.prod(self.pool_shape),
                                self.input_.dtype)
            n_imgs = self.input_.shape[0]
            n_channels = self.input_.shape[1]
            conv_output = T.nnet.conv2d(
                padded_input.reshape((n_imgs * n_channels, 1,
                                      padded_input.shape[2],
                                      padded_input.shape[3])),
                avg_kernel, subsample=self.pool_stride)
            pooled = conv_output.reshape((n_imgs, n_channels,
                                         conv_output.shape[2],
                                         conv_output.shape[3]))

        # A caffe quirk: The output shape is (for width, analogous for h:)
        # ceil((w + 2 * pad_w - kernel_w) / stride_w) + 1, instead of floor
        # With floor, ignore_border=True would have yielded the exact result
        # With ceil, sometimes we need an extra column and/or line. So we do
        # ignore_border=False and then crop to the right shape. Since the
        # shape is dynamic we need to first calculate it:

        # padding gotta be a tuple too
        pad = T.constant(self.padding)
        # pad = T.constant(zero_padder.padding_)
        # supposing here that self.pool_shape is a tuple. Should check
        pool_shape = T.constant(self.pool_shape)
        # stride hopefully a tuple, too
        pool_stride = T.constant(self.pool_stride, dtype='float64')
        float_shape = (self.input_.shape[2:4] + 2 * pad
                       - pool_shape) / pool_stride + 1
        output_shape = T.cast(T.ceil(float_shape), dtype='int64')
        self.expression_ = pooled[:, :, 0:output_shape[0],
                                        0:output_shape[1]]


class ZeroPad(object):
    """Zero-padding using set_subtensor

    """

    def __init__(self, padding=1, input_dtype='float32'):
        self.padding = padding
        self.input_dtype = input_dtype

        self._build_expression()

    def _build_expression(self, input_expression=None):
        if isinstance(self.padding, numbers.Number):
            self.padding_ = (self.padding,) * 4
        elif len(self.padding) == 1:
            self.padding_ = tuple(self.padding) * 4
        elif len(self.padding) == 2:
            self.padding_ = tuple(self.padding) * 2
        elif len(self.padding) == 4:
            self.padding_ = self.padding
        else:
            raise ValueError("padding must be of length 1, 2 or 4")

        p = self.padding_
        shape = (p[0] + 1 + p[2], p[1] + 1 + p[3])

        padding_indicator = np.zeros(shape, dtype=np.dtype(self.input_dtype))
        padding_indicator[p[0], p[1]] = 1.
        self.padding_indicator_ = theano.shared(
            padding_indicator[np.newaxis, np.newaxis])
        if input_expression is None:
            self.input_ = T.tensor4(dtype=self.input_dtype)
        else:
            self.input_ = input_expression
        input_shape = self.input_.shape
        output_shape = (input_shape[0], input_shape[1],
                        input_shape[2] + p[0] + p[2],
                        input_shape[3] + p[1] + p[3])
        output = T.zeros(output_shape, dtype=self.input_.dtype)
        self.expression_ = T.set_subtensor(
            output[:, :, p[0]:output_shape[2] - p[2],
                         p[1]:output_shape[3] - p[3]],
            self.input_)
        return None


class Relu(object):
    def __init__(self, input_type=T.tensor4):
        self.input_type = input_type

        self._build_expression()

    def _build_expression(self, input_tensor=None):
        if input_tensor is None:
            self.input_ = self.input_type()
        else:
            self.input_ = input_tensor
        self.expression_ = T.maximum(self.input_, 0)


class LRN(object):
    def __init__(self, normalization_size,
                 normalization_factor,
                 normalization_exponent,
                 axis='channels', input_type=T.tensor4):
        self.normalization_size = normalization_size
        self.normalization_factor = normalization_factor
        self.normalization_exponent = normalization_exponent
        self.axis = axis
        self.input_type = input_type

        self._build_expression()

    def _build_expression(self, input_tensor=None):
        if self.axis != 'channels':
            raise NotImplementedError("Only implemented for channels "
                                      "at this moment")
        if self.normalization_size % 2 == 0:
            raise ValueError("Only accepting odd sized pooling regions"
                             " in order to be able to identify a midpoint.")

        if input_tensor is None:
            self.input_ = self.input_type()
        else:
            self.input_ = input_tensor

        # Implement local response normalization across channels by reshaping
        # and using conv2d with a 1D filter and cropping properly ...
        nsize = self.normalization_size
        fil = (T.ones((1, 1, nsize, 1),
                     dtype=self.input_.dtype)
               / self.normalization_size
               * self.normalization_factor)
        local_mean = T.nnet.conv2d(
            self.input_.reshape(
                (self.input_.shape[0], 1, self.input_.shape[1], -1)) ** 2,
            fil, border_mode='full')[
            :, :, nsize // 2:-(nsize // 2), :].reshape(
            (self.input_.shape[0], -1,
             self.input_.shape[2], self.input_.shape[3]))

        self.expression_ = self.input_ / ((1 + local_mean) **
                                             self.normalization_exponent)


def fuse(building_blocks, fuse_dim=4, input_variables=None, entry_expression=None,
         output_expressions=-1, input_dtype='float32'):

    num_blocks = len(building_blocks)

    if isinstance(output_expressions, numbers.Number):
        output_expressions = [output_expressions]

    # account for indices -1, -2 etc
    output_expressions = [oe % num_blocks for oe in output_expressions]

    if fuse_dim == 4:
        fuse_block = T.tensor4
    else:
        fuse_block = T.matrix

    if input_variables is None and entry_expression is None:
        input_variables = fuse_block(dtype=input_dtype)
        entry_expression = input_variables

    current_expression = entry_expression
    outputs = []

    for i, block in enumerate(building_blocks):
        if not hasattr(block, "expression_"):
            block._build_expression()
        current_expression = theano.clone(
            block.expression_,
            replace={block.input_: current_expression},
            strict=False)
        if i in output_expressions:
            outputs.append(current_expression)

    return outputs, input_variables

