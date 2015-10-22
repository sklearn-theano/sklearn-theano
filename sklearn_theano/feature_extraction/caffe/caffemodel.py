"""Makes .caffemodel files readable for sklearn-theano"""
# Authors: Michael Eickenberg
#          Kyle Kastner
#          Erfan Noury
#          Li Yao
# License: BSD 3 Clause
from __future__ import print_function
import os
import numpy as np
from collections import OrderedDict
import theano.tensor as T
from ...datasets import get_dataset_dir, download
from sklearn_theano.base import Convolution, Relu, LRN, Feedforward, ZeroPad
from sklearn_theano.base import CaffePool
import warnings


def _get_caffe_dir():
    """Function to find caffe installation. First checks for pycaffe. If not
    present, checks for $CAFFE_DIR environment variable."""

    try:
        import caffe
        from os.path import dirname
        caffe_dir = dirname(dirname(dirname(caffe.__file__)))
    except ImportError:
        caffe_dir = os.environ.get("CAFFE_DIR", None)

    return caffe_dir


def _compile_caffe_protobuf(caffe_proto=None,
                            proto_src_dir=None,
                            python_out_dir=None):
    """Compiles protocol buffer to python_out_dir"""

    if caffe_proto is None:
        caffe_dir = _get_caffe_dir()
        if caffe_dir is None:
            # No CAFFE_DIR found, neither could pycaffe be imported.
            # Search for caffe.proto locally
            caffe_dataset_dir = get_dataset_dir('caffe')
            caffe_proto = os.path.join(caffe_dataset_dir, 'caffe.proto')
            if os.path.exists(caffe_proto):
                # Found caffe.proto, everything fine
                pass
            else:
                print("Downloading caffe.proto")
                url = ('https://raw.githubusercontent.com/'
                       'BVLC/caffe/master/src/caffe/proto/caffe.proto')
                download(url, caffe_proto, progress_update_percentage=1)
            # raise ValueError("Cannot find $CAFFE_DIR environment variable"
            #                  " specifying location of Caffe files."
            #                  " Nor does there seem to be pycaffe. Please"
            #                  " provide path to caffe.proto file in the"
            #                  " caffe_proto kwarg, e.g. "
            #                  "/home/user/caffe/src/caffe/proto/caffe.proto")
        else:
            caffe_proto = os.path.join(caffe_dir, "src", "caffe", "proto",
                                       "caffe.proto")
    if not os.path.exists(caffe_proto):
        raise ValueError(
            ("Could not find {pf}. Please specify the correct"
             " caffe.proto file in the caffe_proto kwarg"
             " e.g. /home/user/caffe/src/caffe/proto/caffe.proto").format(
                pf=caffe_proto))

    if proto_src_dir is None:
        proto_src_dir = os.path.dirname(caffe_proto)

    if python_out_dir is None:
        python_out_dir = os.path.dirname(os.path.abspath(__file__))

    protoc_command = ("protoc -I={srcdir}"
                      " --python_out={outdir} {protofile}").format(
        srcdir=proto_src_dir, outdir=python_out_dir, protofile=caffe_proto)

    import commands
    status, output = commands.getstatusoutput(protoc_command)

    if status != 0:
        raise Exception(
            "Error executing protoc: code {c}, message {m}".format(
                c=status, m=output))


def _get_caffe_pb2():
    import sys
    this_file_path = os.path.realpath(__file__)
    google_dir = str(os.sep).join(
        this_file_path.split(os.sep)[:-3] + ['externals'])
    sys.path.append(google_dir)
    from ...models.bvlc_googlenet import caffe_pb2
    sys.path.remove(google_dir)
    return caffe_pb2


def _open_caffe_model(caffemodel_file):
    """Opens binary format .caffemodel files. Returns protobuf object."""
    caffe_pb2 = _get_caffe_pb2()
    f = open(caffemodel_file, 'rb')
    binary_content = f.read()
    protobuf = caffe_pb2.NetParameter()
    protobuf.ParseFromString(binary_content)

    return protobuf


def _blob_to_ndarray(blob):
    """Converts a caffe protobuf blob into an ndarray"""
    dimnames = ["num", "channels", "height", "width"]
    data = np.array(blob.data)
    shape = tuple([getattr(blob, dimname) for dimname in dimnames])
    return data.reshape(shape)


LAYER_PROPERTIES = dict(
    DATA=None,
    CONVOLUTION=('blobs',
                 ('convolution_param', 'stride'),
                 ('convolution_param', 'stride_h'),
                 ('convolution_param', 'stride_w'),
                 ('convolution_param', 'pad'),
                 ('convolution_param', 'pad_h'),
                 ('convolution_param', 'pad_w')),
    RELU=None,
    POOLING=(('pooling_param', 'kernel_size'),
             ('pooling_param', 'kernel_h'),
             ('pooling_param', 'kernel_w'),
             ('pooling_param', 'stride'),
             ('pooling_param', 'stride_h'),
             ('pooling_param', 'stride_w'),
             ('pooling_param', 'pad'),
             ('pooling_param', 'pad_h'),
             ('pooling_param', 'pad_w'),
             ('pooling_param', 'pool')
             ),
    SPLIT=None,
    LRN=(('lrn_param', 'local_size'),
         ('lrn_param', 'alpha'),
         ('lrn_param', 'beta'),
         ('lrn_param', 'norm_region')),
    CONCAT=(('concat_param', 'concat_dim'),),
    INNER_PRODUCT=('blobs',),
    SOFTMAX_LOSS=None,
    SOFTMAX=None,
    DROPOUT=None
)


def _get_property(obj, property_path):

    if isinstance(property_path, tuple):
        if len(property_path) == 1:
            return getattr(obj, property_path[0])
        else:
            return _get_property(getattr(obj, property_path[0]),
                                 property_path[1:])
    else:
        return getattr(obj, property_path)


def _parse_caffe_model(caffe_model):
    warnings.warn("Caching parse for caffemodel, this may take some time")
    caffe_pb2 = _get_caffe_pb2()  # need to remove this dependence on pb here
    try:
        _layer_types = caffe_pb2.LayerParameter.LayerType.items()
    except AttributeError:
        _layer_types = caffe_pb2.V1LayerParameter.LayerType.items()

    # create a dictionary that indexes both ways, number->name, name->number
    layer_types = dict(_layer_types)
    for v, k in _layer_types:
        layer_types[k] = v

    if not hasattr(caffe_model, "layers"):
        # Consider it a filename
        caffe_model = _open_caffe_model(caffe_model)

    layers_raw = caffe_model.layers
    parsed = []

    for n, layer in enumerate(layers_raw):
        # standard properties
        ltype = layer_types[layer.type]
        if n == 0 and ltype != 'DATA':
            warnings.warn("Caffemodel doesn't start with DATA - adding")
            first_layer_descriptor = dict(
                type='DATA',
                name='data',
                top_blobs=('data',),
                bottom_blobs=tuple())
            parsed.append(first_layer_descriptor)

        layer_descriptor = dict(type=ltype,
                                name=layer.name,
                                top_blobs=tuple(layer.top),
                                bottom_blobs=tuple(layer.bottom))
        parsed.append(layer_descriptor)
        # specific properties
        specifics = LAYER_PROPERTIES[ltype]
        if specifics is None:
            continue
        for param in specifics:
            if param == 'blobs':
                layer_descriptor['blobs'] = list(map(_blob_to_ndarray,
                                                     layer.blobs))
            else:
                param_name = '__'.join(param)
                param_value = _get_property(layer, param)
                layer_descriptor[param_name] = param_value
    return parsed


def parse_caffe_model(caffe_model, convert_fc_to_conv=True,
                      float_dtype='float32', verbose=0):
    if isinstance(caffe_model, str) or not isinstance(caffe_model, list):
        parsed_caffe_model = _parse_caffe_model(caffe_model)
    else:
        parsed_caffe_model = caffe_model

    layers = OrderedDict()
    inputs = OrderedDict()
    blobs = OrderedDict()
    params = OrderedDict()

    for i, layer in enumerate(parsed_caffe_model):
        layer_type = layer['type']
        layer_name = layer['name']
        top_blobs = layer['top_blobs']
        bottom_blobs = layer['bottom_blobs']
        layer_blobs = layer.get('blobs', None)

        if verbose > 0:
            print("%d\t%s\t%s" % (i, layer_type, layer_name))
        if layer_type == 'DATA':
            # DATA layers contain input data in top_blobs, create input
            # variables, float for 'data' and int for 'label'
            for data_blob_name in top_blobs:
                if data_blob_name == 'label':
                    blobs['label'] = T.ivector()
                    inputs['label'] = blobs['label']
                else:
                    blobs[data_blob_name] = T.tensor4(dtype=float_dtype)
                    inputs[data_blob_name] = blobs[data_blob_name]
        elif layer_type == 'CONVOLUTION':
            # CONVOLUTION layers take input from bottom_blob, convolve with
            # layer_blobs[0], and add bias layer_blobs[1]
            stride = layer['convolution_param__stride']
            stride_h = max(layer['convolution_param__stride_h'], stride)
            stride_w = max(layer['convolution_param__stride_w'], stride)
            if stride_h > 1 or stride_w > 1:
                subsample = (stride_h, stride_w)
            else:
                subsample = None
            pad = layer['convolution_param__pad']
            pad_h = max(layer['convolution_param__pad_h'], pad)
            pad_w = max(layer['convolution_param__pad_w'], pad)
            conv_filter = layer_blobs[0].astype(float_dtype)[..., ::-1, ::-1]
            conv_bias = layer_blobs[1].astype(float_dtype).ravel()
            convolution_input = blobs[bottom_blobs[0]]
            convolution = Convolution(conv_filter, biases=conv_bias,
                                      activation=None, subsample=subsample,
                                      input_dtype=float_dtype)
            # If padding is specified, need to pad. In practice, I think
            # caffe prevents padding that would make the filter see only
            # zeros, so technically this can also be obtained by sensibly
            # cropping a border_mode=full convolution. However, subsampling
            # may then be off by 1 and would have to be done separately :/
            if pad_h > 0 or pad_w > 0:
                zp = ZeroPad((pad_h, pad_w))
                zp._build_expression(convolution_input)
                expression = zp.expression_
                layers[layer_name] = (zp, convolution)
            else:
                layers[layer_name] = convolution
                expression = convolution_input
            convolution._build_expression(expression)
            expression = convolution.expression_
            # if subsample is not None:
            #     expression = expression[:, :, ::subsample[0],
            #                                     ::subsample[1]]

            blobs[top_blobs[0]] = expression

            params[layer_name + '_conv_W'] = convolution.convolution_filter_
            params[layer_name + '_conv_b'] = convolution.biases_

        elif layer_type == "RELU":
            # RELU layers take input from bottom_blobs, set everything
            # negative to zero and write the result to top_blobs
            relu_input = blobs[bottom_blobs[0]]
            relu = Relu()
            relu._build_expression(relu_input)
            layers[layer_name] = relu
            blobs[top_blobs[0]] = relu.expression_
        elif layer_type == "POOLING":
            # POOLING layers take input from bottom_blobs, perform max
            # pooling according to stride and kernel size information
            # and write the result to top_blobs
            pooling_input = blobs[bottom_blobs[0]]
            kernel_size = layer['pooling_param__kernel_size']
            kernel_h = max(layer['pooling_param__kernel_h'], kernel_size)
            kernel_w = max(layer['pooling_param__kernel_w'], kernel_size)
            stride = layer['pooling_param__stride']
            stride_h = max(layer['pooling_param__stride_h'], stride)
            stride_w = max(layer['pooling_param__stride_w'], stride)
            pad = layer['pooling_param__pad']
            pad_h = max(layer['pooling_param__pad_h'], pad)
            pad_w = max(layer['pooling_param__pad_w'], pad)
            pool_types = {0: 'max', 1: 'avg'}
            pool_type = pool_types[layer['pooling_param__pool']]
            # print "POOL TYPE is %s" % pool_type
            # pooling = FancyMaxPool((kernel_h, kernel_w),
            #                        (stride_h, stride_w),
            #                        ignore_border=False)
            pooling = CaffePool((kernel_h, kernel_w),
                                (stride_h, stride_w),
                                (pad_h, pad_w),
                                pool_type=pool_type)
            pooling._build_expression(pooling_input)
            layers[layer_name] = pooling
            blobs[top_blobs[0]] = pooling.expression_
        elif layer_type == "DROPOUT":
            # DROPOUT may figure in some networks, but it is only relevant
            # at the learning stage, not at the prediction stage.
            pass
        elif layer_type in ["SOFTMAX_LOSS", "SOFTMAX"]:
            softmax_input = blobs[bottom_blobs[0]]
            # have to write our own softmax expression, because of shape
            # issues
            si = softmax_input.reshape((softmax_input.shape[0],
                                        softmax_input.shape[1], -1))
            shp = (si.shape[0], 1, si.shape[2])
            exp = T.exp(si - si.max(axis=1).reshape(shp))
            softmax_expression = (exp / exp.sum(axis=1).reshape(shp)
                                  ).reshape(softmax_input.shape)
            layers[layer_name] = "SOFTMAX"
            blobs[top_blobs[0]] = softmax_expression
        elif layer_type == "SPLIT":
            split_input = blobs[bottom_blobs[0]]
            for top_blob in top_blobs:
                blobs[top_blob] = split_input
            # Should probably make a class to be able to add to layers
            layers[layer_name] = "SPLIT"
        elif layer_type == "LRN":
            # Local normalization layer
            lrn_input = blobs[bottom_blobs[0]]
            lrn_factor = layer['lrn_param__alpha']
            lrn_exponent = layer['lrn_param__beta']
            axis = {0: 'channels'}[layer['lrn_param__norm_region']]
            nsize = layer['lrn_param__local_size']
            lrn = LRN(nsize, lrn_factor, lrn_exponent, axis=axis)
            lrn._build_expression(lrn_input)
            layers[layer_name] = lrn
            blobs[top_blobs[0]] = lrn.expression_
        elif layer_type == "CONCAT":
            input_expressions = [blobs[bottom_blob] for bottom_blob
                                 in bottom_blobs]
            axis = layer['concat_param__concat_dim']
            output_expression = T.concatenate(input_expressions, axis=axis)
            blobs[top_blobs[0]] = output_expression
            layers[layer_name] = "CONCAT"
        elif layer_type == "INNER_PRODUCT":
            weights = layer_blobs[0].astype(float_dtype)
            biases = layer_blobs[1].astype(float_dtype).squeeze()
            fully_connected_input = blobs[bottom_blobs[0]]
            if not convert_fc_to_conv:
                if fully_connected_input.ndim == 4:
                    m_, t_, x_, y_ = fully_connected_input.shape
                    fully_connected_input = fully_connected_input.reshape(
                        (m_, t_ * x_ * y_))
                fc_layer = Feedforward(weights.squeeze().T, biases,
                                       activation=None)
                params[layer_name + '_fc_W'] = fc_layer.weights
                if fc_layer.biases is not None:
                    params[layer_name + '_fc_b'] = fc_layer.biases
            else:
                fc_layer = Convolution(weights.transpose((2, 3, 0, 1)), biases,
                                       activation=None)
                params[layer_name + '_conv_W'] = convolution.convolution_filter_
                params[layer_name + '_conv_b'] = convolution.biases_

            fc_layer._build_expression(fully_connected_input)
            layers[layer_name] = fc_layer
            blobs[top_blobs[0]] = fc_layer.expression_
        else:
            raise ValueError('layer type %s is not known to sklearn-theano'
                             % layer_type)
    return layers, blobs, inputs, params
