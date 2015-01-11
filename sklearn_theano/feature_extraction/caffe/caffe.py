"""Makes .caffemodel files readable for sklearn-theano"""
import os
import numpy as np
from collections import OrderedDict
import theano
import theano.tensor as T


def _compile_caffe_protobuf(caffe_proto=None,
                           proto_src_dir=None,
                           python_out_dir=None):
    """Compiles protocol buffer to python_out_dir"""

    if caffe_proto is None:
        caffe_dir = os.environ.get("CAFFE", None)
        if caffe_dir is None:
            raise ValueError("Cannot find $CAFFE environment variable"
                             " specifying location of Caffe files. Please"
                             " provide path to caffe.proto file in the"
                             " caffe_proto kwarg")
        caffe_proto = os.path.join(caffe_dir, "src", "caffe", "proto",
                                  "caffe.proto")
    if not os.path.exists(caffe_proto):
        raise ValueError(
            "Could not find {pf}. Please specify the correct"
            " caffe.proto file in the caffe_proto kwarg".format(
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

try:
    import caffe_pb2
except:
    # If compiled protocol buffer does not exist yet, compile it
    _compile_caffe_protobuf()
    import caffe_pb2

_layer_types = caffe_pb2.LayerParameter.LayerType.items()

# create a dictionary that indexes both ways, number->name, name->number
layer_types = dict(_layer_types)
for v, k in _layer_types:
    layer_types[k] = v


def _open_caffe_model(caffemodel_file):
    """Opens binary format .caffemodel files. Returns protobuf object."""
    binary_content = open(caffemodel_file, "rb").read()
    protobuf = caffe_pb2.NetParameter()
    protobuf.ParseFromString(binary_content)

    return protobuf


def _blob_to_ndarray(blob):
    """Converts a caffe protobuf blob into an ndarray"""
    dimnames = ["num", "channels", "height", "width"]
    data = np.array(blob.data)
    shape = tuple([getattr(blob, dimname) for dimname in dimnames])
    return data.reshape(shape)


def _read_pooling_param(pooling_param):
    """Reads info out of pooling_param object. This function is to be kept
    minimal in the information it extracts.

    The current extraction yields:
    (pooling_param.kernel_size, pooling_param.stride)
    """
    property_names = ('kernel_size', 'stride')
    property_values = tuple([getattr(pooling_param, property_name)
                             for property_name in property_names])
    return property_values


def _parse_caffe_model(caffe_model):
    """Reads the relevant information out of the layers of a protobuffer
    object (binary) describing the network or a filename pointing to it."""

    if not hasattr(caffe_model, "layers"):
        # Consider it a filename
        caffe_model = _open_caffe_model(caffe_model)

    layers_raw = caffe_model.layers
    layer_names = [l.name for l in layers_raw]
    layer_type_names = [layer_types[l.type] for l in layers_raw]
    layer_blobs_raw = [l.blobs for l in layers_raw]
    layer_blobs_ndarrays = [map(_blob_to_ndarray, blob)
                            for blob in layer_blobs_raw]
    top_blobs = [l.top for l in layers_raw]
    bottom_blobs = [l.bottom for l in layers_raw]
    pooling_info = [_read_pooling_param(l.pooling_param) for l in layers_raw]

    output = (layer_names, layer_type_names, bottom_blobs,
              top_blobs, layer_blobs_ndarrays, pooling_info)

    return output


from sklearn_theano.base import (Convolution, Relu, MaxPool, FancyMaxPool)


def parse_caffe_model(caffe_model, float_dtype='float32'):
    """Reads a .caffemodel file and returns a list of sklearn-theano
    operators.

    Parameters
    ==========

    caffe_model: string or binary google protobuffer object
        file or binary protobuf object specifying the caffe model.

    Returns
    =======

    parsed_caffe_model: List of sklearn-theano operators ready to be fused.

    Notes
    =====

    This parser understands the caffe layers
    DATA
    CONVOLUTION
    RELU
    POOLING

    """

    raw_parsed = _parse_caffe_model(caffe_model)

    layers = OrderedDict()
    # inputs = OrderedDict()
    blobs = OrderedDict()

    for i, (layer_name, layer_type, bottom_blobs,
            top_blobs, layer_blobs, pooling_info
            ) in enumerate(zip(*raw_parsed)):
        if layer_type == 'DATA':
            # DATA layers contain input data in top_blobs, create input
            # variables, float for 'data' and int for 'label'
            for data_blob_name in top_blobs:
                if data_blob_name == 'label':
                    blobs['label'] = T.ivector()
                else:
                    blobs[data_blob_name] = T.tensor4(dtype=float_dtype)
        elif layer_type == 'CONVOLUTION':
            # CONVOLUTION layers take input from bottom_blob, convolve with
            # layer_blobs[0], and add bias layer_blobs[1]
            conv_filter = layer_blobs[0].astype(float_dtype)
            conv_bias = layer_blobs[1].astype(float_dtype).ravel()
            convolution_input = blobs[bottom_blobs[0]]
            import IPython
            IPython.embed()
            convolution = Convolution(conv_filter, biases=conv_bias,
                                      activation=None, subsample=None,
                                      input_dtype=float_dtype)
            convolution._build_expression(convolution_input)
            layers[layer_name] = convolution
            blobs[top_blobs[0]] = convolution
        elif layer_type == "RELU":
            # RELU layers take input from bottom_blobs, set everything
            # negative to zero and write the result to top_blobs
            relu_input = bottom_blobs[0]
            relu = Relu()
            relu._build_expression(relu_input)
            layers[layer_name] = relu
            blobs[top_blobs[0]] = relu
        elif layer_type == "POOLING":
            # POOLING layers take input from bottom_blobs, perform max
            # pooling according to stride and kernel size information
            # and write the result to top_blobs
            pooling_input = bottom_blobs[0]
            kernel_size, stride = pooling_info
            pooling = FancyMaxPool(kernel_size, stride)
            pooling._build_expression(pooling_input)
            layers[layer_name] = pooling
            blobs[top_blobs[0]] = pooling
        elif layer_type == "DROPOUT":
            # DROPOUT may figure in some networks, but it is only relevant
            # at the learning stage, not at the prediction stage.
            pass
        elif layer_type == "SOFTMAX_LOSS":
            # SOFTMAX_LOSS is used at training time. At prediction time, we
            # should replace it with a soft max.
            pass

    return layers, blobs

if __name__ == "__main__":
    pb = parse_caffe_model("/home/me232320/Downloads/cifar10_nin.caffemodel")
    import IPython
    IPython.embed()

