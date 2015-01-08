"""Makes .caffemodel files readable for sklearn-theano"""
import os
import numpy as np

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

    return layers_raw, layer_names, layer_type_names, bottom_blobs, top_blobs, layer_blobs_ndarrays

from sklearn_theano.base import (Convolution, Relu, MaxPool)



if __name__ == "__main__":
    pb = _parse_caffe_model("/home/me/Downloads/cifar10_nin.caffemodel")
    import IPython
    IPython.embed()

