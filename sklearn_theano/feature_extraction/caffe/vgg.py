"""Parser for VGG caffemodel."""
# Authors: Michael Eickenberg
#          Kyle Kastner
# License: BSD 3 Clause

from sklearn.externals import joblib
from ...datasets import get_dataset_dir, download
from .caffemodel import _parse_caffe_model, parse_caffe_model
import os
import theano

VGG_PATH = get_dataset_dir("caffe/vgg")


def fetch_vgg_protobuffer_file(caffemodel_file=None):
    """Checks for existence of caffemodel protobuffer.
    Downloads it if it cannot be found."""

    default_filename = os.path.join(VGG_PATH,
                                    "VGG_ILSVRC_19_layers.caffemodel")

    if caffemodel_file is not None:
        if os.path.exists(caffemodel_file):
            return caffemodel_file
        else:
            if os.path.exists(default_filename):
                import warnings
                warnings.warn('Did not find %s, but found and returned %s.' %
                              (caffemodel_file, default_filename))
                return default_filename
    else:
        if os.path.exists(default_filename):
            return default_filename

    # We didn't find the file, let's download it. To the specified location
    # if specified, otherwise to the default place
    if caffemodel_file is None:
        caffemodel_file = default_filename
        if not os.path.exists(VGG_PATH):
            os.makedirs(VGG_PATH)

    url = "http://www.robots.ox.ac.uk/%7Evgg/software/very_deep/caffe/"
    url += "VGG_ILSVRC_19_layers.caffemodel"
    download(url, caffemodel_file, progress_update_percentage=1)
    return caffemodel_file


def fetch_vgg_architecture(caffemodel_parsed=None, caffemodel_protobuffer=None):
    """Fetch a pickled version of the caffe model, represented as list of
    dictionaries."""

    default_filename = os.path.join(VGG_PATH, 'vgg.pickle')
    if caffemodel_parsed is not None:
        if os.path.exists(caffemodel_parsed):
            return joblib.load(caffemodel_parsed)
        else:
            if os.path.exists(default_filename):
                import warnings
                warnings.warn('Did not find %s, but found %s. Loading it.' %
                              (caffemodel_parsed, default_filename))
                return joblib.load(default_filename)
    else:
        if os.path.exists(default_filename):
            return joblib.load(default_filename)

    # We didn't find the file: let's create it by parsing the protobuffer
    protobuf_file = fetch_vgg_protobuffer_file(caffemodel_protobuffer)
    model = _parse_caffe_model(protobuf_file)

    if caffemodel_parsed is not None:
        joblib.dump(model, caffemodel_parsed)
    else:
        joblib.dump(model, default_filename)

    return model


def create_theano_expressions(model=None, verbose=0):

    if model is None:
        model = fetch_vgg_architecture()

    layers, blobs, inputs, params = parse_caffe_model(
        model, convert_fc_to_conv=False, verbose=verbose)
    data_input = inputs['data']
    return blobs, data_input


def _get_fprop(output_layers=('prob',), model=None, verbose=0):

    if model is None:
        model = fetch_vgg_architecture(model)

    expressions, input_data = create_theano_expressions(model,
                                                        verbose=verbose)
    to_compile = [expressions[expr] for expr in output_layers]

    return theano.function([input_data], to_compile)
