import numpy as np
import os
from ...sandbox.overfeat_wrapper import get_output
from ..overfeat import _get_fprop
from sklearn.externals.joblib import Memory

from numpy.testing import assert_array_almost_equal

cachedir = os.path.join(os.path.dirname(__file__), "cache")
if not os.path.exists(cachedir):
    os.makedirs(cachedir)
mem = Memory(cachedir=cachedir)

get_output = mem.cache(get_output)


# @mem.cache
def get_theano_output(input_img, layer, largenet, detailed):
    fprop = _get_fprop(largenet, output_layers=[layer], detailed=detailed)
    return fprop(input_img.astype(np.float32
                                  ).transpose(2, 0, 1)[np.newaxis])[0][0]


def _check_overfeat_layer(input_img, theano_layer, binary_layer,
                          largenet, detailed, cropping=None,
                          overfeatcmd=None, net_weight_file=None,
                          overfeat_dir=None, architecture='linux_64'):

    theano_output = get_theano_output(input_img, theano_layer, largenet,
                                      detailed)
    binary_output = get_output(input_img, binary_layer, largenet,
                               overfeatcmd=overfeatcmd,
                               net_weight_file=net_weight_file,
                               overfeat_dir=overfeat_dir,
                               architecture=architecture)
    if cropping is not None:
        binary_output = binary_output[[slice(None)] + cropping]

    assert_array_almost_equal(theano_output, binary_output, decimal=3)


def test_theano_overfeat_against_binary():
    layer_correspondence = dict(
        normal=dict(
            large=[(0, 0, None), (1, 2, None), (2, 3, None), (3, 5, None),
                   (4, 6, None), (5, 9, None), (6, 12, None), (7, 15, None),
                   (8, 18, None), (9, 19, None), (10, 21, None),
                   (11, 23, None), (12, 24, None)],
            small=[(0, 0, None), (1, 2, None), (2, 3, None), (3, 5, None),
                   (4, 6, None), (5, 9, None), (6, 12, None), (7, 15, None),
                   (8, 16, None), (9, 18, None), (10, 20, None), 
                   (11, 21, None)]),
        detailed=dict(
            large=[(i, i, None) for i in range(25)],
            small=[(i, i, None) for i in range(22)]
            )
        )

    rng = np.random.RandomState(42)
    image = (rng.rand(320, 320, 3) * 255).astype(np.uint8)
    from skimage.data import lena
    image = lena()

    for detailed, correspondences in layer_correspondence.items():
        for net_size, correspondence in correspondences.items():
            for theano_layer, binary_layer, cropping in correspondence:
                _check_overfeat_layer(image, theano_layer, binary_layer,
                                      net_size == 'large',
                                      detailed == 'detailed',
                                      cropping=cropping)

