import numpy as np
import skimage.data as skd
from scipy.misc import lena
from path import path
from sklearn.externals.joblib import Memory
from overfeat_binary_wrapper import get_overfeat_layer
from feature_extraction.image.overfeat import _setup_small_network
from feature_extraction.image.utils import fetch_overfeat_weights
import theano
import theano.tensor as T

cachedir = path("cache")
if not cachedir.exists():
    cachedir.makedirs()

mem = Memory(cachedir=cachedir)

get_overfeat_layer = mem.cache(get_overfeat_layer)

test_images = [skd.lena(),
               lena(),
               skd.camera(),
               skd.chelsea(),
               skd.coffee()]

# figuring out layer 0
coefs_from_binary = [get_overfeat_layer(img, layer=0)
                     for img in test_images]

# Global normalization?
global_linreg = []
channel_linreg = []
for img, coefs in zip(test_images, coefs_from_binary):

    if img.ndim == 2:
        img = img[..., np.newaxis] * np.ones((1, 1, 3))

    fimg = img.transpose(2, 0, 1).ravel().astype(float)
    X = np.vstack([fimg.ravel(), np.ones(fimg.size)]).T
    beta = np.linalg.pinv(X).dot(coefs.ravel())
    global_linreg.append(beta)

    cimg = fimg.reshape(3, -1)
    Xc = np.vstack([cimg, np.ones(cimg.shape[1])]).T
    beta_c = np.linalg.pinv(Xc).dot(coefs.reshape(3, -1).T)
    channel_linreg.append(beta_c)

# (CRAZY) Conclusion: Overfeat subtracts the mean of cameraman and divides by its stddev
# (REASONABLE) Confirmation: Numbers from all of ImageNet
#    mean: 118.38094783922
#    std: 61.896912947719
# (CRAZIEST) Cameraman is the ideal representative of all natural images... ever


# figuring out layer 1
coefs_1 = [get_overfeat_layer(img, layer=1)
           for img in test_images]

# make theano layers
small_fprop = _setup_small_network()
#large_fprop = _setup_large_network()

f1 = small_fprop[1]
theano_l1 = [f1(coef[np.newaxis].astype(np.float32))
             for coef in coefs_from_binary]

# Comparing theano_l1 and coefs_1 we can see that we are using the right
# filters in the right order. This first impression is due to easy
# distinction between zero mean and non-zero mean filters. However,
# looking at the details, we see that some things have been reversed somehow.
# Need to check out what is going on.
weights = fetch_overfeat_weights(use_small_network=True)

weights1 = weights[0][0].transpose(0, 1, 2, 3)[:, :, ::-1, ::-1]
biases1 = weights[1][0]
biases2 = weights[1][1]

W1 = theano.shared(weights1, name='W1', borrow=True)
b1 = theano.shared(biases1, name='b1', borrow=True)
input_data = T.tensor4(name='input', dtype='float32')
conv_out = T.nnet.conv2d(input_data, W1, subsample=(4, 4))
conv_plus_bias = conv_out + b1.dimshuffle('x', 0, 'x', 'x')

f_aff = theano.function([input_data], conv_plus_bias)

theano_a1 = [f_aff(coef[np.newaxis].astype(np.float32))
             for coef in coefs_from_binary]

