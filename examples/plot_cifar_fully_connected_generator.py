"""
===========================================
Generative networks for random CIFAR images
===========================================

This demo of a CIFAR generator is based on the work of
I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair,
A. Courville, Y.Bengio. *Generative Adversarial Networks*, June 2014.

The generators trained as part of the published experiment have been wrapped in
sklearn-theano, and can easily be used to fetch an arbitrary number of plausible
CIFAR images.

Additionally, this example also shows how to make an automatically updating plot
with the 'TkAgg' backend to matplotlib.

"""
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn_theano.datasets import fetch_cifar_fully_connected_generated

X = fetch_cifar_fully_connected_generated(n_samples=1600, random_state=1999)


# plotting based on
# http://stackoverflow.com/questions/4098131/matplotlib-update-a-plot
num_updates = len(X) // 16
f, axarr = plt.subplots(4, 4)
objarr = np.empty_like(axarr)
for n, ax in enumerate(axarr.flat):
    objarr.flat[n] = ax.imshow(X[n], interpolation='nearest')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
plt.show(block=False)

for i in range(num_updates):
    for n, obj in enumerate(objarr.flat):
        obj.set_data(X[i * len(objarr.flat) + n])
    plt.draw()
    time.sleep(.08)
    print("Iteration %i" % i)
plt.show()
