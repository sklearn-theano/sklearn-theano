import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn_theano.datasets import fetch_mnist_generated

X_tf = fetch_mnist_generated(n_samples=1600, random_state=1999)

# plotting based on
# http://stackoverflow.com/questions/4098131/matplotlib-update-a-plot
num_updates = len(X_tf) // 16
f, axarr = plt.subplots(4, 4)
objarr = np.empty_like(axarr)
for n, ax in enumerate(axarr.flat):
    objarr.flat[n] = ax.matshow(X_tf[n].reshape(28, 28), cmap='gray')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
plt.show(block=False)

for i in range(num_updates):
    for n, obj in enumerate(objarr.flat):
        obj.set_data(X_tf[i * len(objarr.flat) + n].reshape(28, 28))
    plt.draw()
    time.sleep(.08)
    print("Iteration %i" % i)
plt.show()
