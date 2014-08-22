from sklearn_theano.datasets import fetch_asirra
from sklearn_theano.feature_extraction import OverfeatTransformer
import matplotlib.pyplot as plt
import time
asirra = fetch_asirra()
X = asirra.images.astype('float32')
X = X[0:5]
y = asirra.target
all_times = []
for i in range(0, 15):
    tf = OverfeatTransformer(output_layers=[i])
    t0 = time.time()
    X_tf = tf.transform(X)
    print("Shape of layer %i output" % i)
    print(X_tf.shape)
    t_o = time.time() - t0
    all_times.append(t_o)
    print("Time for layer %i" % i, t_o)
    print()
plt.plot(all_times, marker='o')
plt.title("Runtime for input to layer X")
plt.xlabel("Layer number")
plt.ylabel("Time (seconds)")
plt.show()
