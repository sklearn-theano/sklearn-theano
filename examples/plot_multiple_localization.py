import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn_theano.datasets import load_sample_image
from sklearn_theano.feature_extraction import OverfeatLocalizer
from sklearn.mixture import GMM


def convert_gmm_to_box(gmm, color, alpha):
    midpoint = gmm.means_
    std = 3 * np.sqrt(clf.covars_)
    width = std[:, 0]
    height = std[:, 1]
    upper_left_point = (midpoint[:, 0] - width // 2,
                        midpoint[:, 1] - height // 2)
    return Rectangle(upper_left_point, width, height, ec=color,
                     fc=color, alpha=alpha)

X = load_sample_image("cat_and_dog.jpg")
dog_label = 'dog.n.01'
cat_label = 'cat.n.01'
clf = OverfeatLocalizer(top_n=1,
                        match_strings=[dog_label, cat_label])
points = clf.predict(X.astype('float32'))
dog_points = points[0]
cat_points = points[1]

plt.imshow(X)
plt.title("Cat and dog localization")
ax = plt.gca()
ax.autoscale(enable=False)
clf = GMM()
clf.fit(dog_points)
dog_box = convert_gmm_to_box(clf, "darkred", .6)
clf.fit(cat_points)
cat_box = convert_gmm_to_box(clf, "steelblue", .6)
ax.add_patch(dog_box)
ax.add_patch(cat_box)
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
plt.show()
