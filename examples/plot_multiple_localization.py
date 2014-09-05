"""
=======================================
Localizing multiple objects in an image
=======================================

Extending on the ``plot_localization_tutorial`` example, there are more advanced
ways to search for objects, and draw bounding boxes.

Rather than using the full class label, we can also use the wordnet hierarchy to
describe groups of labels which apply. In this case the high level labels
'cat' and 'dog' are used to search through the image. By specifying 'cat.n.01',
the OverfeatLocalizer will use class tags which fall into the group of
'cat.n.01' based on the WordNet hierarchy. Dogs work the same same way, using
'dog.n.01'.

By allowing specification of higher level groupings, it is easier
to search for a variety of subjects, but may lead to confusion when drawing
bounding boxes if the subjects both fall under the same WordNet group, but are
not the same class. Imagine an image of two different species cats -
in this case the exact class labels may be better at creating bounding boxes.

It is also beneficial to use scikit-learn's Gaussian Mixture Models to
better estimate bounding boxes without the influence of outliers. In this case,
one Gaussian will be placed to try and describe all of the matched points. Using
a width and height bound of 3 standard deviations, it is possible to draw
bounding boxes which are more robust to spurious points than the simple
method used in ``plot_single_localization``.

"""
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
points = clf.predict(X)
dog_points = points[0]
cat_points = points[1]

plt.imshow(X)
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
