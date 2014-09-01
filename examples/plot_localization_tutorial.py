"""
=====================================
Localizing an object in a large image
=====================================

Convolutional neural networks can also be used to localize an object in a large
image. This example will show the basic steps taken to find objects in images
with convolutional neural networks, using the OverfeatTransformer and
OverfeatLocalizer classes.

Step 1: Input
=============

The first image in the first row shows the image input to the OverfeatLocalizer.

Step 2: Process windows over input
==================================

The second image in the first row shows one processing window for the image.
This box will be used as input to the OverfeatLocalizer, which internally gets
processed and retrieves the top_n classes for that window out of 1000 possible
classes.

The last image in the first row shows *all* processing windows - each one of
these windows has the entire convolutional neural network applied to it!

Step 3: Identify matching points
================================

After processing the entire image, there are (n_X_windows, n_Y_windows, top_n)
classification results. By finding only the points which match the desired class
label, there is an approximate "detection region" for that object, formed by the
matching points. These are plotted in the first image of the second row.

Step 4: Draw bounding region
============================

There are a variety of ways to draw a bounding region given a set of points, but
one of the simplest is to take the extreme values (in this case min and max for
both X and y) and draw a box. This method is very susceptible to outliers and
spurious classifications, but works fine for this example.

Ultimately bounding regions are a form of density estimation, which means a wide
variety of scikit-learn estimators can be used. Examples using scikit-learn's
GMM class can be found in ``plot_multiple_localization``.

"""
print(__doc__)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn_theano.datasets import load_sample_image
from sklearn_theano.feature_extraction import OverfeatLocalizer
from sklearn_theano.feature_extraction import OverfeatTransformer


def convert_points_to_box(points, color, alpha):
    upper_left_point = (points[:, 0].min(), points[:, 1].min())
    width = points[:, 0].max() - points[:, 0].min()
    height = points[:, 1].max() - points[:, 1].min()
    return Rectangle(upper_left_point, width, height, ec=color,
                     fc=color, alpha=alpha)

# Show the original image
f, axarr = plt.subplots(2, 3)
X = load_sample_image("sloth.jpg")
axarr[0, 0].imshow(X)
axarr[0, 0].axis('off')

# Show a single box
axarr[0, 1].imshow(X)
axarr[0, 1].axis('off')
r = Rectangle((0, 0), 231, 231, fc='yellow', ec='black', alpha=.8)
axarr[0, 1].add_patch(r)

# Show all the boxes being processed
axarr[0, 2].imshow(X)
axarr[0, 2].axis('off')
clf = OverfeatTransformer(force_reshape=False)
X_tf = clf.transform(X)
x_points = np.linspace(0, X.shape[1] - 231, X_tf[0].shape[3])
y_points = np.linspace(0, X.shape[0] - 231, X_tf[0].shape[2])
xx, yy = np.meshgrid(x_points, y_points)
for x, y in zip(xx.flat, yy.flat):
    axarr[0, 2].add_patch(Rectangle((x, y), 231, 231, fc='yellow', ec='black',
                          alpha=.4))

# Get all points with sloth in the top 5 labels
sloth_label = "three-toed sloth, ai, Bradypus tridactylus"
clf = OverfeatLocalizer(match_strings=[sloth_label])
sloth_points = clf.predict(X)[0]
axarr[1, 0].imshow(X)
axarr[1, 0].axis('off')
axarr[1, 0].autoscale(enable=False)
axarr[1, 0].scatter(sloth_points[:, 0], sloth_points[:, 1], color='orange',
                    s=50)

# Final localization box
sloth_box = convert_points_to_box(sloth_points, 'orange', .6)
axarr[1, 1].imshow(X)
axarr[1, 1].autoscale(enable=False)
axarr[1, 1].add_patch(sloth_box)
axarr[1, 1].axis('off')

# Remove the unused box
axarr[1, 2].axis('off')

plt.show()
