import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn_theano.datasets import load_sample_image
from sklearn_theano.feature_extraction import OverfeatLocalizer
from sklearn_theano.feature_extraction import OverfeatTransformer
from sklearn_theano.feature_extraction import get_all_overfeat_labels


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
X_tf = clf.transform(X[None].astype('float32'))
x_points = np.linspace(0, X.shape[1] - 231, X_tf[0].shape[3])
y_points = np.linspace(0, X.shape[0] - 231, X_tf[0].shape[2])
xx, yy = np.meshgrid(x_points, y_points)
for x, y in zip(xx.flat, yy.flat):
    axarr[0, 2].add_patch(Rectangle((x, y), 231, 231, fc='yellow', ec='black',
                          alpha=.4))

# Get all points with sloth in the top 5 labels
sloth_label = [label for label in get_all_overfeat_labels()
               if 'three-toed sloth' in label][0]
clf = OverfeatLocalizer(match_strings=[sloth_label])
sloth_points = clf.predict(X.astype('float32'))[0]
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
