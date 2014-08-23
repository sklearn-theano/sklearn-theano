import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn_theano.datasets import load_sample_image
from sklearn_theano.feature_extraction import OverfeatLocalizer
from sklearn_theano.feature_extraction import get_all_overfeat_labels

X = load_sample_image("sloth.jpg")
sloth_label = [label for label in get_all_overfeat_labels()
               if 'three-toed sloth' in label][0]
clf = OverfeatLocalizer(match_strings=[sloth_label])

sloth_points = clf.predict(X.astype('float32'))[0]
upper_left_point = (sloth_points[:, 0].min(), sloth_points[:, 1].min())
sloth_width = sloth_points[:, 0].max() - sloth_points[:, 0].min()
sloth_height = sloth_points[:, 1].max() - sloth_points[:, 1].min()

plt.imshow(X)
plt.title('Sloth detection')
ax = plt.gca()
ax.autoscale(enable=False)
sloth_box = Rectangle(upper_left_point, sloth_width, sloth_height, ec='orange',
                      fc='orange', alpha=.4)
ax.add_patch(sloth_box)
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
plt.show()
