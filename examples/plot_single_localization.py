import matplotlib.pyplot as plt
from sklearn_theano.datasets import load_sample_images
from sklearn_theano.feature_extraction import OverfeatLocalizer
from sklearn_theano.feature_extraction import get_all_overfeat_labels

s = load_sample_images()
X = s.images[0]
sloth_label = [label for label in get_all_overfeat_labels()
               if 'three-toed sloth' in label][0]
clf = OverfeatLocalizer(match_strings=[sloth_label])
points = clf.predict(X.astype('float32'))
sloth_points = points[0]
plt.imshow(X)
ax = plt.gca()
ax.autoscale(enable=False)
plt.scatter(sloth_points[:, 0], sloth_points[:, 1], color='darkred',
            label='sloth')
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
plt.legend()
plt.show()
