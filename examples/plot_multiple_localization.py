import matplotlib.pyplot as plt
from sklearn_theano.datasets import load_sample_image
from sklearn_theano.feature_extraction import OverfeatLocalizer
from sklearn_theano.feature_extraction import get_all_overfeat_labels

X = load_sample_image("cat_and_dog.jpg")
dog_label = [label for label in get_all_overfeat_labels()
             if 'dog' in label][0]
cat_label = [label for label in get_all_overfeat_labels()
             if 'cat' in label][0]
clf = OverfeatLocalizer(match_strings=[dog_label, cat_label])
points = clf.predict(X.astype('float32'))
dog_points = points[0]
cat_points = points[1]
plt.imshow(X)
ax = plt.gca()
ax.autoscale(enable=False)
plt.scatter(dog_points[:, 0], dog_points[:, 1], color='darkred', label='dog')
plt.scatter(cat_points[:, 0], cat_points[:, 1], color='steelblue', label='cat')
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
plt.legend(loc='lower right')
plt.show()
