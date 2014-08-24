import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn_theano.datasets import load_sample_image
from sklearn_theano.feature_extraction import OverfeatLocalizer


def convert_points_to_box(points, color, alpha):
    upper_left_point = (points[:, 0].min(), points[:, 1].min())
    width = points[:, 0].max() - points[:, 0].min()
    height = points[:, 1].max() - points[:, 1].min()
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
cat_box = convert_points_to_box(cat_points, 'steelblue', .4)
dog_box = convert_points_to_box(dog_points, 'darkred', .4)

plt.imshow(X)
plt.title("Cat and dog localization")
ax = plt.gca()
ax.autoscale(enable=False)
ax.add_patch(cat_box)
ax.add_patch(dog_box)
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
plt.show()
