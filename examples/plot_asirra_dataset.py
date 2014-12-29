"""
===============================================
Asirra dataset classification using transformer
===============================================

This example shows a basic use of the OverfeatTransformer in a scikit-learn
pipeline in order to do classification of natural images.

In this case, the images come from the Asirra dataset functionality built into
sklearn-theano. Plots show one example of each class (cats and dogs).

"""
print(__doc__)

from sklearn_theano.datasets import fetch_asirra
from sklearn_theano.feature_extraction import OverfeatTransformer
from sklearn_theano.utils import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import time

asirra = fetch_asirra(image_count=20)
X = asirra.images.astype('float32')
y = asirra.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=.6, random_state=1999)
tf = OverfeatTransformer(output_layers=[-3])
clf = LogisticRegression()
pipe = make_pipeline(tf, clf)
t0 = time.time()
pipe.fit(X_train, y_train)
print("Total transform time")
print("====================")
print(time.time() - t0)
print()
y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred))
print()
print("Accuracy score")
print("==============")
print(accuracy_score(y_test, y_pred))
f, axarr = plt.subplots(1, 2)
axarr[0].imshow(asirra.images[asirra.target == 0][-1])
axarr[0].axis('off')
axarr[1].imshow(asirra.images[asirra.target == 1][0])
axarr[1].axis('off')
plt.show()
