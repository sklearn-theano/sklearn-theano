from sklearn_theano.datasets import fetch_asirra
from sklearn_theano.feature_extraction import OverfeatTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score
import time

asirra = fetch_asirra()
X = asirra.images
y = asirra.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=.8, random_state=1999)
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
