import numpy as np
from sklearn_theano.feature_extraction import GoogLeNetClassifier
from sklearn_theano.datasets import load_sample_image

X = load_sample_image("sloth_closeup.jpg")
top_n_classes = 5
goog_clf = GoogLeNetClassifier(top_n=top_n_classes)
goog_preds = goog_clf.predict(X)[0]
goog_probs = goog_clf.predict_proba(X)[0]
# Want the sorted from greatest probability to least
sort_indices = np.argsort(goog_probs)[::-1]

for n, (pred, prob) in enumerate(zip(goog_preds[sort_indices],
                                     goog_probs[sort_indices])):
    print("Class prediction (probability): %s (%.4f)" % (pred, prob))
