import numpy as np
from theano import tensor as T
import theano
from sklearn.datasets import make_classification
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

rng = np.random.RandomState(1999)
X, y = make_classification(n_samples=400, n_features=25, n_informative=10,
                           n_classes=2, n_clusters_per_class=2,
                           random_state=1999)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8)
n_samples, n_features = X_train.shape
x = T.matrix('x')
y = T.vector('y')
w = theano.shared(rng.randn(n_features), name='w')
b = theano.shared(0., name='b')

print("Initial model")
print(w.get_value(), b.get_value())

learning_rate = 0.01
reg = .1
n_iter = 10000
prob = 1 / (1 + T.exp(-T.dot(x, w) - b))
pred = prob > 0.5
loss = -y * T.log(prob) - (1 - y) * T.log(1 - prob)

# l2
# penalty = reg * (w ** 2).sum()

# l1
penalty = reg * abs(w).sum()

# l0
# penalty = reg * T.neq(w, 0).sum()
cost = loss.mean() + penalty
gw, gb = T.grad(cost, [w, b])

train = theano.function(inputs=[x, y], outputs=[pred, loss],
                        updates=((w, w - learning_rate * gw),
                                 (b, b - learning_rate * gb)))
predict = theano.function(inputs=[x], outputs=pred)

for i in range(n_iter):
    pred, err = train(X_train, y_train)

print("Final model:")
print(w.get_value(), b.get_value())
print("Report:")
y_pred = predict(X_test)
report = classification_report(y_test, y_pred)
print(report)
