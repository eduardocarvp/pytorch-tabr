import numpy as np
from pytorch_tabr import TabRClassifier, TabRRegressor

X_train = np.random.rand(1000, 10)
y_train_classification = np.array([int(x[0] > 0.5) for x in X_train])
y_train_regression = np.array([x[0] * 2 for x in X_train]).reshape(-1, 1)

X_valid = np.random.rand(1000, 10)
y_valid_classification = np.array([int(x[0] > 0.5) for x in X_valid])
y_valid_regression = np.array([x[0] * 2 for x in X_valid]).reshape(-1, 1)

X_test = np.random.rand(1000, 10)
y_test_classification = np.array([int(x[0] > 0.5) for x in X_test])
y_test_regression = np.array([x[0] * 2 for x in X_test]).reshape(-1, 1)


def test_integration_classification():
    clf = TabRClassifier()
    clf.fit(
        X_train, y_train_classification,
        eval_set=[(X_valid, y_valid_classification)]
    )
    clf.predict_proba(X_test)


def test_integration_regression():
    clf = TabRRegressor()
    clf.fit(
        X_train, y_train_regression,
        eval_set=[(X_valid, y_valid_regression)]
    )
    clf.predict(X_test)