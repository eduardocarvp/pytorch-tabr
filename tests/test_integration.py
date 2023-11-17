import numpy as np
from pytorch_tabr import TabRClassifier, TabRRegressor

X_train = np.random.rand(1000, 10)
y_train_classification = np.array([int(x[0] > 0.5) for x in X_train])
y_train_regression = np.array([float(x[0] * 2) for x in X_train])

X_valid = np.random.rand(1000, 10)
y_valid_classification = np.array([int(x[0] > 0.5) for x in X_valid])
y_valid_regression = np.array([float(x[0] * 2) for x in X_valid])

X_test = np.random.rand(1000, 10)
y_test_classification = np.array([int(x[0] > 0.5) for x in X_test])
y_test_regression = np.array([float(x[0] * 2) for x in X_test])


def test_integration_classification():
    clf = TabRClassifier()
    clf.fit(
        X_train, y_train_classification,
        eval_set=[(X_valid, y_valid_classification)],
        max_epochs=5,
        patience=2
    )
    clf.predict_proba(X_test)


def test_overfit_classification():
    from sklearn.metrics import roc_auc_score
    clf = TabRClassifier(
        encoder_n_blocks=0,
        predictor_n_blocks=1,
    )
    clf.fit(
        X_train, y_train_classification,
        eval_set=[(X_valid, y_valid_classification)],
        max_epochs=20,
        patience=5,
        batch_size=64,
    )
    preds = clf.predict_proba(X_train)
    assert (1 - roc_auc_score(y_train_classification, preds[:, 1])) <= 0.01


def test_integration_regression():
    clf = TabRRegressor()
    clf.fit(
        X_train, y_train_regression.reshape(-1, 1),
        eval_set=[(X_valid, y_valid_regression.reshape(-1, 1))],
        max_epochs=2,
        patience=2
    )
    clf.predict_proba(X_test)


def test_overfit_regression():
    from sklearn.metrics import mean_squared_error
    clf = TabRRegressor(
        encoder_n_blocks=0,
        predictor_n_blocks=1,
    )
    clf.fit(
        X_train, y_train_regression.reshape(-1, 1),
        eval_set=[(X_valid, y_valid_regression.reshape(-1, 1))],
        max_epochs=100,
        patience=20,
        batch_size=64,
    )
    preds = clf.predict(X_train)
    assert mean_squared_error(y_train_regression, preds.flatten()) <= 0.01
