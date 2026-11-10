"""Classical outlier detectors for Tables 1 and 6 and Figures 15–16."""
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler

METHODS = ('IsolationForest', 'LocalOutlierFactor', 'OneClassSVM', 'EllipticEnvelope')


def flags(X):
    """X (M, D) L2-normalised keys -> {method: bool (M,) outlier flags}"""
    if tuple(map(int, sklearn.__version__.split('.')[:2])) >= (1, 6):
        raise RuntimeError('Use scikit-learn==1.5.2 from requirements.txt for these detectors.')
    Xs = StandardScaler().fit_transform(X)
    return {
        'IsolationForest': IsolationForest(n_estimators=300, max_samples=20, contamination='auto',
                                           random_state=42).fit(Xs).decision_function(Xs) < 0,
        'LocalOutlierFactor': LocalOutlierFactor(n_neighbors=10, contamination='auto').fit_predict(Xs) == -1,
        'OneClassSVM': OneClassSVM(kernel='rbf', gamma='scale', nu=0.2).fit(Xs).decision_function(Xs) < 0,
        'EllipticEnvelope': EllipticEnvelope(contamination=0.2, random_state=42).fit(Xs).decision_function(Xs) < 0,
    }
