import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class L1Regression:
    def __init__(self, alpha=1.0, fit_intercept=True, max_iter=10000, normalize=True):
        self.alpha = alpha
        self.normalize = normalize
        steps = []
        if normalize:
            steps.append(("scaler", StandardScaler()))
        steps.append(("lasso", Lasso(alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter)))
        self.model = Pipeline(steps)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
