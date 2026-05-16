from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class L2Regression:
    def __init__(self, alpha=1.0, fit_intercept=True, normalize=True):
        self.alpha = alpha
        self.normalize = normalize
        steps = []
        if normalize:
            steps.append(("scaler", StandardScaler()))
        steps.append(("ridge", Ridge(alpha=alpha, fit_intercept=fit_intercept)))
        self.model = Pipeline(steps)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
