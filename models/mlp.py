from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class MLP:
    def __init__(
        self,
        hidden_layer_sizes=(64, 32),
        activation="relu",
        learning_rate_init=1e-3,
        max_iter=500,
        random_state=None,
        normalize=True,
    ):
        steps = []
        if normalize:
            steps.append(("scaler", StandardScaler()))
        steps.append((
            "mlp",
            MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                learning_rate_init=learning_rate_init,
                max_iter=max_iter,
                random_state=random_state,
            ),
        ))
        self.model = Pipeline(steps)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
