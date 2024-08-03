# my_package/regression_estimators.py

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor

class CustomRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, model, **kwargs):
        """
        Custom Regressor Class.

        Parameters:
        model (object): A scikit-learn compatible regressor model (e.g., LinearRegression, RandomForestRegressor).
        kwargs: Additional keyword arguments for the model.
        """
        self.model = model
        self.kwargs = kwargs

    def fit(self, X, y):
        """
        Fit the model to the training data.

        Parameters:
        X (array-like): Feature matrix.
        y (array-like): Target vector.

        Returns:
        self: Returns the instance itself.
        """
        self.model.set_params(**self.kwargs)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict using the fitted model.

        Parameters:
        X (array-like): Feature matrix.

        Returns:
        array: Predicted values.
        """
        return self.model.predict(X)

    def score(self, X, y):
        """
        Evaluate the model using custom metrics.

        Parameters:
        X (array-like): Feature matrix.
        y (array-like): Target vector.

        Returns:
        dict: A dictionary of metrics.
        """
        y_pred = self.predict(X)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        mrae_val = self.mrae(y, y_pred)
        
        return {
            'MAE': mae,
            'MRAE': mrae_val,
            'R2 Score': r2
        }

    def mrae(self, y_true, y_pred):
        """
        Calculate Mean Relative Absolute Error (MRAE).

        Parameters:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted values.

        Returns:
        float: Mean Relative Absolute Error.
        """
        return np.mean(np.abs((y_true - y_pred) / np.mean(y_true)))

def run_regressors(X, y):
    regressors = {
        "Linear Regression": CustomRegressor(model=LinearRegression()),
        "K-Nearest Neighbors": CustomRegressor(model=KNeighborsRegressor()),
        "Random Forest": CustomRegressor(model=RandomForestRegressor()),
        "Gradient Boosting": CustomRegressor(model=GradientBoostingRegressor()),
        "AdaBoost": CustomRegressor(model=AdaBoostRegressor()),
        "Multi-Layer Perceptron": CustomRegressor(model=MLPRegressor(max_iter=1000))
    }

    results = {}
    for name, reg in regressors.items():
        reg.fit(X, y)
        scores = reg.score(X, y)
        results[name] = scores
        print(f"{name} - MAE: {scores['MAE']:.4f}, MRAE: {scores['MRAE']:.4f}, R2 Score: {scores['R2 Score']:.4f}")

    return results
