import numpy as np


class CustomLinearRegressionClass:

    """A custom Linear Regression class that does linear regression on a numpy 2d array of the shape
    n_observations x n_features.

    === Private Attributes ===
    _learning_rate: a hyperparameter that controls how much the model updates the weights during each training cycle
    _iterations: the number of times the algorithm updates the weights parameters
    _weights: set of coefficients that quantify the impact for each dependent variable on the independent variable
    _bias: intercept terms

    """

    _learning_rate: float
    _iterations: int
    _weights: np.ndarray
    _bias: float

    def __init__(self, learning_rate: float, iterations: int) -> None:
        """Initialize a new CustomLinearRegressionClass with _learning_rate <learning_rate> and
        _iterations <iterations>"""
        self._learning_rate = learning_rate
        self._iterations = iterations
        self._weights = None
        self._bias = None

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @property
    def iterations(self) -> int:
        return self._iterations

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @property
    def bias(self) -> float:
        return self._bias
