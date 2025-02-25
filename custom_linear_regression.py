import numpy as np
import matplotlib.pyplot as plt


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
    _history: dict

    def __init__(self, learning_rate: float, iterations: int) -> None:
        """Initialize a new CustomLinearRegressionClass with _learning_rate <learning_rate> and
        _iterations <iterations>"""

        if learning_rate <= 0:
            raise ValueError("learning rate should be greater than 0")
        if int(iterations) != iterations or iterations <= 0:
            raise ValueError("iterations should be a positive non-zero integer")

        self._learning_rate = learning_rate
        self._iterations = iterations
        self._weights = None
        self._bias = None
        self._history = {"loss": [], "R_squared": []}

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

    @property
    def history(self) -> dict:
        return self._history

    def fit(self, X: np.ndarray, y: np.ndarray, method: str = "GD", batch_size: int = 1) -> None:
        """Update the _weights and _bias terms trained on the training set <X> of shape n_observations x n_features and
        the target variables for each observation <y> using Mean Squared Error (MSE) as the cost function and <method>
        as the optimization algorithm.

        If <print_loss> is true (default is false), then the loss value would be printed at each iteration in this format:
        Loss at iteration <number of current iteration>: <value_of_MSE>

        <method> can take two values:
        GD: Gradient Descent (default)
        SGD: Stochastic Gradient Descent

        If <method> is "SGD", then the size of the batches used can be set using the <batch_size> parameter. The default
        for it is 1. batch_size should be a non-zero positive integer that is less than n_observations of <X>.
        """

        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of observations in matrix X doesn't match number of target values in array y")

        n_observations, n_features = X.shape
        self._weights = np.zeros(n_features)
        self._bias = 0

        for iteration in range(self._iterations):

            if method == "GD":

                y_predicted = X @ self._weights + self._bias

                # Determine the derivative of the MSE with respect to the weights
                # and update the <self._weights> parameters
                d_weights = - (2 / n_observations) * (X.T @ (y - y_predicted))
                self._weights = self._weights - self._learning_rate * d_weights

                # Determine the derivative of the MSE with respect to the bias term
                # and update the <self._bias> parameter
                d_bias = - (2 / n_observations) * np.sum(y - y_predicted)
                self._bias = self._bias - self._learning_rate * d_bias

            elif method == "SGD":

                if not 0 < batch_size <= n_observations:
                    raise ValueError("When  using stochastic gradient descent, batch size should be greater than 0 and "
                                     "less than or equal to the number of observations ")

                # Shuffle <X> and <y> in unison
                np.random.seed(36)
                random_shuffle = np.random.permutation(len(y))
                shuffled_X = X[random_shuffle]
                shuffled_y = y[random_shuffle]

                # Iterate through the shuffled datasets in batches of size <batch_size>
                for start in range(0, n_observations, batch_size):
                    stop = start + batch_size
                    X_batch, y_batch = shuffled_X[start:stop], shuffled_y[start:stop]
                    y_batch_predicted = X_batch @ self._weights + self._bias

                    # Determine the derivative of the MSE with respect to the weights
                    # and update the <self._weights> parameters based on this batch
                    d_weights = - (2 / n_observations) * (X_batch.T @ (y_batch - y_batch_predicted))
                    self._weights = self._weights - self._learning_rate * d_weights

                    # Determine the derivative of the MSE with respect to the bias term
                    # and update the <self._bias> parameter based on this batch
                    d_bias = - (2 / n_observations) * np.sum(y_batch - y_batch_predicted)
                    self._bias = self._bias - self._learning_rate * d_bias

            else:

                raise ValueError("method can only take two values; GD (for Gradient Descent) or "
                                 "SGD (for Stochastic Gradient Descent)")

            # Calculate the predicted y values after every training iteration
            y_predicted = X @ self._weights + self._bias

            # Save the value of the loss for each iteration
            mse_loss = (1 / n_observations) * np.sum((y - y_predicted) ** 2)
            self._history["loss"].append(mse_loss)
            # Save the value of the R squared for each iteration
            res_sum_squares = np.sum((y - y_predicted) ** 2)
            res_total = np.sum((y - np.mean(y)) ** 2)
            res = 1 - (res_sum_squares / res_total)
            self._history["R_squared"].append(res)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the predicted values of the dependent (target) variables based on the test set <X> and the fit
        parameters self._weights and self._bias

        Can only be called after calling the self.fit method"""

        y_predicted = X @ self._weights + self._bias
        return y_predicted

    def plot_loss(self) -> None:
        plt.plot(self._history["loss"])
        plt.title("Model MSE loss")
        plt.ylabel("MSE loss")
        plt.xlabel("Iteration")
        plt.legend(["train"], loc="upper left")
        plt.show()

    def plot_R_squared(self) -> None:
        plt.plot(self._history["R_squared"])
        plt.axhline(y=1, color="black", linestyle="--")
        plt.title("Model R-Squared")
        plt.ylabel("R-Squared")
        plt.xlabel("Iteration")
        plt.legend(["train"], loc="upper left")
        plt.show()
