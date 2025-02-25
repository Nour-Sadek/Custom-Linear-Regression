import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from custom_linear_regression import CustomLinearRegressionClass

"""Compare the custom linear regression model's performance to scikit-learn's using the Advertising dataset where
the features are the advertising spent on TV, radio, and newspaper (3 features) and the target values are the 
sales of the product (in thousands of units). It is made up of 200 observations."""

# URL for the advertising dataset
url = "https://media.geeksforgeeks.org/wp-content/uploads/20240522145649/advertising.csv"
df = pd.read_csv(url)

# Splitting the dataset
X, y = df.drop(columns=["Sales"]), df["Sales"]
X, y = X.to_numpy(), y.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=36)


# Center the train and test sets
def center_data(X: np.ndarray) -> np.ndarray:
    """Return a modified numpy array where <X> is centered"""
    cols_mean = np.mean(X, axis=0)
    cols_mean_mat = cols_mean * np.ones((X.shape[0], X.shape[1]))
    centered_data = X - cols_mean_mat
    return centered_data


X_train, X_test = center_data(X_train), center_data(X_test)

# Running Scikit-learn's Linear Regression model and predicting the target values of the test dataset
scikit_model = LinearRegression()
scikit_model.fit(X_train, y_train)
scikit_y_predict = scikit_model.predict(X_test)

# Running the custom Linear Regression model and doing predictions
custom_model = CustomLinearRegressionClass(0.001, 2000)
custom_model.fit(X_train, y_train, "SGD")
custom_y_predict = custom_model.predict(X_test)

custom_model.plot_loss()
custom_model.plot_R_squared()

# Plot a scatter plot of actual vs predicted values for the test dataset
plt.figure(figsize=(8, 6))
plt.scatter(y_test, custom_y_predict, color="blue", label="Custom Model")
plt.scatter(y_test, scikit_y_predict, color="red", label="Scikit-learn")
plt.plot(y_test, y_test, color="black", linestyle="--", label="Perfect Fit")

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.legend()
plt.show()

# Residual Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test - custom_y_predict, color="blue", label="Custom Model")
plt.scatter(y_test, y_test - scikit_y_predict, color="red", label="Scikit-learn")
plt.axhline(y=0, color="black", linestyle="--")

plt.xlabel("Actual Values")
plt.ylabel("Residuals (Error between actual and predicted)")
plt.title("Residual Plot")
plt.legend()
plt.show()
