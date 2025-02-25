# Custom-Linear-Regression

This repository compares a custom Linear Regression iterative implementation to Scikit-learn's non-iterative Linear Regression.
It allows using either Gradient Descent or Stochastic Gradient Descent for optimization of the weights.

The comparison is performed on the Advertising dataset, which is made up of 3 features and 200 observations and can be found here: [Advertising Dataset](https://media.geeksforgeeks.org/wp-content/uploads/20240522145649/advertising.csv)

# Usage

Run the evaluation script:
    
    python evaluate.py

Expected outputs:
- Plot of the value of the MSE cost function over each iteration
- Plot of the R-squared value over each iteration
- Scatter plot of actual versus predicted values of the target variable compared between the custom model and Scikit-learn's
- Scatter plot of the residuals (difference between the actual and predicted target values) plotted against the actual values compared between the custom and Scikit-learn's

# Results

- The MSE loss function converges to 0 as the number of iterations increases using Stochastic Gradient Descent as the optimization algorithm:
    ![Image](https://github.com/user-attachments/assets/94e7fc3a-d9a5-47cb-892f-6b4808da801f)


- The R-Squared function converges to 1 as the number of iterations increases
    ![Image](https://github.com/user-attachments/assets/2cd031c0-98b8-4104-9ac7-01674e0c18e2)


- The custom Linear Regression model that uses an iterative training process performs similarly to Scikit-learn's Linear Regression that uses the Normal Equation:
    ![Image](https://github.com/user-attachments/assets/5479b6eb-5d2b-4aad-80cb-06c0ed979f45)


- The residuals for both the custom Linear Regression and Scikit-learn's scatter randomly around zero.
    ![Image](https://github.com/user-attachments/assets/f5ff123e-7510-4922-975e-436bf785cb41)

# Repository Structure

This repository contains:

    custom_linear_regression.py: Implementation of the CustomLinearRegressionClass
    evaluate.py: Main script for performing the comparisons on the Advertising Dataset and generating plots
    requirements.txt: List of required Python packages

Python 3.12 version was used