# Linear Regression in Machine Learning.

Linear Regression is a fundamental supervised learning algorithm used for predicting a continuous outcome variable (also called the dependent variable) based on one or more predictor variables (independent variables). It assumes a linear relationship between the input features and the target variable.

## Simple Linear Regression

### Model Representation

The simplest form of Linear Regression is the **Simple Linear Regression**, where there is only one predictor variable. The model can be represented as:

\[ y = mx + b \]

- \( y \): Dependent variable
- \( x \): Independent variable
- \( m \): Slope of the line
- \( b \): Y-intercept

### Objective

The objective of Simple Linear Regression is to find the best-fitting line that minimizes the sum of squared differences between the observed and predicted values.

### Training

Training involves finding the optimal values for \( m \) and \( b \) to minimize the cost function, often using techniques like the least squares method.

## Multiple Linear Regression

### Model Representation

For scenarios with multiple predictor variables, we use **Multiple Linear Regression**:

\[ y = b_0 + b_1x_1 + b_2x_2 + \ldots + b_nx_n \]

- \( y \): Dependent variable
- \( x_1, x_2, \ldots, x_n \): Independent variables
- \( b_0 \): Y-intercept
- \( b_1, b_2, \ldots, b_n \): Coefficients for the respective independent variables

### Objective

The objective remains the same: minimize the sum of squared differences between the observed and predicted values.

### Training

Training involves finding the optimal values for \( b_0, b_1, \ldots, b_n \) to minimize the cost function.

## Polynomial Regression

In **Polynomial Regression**, the relationship between the independent and dependent variables is modeled as an \(n\)-degree polynomial. The model can be expressed as:

\[ y = b_0 + b_1x + b_2x^2 + \ldots + b_nx^n \]

Polynomial Regression is an extension of Linear Regression and is useful when the relationship between variables is nonlinear.

## Multivariate Regression

**Multivariate Regression** involves multiple predictor variables, similar to Multiple Linear Regression. However, it considers relationships between multiple independent variables and the dependent variable simultaneously.

The model can be represented as:

\[ y = b_0 + b_1x_1 + b_2x_2 + \ldots + b_nx_n \]

- \( y \): Dependent variable
- \( x_1, x_2, \ldots, x_n \): Independent variables
- \( b_0 \): Y-intercept
- \( b_1, b_2, \ldots, b_n \): Coefficients for the respective independent variables

### Objective

The objective remains the same: minimize the sum of squared differences between the observed and predicted values.

### Training

Training involves finding the optimal values for \( b_0, b_1, \ldots, b_n \) to minimize the cost function.

## Evaluation

### Metrics

Common metrics for evaluating Linear Regression models include:

#### Mean Squared Error (MSE)

\[ MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \]

#### Root Mean Squared Error (RMSE)

\[ RMSE = \sqrt{MSE} \]

#### Mean Absolute Error (MAE)

\[ MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| \]

#### \( R^2 \) (coefficient of determination)

\[ R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2} \]

### Loss Function

The cost function, often used during training, can be represented as:

```python
# Our cost function
def cost_function(m, b, x, y):
    totalError = 0
    for i in range(0, len(x)):
        totalError += (y[i] - (m * x[i] + b))**2
    return totalError / float(len(x))
