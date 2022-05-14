from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model
    # f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) +
    # eps for eps Gaussian noise
    # and split into training- and testing portions
    X = np.linspace(-2.5, 2.5, n_samples)
    y = (X + 3) * (X + 2) * (X + 1) * (X - 1) * (X - 2)
    y_with_noise = y + np.random.normal(0, noise, y.shape)
    X_train, y_train, X_test, y_test = split_train_test(X, y_with_noise)

    X_train = X_train.values.reshape(1, -1)
    y_train = y_train.values.reshape(1, -1)
    X_test = X_test.values.reshape(1, -1)
    y_test = y_test.values.reshape(1, -1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X, y=y, mode='lines',
                             name='True Polynomial Function',
                             line=dict(color='black')))
    fig.add_trace(go.Scatter(x=X_train[0], y=y_train[0], mode='markers',
                             name='Training data', marker=dict(
            color='lightsalmon', symbol='circle')))

    fig.add_trace(go.Scatter(x=X_test[0], y=y_test[0], mode='markers',
                             name='Testing data', marker=dict(
            color='cornflowerblue', symbol='diamond')))

    fig.update_layout(title='Training and testing data', title_x=0.5)
    # fig.show()


    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    degree = pd.Series(np.arange(0, 11))
    res = degree.apply(lambda k: cross_validate(
        PolynomialFitting(k), X_train, y_train,
        scoring=mean_square_error, cv=5))

    train_score = res.apply(lambda r: r[0])
    validation_score = res.apply(lambda r: r[1])

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=degree, y=train_score, mode='lines',
                              name='Training error'))
    fig2.add_trace(go.Scatter(x=degree, y=validation_score, mode='lines',
                              name='Validation error'))
    fig2.update_layout(title='Cross-validation for polynomial fitting',
                       title_x=0.5)
    # fig2.show()

    # Question 3 - Using best value of k,
    # fit a k-degree polynomial model and report test error

    k = int(validation_score.idxmin())
    print('Best k:', k)
    model = PolynomialFitting(k)
    model.fit(X_train, y_train.transpose())
    loss = model.loss(X_test, y_test.transpose())
    print(f'Test error: {round(loss, 2)}')


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    raise NotImplementedError()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    raise NotImplementedError()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    raise NotImplementedError()
