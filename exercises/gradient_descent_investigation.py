import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test

import plotly.graph_objects as go
import plotly.express as px

from sklearn.metrics import roc_curve, auc

from IMLearn.model_selection import cross_validate
from IMLearn.metrics.loss_functions import misclassification_error

MAX_ITER = 20000
LEARNING_RATE = 1e-4
LAMBDAS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
FOLDS = 5


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange,
                                       density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1],
                                 mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[
    Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """

    values_array = []
    weights_array = []

    def callback(model, weights, val, grad, t, eta,
                 delta):
        values_array.append(val)
        weights_array.append(weights)

    return callback, values_array, weights_array


def compare_fixed_learning_rates(
        init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
        etas: Tuple[float] = (1, .1, .01, .001)):
    max_iter = 1000

    # loop over the modules:
    for name, penalty_module in {"L1": L1, "L2": L2}.items():
        losses = pd.DataFrame(columns=etas)

        # loop over the learning rates:
        for eta in etas:
            # initialize module's weights:
            module = penalty_module(weights=init.copy())

            # initialize the fixed learning rate class:
            learning_rate = FixedLR(eta)

            # get callback function for recording the state of the gradient
            # descent algorithm:
            callback, values, weights = get_gd_state_recorder_callback()

            # initialize the gradient descent algorithm:
            GD = GradientDescent(learning_rate=learning_rate,
                                 callback=callback, max_iter=max_iter)

            GD.fit(module, np.nan, np.nan)

            if len(values) < max_iter:
                values = values + ([np.nan] * (max_iter - len(values)))

            losses[eta] = values

            fig = plot_descent_path(module=penalty_module,
                                    descent_path=np.array(weights),
                                    title=f"{name}: learning rate = {eta}")
            fig.show()

        val_fig = px.line(losses)
        val_fig.update_layout(title=f"{name}: learning rate comparison",
                              title_x=0.5,
                              legend_title="etas:", )
        val_fig.show()


def compare_exponential_decay_rates(
        init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
        eta: float = .1,
        gammas: Tuple[float] = (.9, .95, .99, 1)):
    max_iter = 1000

    # loop over the modules:
    for name, base_module in {"L1": L1, "L2": L2}.items():
        losses = pd.DataFrame(columns=gammas)

        # loop over the learning rates:
        for gamma in gammas:
            # initialize module's weights:
            module = base_module(weights=init.copy())

            # initialize the fixed learning rate class:
            learning_rate = ExponentialLR(eta, gamma)

            # get callback function for recording the state of the gradient
            # descent algorithm:
            callback, values, weights = get_gd_state_recorder_callback()

            # initialize the gradient descent algorithm:
            GD = GradientDescent(learning_rate=learning_rate,
                                 callback=callback)

            GD.fit(module, np.nan, np.nan)

            if len(values) < max_iter:
                values = values + ([np.nan] * (max_iter - len(values)))

            losses[gamma] = values

            fig = plot_descent_path(module=base_module,
                                    descent_path=np.array(weights),
                                    title=f"{name}: learning rate = {eta},"
                                          f" decay = {gamma}")
            fig.show()
        val_fig = px.line(losses)
        val_fig.update_layout(title=f"{name}: learning rate comparison",
                              title_x=0.5,
                              legend_title="etas:", )
        val_fig.show()


def load_data(path: str = "../datasets/SAheart.data",
              train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd,
                            train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Initialize the gradient descent algorithm:
    GD = GradientDescent(learning_rate=FixedLR(LEARNING_RATE),
                         max_iter=MAX_ITER)

    # Initialize logistic regression module:
    log_reg = LogisticRegression(solver=GD)
    log_reg.fit(X_train.to_numpy(), y_train.to_numpy())

    # predict:
    y_pred_proba = log_reg.predict_proba(X_train.to_numpy())
    fpr, tpr, thresholds = roc_curve(y_train, y_pred_proba)

    # find the best alpha and loss:
    best_alpha = thresholds[np.argmax(tpr - fpr)]
    loss_for_best_alpha = LogisticRegression(solver=GD, alpha=best_alpha).fit(
        X_train.to_numpy(), y_train.to_numpy()).loss(
        X_test.to_numpy(), y_test.to_numpy())

    # q.8 prints:
    print('fit_logistic_regression:')
    print(f'\t- Optimal ROC alpha from argmax[TPR - FPR] is {round(best_alpha, 3)}')
    print(f'\t- Loss for that alpha is {round(loss_for_best_alpha, 3)}')

    go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                         line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds,
                         name="", showlegend=False,
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(
            title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
            xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
            yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$"))).show()


    # Fitting l1- and l2-regularized logistic regression models,
    # using cross-validation to specify values of regularization parameter
    print()
    print('Cross Validation: ')
    for regularization in ['l1', 'l2']:
        GD = GradientDescent(learning_rate=FixedLR(LEARNING_RATE),
                             max_iter=MAX_ITER)
                             # max_iter=2000)  # works very similar, but way faster

        lambdas = pd.Series(LAMBDAS)
        res = lambdas.apply(lambda l: cross_validate(LogisticRegression(solver=GD,
                                                                        lam=l,
                                                                        penalty=regularization),
                                                     X_train.to_numpy(),
                                                     y_train.to_numpy(),
                                                     scoring=misclassification_error,
                                                     cv=FOLDS))

        train_score = res.apply(lambda r: r[0])
        validation_score = res.apply(lambda r: r[1])

        # cross validation results:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=lambdas, y=train_score, mode='lines',
                                  name='Training error'))
        fig2.add_trace(go.Scatter(x=lambdas, y=validation_score, mode='lines',
                                  name='Validation error'))
        fig2.update_layout(
            title=f'Cross-validation for {regularization.upper()} Regularized Logistic regression',
            title_x=0.5)
        # fig2.show()

        best_lambda = lambdas[validation_score.idxmin()]
        print(f'\t- Best lambda for {regularization.upper()} regularization is {best_lambda}')
        print(f'\t- Test error for that lambda is {round(validation_score.min(), 3)}\n')


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
