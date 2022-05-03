import numpy as np
from typing import Tuple

import pandas as pd

from IMLearn.metalearners.adaboost import AdaBoost
# from IMLearn.learners.metalearners.adaboost import AdaBoost # todo make sure
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IMLearn.metrics.loss_functions import accuracy


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000,
                              test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size,
                                                         noise), generate_data(
        test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(DecisionStump, n_learners)
    adaboost.fit(train_X, train_y)

    iterations = list(range(1, n_learners + 1))
    error_df = pd.DataFrame({'iteration': iterations,
                             'train': [
                                 adaboost.partial_loss(train_X, train_y, i) for
                                 i in iterations],
                             'test': [adaboost.partial_loss(test_X, test_y, i)
                                      for i in iterations]})

    fig1 = px.line(error_df, x='iteration', y=['train', 'test'],
                   title='Train, Test errors of AdaBoost in noiseless case by iteration')
    fig1.update_layout(title_x=0.5)
    fig1.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0),
                     np.r_[train_X, test_X].max(axis=0)]).T + np.array(
        [-.1, .1])

    colors_dict = {-1: 'lightsalmon', 1: 'cornflowerblue'}
    symbols_dict = {-1: 'circle', 1: 'diamond'}
    true_test_symbols = [symbols_dict[f] for f in test_y]
    true_test_colors = [colors_dict[f] for f in test_y]
    true_train_colors = [colors_dict[f] for f in train_y]

    q2_figures = []
    for t in T:
        q2_figures.append(go.Figure())
        q2_figures[-1].add_trace(go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                                  mode='markers',
                                  marker=dict(color=true_test_colors,
                                              symbol=true_test_symbols)
                                  ))

        q2_figures[-1].add_trace(decision_surface(
            lambda x: adaboost.partial_predict(x, t),
            lims[0], lims[1],
            showscale=False))

        q2_figures[-1].update_layout(
            title_text=f'Decision boundary for size ({t})', title_x=0.5,
            yaxis_range=[-1, 1], xaxis_range=[-1, 1])

        q2_figures[-1].show()

    # Question 3: Decision surface of best performing ensemble

    # calc plot params:
    errors = [adaboost.partial_loss(test_X, test_y, t) for t in iterations]
    min_err_size = np.argmin(errors)
    min_err_acc = accuracy(test_y, adaboost.partial_predict(test_X, min_err_size))

    # plot fig:
    q3_fig = go.Figure()
    q3_fig.add_trace(go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                                  mode='markers',
                                  marker=dict(color=true_test_colors,
                                              symbol=true_test_symbols)
                                  ))
    q3_fig.add_trace(decision_surface(
        lambda x: adaboost.partial_predict(x, min_err_size),
        lims[0], lims[1],
        showscale=False))

    q3_fig.update_layout(
        title_text=f'Decision boundary for Lowest error'
                   f'[size := {min_err_size}, accuracy := {min_err_acc}]',
        title_x=0.5, yaxis_range=[-1, 1], xaxis_range=[-1, 1])

    q3_fig.show()

    # Question 4: Decision surface with weighted samples
    weights = adaboost.D_   # todo d as array of vectors?
    weights = (weights / np.max(weights)) * 40 # todo needs to be 5...

    # plot fig:
    q4_fig = go.Figure()
    q4_fig.add_trace(go.Scatter(x=train_X[:, 0], y=train_X[:, 1],
                                mode='markers',
                                marker=dict(color=true_train_colors,
                                            size=weights,
                                            line=dict(width=2,
                                                      color='DarkSlateGrey')
                                            )
                                ))
    q4_fig.add_trace(decision_surface(
        lambda x: adaboost.predict(x),
        lims[0], lims[1],
        showscale=False))

    q4_fig.update_layout(
        title_text=f'Importance of points in train for classification:',
        title_x=0.5, yaxis_range=[-1, 1], xaxis_range=[-1, 1])

    q4_fig.show()

if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
