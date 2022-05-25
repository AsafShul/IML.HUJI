from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator,
                   X: np.ndarray,
                   y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float],
                   cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """


    X = X.reshape(-1, 1) if X.shape[0] == 1 else X
    data = deepcopy(np.concatenate((X, y.reshape(-1, 1)), axis=1))
    # np.random.shuffle(data)
    sectioned_data = np.array_split(data, cv)
    folded_data = [[np.concatenate([sectioned_data[j] for j in range(cv) if j != i], axis=0),
                   sectioned_data[i]] for i in range(cv)]
    train_scores = []
    validation_scores = []

    for train_data, validation_data in folded_data:
        temp_estimator = deepcopy(estimator)
        temp_estimator.fit(train_data[:, :-1], train_data[:, -1])

        train_scores.append(scoring(
            temp_estimator.predict(train_data[:, :-1]),
            train_data[:, -1]))

        validation_scores.append(scoring(
            temp_estimator.predict(validation_data[:, :-1]),
            validation_data[:, -1]))

    return np.mean(train_scores), np.mean(validation_scores)
