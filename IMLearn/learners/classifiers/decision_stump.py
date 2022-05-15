from __future__ import annotations
from typing import NoReturn

from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product

from IMLearn.metrics.loss_functions import misclassification_error
import pandas as pd


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None
        self.labels_ = np.array([-1, 1])

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        min_error = np.inf
        combinations = product(self.labels_, range(X.shape[1]))
        for label, size in combinations:
            threshold, error = self._find_threshold(X[:, size], y, label)
            min_error, self.threshold_, self.j_, self.sign_ = \
                (error, threshold, size, label) if error < min_error else \
                    (min_error, self.threshold_, self.j_, self.sign_)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """

        return np.where(X[:, int(self.j_)] < self.threshold_, -self.sign_,
                        self.sign_)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray,
                        sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """

        data = sorted(zip(values, labels), key=lambda tup: tup[0])

        values = np.array([x[0] for x in data])
        labels = np.array([x[1] for x in data])

        threshold = np.concatenate(
            [[-np.inf], list(pd.Series(values).rolling(2).mean().dropna()),
             [np.inf]])
        loss = np.sum(np.abs(labels[np.sign(labels) == sign]))
        error = np.append(loss, loss - np.cumsum(labels * sign))
        index = np.argmin(error)
        return threshold[index], error[index]

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """

        return misclassification_error(y, self.predict(X))
