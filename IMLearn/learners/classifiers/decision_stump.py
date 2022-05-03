from __future__ import annotations
from typing import NoReturn

from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product

# todo remove?
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
        # initialize loop parameters:
        # min_error = np.inf
        signs = np.unique(y)  # todo make sure

        cols = ['sign', 'j', 'thresh', 'error']
        res = pd.DataFrame(columns=cols)
        res[['sign', 'j']] = list(product(signs, range(X.shape[1])))
        res['thresh'], res['error'] = \
            zip(*(res.apply(
                lambda row: self._find_threshold
                (X[:, int(row[1])], y, row[0]), axis=1)))

        self.threshold_, self.j_, self.sign_ = res.loc[res['error'].idxmin(),
                                                       ['thresh', 'j', 'sign']]
        self.fitted_ = True

        # # loop over all features and all possible thresholds:
        # for sign, feature in product(signs, range(X.shape[1])):
        #     thresh, error = self._find_threshold(X[:, feature], y, sign)
        #     if error < min_error:
        #         min_error = error
        #         self.threshold_, self.j_, self.sign_ = thresh, feature, sign
        #
        # self.fitted_ = True

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

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
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

        thresholds = pd.Series(sorted(values)).rolling(2).mean().dropna() # todo check, need gini?
        err = thresholds.apply(lambda t: misclassification_error(
                             labels, np.where(values < t, -sign, sign)))

        return thresholds[err.idxmin()], err.min()

        # df = pd.DataFrame(data={'values': values,
        #                         'labels': labels,
        #                         'errors': np.nan})
        #
        # df.errors = df['values'].apply(
        #     lambda row: misclassification_error(
        #         df.labels, np.where(values > row, sign, -sign)))
        #
        # return df['values'][df.errors.idxmin()], df.errors.min()
        # todo check return values

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
