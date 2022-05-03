import numpy as np
# from ...base import BaseEstimator # todo check why not working?
from IMLearn.base import BaseEstimator
from IMLearn.metrics.loss_functions import misclassification_error
from typing import Callable, NoReturn


class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = [], np.zeros(
            # iterations), np.zeros(iterations)
            iterations), np.zeros(iterations) # todo check !!!!

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # Initialize params:
        sample_size = X.shape[0]
        data = np.concatenate((X, y.reshape(-1, 1)), axis=1)

        # Initialize weights and Distribution uniformly:
        initial_distribution = list(np.ones(sample_size) / sample_size)
        self.D_ = initial_distribution

        # Iterate over boosting iterations
        for t in range(self.iterations_):
            # sample data:
            dataD = data[np.random.choice(np.arange(len(data)),
                                          size=sample_size, p=self.D_)]

            xD, yD = dataD[:, :-1], dataD[:, -1]

            # generate a weak learner:
            model = self.wl_()
            model.fit(xD, yD)
            self.models_.append(model)

            # predict:
            pred = model.predict(X)

            # calculate error:
            error = (np.where(pred != y, 1, 0) * self.D_).sum()

            # calculate weak learner weight:
            self.weights_[t] = 0.5 * np.log((1 - error) / error)

            # Update sample weights:
            self.D_ *= np.exp(-self.weights_[t] * y * pred)
            self.D_ /= np.sum(self.D_)
            print(t)

        self.fitted_ = True

    def _predict(self, X):
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return self.partial_predict(X, X.shape[1])  # todo make sure -1 or not

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
        self.partial_loss(X, y, X.shape[1])  # todo make sure -1 or not

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        predictions = np.array(
            [model.predict(X) for model in self.models_[:T]])

        return np.sign(np.sum(predictions * self.weights_[:T].reshape(-1, 1), axis=0))

    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """

        return misclassification_error(y, self.partial_predict(X, T))
