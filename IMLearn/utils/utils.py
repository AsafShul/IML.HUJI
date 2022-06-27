from typing import Tuple
import numpy as np
import pandas as pd


def split_train_test(X: pd.DataFrame, y: pd.Series, train_proportion: float = .75) \
        -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Randomly split given sample to a training- and testing sample

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Data frame of samples and feature values.

    y : Series of shape (n_samples, )
        Responses corresponding samples in data frame.

    train_proportion: Fraction of samples to be split as training set

    Returns
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
    x_df = pd.DataFrame(X)
    y_df = pd.DataFrame(y)
    train_size = int(np.floor(x_df.shape[0] * train_proportion))
    temp_df_rand = pd.concat([x_df, y_df], axis=1).sample(frac=1).reset_index(drop=True)
    temp_x = temp_df_rand.iloc[:, :-1]
    temp_y = temp_df_rand.iloc[:, -1]
    return temp_x.iloc[:train_size], temp_y.iloc[:train_size], temp_x.iloc[train_size:], temp_y.iloc[train_size:]


def confusion_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute a confusion matrix between two sets of integer vectors

    Parameters
    ----------
    a: ndarray of shape (n_samples,)
        First vector of integers

    b: ndarray of shape (n_samples,)
        Second vector of integers

    Returns
    -------
    confusion_matrix: ndarray of shape (a_unique_values, b_unique_values)
        A confusion matrix where the value of the i,j index shows the number of times value `i` was found in vector `a`
        while value `j` vas found in vector `b`
    """
    y_true = pd.Series(a)
    y_pred = pd.Series(b) # todo makesure
    return pd.crosstab(y_true, y_pred).to_numpy()
