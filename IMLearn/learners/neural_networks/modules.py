import numpy as np
from IMLearn.base.base_module import BaseModule
from IMLearn.metrics.loss_functions import cross_entropy, softmax
import pandas as pd

def linear_activation(X: np.ndarray) -> np.ndarray: # todo module class?
    """
    Default - Linear activation function
    """
    return X


class FullyConnectedLayer(BaseModule):
    """
    Module of a fully connected layer in a neural network

    Attributes:
    -----------
    input_dim_: int
        Size of input to layer (number of neurons in preceding layer

    output_dim_: int
        Size of layer output (number of neurons in layer_)

    activation_: BaseModule
        Activation function to be performed after integration of inputs and weights

    weights: ndarray of shape (input_dim_, outout_din_)
        Parameters of function with respect to which the function is optimized.

    include_intercept: bool
        Should layer include an intercept or not
    """
    def __init__(self, input_dim: int, output_dim: int,
                 activation: BaseModule = None, include_intercept: bool = True):
        """
        Initialize a module of a fully connected layer

        Parameters:
        -----------
        input_dim: int
            Size of input to layer (number of neurons in preceding layer

        output_dim: int
            Size of layer output (number of neurons in layer_)

        activation_: BaseModule, default=None
            Activation function to be performed after integration of inputs and weights. If
            none is specified functions as a linear layer

        include_intercept: bool, default=True
            Should layer include an intercept or not

        Notes:
        ------
        Weights are randomly initialized following N(0, 1/input_dim)
        """

        super().__init__()
        # todo:
        # self.input_dim_ = input_dim if not include_intercept else input_dim + 1
        self.input_dim_ = input_dim
        self.output_dim_ = output_dim

        # self.activation_ = activation if activation is not None else linear_activation
        self.activation_ = activation
        self.include_intercept_ = include_intercept

        weights_dim = (input_dim, output_dim) if not include_intercept else (
        input_dim + 1, output_dim)

        # self.weights = np.random.randn(*weights_dim) / np.sqrt(self.input_dim_)
        # set weights to Normal(0, 1/input_dim)
        self.weights = np.random.normal(0, 1 / input_dim, weights_dim)

        # bias is the first column of weights, one bias per output neuron



    # todo
    # def compute_output(self, X: np.ndarray, **kwargs) -> np.ndarray:
    # def compute_output(self, X: np.ndarray, no_activation: bool=False, **kwargs) -> np.ndarray:
    def compute_output(self, X: np.ndarray, pre_activations, post_activations, **kwargs) -> np.ndarray:
        """
        Compute activation(weights @ x) for every sample x: output value of layer at point
        self.weights and given input

        Parameters:
        -----------
        X: ndarray of shape (n_samples, input_dim)
            Input data to be integrated with weights

        Returns:
        --------
        output: ndarray of shape (n_samples, output_dim)
            Value of function at point self.weights

        Returns
        -------
        object
        """

        # if self.include_intercept_:
        #     X = np.c_[np.ones(X.shape[0]), X]
        # w_X = X @ self.weights
        #
        # if kwargs.get("pre_active"):
        #     kwargs["pre_active"][0] = w_X
        #
        # if self.activation_:
        #     return self.activation_.compute_output(w_X)
        # return w_X



        m = X.shape[0]

        if self.include_intercept_:
            X = np.c_[np.ones(m), X]

        z = X @ self.weights  # todo reverse?
        a = self.activation_.compute_output(X=z, **kwargs) if not self.activation_ else z

        pre_activations.append(z)
        post_activations.append(a)
        return a


    def compute_jacobian(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute module derivative with respect to self.weights at point self.weights

        Parameters:
        -----------
        X: ndarray of shape (n_samples, input_dim)
            Input data to be integrated with weights

        Returns:
        -------
        output: ndarray of shape (input_dim, n_samples)
            Derivative with respect to self.weights at point self.weights
        """

        if self.activation_:
            return self.activation_.compute_jacobian(X=X)
        return np.ones(X.shape)


        # z = X @ self.weights.T
        # # dz = self.activation_.compute_jacobian(X=z, **kwargs) if self.activation_ is not None else np.eye(X.shape[0])
        # dz = self.activation_.compute_jacobian(X=z, **kwargs) if self.activation_ is not None else z
        # # dz = self.activation_.compute_jacobian(X=z, **kwargs) if self.activation_ is not None else X
        # dw = X.T @ dz
        # # dw = dz @ X.T
        # return dw

        # # todo intercept?, X.T?
        # m = X.shape[0]
        #
        # if self.include_intercept_:
        #     X = np.c_[np.ones(m), X]
        #
        # # z = self.weights @ X.T
        # # z_ = X @ self.weights
        #
        # # TODO !!!
        #
        # # z_ = X.T @ self.weights
        #
        # # return self.activation_.compute_jacobian(X=z_, **kwargs) @ X
        # # return X.T @ self.activation_.compute_jacobian(X=z_, **kwargs)
        # # return self.activation_.compute_jacobian(X=z, **kwargs) if self.activation_ is not None else np.eye(m)
        # return self.activation_.compute_jacobian(X=X, **kwargs) if self.activation_ is not None else np.eye(m)


class ReLU(BaseModule):
    """
    Module of a ReLU activation function computing the element-wise function ReLU(x)=max(x,0)
    """

    def compute_output(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute element-wise value of activation

        Parameters:
        -----------
        X: ndarray of shape (n_samples, input_dim)
            Input data to be passed through activation

        Returns:
        --------
        output: ndarray of shape (n_samples, input_dim)
            Data after performing the ReLU activation function
        """
        # return np.max(X, 0)
        # todo ??

        return np.where(X > 0, X, 0)

    def compute_jacobian(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute module derivative with respect to given data

        Parameters:
        -----------
        X: ndarray of shape (n_samples, input_dim)
            Input data to compute derivative with respect to

        Returns:
        -------
        output: ndarray of shape (n_samples,)
            Element-wise derivative of ReLU with respect to given data
        """
        return np.greater(X, 0).astype(int)
        # return np.where(X > 0, 1, 0)


class CrossEntropyLoss(BaseModule):
    """
    Module of Cross-Entropy Loss: The Cross-Entropy between the Softmax of a sample x and e_k for a true class k
    """
    def compute_output(self, X: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Computes the Cross-Entropy over the Softmax of given data, with respect to every

        CrossEntropy(Softmax(x),e_k) for every sample x

        Parameters:
        -----------
        X: ndarray of shape (n_samples, input_dim)
            Input data for which to compute the cross entropy loss

        y: ndarray of shape (n_samples,)
            Values with respect to which cross-entropy loss is computed

        Returns:
        --------
        output: ndarray of shape (n_samples,)
            cross-entropy loss value of given X and y
        """

        # one_hot_y = np.eye(np.max(y) + 1)[y]
        # softmax_X = softmax(X)
        # # check if softmax_X is flat
        # if softmax_X.ndim == 1:
        #     return -np.sum(one_hot_y * np.log(softmax_X))
        #
        # res = -np.sum(one_hot_y * np.log(softmax_X), axis=1)
        # return res

        return cross_entropy(pd.get_dummies(y).to_numpy(), softmax(X))

        # one_hot_y = np.eye(np.max(y) + 1)[y]
        # softmax_X = softmax(X)
        # return -np.diag((np.log(softmax_X) @ one_hot_y.T))
        #
        #
        # #
        # #
        # # return cross_entropy(softmax_X, one_hot_y)
        #
        # # y_pred = np.apply_along_axis(softmax, 1, X)
        # # return np.apply_along_axis(cross_entropy, 1, y, y_pred)

    def compute_jacobian(self, X: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Computes the derivative of the cross-entropy loss function with respect to every given sample

        Parameters:
        -----------
        X: ndarray of shape (n_samples, input_dim)
            Input data with respect to which to compute derivative of the cross entropy loss

        y: ndarray of shape (n_samples,)
            Values with respect to which cross-entropy loss is computed

        Returns:
        --------
        output: ndarray of shape (n_samples, input_dim)
            derivative of cross-entropy loss with respect to given input
        """

        return softmax(X) - pd.get_dummies(y).to_numpy()

        # one_hot_y = np.eye(np.max(y) + 1)[y]
        # softmax_X = softmax(X)
        #
        # res = softmax_X - one_hot_y  # todo ooooooooooooooooooooooooooooooooo!!!!!!!!!!!!!!
        # # res = np.argmax(softmax_X, axis=1) - y
        # return res

        #
        # one_hot_y = np.eye(np.max(y) + 1)[y]
        # return np.sum(one_hot_y - softmax(X), axis=1) # todo sum????????
        #
        #
        #
        #
        # #
        # # # one_hot_y = np.eye(np.max(y) + 1)[y]
        # # y_pred = np.argmax(softmax(X), axis=1)
        # # # S = -(np.log(softmax_X) @ one_hot_y.T)
        # # # return np.diag(S) - S @ softmax_X.T
        # #
        # # return -(y / y_pred).T
        # #
        # #
        # #
        # #
        # #
        #
        #
        # #
        # #
        # #
        # #
        # # # todo make sure !
        # # X_softmax = softmax(X)
        # # X_softmax_diag = np.diag(X_softmax)
        # # softmax_jacobian_diag = X_softmax_diag * (1 - X_softmax_diag)
        # #
        # # softmax_jacobian = X_softmax @ X_softmax.T
        # # np.fill_diagonal(softmax_jacobian, softmax_jacobian_diag)
        # #
        # # return softmax_jacobian
        #
