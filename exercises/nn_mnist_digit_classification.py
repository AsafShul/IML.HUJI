import time
import numpy as np
import gzip
from typing import Tuple

from IMLearn.metrics.loss_functions import accuracy
from IMLearn.learners.neural_networks.modules import FullyConnectedLayer, ReLU, \
    CrossEntropyLoss, softmax
from IMLearn.learners.neural_networks.neural_network import NeuralNetwork
from IMLearn.desent_methods import GradientDescent, StochasticGradientDescent, \
    FixedLR
from IMLearn.utils.utils import confusion_matrix

import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def load_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the MNIST dataset

    Returns:
    --------
    train_X : ndarray of shape (60,000, 784)
        Design matrix of train set

    train_y : ndarray of shape (60,000,)
        Responses of training samples

    test_X : ndarray of shape (10,000, 784)
        Design matrix of test set

    test_y : ndarray of shape (10,000, )
        Responses of test samples
    """

    def load_images(path):
        with gzip.open(path) as f:
            # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
            raw_data = np.frombuffer(f.read(), 'B', offset=16)
        # converting raw data to images (flattening 28x28 to 784 vector)
        return raw_data.reshape(-1, 784).astype('float32') / 255

    def load_labels(path):
        with gzip.open(path) as f:
            # First 8 bytes are magic_number, n_labels
            return np.frombuffer(f.read(), 'B', offset=8)

    return (load_images('../datasets/mnist-train-images.gz'),
            load_labels('../datasets/mnist-train-labels.gz'),
            load_images('../datasets/mnist-test-images.gz'),
            load_labels('../datasets/mnist-test-labels.gz'))


def plot_images_grid(images: np.ndarray, title: str = ""):
    """
    Plot a grid of images

    Parameters
    ----------
    images : ndarray of shape (n_images, 784)
        List of images to print in grid

    title : str, default="
        Title to add to figure

    Returns
    -------
    fig : plotly figure with grid of given images in gray scale
    """
    side = int(len(images) ** 0.5)
    subset_images = images.reshape(-1, 28, 28)

    height, width = subset_images.shape[1:]
    grid = subset_images.reshape(side, side, height, width).swapaxes(1,
                                                                     2).reshape(
        height * side, width * side)

    return px.imshow(grid, color_continuous_scale="gray") \
        .update_layout(
        title=dict(text=title, y=0.97, x=0.5, xanchor="center", yanchor="top"),
        font=dict(size=16), coloraxis_showscale=False) \
        .update_xaxes(showticklabels=False) \
        .update_yaxes(showticklabels=False)


if __name__ == '__main__':
    train_X, train_y, test_X, test_y = load_mnist()
    (n_samples, n_features), n_classes = train_X.shape, 10

    # ---------------------------------------------------------------------------------------------#
    # Question 5+6+7: Network with ReLU activations using SGD + recording convergence              #
    # ---------------------------------------------------------------------------------------------#
    # Initialize, fit and test network
    # Create simple network:
    # hidden_layer_size = 64
    #
    # input_layer = FullyConnectedLayer(input_dim=n_features,
    #                                   output_dim=hidden_layer_size,
    #                                   activation=ReLU(),
    #                                   include_intercept=True)
    #
    # mid_layer = FullyConnectedLayer(input_dim=hidden_layer_size,
    #                                 output_dim=hidden_layer_size,
    #                                 activation=ReLU(),
    #                                 include_intercept=True)
    #
    # output_layer = FullyConnectedLayer(input_dim=hidden_layer_size,
    #                                    output_dim=n_classes,
    #                                    activation=ReLU(),
    #                                    include_intercept=True)
    #
    # layers = [input_layer, mid_layer, output_layer]
    #
    # from exercises.gradient_descent_investigation import \
    #     get_gd_state_recorder_callback
    #
    # callback, values, weights = get_gd_state_recorder_callback()
    #
    # gd_solver = GradientDescent(learning_rate=FixedLR(0.1), max_iter=100,
    #                             callback=callback)
    #
    # nn = NeuralNetwork(modules=layers, loss_fn=CrossEntropyLoss(),
    #                    solver=gd_solver)
    #
    # # Fit network to data:
    # nn.fit(train_X, train_y)
    #
    # print(
    #     f'MNST NN [2x64x64x3] accuracy: {round(accuracy(nn.predict(test_X), test_y) * 100, 3)}%')
    #
    # # Plotting convergence process
    # px.line(x=list(range(len(values))), y=values).show()
    #
    # # Plotting test true- vs predicted confusion matrix
    # from sklearn.metrics import confusion_matrix
    #
    # cm = confusion_matrix(test_y, nn.predict(test_X))
    # fig = px.imshow(cm, color_continuous_scale="Viridis")
    # fig.update_layout(title="MNIST NN [2x64x64x3] confusion matrix",
    #                   font=dict(size=16), coloraxis_showscale=False)
    # fig.update_xaxes(showticklabels=False)
    # fig.update_yaxes(showticklabels=False)
    # fig.show()

    # ---------------------------------------------------------------------------------------------#
    # Question 8: Network without hidden layers using SGD                                          #
    # ---------------------------------------------------------------------------------------------#
    simple_layers = [FullyConnectedLayer(input_dim=n_features,
                                  output_dim=n_classes,
                                  activation=ReLU(),
                                  include_intercept=True)]



    gd_solver_2 = GradientDescent(learning_rate=FixedLR(0.1), max_iter=100)

    simple_nn = NeuralNetwork(modules=simple_layers, loss_fn=CrossEntropyLoss(),
                                solver=gd_solver_2)

    # Fit network to data:
    simple_nn.fit(train_X, train_y)

    print(f'MNST NN [2x3] accuracy: {round(accuracy(simple_nn.predict(test_X), test_y) * 100, 3)}%')


    # ---------------------------------------------------------------------------------------------#
    # Question 9: Most/Least confident predictions                                                 #
    # ---------------------------------------------------------------------------------------------#
    # raise NotImplementedError()

    # ---------------------------------------------------------------------------------------------#
    # Question 10: GD vs GDS Running times                                                         #
    # ---------------------------------------------------------------------------------------------#
    # raise NotImplementedError()
