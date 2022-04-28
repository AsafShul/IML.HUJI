from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
from os import path

pio.templates.default = "simple_white"
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)

def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        samples, labels = load_dataset(path.join('../datasets', f))
        losses = []

        # Fit Perceptron and record loss in each fit iteration
        perceptron_classifier = Perceptron(callback=
                                           lambda fit, x, y: losses.append(
                                               fit.loss(x, y)))
        perceptron_classifier.fit(samples, labels)

        # Plot figure
        perceptron_loss_fig = px.line(x=np.arange(len(losses)), y=losses,
                                      labels=dict(x='Iteration',
                                                  y='Normalized Loss'))
        perceptron_loss_fig.update_layout(title_text=f'{n} Perceptron Loss',
                                          title_x=0.5)
        perceptron_loss_fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (
        np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines",
                      marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    from IMLearn.metrics import accuracy

    # initialize parameters:
    symbols_dict = {0: 'circle', 1: 'square', 2: 'diamond'}
    colors_dict = {0: 'lightsalmon', 1: 'gold', 2: 'cornflowerblue'}

    # loop over datasets:
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        samples, labels = load_dataset(path.join('../datasets', f))
        true_symbols = [symbols_dict[f] for f in labels]
        lims = np.array([samples.min(axis=0),
                         samples.max(axis=0)]).T + np.array([-.4, .4])

        # initialize classifiers:
        gaussian_classifier = GaussianNaiveBayes()
        lda_classifier = LDA()

        # Fit models and predict over training set
        gaussian_classifier.fit(samples, labels)
        lda_classifier.fit(samples, labels)

        # Predict over samples:
        gc_response = gaussian_classifier.predict(samples)
        lda_response = lda_classifier.predict(samples)

        # initialize figure:
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=(
                                f'Gaussian Naive Bayes, Accuracy:'
                                f' {round(accuracy(labels, gc_response), 3) * 100}%',
                                f'LDA, Accuracy:'
                                f' {round(accuracy(labels, lda_response), 3) * 100}%'))
        fig.update_layout(
            title_text=f'Comparing classifiers over "{f}" dataset',
            title_x=0.5)
        fig.update_layout(showlegend=False)

        # set parameters:
        responses = [gc_response, lda_response]
        classifiers = [gaussian_classifier, lda_classifier]

        # Plot the figures (scatter, ellipse, and markers)
        for i, item in enumerate(zip(classifiers, responses)):
            # extract params:
            classifier, response = item

            # Plot scatter of the classifier:
            scatter_trace = go.Scatter(x=samples[:, 0], y=samples[:, 1],
                                       mode='markers',
                                       marker=dict(color=
                                                   [colors_dict[f] for f in
                                                    response.reshape(-1, )],
                                                   symbol=true_symbols,
                                                   size=10,
                                                   line=dict(width=2)
                                                   )
                                       )

            # plot classes results:
            ellipse_traces = []
            gaussian_markers = []
            for _class in classifier.classes_:
                cov_mat = np.diag(classifier.vars_[_class]) if not i \
                    else classifier.cov_

                # add ellipse:
                coords = get_ellipse_coordinates(classifier.mu_[_class],
                                                 cov_mat)
                ellipse_traces.append(go.Scatter(x=coords[:, 0],
                                                 y=coords[:, 1],
                                                 line=dict(color="black"))
                                      )
                # if the given ellipse function needs to be used, uncomment this line:
                # ellipse_traces.append(get_ellipse(classifier.mu_[_class], cov_mat))

                # add middle gaussian markers:
                gaussian_markers.append(go.Scatter(
                    x=[classifier.mu_[_class][0]],
                    y=[classifier.mu_[_class][1]],
                    mode='markers',
                    marker=dict(color='black',
                                symbol='x',
                                size=15,
                                ))
                )

            # add decision boundary:
            # (wasn't asked, but I found it really useful)
            decision_boundary = decision_surface(classifier.predict,
                                                 lims[0], lims[1],
                                                 showscale=False)
            # add traces:
            traces = [decision_boundary] + [scatter_trace] + \
                      ellipse_traces + gaussian_markers

            [fig.add_trace(trace, row=1, col=i + 1) for trace in traces]
        fig.show()


# I did this function before I pulled the latest version, and saw you
# implemented it for me, so I left it here.
def get_ellipse_coordinates(mu, cov):
    # radius:
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ellipse_r_x = np.sqrt(1 + pearson)
    ellipse_r_y = np.sqrt(1 - pearson)

    # base coordinates:
    step = np.linspace(0, 2 * np.pi, 100)
    ellipse_coords = np.column_stack(
        [ellipse_r_x * np.cos(step), ellipse_r_y * np.sin(step)])

    # translation:
    translation_matrix = np.tile([mu[0], mu[1]],
                                 (ellipse_coords.shape[0], 1))
    # rotation:
    quarter_pi = (np.pi / 4)
    rotation_matrix = np.array([[np.cos(quarter_pi), np.sin(quarter_pi)],
                                [-np.sin(quarter_pi), np.cos(quarter_pi)]])
    # scale:
    std_num = 2
    scale_matrix = np.array([[np.sqrt(cov[0, 0]) * std_num, 0],
                             [0, np.sqrt(cov[1, 1]) * std_num]])
    ellipse_coords = ellipse_coords @ rotation_matrix @ scale_matrix + \
                     translation_matrix

    return ellipse_coords


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
