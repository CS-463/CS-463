"""Data and plotting helpers for Lab 7 (ch_6_and_10_lab_7.ipynb).

You don't need to read this file. It holds the matplotlib details so the notebook can stay
focused on training and diagnosing networks. Each function's docstring says what it takes
and what it returns; the notebook shows an example call before you use one.

On Colab: upload this file and mnist.npz next to the notebook (Files panel, then Upload).
"""
import warnings

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

# Okabe-Ito colors, which stay distinguishable under common color blindness.
BLUE, ORANGE, GREEN, PURPLE, SKY, VERMILLION = '#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00'
CURVE_COLORS = [BLUE, ORANGE, GREEN, PURPLE, SKY, VERMILLION]


def xor_data(n=25, spread=0.12, seed=0):
    """Wednesday's noisy XOR data: n points around each corner of the unit square.

    Returns X, shape (4n, 2), and y, shape (4n,): 1 at the corners (1, 0) and (0, 1),
    where exactly one feature is 'on', and 0 at (0, 0) and (1, 1).
    """
    rng = np.random.default_rng(seed)
    corners = np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=float)
    labels = np.array([0, 0, 1, 1], dtype=float)
    X = np.vstack([c + rng.normal(0, spread, (n, 2)) for c in corners])
    y = np.repeat(labels, n)
    order = rng.permutation(len(y))
    return X[order], y[order]


def wednesday_start(scale=1.0):
    """Wednesday's starting weights for the XOR network, multiplied by scale.

    These are the weights Training a Network drew with random_params(seed=1): every weight
    and bias from a standard normal distribution. Multiplying by scale keeps the starting
    point's direction in parameter space and changes only its distance from the origin.

    Returns a dict with 'coefs' and 'intercepts', in scikit-learn's layout (the same layout
    as a fitted MLPClassifier's coefs_ and intercepts_), ready for fit_from_start.
    """
    rng = np.random.default_rng(1)
    W1, b1 = rng.normal(size=(2, 2)), rng.normal(size=2)      # same draws, same order as Wednesday
    W2, b2 = rng.normal(size=(1, 2)), rng.normal(size=1)
    return {'coefs': [scale * W1.T, scale * W2.T], 'intercepts': [scale * b1, scale * b2]}


def fit_from_start(model, start, X, y):
    """Fit an unfitted MLPClassifier, starting from the weights in start instead of random ones.

    model   an unfitted MLPClassifier, e.g. from xor_net. Its settings (learning rate,
            max_iter, ...) are used; its random_state no longer matters for the start.
    start   a dict like wednesday_start's: {'coefs': [...], 'intercepts': [...]}.

    Returns the fitted model, with loss_curve_[0] the loss at the starting weights, like
    any fit. (scikit-learn has no option for this, so we let it set up a fitted model
    with a negligible step, swap in the starting weights, and continue with warm_start.)
    """
    from sklearn.base import clone
    model = clone(model)
    settings = model.get_params()
    model.set_params(max_iter=1, learning_rate_init=1e-300, shuffle=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(X, y)
    model.coefs_ = [np.array(W, dtype=float) for W in start['coefs']]
    model.intercepts_ = [np.array(b, dtype=float) for b in start['intercepts']]
    model.loss_curve_, model.n_iter_, model.t_ = [], 0, 0
    model.best_loss_, model._no_improvement_count = np.inf, 0
    model.set_params(**{k: settings[k] for k in ('max_iter', 'learning_rate_init', 'shuffle')}, warm_start=True)
    return model.fit(X, y)


def plot_loss_curves(curves, labels=None, x='epoch', steps_per_epoch=None, baseline=None,
                     title='Loss curve'):
    """Plot one or more training-loss curves on one set of axes.

    curves          a list of 1-D sequences of losses, usually each fitted model's
                    loss_curve_ (one entry per epoch). The curves may differ in length.
                    For a single curve, pass a one-item list: [model.loss_curve_].
    labels          optional list of strings, one per curve, for the legend. Leave it out
                    when plotting many curves; they're drawn thin, with no legend.
    x               'epoch' (the default) or 'step'. With 'step', give steps_per_epoch.
    steps_per_epoch with x='step': a list with one number per curve, the gradient steps
                    each curve's model took per epoch (1 for full batch).
    baseline        optional number drawn as a dashed horizontal line, e.g. np.log(2).
    title           the plot's title.

    Returns the matplotlib Axes, so you can add to the plot.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    many = labels is None and len(curves) > 6
    for k, curve in enumerate(curves):
        curve = np.asarray(curve)
        scale = 1 if x == 'epoch' else steps_per_epoch[k]
        xs = (np.arange(len(curve)) + 1) * scale
        if many:
            ax.plot(xs, curve, lw=1, alpha=0.7, color=BLUE)
        else:
            # Earlier curves are drawn on top, and a dot marks where each curve ends,
            # so a short run (one that stopped early) stays visible.
            color = CURVE_COLORS[k % 6]
            ax.plot(xs, curve, lw=2.5 - 0.5 * min(k, 2), color=color, zorder=10 - k,
                    label=None if labels is None else labels[k])
            ax.plot(xs[-1], curve[-1], 'o', ms=7, color=color, zorder=20 - k)
    if baseline is not None:
        ax.axhline(baseline, color='gray', ls='--', lw=1.5,
                   label=f'{baseline:.3f}' + (': predict 0.5 everywhere' if abs(baseline - np.log(2)) < 1e-9 else ''))
    ax.set_xlabel('epoch' if x == 'epoch' else 'gradient step')
    ax.set_ylabel('training loss (log-loss)')
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    if labels is not None or baseline is not None:
        ax.legend()
    ax.grid(alpha=0.3)
    plt.show()
    return ax


def show_digits(images, labels, predicted=None, shape=(8, 8), ncols=10, title=None):
    """Show images in a grid. images: shape (k, pixels) or (k, rows, cols); labels: shape (k,).

    With predicted (shape (k,)), each title reads 'true → predicted'.
    """
    images = np.asarray(images).reshape(len(images), *shape)
    nrows = int(np.ceil(len(images) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.3 * ncols, 1.5 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for k, (ax, image) in enumerate(zip(axes.flat, images)):
        ax.imshow(image, cmap='gray_r')
        ax.set_title(f'{labels[k]}' if predicted is None else f'{labels[k]} → {predicted[k]}', fontsize=11)
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_confusion(y_true, y_pred, title='Confusion matrix (test set)'):
    """Draw the confusion matrix of true vs. predicted digits, with counts in the cells."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.imshow(cm, cmap='Blues')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=10)
    ax.set_xticks(range(cm.shape[1]))
    ax.set_yticks(range(cm.shape[0]))
    ax.set_xlabel('predicted digit')
    ax.set_ylabel('true digit')
    ax.set_title(title)
    plt.show()


def plot_accuracy_vs_width(sizes, train_acc, test_acc, reference=None,
                           reference_label='logistic regression (test)',
                           title='Accuracy vs. hidden-layer width'):
    """Plot training and test accuracy against the number of hidden units (log-scale x-axis).

    sizes, train_acc, test_acc: sequences of the same length. reference: an optional
    accuracy drawn as a dashed horizontal line (e.g. logistic regression's test accuracy).
    """
    if len(train_acc) == 0 or len(test_acc) == 0:
        print('Nothing to plot yet: train_acc and test_acc are empty. Fill in the loop above, then rerun.')
        return None
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sizes, train_acc, 'o-', color=BLUE, lw=2, label='network, training')
    ax.plot(sizes, test_acc, 's-', color=ORANGE, lw=2, label='network, test')
    if reference is not None:
        ax.axhline(reference, color='gray', ls='--', lw=1.5, label=reference_label)
    ax.set_xscale('log')
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel('hidden units (log scale)')
    ax.set_ylabel('accuracy')
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.show()


def _network(model):
    """Split a fitted pipeline into (preprocessing steps, MLPClassifier); a bare model has none."""
    if isinstance(model, Pipeline):
        return model[:-1], model[-1]
    return None, model


def hidden_activations(model, X):
    """The first hidden layer's activations for each row of X: shape (n, hidden units)."""
    prep, net = _network(model)
    if prep is not None:
        X = prep.transform(X)
    Z = X @ net.coefs_[0] + net.intercepts_[0]
    if net.activation == 'relu':
        return np.maximum(0, Z)
    if net.activation == 'logistic':
        return 1 / (1 + np.exp(-Z))
    if net.activation == 'tanh':
        return np.tanh(Z)
    return Z


def plot_h_space(model, X, y, title=None, units=(0, 1), ax=None):
    """Plot each row of X at two of its hidden activations, colored by its class y.

    model: a fitted MLPClassifier (or a pipeline ending in one).
    units: which two hidden units to put on the axes (default: the first two). A network
           with exactly 2 hidden units shows its whole h-space; a wider one shows only
           those two of its activations.
    ax:    optional matplotlib Axes to draw on (e.g. one panel of plt.subplots). Without
           it, the function makes its own figure and shows it.
    For a binary network with 2 hidden units, also draws the output unit's boundary, p = 0.5.
    """
    prep, net = _network(model)
    H = hidden_activations(model, X)
    if H.shape[1] < 2:
        raise ValueError(f'plot_h_space needs at least 2 hidden units; this network has {H.shape[1]}.')
    i, j = units
    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=(7, 6))
    classes = np.unique(y)
    if len(classes) == 2:
        for c, color, marker in zip(classes, (ORANGE, BLUE), ('^', 'o')):
            ax.scatter(H[y == c, i], H[y == c, j], s=30, color=color, marker=marker, label=f'class {int(c)}', alpha=0.8)
        if H.shape[1] == 2:
            w, b = net.coefs_[1][:, 0], net.intercepts_[1][0]
            lo, hi = H.min(axis=0) - 0.1, H.max(axis=0) + 0.1
            if abs(w[1]) > 1e-12:
                h1 = np.linspace(lo[0], hi[0], 100)
                ax.plot(h1, -(w[0] * h1 + b) / w[1], 'k-', lw=2, label='output unit: p = 0.5')
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])
        ax.legend(loc='best')
    else:
        cmap = plt.get_cmap('tab10')
        for k, c in enumerate(classes):
            ax.scatter(H[y == c, i], H[y == c, j], s=4, color=cmap(k % 10), alpha=0.5, label=str(c))
        ax.legend(title='digit', markerscale=4, loc='upper right', fontsize=9)
    ax.set_xlabel(f'$h_{{{i + 1}}}$')
    ax.set_ylabel(f'$h_{{{j + 1}}}$')
    ax.set_title(title or 'h-space: each point at two hidden activations')
    if own_figure:
        plt.show()
    return ax
