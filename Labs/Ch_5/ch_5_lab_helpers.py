"""Plotting helpers for ch_5_lab.ipynb, plus two functions that repeat your own loops.

You don't need to read or understand the plotting code. `validation_curves` and `cv_curves`
are worth a look: they run the same loops you write in 1(a) and 2(a), once per random seed.
"""
import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor

BLUE, ORANGE, GREEN, GRAY = '#0072B2', '#D55E00', '#009E73', '0.6'


def validation_curves(X, y, K_values, seeds):
    """One validation-MSE curve per seed: split in half, fit KNN for each K, score on the other half."""
    curves = []
    for seed in seeds:
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.5, random_state=seed)
        curves.append([mean_squared_error(y_valid, KNeighborsRegressor(n_neighbors=K)
                                          .fit(X_train, y_train).predict(X_valid))
                       for K in K_values])
    return curves


def cv_curves(X, y, K_values, seeds):
    """One 10-fold cross-validation MSE curve per shuffle of the rows into folds."""
    curves = []
    for seed in seeds:
        folds = KFold(n_splits=10, shuffle=True, random_state=seed)
        curves.append([-cross_val_score(KNeighborsRegressor(n_neighbors=K), X, y, cv=folds,
                                        scoring='neg_mean_squared_error').mean()
                       for K in K_values])
    return curves


def plot_knn_fits(X, y, K_values=(1, 20, 150)):
    """Scatter of mpg against horsepower, with the KNN regression fit for a few values of K."""
    fig, axes = subplots(1, len(K_values), figsize=(4.2 * len(K_values), 3.8), sharey=True)
    grid = np.linspace(X.iloc[:, 0].min(), X.iloc[:, 0].max(), 400)
    grid_df = pd.DataFrame({X.columns[0]: grid})
    for ax, K in zip(axes, K_values):
        model = KNeighborsRegressor(n_neighbors=K).fit(X, y)
        ax.scatter(X.iloc[:, 0], y, s=10, alpha=0.4, color=GRAY)
        ax.plot(grid, model.predict(grid_df), color=BLUE, lw=2)
        ax.set(title=f'KNN regression, K = {K}', xlabel=X.columns[0])
    axes[0].set_ylabel(y.name)
    fig.tight_layout()
    return fig


def plot_validation_curves(K_values, curves, title='Validation set approach'):
    """Validation MSE against K. The first curve is drawn darker; each curve's minimum is marked."""
    fig, ax = subplots(figsize=(7, 4))
    for i, curve in enumerate(curves):
        ax.plot(K_values, curve, color=BLUE if i == 0 else GRAY, lw=2.5 if i == 0 else 1,
                alpha=1 if i == 0 else 0.7, label='first split' if i == 0 else None)
        best = int(np.argmin(curve))
        ax.plot(K_values[best], curve[best], 'o', color=ORANGE, ms=6)
    ax.plot([], [], 'o', color=ORANGE, label="each curve's lowest MSE")
    ax.set_xscale('log')
    ax.set(xlabel='K (number of neighbors, log scale)', ylabel='validation MSE', title=title)
    ax.legend()
    return fig


def plot_cv_vs_validation(K_values, cv, valid):
    """Ten-fold CV curves (one per shuffle) next to the validation-set curves, on the same axes."""
    fig, axes = subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, curves, color, title in [(axes[0], valid, GRAY, 'Validation set approach, 10 splits'),
                                     (axes[1], cv, GREEN, '10-fold cross-validation, 10 shuffles')]:
        for curve in curves:
            ax.plot(K_values, curve, color=color, lw=1.2)
            best = int(np.argmin(curve))
            ax.plot(K_values[best], curve[best], 'o', color=ORANGE, ms=5)
        ax.set_xscale('log')
        ax.set(xlabel='K (number of neighbors, log scale)', title=title)
    axes[0].set_ylabel('estimated test MSE')
    fig.tight_layout()
    return fig


def plot_bootstrap(values, estimate, xlabel, title, interval=None, zero_line=False, ax=None):
    """Histogram of bootstrap values, with the estimate from the original data and an optional interval."""
    if ax is None:
        _, ax = subplots(figsize=(7, 3.8))
    ax.hist(values, bins=40, color=BLUE, alpha=0.6, edgecolor='white')
    ax.axvline(estimate, color='black', lw=2, label='from the original data')
    if interval is not None:
        for edge in interval:
            ax.axvline(edge, color=ORANGE, lw=2, ls='--')
        ax.plot([], [], color=ORANGE, ls='--', label='95% percentile interval')
    if zero_line:
        ax.axvline(0, color=GRAY, lw=2, ls=':', label='no difference')
    ax.set(xlabel=xlabel, ylabel='number of bootstrap samples', title=title)
    ax.legend(fontsize=9)
    return ax


def plot_line_fit(df, x, y, model):
    """Scatter of y against x with the least squares line from a fitted statsmodels model."""
    fig, ax = subplots(figsize=(7, 4))
    ax.scatter(df[x], df[y], s=10, alpha=0.4, color=GRAY)
    xs = np.linspace(df[x].min(), df[x].max(), 100)
    ax.plot(xs, model.params['Intercept'] + model.params[x] * xs, color=BLUE, lw=2,
            label='least squares line')
    ax.set(xlabel=x, ylabel=y, title=f'{y} vs. {x}, with the least squares line')
    ax.legend()
    return fig
