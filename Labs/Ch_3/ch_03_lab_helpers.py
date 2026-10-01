"""Plotting and table helpers for ch_03_lab.ipynb.

You don't need to read or understand this file to do the lab. It keeps the
plotting code out of the notebook so you can focus on the regression ideas.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.pyplot import subplots


def compare_coefficient(models, term):
    """One row per model: the coefficient of `term`, its standard error, and its p-value.

    models: dict mapping a label to a fitted statsmodels results object.
    """
    return pd.DataFrame({label: {f'{term} coef': m.params[term], 'std err': m.bse[term],
                                 'p-value': m.pvalues[term]}
                         for label, m in models.items()}).T


def plot_fit(df, x, y, model, ax=None):
    """Scatter of y against x with the fitted least squares line."""
    if ax is None:
        _, ax = subplots(figsize=(7, 5))
    ax.scatter(df[x], df[y], alpha=0.5)
    xs = np.linspace(df[x].min(), df[x].max(), 100)
    ax.plot(xs, model.predict(pd.DataFrame({x: xs})), color='C3', linewidth=2,
            label='least squares line')
    ax.set(xlabel=x, ylabel=y, title=f'{y} vs. {x}, with the least squares line')
    ax.legend()
    return ax


def _with_correlation(x, r, rng):
    """Standardized y whose sample correlation with x is exactly r."""
    zx = (x - x.mean()) / x.std()
    noise = rng.normal(size=x.size)
    noise -= noise.mean() + (noise @ zx) / (zx @ zx) * zx   # remove any part of the noise that tracks x
    noise /= noise.std()
    return r * zx + np.sqrt(1 - r ** 2) * noise


def _scatter_panels(panels, suptitle):
    fig, axes = subplots(1, len(panels), figsize=(3.6 * len(panels), 3.8))
    for ax, (title, xs, ys) in zip(axes, panels):
        ax.scatter(xs, ys, alpha=0.6, s=15)
        r = round(np.corrcoef(xs, ys)[0, 1], 2) + 0.0   # + 0.0 turns -0.0 into 0.0
        ax.set(title=f'{title}\nr = {r:.2f}', xlabel='x', ylabel='y')
    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()
    return fig


def plot_correlation_strength(seed=463):
    """Five simulated data sets with r from -0.9 to 0.9: the sign is the direction, the size is the tightness."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=100)
    panels = [(label, x, _with_correlation(x, r, rng))
              for label, r in [('Strong negative', -0.9), ('Moderate negative', -0.5),
                               ('No linear relationship', 0.0),
                               ('Moderate positive', 0.5), ('Strong positive', 0.9)]]
    return _scatter_panels(panels, 'What r measures: direction (sign) and how tightly points hug a line (size)')


def plot_correlation_pitfalls(seed=463):
    """Same r with different slopes, a strong curve with r near 0, and an outlier that creates r."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=100)
    y = _with_correlation(x, 0.9, rng)
    curve_x = np.linspace(-3, 3, 100)
    cloud_x = rng.normal(size=30)
    cloud_y = _with_correlation(cloud_x, 0.0, rng)
    panels = [('Steep line', x, 3 * y),
              ('Shallow line', x, 0.3 * y),
              ('Strong curve', curve_x, curve_x ** 2 + 0.5 * rng.normal(size=100)),
              ('No relationship + 1 outlier', np.append(cloud_x, 8), np.append(cloud_y, 8))]
    fig = _scatter_panels(panels, "What r doesn't tell you")
    steep, shallow = fig.axes[0], fig.axes[1]
    shallow.set_ylim(steep.get_ylim())   # same vertical scale, so the slopes can be compared
    return fig


def plot_plane(df, x1, x2, y, model):
    """Interactive 3-D scatter of y against x1 and x2, with the fitted plane. Drag to rotate."""
    g1 = np.linspace(df[x1].min(), df[x1].max(), 20)
    g2 = np.linspace(df[x2].min(), df[x2].max(), 20)
    G1, G2 = np.meshgrid(g1, g2)
    Z = model.predict(pd.DataFrame({x1: G1.ravel(), x2: G2.ravel()})).to_numpy().reshape(G1.shape)
    fig = go.Figure([
        go.Scatter3d(x=df[x1], y=df[x2], z=df[y], mode='markers', name='stores',
                     marker=dict(size=3, opacity=0.6)),
        go.Surface(x=G1, y=G2, z=Z, name='fitted plane', opacity=0.5,
                   colorscale='Reds', showscale=False),
    ])
    fig.update_layout(title=f'{y} ~ {x1} + {x2}: the least squares plane',
                      scene=dict(xaxis_title=x1, yaxis_title=x2, zaxis_title=y,
                                 camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8),
                                             projection=dict(type='orthographic'))),
                      width=800, height=600, margin=dict(l=0, r=0, t=40, b=0))
    return fig


def plot_slices(df, x, fixed, y, model, values, ax=None):
    """The fitted plane sliced at a few values of `fixed`: one line of y against x per value.

    Each point is colored like the slice whose `fixed` value is closest to its own.
    """
    if ax is None:
        _, ax = subplots(figsize=(8, 5.5))
    values = np.asarray(values)
    nearest = np.abs(df[fixed].to_numpy()[:, None] - values[None, :]).argmin(axis=1)
    xs = np.linspace(df[x].min(), df[x].max(), 100)
    slope = model.params[x]
    for i, value in enumerate(values):
        rows = nearest == i
        ax.scatter(df.loc[rows, x], df.loc[rows, y], alpha=0.45, s=18, color=f'C{i}')
        ax.plot(xs, model.predict(pd.DataFrame({x: xs, fixed: value})), color=f'C{i}', linewidth=2.5,
                label=f'{fixed} held at {value} (slope {slope:.3f})')
    ax.set(xlabel=x, ylabel=y,
           title=f'Slices of the plane: {y} vs. {x}, holding {fixed} fixed\n'
                 f'(each store is colored like the slice its {fixed} is closest to)')
    ax.legend(fontsize=9)
    return ax


def plot_correlation_matrix(df, columns, ax=None):
    """Heat map of the correlations among `columns`, with each r printed in its square."""
    corr = df[columns].corr()
    if ax is None:
        _, ax = subplots(figsize=(7, 6))
    image = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    for i in range(len(columns)):
        for j in range(len(columns)):
            r = round(corr.iloc[i, j], 2) + 0.0   # + 0.0 turns -0.0 into 0.0
            ax.text(j, i, f'{r:.2f}', ha='center', va='center',
                    color='white' if abs(r) > 0.7 else 'black')
    ax.set_xticks(range(len(columns)), columns, rotation=45, ha='right')
    ax.set_yticks(range(len(columns)), columns)
    ax.set_title('Correlation matrix (r)')
    ax.figure.colorbar(image, ax=ax, label='r')
    return ax


def plot_parallel_lines(df, x, y, group, model, ax=None):
    """Scatter colored by a categorical `group`, with the model's fitted line for each level."""
    if ax is None:
        _, ax = subplots(figsize=(7, 5))
    xs = np.linspace(df[x].min(), df[x].max(), 100)
    for i, level in enumerate(df[group].cat.categories):
        rows = df[group] == level
        ax.scatter(df.loc[rows, x], df.loc[rows, y], alpha=0.4, color=f'C{i}')
        line = pd.DataFrame({x: xs, group: pd.Categorical([level] * len(xs),
                                                           categories=df[group].cat.categories)})
        ax.plot(xs, model.predict(line), color=f'C{i}', linewidth=2.5, label=f'{group} = {level}')
    ax.set(xlabel=x, ylabel=y, title=f'{y} vs. {x}: one fitted line per level of {group}')
    ax.legend()
    return ax
