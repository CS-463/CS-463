"""Plotting helpers for regression_demos.ipynb.

The notebook stays focused on regression ideas; the matplotlib details live here.
Lines are passed around as a dict mapping a label to its (slope, intercept),
e.g. {'Line A': (0.1, 3.0)}.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.pyplot import subplots
from plotly.subplots import make_subplots
from scipy.stats import linregress, t as t_dist


def pick_spread_points(df, x='TV', n=3, low=0.1, high=0.9):
    """Return n rows of df spread evenly along column x.

    Rows are sorted by x and taken at evenly spaced quantile positions between
    low and high, so the same rows are chosen every time.
    """
    ordered = df.sort_values(x)
    positions = np.round(np.linspace(low, high, n) * (len(ordered) - 1)).astype(int)
    return ordered.iloc[positions]


def fit_line(x, y):
    """Return the slope and intercept (m, b) of the least squares line."""
    m, b = np.polyfit(x, y, deg=1)
    return m, b


def residuals(pts, m, b, x='TV', y='sales'):
    """Return the residuals e_i = y_i - (m x_i + b) for the line (m, b)."""
    return pts[y] - (m * pts[x] + b)


def score_table(pts, lines, x='TV', y='sales'):
    """Return a DataFrame with one row per line and one column per way of combining residuals."""
    rows = {}
    for label, (m, b) in lines.items():
        e = residuals(pts, m, b, x, y)
        rows[label] = {'sum of e': e.sum(),
                       'sum of |e|': e.abs().sum(),
                       'sum of e²': (e ** 2).sum()}
    return pd.DataFrame(rows).T.round(2)


def _equation(m, b):
    return f'y = {m:.3f}x + {b:.2f}'


def _color(lines, label):
    return f'C{list(lines).index(label)}'


def _base_axes(pts, x, y, lines, ax):
    """Scatter the points and set limits that keep every prediction in view."""
    if ax is None:
        _, ax = subplots(figsize=(7, 5))
    ax.scatter(pts[x], pts[y], s=60, color='black', zorder=3)
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    predictions = [m * pts[x] + b for m, b in lines.values()]
    values = np.concatenate([pts[y].to_numpy()] + [p.to_numpy() for p in predictions])
    ax.set_xlim(0, pts[x].max() * 1.2)   # room for residual labels on the right
    ax.set_ylim(min(0, values.min()), values.max() * 1.2)
    return ax


def _draw_line(ax, m, b, **kwargs):
    grid = np.linspace(*ax.get_xlim(), 100)
    ax.plot(grid, m * grid + b, **kwargs)


def plot_lines(pts, lines, x='TV', y='sales', ax=None):
    """Plot the points with every candidate line."""
    ax = _base_axes(pts, x, y, lines, ax)
    for label, (m, b) in lines.items():
        _draw_line(ax, m, b, color=_color(lines, label),
                   label=f'{label}: {_equation(m, b)}')
    ax.legend(loc='upper left')
    return ax


def plot_residuals(pts, lines, label, x='TV', y='sales', ax=None, show_rss=False):
    """Plot one line from lines, with each residual drawn and labeled.

    Axis limits come from all of lines, so plots for different lines share a scale.
    With show_rss=True, the title also shows the sum of squared residuals.
    """
    m, b = lines[label]
    ax = _base_axes(pts, x, y, lines, ax)
    _draw_line(ax, m, b, color=_color(lines, label),
               label=f'{label}: {_equation(m, b)}')

    for xi, yi in zip(pts[x], pts[y]):
        y_hat = m * xi + b
        ax.plot([xi, xi], [y_hat, yi], color='red', linestyle='--')
        ax.annotate(f'e = {yi - y_hat:+.2f}', xy=(xi, (yi + y_hat) / 2),
                    xytext=(10, 0), textcoords='offset points',
                    va='center', color='red', zorder=4,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor='none', alpha=0.85))

    title = f'Residuals for {label}'
    if show_rss:
        title += f'\nRSS = {(residuals(pts, m, b, x, y) ** 2).sum():.1f}'
    ax.set_title(title)
    ax.legend(loc='upper left')
    return ax


LOSSES = {
    'RSS (sum of e²)': lambda e: np.sum(e ** 2),
    'sum of |e|': lambda e: np.sum(np.abs(e)),
}


def _losses_along(xs, ys, params):
    """Return {loss name: array of that loss for each (m, b) in params}."""
    return {name: np.array([loss(ys - (m * xs + b)) for m, b in params])
            for name, loss in LOSSES.items()}


def loss_curves(pts, m, b_values, x='TV', y='sales'):
    """Return {loss name: array of that loss at each b in b_values}, with slope fixed at m."""
    return _losses_along(pts[x].to_numpy(), pts[y].to_numpy(), [(m, b) for b in b_values])


def loss_explorer(pts, sweep, values, m=None, b=None, x='TV', y='sales'):
    """Interactive figure: sweep one line parameter while the other stays fixed.

    sweep='b' varies the intercept over values (pass the fixed slope m);
    sweep='m' varies the slope over values (pass the fixed intercept b).

    Left: the points, the line y = m x + b, and its residuals.
    Right: the loss at every value visited so far, one new point per slider step.
    Buttons switch the right plot between the losses in LOSSES.
    """
    xs, ys = pts[x].to_numpy(), pts[y].to_numpy()
    values = np.asarray(values, dtype=float)
    if sweep == 'b':
        params = [(m, v) for v in values]
        axis_title, fixed_text, fmt = 'intercept b', f'slope fixed at {m:.3f}', '.1f'
    elif sweep == 'm':
        params = [(v, b) for v in values]
        axis_title, fixed_text, fmt = 'slope m', f'intercept fixed at {b:.2f}', '.3f'
    else:
        raise ValueError("sweep must be 'b' or 'm'")
    curves = _losses_along(xs, ys, params)
    names = list(curves)

    x_max = xs.max() * 1.1
    grid = np.array([0.0, x_max])

    def line_xy(mi, bi):
        return grid.tolist(), (mi * grid + bi).tolist()

    def residual_xy(mi, bi):
        rx, ry = [], []
        for xi, yi in zip(xs, ys):
            rx += [float(xi), float(xi), None]
            ry += [float(yi), float(mi * xi + bi), None]
        return rx, ry

    def trace_data(i):
        """x and y for every trace that changes with the slider (traces 1 onward)."""
        mi, bi = params[i]
        lx, ly = line_xy(mi, bi)
        rx, ry = residual_xy(mi, bi)
        xs_upd, ys_upd = [lx, rx], [ly, ry]
        for name in names:
            xs_upd += [values[:i + 1].tolist(), [float(values[i])]]
            ys_upd += [curves[name][:i + 1].tolist(), [float(curves[name][i])]]
        return xs_upd, ys_upd

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=(f'{y} vs. {x}  ({fixed_text})',
                                        f'Loss vs. {axis_title}'))
    x0, y0 = trace_data(0)

    # Trace 0: data (never changes). Traces 1-2: line and residuals.
    fig.add_trace(go.Scatter(x=xs, y=ys, mode='markers',
                             marker=dict(size=10, color='black')), 1, 1)
    fig.add_trace(go.Scatter(x=x0[0], y=y0[0], mode='lines',
                             line=dict(color='#1f77b4', width=3)), 1, 1)
    fig.add_trace(go.Scatter(x=x0[1], y=y0[1], mode='lines',
                             line=dict(color='red', dash='dash')), 1, 1)

    # Traces 3 onward: for each loss, the curve so far and the current point.
    for k, name in enumerate(names):
        fig.add_trace(go.Scatter(x=x0[2 + 2 * k], y=y0[2 + 2 * k], mode='lines+markers',
                                 line=dict(color='gray'), marker=dict(size=5),
                                 visible=(k == 0)), 1, 2)
        fig.add_trace(go.Scatter(x=x0[3 + 2 * k], y=y0[3 + 2 * k], mode='markers',
                                 marker=dict(size=12, color='#1f77b4'),
                                 visible=(k == 0)), 1, 2)

    changing = list(range(1, len(fig.data)))
    steps = []
    for i, v in enumerate(values):
        xs_upd, ys_upd = trace_data(i)
        steps.append(dict(method='restyle', label=format(v, fmt),
                          args=[{'x': xs_upd, 'y': ys_upd}, changing]))

    loss_range = {name: [0, curves[name].max() * 1.05] for name in names}
    buttons = []
    for k, name in enumerate(names):
        visible = [True, True, True] + [j == k for j in range(len(names)) for _ in range(2)]
        buttons.append(dict(label=name, method='update',
                            args=[{'visible': visible},
                                  {'yaxis2.range': loss_range[name],
                                   'yaxis2.title.text': name}]))

    line_values = np.concatenate([ys] + [mi * grid + bi for mi, bi in params])
    pad = (values.max() - values.min()) * 0.05
    fig.update_layout(
        height=500, showlegend=False,
        xaxis=dict(title=x, range=[0, x_max]),
        yaxis=dict(title=y, range=[min(0, line_values.min()), line_values.max() * 1.05]),
        xaxis2=dict(title=axis_title, range=[values.min() - pad, values.max() + pad]),
        yaxis2=dict(title=names[0], range=loss_range[names[0]]),
        updatemenus=[dict(type='buttons', direction='right', buttons=buttons,
                          x=1.0, xanchor='right', y=1.18, yanchor='top')],
        sliders=[dict(active=0, steps=steps, currentvalue=dict(prefix=f'{sweep} = '),
                      pad=dict(t=50))],
    )
    return fig


def rss_grid(pts, m_values, b_values, x='TV', y='sales'):
    """Return RSS for every (m, b) pair; rows follow b_values, columns follow m_values."""
    xs, ys = pts[x].to_numpy(), pts[y].to_numpy()
    M, B = np.meshgrid(m_values, b_values)
    predictions = M[..., None] * xs + B[..., None]
    return ((ys - predictions) ** 2).sum(axis=-1)


def rss_surface_3d(pts, m_values, b_values, x='TV', y='sales'):
    """Interactive 3-D plot of RSS over (m, b), with the minimum marked in red."""
    Z = rss_grid(pts, m_values, b_values, x, y)
    m_hat, b_hat = fit_line(pts[x], pts[y])
    rss_min = float(rss_grid(pts, [m_hat], [b_hat], x, y)[0, 0])

    fig = go.Figure(go.Surface(x=m_values, y=b_values, z=Z, colorscale='Viridis',
                               colorbar=dict(title='RSS'), opacity=0.9))
    fig.add_trace(go.Scatter3d(x=[m_hat], y=[b_hat], z=[rss_min], mode='markers',
                               marker=dict(size=6, color='red')))
    fig.update_layout(height=600, showlegend=False,
                      scene=dict(xaxis_title='slope m', yaxis_title='intercept b',
                                 zaxis_title='RSS'))
    return fig


def rss_contour(pts, m_values, b_values, lines=None, x='TV', y='sales', ax=None):
    """Contour map of RSS over (m, b), with any lines inside the grid marked as points."""
    Z = rss_grid(pts, m_values, b_values, x, y)
    if ax is None:
        _, ax = subplots(figsize=(7, 5))
    levels = np.geomspace(Z.min() * 1.05, Z.max(), 15)   # RSS spans orders of magnitude
    contours = ax.contour(m_values, b_values, Z, levels=levels, cmap='viridis')
    ax.clabel(contours, fmt='%.0f', fontsize=8)

    lines = lines or {}
    for label, (m, b) in lines.items():
        if min(m_values) <= m <= max(m_values) and min(b_values) <= b <= max(b_values):
            ax.plot(m, b, 'o', markersize=9, color=_color(lines, label))
            ax.annotate(label, (m, b), xytext=(6, 6), textcoords='offset points')

    ax.set_xlabel('slope m')
    ax.set_ylabel('intercept b')
    ax.set_title('RSS contours')
    return ax


def r_squared(x, y):
    """Return R² = 1 - RSS/TSS for the least squares line through (x, y)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    m, b = fit_line(x, y)
    rss = np.sum((y - (m * x + b)) ** 2)
    tss = np.sum((y - y.mean()) ** 2)
    return 1 - rss / tss


def _true_line(x):
    return 2 + 0.5 * x


def _true_curve(x):
    return 0.5 * (x - 5) ** 2


SCENARIOS = {
    'baseline': 'Baseline: straight line, a little noise',
    'noisy': 'Noisy y: straight line, lots of noise',
    'curved': 'Wrong form: straight line fit to a curve',
    'narrow': 'Narrow range of x',
    'outliers': 'Outliers in y',
}


def make_scenario(name, n=50, seed=463):
    """Return simulated (x, y, true_fn) for one of the SCENARIOS.

    All data are generated here with a fixed seed; none of it comes from Advertising.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, n)
    true_fn = _true_line
    if name == 'baseline':
        y = true_fn(x) + rng.normal(0, 0.5, n)
    elif name == 'noisy':
        y = true_fn(x) + rng.normal(0, 4, n)
    elif name == 'curved':
        true_fn = _true_curve
        y = true_fn(x) + rng.normal(0, 1, n)
    elif name == 'narrow':
        x = rng.uniform(4.5, 5.5, n)
        y = true_fn(x) + rng.normal(0, 0.5, n)
    elif name == 'outliers':
        y = true_fn(x) + rng.normal(0, 0.5, n)
        y[rng.choice(n, size=3, replace=False)] += 12
    else:
        raise ValueError(f'unknown scenario {name!r}; choose from {list(SCENARIOS)}')
    return x, y, true_fn


def plot_scenario(name, ax=None):
    """Plot a simulated scenario: data, true function, fitted line, and R²."""
    x, y, true_fn = make_scenario(name)
    m, b = fit_line(x, y)
    if ax is None:
        _, ax = subplots(figsize=(7, 5))

    ax.scatter(x, y, s=20, color='black', alpha=0.7, label='data')
    full = np.linspace(0, 10, 200)
    ax.plot(full, true_fn(full), color='gray', linestyle=':', linewidth=2.5,
            label='true function')
    fitted = np.linspace(x.min(), x.max(), 200)
    ax.plot(fitted, m * fitted + b, color='C0', linewidth=2,
            label=f'fitted line: {_equation(m, b)}')

    ax.set_xlim(0, 10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'{SCENARIOS[name]}   (R² = {r_squared(x, y):.2f})')
    ax.legend(loc='best', fontsize=9)
    return ax


def sample_r2(df, n=3, draws=1000, x='TV', y='sales', seed=463):
    """Draw random n-row samples from df; return the samples and each one's R²."""
    rng = np.random.default_rng(seed)
    samples = [df.iloc[rng.choice(len(df), size=n, replace=False)] for _ in range(draws)]
    return samples, np.array([r_squared(s[x], s[y]) for s in samples])


def null_r2(df, n=3, draws=1000, x='TV', y='sales', seed=463):
    """Draw "no relationship" samples and return them with each one's R².

    Each sample pairs the x values of n random rows with the y values of n *other*
    random rows, so any pattern in it is pure coincidence. Samples are (xs, ys) arrays.
    """
    rng = np.random.default_rng(seed)
    x_all, y_all = df[x].to_numpy(), df[y].to_numpy()
    samples = []
    for _ in range(draws):
        rows = rng.choice(len(df), size=2 * n, replace=False)
        samples.append((x_all[rows[:n]], y_all[rows[n:]]))
    return samples, np.array([r_squared(xs, ys) for xs, ys in samples])


R2_BINS = np.linspace(0, 1, 21)


def plot_r2_histogram(r2, marks=(), reference=None, shade_above=None, ax=None):
    """Histogram of R² values.

    marks: dotted black lines. reference: a dashed red line.
    shade_above: shade R² at or above this value and report the fraction of samples there.
    """
    r2 = np.asarray(r2)
    if ax is None:
        _, ax = subplots(figsize=(7, 5))
    _, _, bars = ax.hist(r2, bins=R2_BINS, color='C0', alpha=0.7)
    for value in marks:
        ax.axvline(value, color='black', linestyle=':')
    if reference is not None:
        ax.axvline(reference, color='red', linestyle='--', label=f'reference: R² = {reference:.2f}')

    title = f'R² from {len(r2)} random samples'
    if shade_above is not None:
        for bar in bars:
            if bar.get_x() + bar.get_width() / 2 >= shade_above:
                bar.set_color('C3')
        ax.axvspan(shade_above, 1, color='C3', alpha=0.1)
        ax.axvline(shade_above, color='C3', linestyle='--', label=f'R² = {shade_above:.3f}')
        title += f'\nfraction with R² ≥ {shade_above:.3f}: {np.mean(r2 >= shade_above):.3f}'
    if reference is not None or shade_above is not None:
        ax.legend(fontsize=9)

    ax.set_xlim(0, 1)
    ax.set_xlabel('R²')
    ax.set_ylabel(f'count out of {len(r2)} samples')
    ax.set_title(title)
    return ax


def _draw_sample(ax, xs, ys, df, x, y, title):
    """A sample's points and least squares line, drawn over all of df in light gray."""
    m, b = fit_line(xs, ys)
    full = np.linspace(df[x].min(), df[x].max(), 100)
    ax.scatter(df[x], df[y], s=10, color='lightgray')
    ax.scatter(xs, ys, s=60, color='black', zorder=3)
    ax.plot(full, m * full + b, color='C0', linewidth=2)
    ax.set_ylim(df[y].min() - 2, df[y].max() + 2)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)


def plot_sample_luck(df, n=3, draws=1000, x='TV', y='sales', seed=463):
    """Six random n-row samples: five spanning the range of R², then one more at random."""
    samples, r2 = sample_r2(df, n, draws, x, y, seed)
    order = np.argsort(r2)
    picks = list(order[np.round(np.linspace(0, draws - 1, 5)).astype(int)])
    picks.append(next(i for i in range(draws) if i not in picks))   # draws are already random

    fig, axes = subplots(2, 3, figsize=(15, 9))
    for k, (ax, i) in enumerate(zip(axes.flat, picks)):
        s = samples[i]
        _draw_sample(ax, s[x], s[y], df, x, y,
                     f'{"Another random sample:  " if k == 5 else ""}R² = {r2[i]:.2f}')
    fig.tight_layout()
    return fig


def plot_null_samples(df, n=3, k=3, x='TV', y='sales', seed=463):
    """The first k "no relationship" samples from null_r2, side by side."""
    samples, r2 = null_r2(df, n, k, x, y, seed)
    fig, axes = subplots(1, k, figsize=(5 * k, 4.5), sharey=True)
    for i, (ax, (xs, ys)) in enumerate(zip(np.atleast_1d(axes), samples)):
        _draw_sample(ax, xs, ys, df, x, y, f'No-relationship sample {i + 1}:  R² = {r2[i]:.2f}')
    fig.tight_layout()
    return fig


def null_r2_explorer(df, n=3, draws=1000, x='TV', y='sales', seed=463):
    """Interactive figure: build up a histogram of R² from "no relationship" samples.

    Left: the current sample and its fitted line. Right: the histogram of R² for every
    sample so far, with the current sample's bar in orange. The first 10 draws are added
    one at a time, then in bigger jumps. Drag the slider or press Play.
    """
    samples, r2 = null_r2(df, n, draws, x, y, seed)
    shown = _build_up_steps(draws)
    centers = (R2_BINS[:-1] + R2_BINS[1:]) / 2
    grid = np.array([df[x].min(), df[x].max()])
    blue, orange = '#1f77b4', '#ff7f0e'

    def frame(k):
        """Traces 1-3 and the layout changes after k draws."""
        xs, ys = samples[k - 1]
        m, b = fit_line(xs, ys)
        counts, _ = np.histogram(r2[:k], bins=R2_BINS)
        current = np.clip(np.digitize(r2[k - 1], R2_BINS) - 1, 0, len(counts) - 1)
        data = [go.Scatter(x=xs, y=ys),
                go.Scatter(x=grid, y=m * grid + b),
                go.Bar(x=centers, y=counts,
                       marker_color=[orange if j == current else blue for j in range(len(counts))])]
        layout = go.Layout(
            title=dict(text=f'Draw {k}: R² = {r2[k - 1]:.2f} (orange bar)'
                            f'      Histogram of all {k} draws so far'),
            yaxis2=dict(range=[0, counts.max() * 1.15 + 1]))
        return data, layout

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                        subplot_titles=(f'One "no relationship" sample of {n}',
                                        'R² of every sample so far'))
    data0, layout0 = frame(shown[0])
    # Trace 0: all rows in light gray (never changes). Traces 1-3: sample, line, histogram.
    fig.add_trace(go.Scatter(x=df[x], y=df[y], mode='markers',
                             marker=dict(size=5, color='lightgray')), 1, 1)
    fig.add_trace(go.Scatter(x=data0[0].x, y=data0[0].y, mode='markers',
                             marker=dict(size=12, color='black')), 1, 1)
    fig.add_trace(go.Scatter(x=data0[1].x, y=data0[1].y, mode='lines',
                             line=dict(color=blue, width=3)), 1, 1)
    fig.add_trace(go.Bar(x=centers, y=data0[2].y, width=R2_BINS[1] - R2_BINS[0],
                         marker_color=data0[2].marker.color), 1, 2)

    frames = []
    for k in shown:
        data, layout = frame(k)
        frames.append(go.Frame(data=data, traces=[1, 2, 3], layout=layout, name=str(k)))
    fig.frames = frames

    fig.update_layout(
        height=550, showlegend=False, title=layout0.title,
        xaxis=dict(title=x), yaxis=dict(title=y, range=[df[y].min() - 2, df[y].max() + 2]),
        xaxis2=dict(title='R²', range=[0, 1]),
        yaxis2=dict(title='count', range=layout0.yaxis2.range),
        **_animation_controls(shown, 'draws so far: '),
    )
    return fig


def _build_up_steps(draws):
    """Slider stops: 1 through 10 one at a time, then bigger jumps up to draws."""
    return [k for k in list(range(1, 11)) + [20, 50, 100, 200, 500, 1000] if k < draws] + [draws]


def _animation_controls(shown, prefix):
    """Play/Pause buttons and a slider for frames named str(k) for each k in shown."""
    instant = dict(mode='immediate', frame=dict(duration=0, redraw=True), transition=dict(duration=0))
    steps = [dict(method='animate', label=str(k), args=[[str(k)], instant]) for k in shown]
    play = dict(label='Play', method='animate',
                args=[None, dict(frame=dict(duration=600, redraw=True), fromcurrent=True,
                                 transition=dict(duration=0))])
    pause = dict(label='Pause', method='animate', args=[[None], instant])
    return dict(
        updatemenus=[dict(type='buttons', buttons=[play, pause], showactive=False,
                          direction='left', x=0.1, xanchor='right', y=0, yanchor='top',
                          pad=dict(r=10, t=87))],
        sliders=[dict(active=0, steps=steps, x=0.1, len=0.9, y=0, yanchor='top',
                      pad=dict(t=50), currentvalue=dict(prefix=prefix))],
    )


def make_truth(df, n=30, x='TV', y='sales', seed=463):
    """A known "true" line for simulations, with its numbers borrowed from all of df.

    The true slope and intercept are the least squares fit to every row, the noise sd
    is that fit's residual standard error, and x holds n fixed values drawn from df[x].
    """
    m, b = fit_line(df[x], df[y])
    rss = np.sum((df[y] - (m * df[x] + b)) ** 2)
    rng = np.random.default_rng(seed)
    return dict(m=m, b=b, sigma=np.sqrt(rss / (len(df) - 2)), x_name=x, y_name=y,
                x=rng.choice(df[x].to_numpy(), size=n, replace=False))


def true_se(truth, sigma=None, x=None):
    """The slope's actual standard error for this truth: sigma / sqrt(sum of (x - mean)²)."""
    xs = truth['x'] if x is None else np.asarray(x)
    sigma = truth['sigma'] if sigma is None else sigma
    return sigma / np.sqrt(np.sum((xs - xs.mean()) ** 2))


def simulate_fits(truth, draws=1000, sigma=None, x=None, seed=463):
    """Simulate samples of y = m x + b + noise at fixed x, and fit a line to each.

    Returns the simulated y values (one row per sample) and a DataFrame with each
    fit's slope, intercept, and slope standard error as scipy's linregress reports it.
    """
    xs = truth['x'] if x is None else np.asarray(x)
    sigma = truth['sigma'] if sigma is None else sigma
    rng = np.random.default_rng(seed)
    ys = truth['m'] * xs + truth['b'] + rng.normal(0, sigma, size=(draws, len(xs)))
    fits = [linregress(xs, row) for row in ys]
    return ys, pd.DataFrame({'slope': [f.slope for f in fits],
                             'intercept': [f.intercept for f in fits],
                             'stderr': [f.stderr for f in fits]})


def slope_explorer(truth, draws=1000, seed=463):
    """Interactive figure: build up a histogram of slope estimates from simulated samples.

    Left: the true line (dashed), the current sample and its fitted line, and a faint fan
    of recent fitted lines. Right: the histogram of every slope estimate so far, with the
    current one's bar in orange and the true slope dashed. Drag the slider or press Play.
    """
    ys, fits = simulate_fits(truth, draws, seed=seed)
    xs, slopes, intercepts = truth['x'], fits['slope'].to_numpy(), fits['intercept'].to_numpy()
    se = true_se(truth)
    bins = np.linspace(truth['m'] - 4 * se, truth['m'] + 4 * se, 33)
    centers = (bins[:-1] + bins[1:]) / 2
    grid = np.array([0.0, xs.max() * 1.05])
    shown = _build_up_steps(draws)
    blue, orange = '#1f77b4', '#ff7f0e'

    def frame(k):
        """Traces 1-4 and the layout changes after k samples."""
        fan_x, fan_y = [], []
        for i in range(max(0, k - 200), k):   # at most 200 lines, so the fan stays readable
            fan_x += [grid[0], grid[1], None]
            fan_y += [intercepts[i] + slopes[i] * grid[0], intercepts[i] + slopes[i] * grid[1], None]
        counts, _ = np.histogram(slopes[:k], bins=bins)
        current = np.clip(np.digitize(slopes[k - 1], bins) - 1, 0, len(counts) - 1)
        sd_text = f', SD = {slopes[:k].std(ddof=1):.4f}' if k > 1 else ''
        data = [go.Scatter(x=fan_x, y=fan_y),
                go.Scatter(x=xs, y=ys[k - 1]),
                go.Scatter(x=grid, y=intercepts[k - 1] + slopes[k - 1] * grid),
                go.Bar(x=centers, y=counts,
                       marker_color=[orange if j == current else blue for j in range(len(counts))])]
        layout = go.Layout(
            title=dict(text=f'Sample {k}: slope = {slopes[k - 1]:.4f} (orange bar)'
                            f'      {k} slope estimates so far{sd_text}'),
            yaxis2=dict(range=[0, counts.max() * 1.15 + 1]))
        return data, layout

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                        subplot_titles=(f'Simulated samples of {len(xs)} stores',
                                        'Slope estimates so far'))
    data0, layout0 = frame(shown[0])
    # Trace 0: the true line (never changes). Traces 1-4: fan, sample, fitted line, histogram.
    fig.add_trace(go.Scatter(x=grid, y=truth['m'] * grid + truth['b'], mode='lines',
                             line=dict(color='black', dash='dash', width=2)), 1, 1)
    fig.add_trace(go.Scatter(x=data0[0].x, y=data0[0].y, mode='lines',
                             line=dict(color='rgba(31, 119, 180, 0.2)', width=1)), 1, 1)
    fig.add_trace(go.Scatter(x=data0[1].x, y=data0[1].y, mode='markers',
                             marker=dict(size=8, color='black')), 1, 1)
    fig.add_trace(go.Scatter(x=data0[2].x, y=data0[2].y, mode='lines',
                             line=dict(color=blue, width=3)), 1, 1)
    fig.add_trace(go.Bar(x=centers, y=data0[3].y, width=bins[1] - bins[0],
                         marker_color=data0[3].marker.color), 1, 2)
    fig.add_vline(x=truth['m'], line=dict(color='black', dash='dash'), row=1, col=2)

    frames = []
    for k in shown:
        data, layout = frame(k)
        frames.append(go.Frame(data=data, traces=[1, 2, 3, 4], layout=layout, name=str(k)))
    fig.frames = frames

    line_ends = truth['b'] + truth['m'] * grid
    fig.update_layout(
        height=550, showlegend=False, title=layout0.title,
        xaxis=dict(title=truth['x_name'], range=list(grid)),
        yaxis=dict(title=f"{truth['y_name']} (simulated)",
                   range=[line_ends.min() - 3 * truth['sigma'], line_ends.max() + 3 * truth['sigma']]),
        xaxis2=dict(title='slope estimate', range=[bins[0], bins[-1]]),
        yaxis2=dict(title='count', range=layout0.yaxis2.range),
        **_animation_controls(shown, 'samples so far: '),
    )
    return fig


def plot_slope_histogram(slopes, true_slope, one_se=None, ax=None):
    """Histogram of slope estimates: true slope dashed, ±1 SD of the estimates shaded.

    one_se: a standard error from a single sample, drawn as an orange ±1 SE bar for comparison.
    """
    slopes = np.asarray(slopes)
    sd = slopes.std(ddof=1)
    if ax is None:
        _, ax = subplots(figsize=(8, 5))
    ax.hist(slopes, bins=30, color='C0', alpha=0.7)
    ax.axvline(true_slope, color='black', linestyle='--', label=f'true slope = {true_slope:.4f}')
    ax.axvspan(true_slope - sd, true_slope + sd, color='C0', alpha=0.15,
               label=f'±1 SD of the estimates ({sd:.4f})')
    if one_se is not None:
        ax.errorbar(true_slope, ax.get_ylim()[1] * 0.9, xerr=one_se, color='C1', capsize=8,
                    linewidth=3, label=f'±1 SE from sample 1 alone ({one_se:.4f})')
    ax.set_xlabel('slope estimate')
    ax.set_ylabel(f'count out of {len(slopes)} samples')
    ax.set_title(f'{len(slopes)} slope estimates: mean {slopes.mean():.4f}, SD {sd:.4f}')
    ax.legend(fontsize=9, loc='upper left')
    return ax


def plot_intervals(fits, true_slope, k=50, ax=None):
    """Slope ± 2 SE for the first k fits, gray if the interval contains the true slope, red if not."""
    fits = fits.iloc[:k]
    low, high = fits['slope'] - 2 * fits['stderr'], fits['slope'] + 2 * fits['stderr']
    covers = (low <= true_slope) & (true_slope <= high)
    if ax is None:
        _, ax = subplots(figsize=(8, 7))
    for i in range(len(fits)):
        color = 'gray' if covers.iloc[i] else 'C3'
        ax.plot([low.iloc[i], high.iloc[i]], [i + 1, i + 1], color=color, linewidth=2)
        ax.plot(fits['slope'].iloc[i], i + 1, 'o', color=color, markersize=4)
    ax.axvline(true_slope, color='black', linestyle='--', label=f'true slope = {true_slope:.4f}')
    ax.set_xlabel('slope')
    ax.set_ylabel('sample')
    ax.set_title(f'Slope ± 2 SE for {len(fits)} samples: {covers.sum()} contain the true slope')
    ax.legend(fontsize=9, loc='upper right')
    return ax


def plot_se_factors(truth, draws=1000, seed=463):
    """Three panels comparing slope estimates for the baseline truth vs. one change each.

    Less noise (sd ÷ 2), more data (each x used 4 times), and less spread in x (squeezed
    halfway toward its mean). The first two halve the standard error; the last doubles it.
    """
    xs = truth['x']
    changes = {'Less noise (noise sd ÷ 2)': dict(sigma=truth['sigma'] / 2),
               'More data (4× as many stores)': dict(x=np.tile(xs, 4)),
               'Less spread in TV (squeezed by half)': dict(x=xs.mean() + (xs - xs.mean()) / 2)}
    _, base = simulate_fits(truth, draws, seed=seed)
    se = true_se(truth)
    bins = np.linspace(truth['m'] - 8 * se, truth['m'] + 8 * se, 61)

    fig, axes = subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for i, (ax, (title, change)) in enumerate(zip(axes, changes.items())):
        _, changed = simulate_fits(truth, draws, seed=seed + i + 1, **change)
        ax.hist(base['slope'], bins=bins, color='gray', alpha=0.5,
                label=f"baseline: SD {base['slope'].std():.4f}")
        ax.hist(changed['slope'], bins=bins, color='C0', alpha=0.6,
                label=f"changed: SD {changed['slope'].std():.4f}")
        ax.axvline(truth['m'], color='black', linestyle='--')
        ax.set_xlabel('slope estimate')
        ax.set_title(title)
        ax.legend(fontsize=9)
    axes[0].set_ylabel(f'count out of {draws} samples')
    fig.tight_layout()
    return fig


def confidence_band(x, y, x0, level=0.95):
    """Least squares prediction of the *average* y at each x0, and its confidence half-width."""
    x, y, x0 = (np.asarray(v, dtype=float) for v in (x, y, x0))
    n = len(x)
    m, b = fit_line(x, y)
    s = np.sqrt(np.sum((y - (m * x + b)) ** 2) / (n - 2))
    se = s * np.sqrt(1 / n + (x0 - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
    return m * x0 + b, t_dist.ppf((1 + level) / 2, n - 2) * se


def plot_tv_decision(df, pts, extra=100, x='TV', y='sales', ax=None):
    """All of df with its line and 95% confidence band, Line C from pts, and the prediction at max x + extra."""
    x_max = df[x].max()
    x_new = x_max + extra
    grid = np.linspace(0, x_new * 1.05, 200)
    fit, half = confidence_band(df[x], df[y], grid)
    m3, b3 = fit_line(pts[x], pts[y])
    (y_new,), (h_new,) = confidence_band(df[x], df[y], [x_new])

    if ax is None:
        _, ax = subplots(figsize=(10, 6))
    ax.axvspan(x_max, grid[-1], color='gray', alpha=0.12, label='beyond our data')
    ax.scatter(df[x], df[y], s=12, color='C0', alpha=0.5, label=f'all {len(df)} markets')
    ax.plot(grid, fit, color='C0', linewidth=2, label='least squares line (all markets)')
    ax.fill_between(grid, fit - half, fit + half, color='C0', alpha=0.3,
                    label='95% confidence band for average sales')
    ax.plot(grid, m3 * grid + b3, color='C1', linestyle='--', label='Line C (3 stores)')
    ax.scatter(pts[x], pts[y], s=70, color='C1', edgecolor='black', zorder=3, label='the 3 stores')
    ax.errorbar(x_new, y_new, yerr=h_new, fmt='o', color='C3', capsize=6, markersize=8, zorder=4,
                label=f'prediction at {x} = {x_new:.0f}: {y_new:.1f} ± {h_new:.1f}')

    ax.set_xlim(0, grid[-1])
    ax.set_xlabel(f'{x} (thousands of dollars)')
    ax.set_ylabel(f'{y} (thousands of units)')
    ax.set_title(f'{y} vs. {x}: predicting {extra:g} beyond the largest budget')
    ax.legend(loc='upper left', fontsize=9)
    return ax
