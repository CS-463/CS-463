"""Plotting and widget helpers for this week's notebooks.

ch_6_and_10_01_ridge_and_lasso.ipynb, ch_6_and_10_01b_linear_algebra_refresher.ipynb, and
ch_6_and_10_02_neurons_to_networks.ipynb use this module. The notebook stays focused on
ideas; the matplotlib and ipywidgets details live here. The shared styling pieces
(colors, big fonts, the figure-redraw helper) and Demos F, G, and I are adapted from
ch_6_and_8_demos_helpers.py (not imported), so this folder works on its own, for
example on Colab.
"""
import warnings
from io import BytesIO
from types import SimpleNamespace

import ipywidgets as widgets
import numpy as np
from IPython.display import display
from matplotlib import get_backend, rc_context
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.pyplot import close, figure, subplots, switch_backend
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Load the backend now, outside any rc_context. Loading Jupyter's inline backend
# turns on interactive mode; if that first happens inside rc_context, exiting the
# context turns it back off, and later figures stop displaying automatically.
switch_backend(get_backend())

BIG_FONTS = {'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 15,
             'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 13}
HTML_STYLE = 'font-size: 18px; line-height: 1.5;'

# Okabe-Ito colors, which stay distinguishable under common color blindness.
CLASS_COLORS = {1: '#0072B2', 2: '#E69F00'}
CURVE_COLORS = {'training error': '#0072B2', 'test error': '#D55E00',
                'cross-validation error': '#009E73'}
OKABE_ITO = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00', '#F0E442',
             '#000000']
PENALTY_COLOR = '#D55E00'
BAYES_LINE = dict(colors='black', linestyles='--', linewidths=2.5)


def _light(color, amount=0.3):
    """The color mixed with white: amount = 1 is the color itself, 0 is white."""
    return tuple(amount * c + (1 - amount) for c in to_rgb(color))


def _figure_area():
    return widgets.Image(format='png', layout=widgets.Layout(max_width='100%'))


def _show(out, draw, size_percent=100):
    """Redraw a matplotlib figure into a _figure_area() Image widget.

    size_percent shrinks the displayed image to that percent of its full size; it
    still never grows past the width of the cell.

    Deliberately avoids ipywidgets.Output: some notebook frontends (VS Code, Cursor)
    also echo Output-captured figures into the cell, so the demo appears twice."""
    with rc_context(BIG_FONTS):
        fig = draw()
    buffer = BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    close(fig)
    png = buffer.getvalue()
    if size_percent != 100:
        width = int.from_bytes(png[16:20], 'big')   # PNG header: IHDR width in pixels
        out.layout.width = f'{round(width * size_percent / 100)}px'
    out.value = png


def _html(text):
    return f'<div style="{HTML_STYLE}">{text}</div>'


def _flexibility_row(slider, left='More flexible', right='Less flexible'):
    """A slider with bold end labels saying which direction is more flexible."""
    return widgets.HBox([widgets.HTML(_html(f'<b>{left}</b>')), slider,
                         widgets.HTML(_html(f'<b>{right}</b>'))],
                        layout=widgets.Layout(align_items='center'))


def _flexibility_ends(ax, left='← More flexible', right='Less flexible →', height=-0.11):
    for position, label, align in ((0, left, 'left'), (1, right, 'right')):
        ax.text(position, height, label, transform=ax.transAxes, ha=align, va='top',
                fontsize=15, fontweight='bold')



# --- Demo F: shrinking a wiggly fit ------------------------

WIGGLE_DEGREE = 12
WIGGLE_LAMBDAS = np.r_[0, np.logspace(-6, 3, 19)]    # 0 is plain least squares


def _wiggle_truth(x):
    return np.sin(2 * np.pi * x) + 0.5 * x


def wiggle_data(n=20, seed=472):
    """About 20 points from a smooth curve plus noise, and a large set of new points."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 1, n))
    x_new = rng.uniform(0, 1, 2000)
    return SimpleNamespace(x=x, y=_wiggle_truth(x) + rng.normal(0, 0.3, n),
                           x_new=x_new, y_new=_wiggle_truth(x_new) + rng.normal(0, 0.3, 2000))


def ridge_wiggle_demo(size_percent=100):
    """Demo F: ridge regression on 12 features made from one x, for each lambda.

    The model is make_pipeline(PolynomialFeatures(12), StandardScaler(), Ridge(alpha=λ)).
    Left: the points, the true curve, and the fitted curve. Right: the 12
    coefficients on a symmetric log scale, so both huge and tiny values show."""
    data = wiggle_data()
    curve_x = np.linspace(0, 1, 400)
    fits = []
    for lam in WIGGLE_LAMBDAS:
        model = make_pipeline(PolynomialFeatures(WIGGLE_DEGREE, include_bias=False),
                              StandardScaler(), Ridge(alpha=lam) if lam > 0 else LinearRegression())
        model.fit(data.x[:, None], data.y)
        fits.append(SimpleNamespace(
            curve=model.predict(curve_x[:, None]), coef=model[-1].coef_,
            train_mse=np.mean((model.predict(data.x[:, None]) - data.y)**2),
            new_mse=np.mean((model.predict(data.x_new[:, None]) - data.y_new)**2)))
    biggest = max(np.abs(f.coef).max() for f in fits)
    lam_slider = widgets.SelectionSlider(
        options=[('0' if lam == 0 else f'{lam:.0e}' if lam < 0.01 else f'{lam:g}', i)
                 for i, lam in enumerate(WIGGLE_LAMBDAS)],
        value=0, description='λ', style={'description_width': '30px'},
        layout=widgets.Layout(width='520px'))
    readout = widgets.HTML()
    out = _figure_area()

    def update(change=None):
        fit = fits[lam_slider.value]
        readout.value = _html(
            f'Sum of squared coefficients: <b>{np.sum(fit.coef**2):,.1f}</b> &nbsp; '
            f'Training MSE: <b>{fit.train_mse:.3f}</b> &nbsp; '
            f'MSE on 2,000 new points: <b>{fit.new_mse:.3f}</b>')

        def draw():
            fig, (ax, bars) = subplots(1, 2, figsize=(16, 6.5),
                                       gridspec_kw={'width_ratios': [1.5, 1]})
            ax.plot(curve_x, _wiggle_truth(curve_x), 'k--', lw=2, label='true curve')
            ax.plot(curve_x, fit.curve, color=PENALTY_COLOR, lw=3, label='fitted curve')
            ax.plot(data.x, data.y, 'o', ms=9, color=CLASS_COLORS[1], mec='black',
                    label='training points')
            ax.set_ylim(-2, 2.5)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f'Ridge with λ = {WIGGLE_LAMBDAS[lam_slider.value]:.2g}')
            ax.legend(loc='upper right')
            bars.bar(range(1, WIGGLE_DEGREE + 1), fit.coef, color=CLASS_COLORS[1])
            bars.set_yscale('symlog', linthresh=1)
            bars.set_ylim(-biggest * 2, biggest * 2)
            bars.axhline(0, color='black', lw=1)
            bars.set_xticks(range(1, WIGGLE_DEGREE + 1))
            bars.set_xlabel('feature (x, x², …, x¹²)')
            bars.set_ylabel('coefficient (symmetric log scale)')
            bars.set_title('The 12 coefficients')
            fig.tight_layout()
            return fig
        _show(out, draw, size_percent)

    lam_slider.observe(update, names='value')
    update()
    display(widgets.VBox([_flexibility_row(lam_slider), readout, out]))


# --- Demo G: ridge vs. lasso coefficient paths -------------

PATH_LAMBDAS = {'Ridge': np.logspace(-2, 5, 29), 'Lasso': np.logspace(-2, 3, 21)}
PATH_HIGHLIGHTED = 4


def _coefficient_paths(X, y):
    paths = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        for method, lambdas in PATH_LAMBDAS.items():
            make = (lambda lam: Ridge(alpha=lam)) if method == 'Ridge' else \
                (lambda lam: Lasso(alpha=lam, max_iter=100_000))
            paths[method] = np.array([make(lam).fit(X, y).coef_ for lam in lambdas])
    return paths


def coefficient_path_demo(X, y, names, size_percent=100):
    """Demo G: standardized coefficients against lambda, one line per feature.

    X must already be standardized (and one-hot encoded); names labels its columns.
    The four features with the biggest coefficients along the path get a color and
    a legend entry; the rest are gray. A vertical line marks the slider's lambda."""
    paths = _coefficient_paths(X, y)
    highlighted = {m: np.argsort(-np.abs(p).mean(axis=0))[:PATH_HIGHLIGHTED]
                   for m, p in paths.items()}
    method = widgets.ToggleButtons(options=list(PATH_LAMBDAS), value='Ridge')
    lam_slider = widgets.SelectionSlider(options=[0], description='λ',
                                         style={'description_width': '30px'},
                                         layout=widgets.Layout(width='520px'))
    readout = widgets.HTML()
    out = _figure_area()

    def set_options(change=None):
        lambdas = PATH_LAMBDAS[method.value]
        lam_slider.options = [(f'{lam:.3g}', i) for i, lam in enumerate(lambdas)]
        lam_slider.value = int(np.argmin(np.abs(np.log10(lambdas) - 1)))   # start at λ = 10

    def update(change=None):
        lambdas, path = PATH_LAMBDAS[method.value], paths[method.value]
        i = lam_slider.value
        n_zero = int(np.sum(path[i] == 0))
        readout.value = _html(f'{method.value}, λ = {lambdas[i]:.3g}: features with coefficient '
                              f'exactly 0: <b>{n_zero}</b> of {path.shape[1]}')

        def draw():
            fig, ax = subplots(figsize=(12, 7))
            for j in range(path.shape[1]):
                if j not in highlighted[method.value]:
                    ax.plot(lambdas, path[:, j], color='0.7', lw=1.5)
            for color, j in zip(OKABE_ITO, highlighted[method.value]):
                ax.plot(lambdas, path[:, j], color=color, lw=3, label=names[j])
            ax.axvline(lambdas[i], color='black', lw=1.5, ls=':')
            ax.axhline(0, color='black', lw=1)
            ax.set_xscale('log')
            ax.set_xlabel('λ (scikit-learn: alpha)')
            ax.set_ylabel('standardized coefficient')
            _flexibility_ends(ax)
            ax.set_title(f'{method.value}: every coefficient as λ grows')
            ax.legend(loc='upper right', title='largest along the path')
            fig.tight_layout()
            return fig
        _show(out, draw, size_percent)

    method.observe(set_options, names='value')
    set_options()
    lam_slider.observe(update, names='value')
    method.observe(update, names='value')
    update()
    display(widgets.VBox([method, _flexibility_row(lam_slider), readout, out]))


# --- Section 7: picturing ridge (Demos K and L) -----------------------------
# --- Section 8: ridge and the lasso as a budget (Demo M) --------------------

PIC_SWEEP = np.r_[0, np.logspace(-0.5, 3.2, 44)]     # Demo L
PIC_STEP = 1.0                            # contour heights: PIC_STEP * k**2
PIC_XLIM = (-0.7, 2.7)
PIC_YLIM = (-1.1, 2.3)
TOTAL_COLOR = '#0072B2'
SIDE_LAMBDAS = np.r_[0, np.logspace(-0.5, 3.2, 26)]   # Demo K
SIDE_XLIM = (-0.6, 2.8)
SIDE_YMAX = 240
SURFACE_TOP = 450                         # Demo L: the surface is cut off here
SURFACE_STEPS = np.array([5, 15, 35, 70, 120, 200])   # contour heights above the bottom
SURFACE_CMAP = 'viridis'
SURFACE_VIEW = (35, -60)                  # elevation and azimuth of the 3D view
BUDGET_REACH = np.round(np.arange(0.1, 2.45, 0.1), 2)   # Demo M: where the regions cross the axes
BUDGET_START = 1.0


def two_feature_data(n=40, r=0.8, seed=1):
    """Forty rows, two standardized features with correlation about 0.9, and a
    centered y (so there is no intercept to fit). The truth is y = 2 x1 + 0.3 x2
    + noise."""
    rng = np.random.default_rng(seed)
    z1, z2 = rng.normal(size=(2, n))
    X = np.column_stack([z1, r * z1 + np.sqrt(1 - r**2) * z2])
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = X @ np.array([2.0, 0.3]) + rng.normal(scale=0.5, size=n)
    return X, y - y.mean()


class _Bowls:
    """RSS, penalty, and their total for a two-feature, no-intercept fit.

    RSS(b) = rss_min + (b - ls)' A (b - ls), with A = X'X and ls the least squares
    coefficients, so every RSS value comes from the formula without refitting."""

    def __init__(self, X, y):
        self.A = X.T @ X
        self.b = X.T @ y
        self.ls = np.linalg.solve(self.A, self.b)
        self.rss_min = float(np.sum((y - X @ self.ls) ** 2))
        g1 = np.linspace(*PIC_XLIM, 260)
        g2 = np.linspace(*PIC_YLIM, 260)
        self.B1, self.B2 = np.meshgrid(g1, g2)

    def rss(self, b1, b2):
        d1, d2 = b1 - self.ls[0], b2 - self.ls[1]
        a = self.A
        return self.rss_min + a[0, 0] * d1**2 + 2 * a[0, 1] * d1 * d2 + a[1, 1] * d2**2

    def ridge(self, lam):
        return np.linalg.solve(self.A + lam * np.eye(2), self.b)

    def lasso(self, lam, sweeps=400):
        """Coordinate descent on RSS + lam * (|b1| + |b2|)."""
        beta = self.ls.copy()
        for _ in range(sweeps):
            for j in range(2):
                k = 1 - j
                z = self.b[j] - self.A[j, k] * beta[k]
                beta[j] = np.sign(z) * max(abs(z) - lam / 2, 0) / self.A[j, j]
        return beta


def _heights(k_max=10):
    return PIC_STEP * np.arange(1, k_max + 1) ** 2


def _coefficient_axes(ax, title=None):
    ax.axhline(0, color='black', lw=0.8)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlim(*PIC_XLIM)
    ax.set_ylim(*PIC_YLIM)
    ax.set_aspect('equal')
    ax.set_xlabel('β₁')
    ax.set_ylabel('β₂')
    if title:
        ax.set_title(title)


def _mark_least_squares(ax, bowls, label=True):
    ax.plot(*bowls.ls, 'o', ms=10, color='black', zorder=6)
    if label:
        ax.annotate('least squares', bowls.ls, xytext=(8, 8), textcoords='offset points',
                    fontsize=13)


def _star(ax, point, size=22):
    ax.plot(*point, '*', ms=size, color=TOTAL_COLOR, mec='black', zorder=7)


def _lam_label(lam):
    return '0' if lam == 0 else f'{lam:.2g}' if lam < 100 else f'{lam:,.0f}'


def _lam_slider(lambdas):
    return widgets.SelectionSlider(
        options=[(_lam_label(lam), i) for i, lam in enumerate(lambdas)],
        value=0, description='λ', style={'description_width': '30px'},
        layout=widgets.Layout(width='520px'))


def penalty_side_view_demo(x, y, size_percent=100):
    """Demo K: ridge with one coefficient, seen from the side.

    With one feature x (standardized) and a centered y, RSS(β) is a parabola with its
    bottom at least squares, λβ² is a parabola with its bottom at 0, and ridge's
    answer is the bottom of their sum, the loss. The slider is λ."""
    a, b = float(x @ x), float(x @ y)
    ls = b / a
    rss_min = float(np.sum((y - ls * x) ** 2))
    beta = np.linspace(*SIDE_XLIM, 400)
    rss = rss_min + a * (beta - ls) ** 2
    lam_slider = _lam_slider(SIDE_LAMBDAS)
    readout = widgets.HTML()
    out = _figure_area()

    def update(change=None):
        lam = SIDE_LAMBDAS[lam_slider.value]
        best = b / (a + lam)
        best_rss = rss_min + a * (best - ls) ** 2
        best_pen = lam * best ** 2
        readout.value = _html(
            f'Ridge\'s coefficient: <b>{best:.2f}</b> &nbsp; (least squares: {ls:.2f}) '
            f'&nbsp; At the star: RSS <b>{best_rss:.1f}</b> + penalty <b>{best_pen:.1f}</b> '
            f'= loss <b>{best_rss + best_pen:.1f}</b>')

        def draw():
            fig, ax = subplots(figsize=(12, 6.5))
            ax.plot(beta, rss, color='0.4', lw=3, label='RSS (from the data)')
            ax.plot(beta, lam * beta ** 2, color=PENALTY_COLOR, lw=3, label='penalty: λβ²')
            ax.plot(beta, rss + lam * beta ** 2, color=TOTAL_COLOR, lw=3,
                    label='loss = RSS + λβ²')
            ax.plot(ls, rss_min, 'o', ms=11, color='black', zorder=6)
            ax.annotate('least squares', (ls, rss_min), xytext=(-45, 14),
                        textcoords='offset points', fontsize=13)
            ax.axvline(best, color='0.3', ls=':', lw=1.5)
            ax.plot(best, best_rss, 'o', ms=9, color='0.4', mec='black', zorder=6)
            ax.plot(best, best_pen, 'o', ms=9, color=PENALTY_COLOR, mec='black', zorder=6)
            _star(ax, (best, best_rss + best_pen))
            ax.axhline(0, color='black', lw=0.8)
            ax.axvline(0, color='black', lw=0.8)
            ax.set_xlim(*SIDE_XLIM)
            ax.set_ylim(-5, SIDE_YMAX)
            ax.set_xlabel('β (the one coefficient)')
            ax.set_ylabel('height (cost)')
            ax.set_title(f'λ = {_lam_label(lam)}.  Star: the bottom of the loss.  '
                         'Dots: RSS and penalty at the star')
            ax.legend(loc='upper center', framealpha=0.95)
            fig.tight_layout()
            return fig
        _show(out, draw, size_percent)

    lam_slider.observe(update, names='value')
    update()
    display(widgets.VBox([_flexibility_row(lam_slider), readout, out]))


def loss_surface_demo(X, y, size_percent=100):
    """Demo L: the loss itself, L = RSS + λ(β₁² + β₂²), as one surface.

    Left: the surface in 3D (cut off at SURFACE_TOP). Right: the same surface from
    above, colored by height, with each contour labeled with its loss value. The
    colors use one fixed scale for every λ, so a rising floor shows up as a color
    change. The slider is λ."""
    bowls = _Bowls(X, y)
    rss = bowls.rss(bowls.B1, bowls.B2)
    pen = bowls.B1 ** 2 + bowls.B2 ** 2
    g1 = np.linspace(*PIC_XLIM, 140)
    g2 = np.linspace(*PIC_YLIM, 140)
    S1, S2 = np.meshgrid(g1, g2)
    s_rss = bowls.rss(S1, S2)
    s_pen = S1 ** 2 + S2 ** 2
    lam_slider = _lam_slider(PIC_SWEEP)
    readout = widgets.HTML()
    out = _figure_area()

    def update(change=None):
        lam = PIC_SWEEP[lam_slider.value]
        star = bowls.ridge(lam)
        low = float(bowls.rss(*star) + lam * star @ star)
        readout.value = _html(
            f'Bottom of the loss: <b>β₁ = {star[0]:.2f}, β₂ = {star[1]:.2f}</b> &nbsp; '
            f'loss there: <b>{low:.1f}</b>')

        def draw():
            fig = figure(figsize=(17, 7.4))
            ax3 = fig.add_subplot(1, 2, 1, projection='3d')
            ax = fig.add_subplot(1, 2, 2)
            surface = s_rss + lam * s_pen
            ax3.plot_surface(S1, S2, np.where(surface <= SURFACE_TOP, surface, np.nan),
                             cmap=SURFACE_CMAP, vmin=0, vmax=SURFACE_TOP, linewidth=0,
                             rstride=2, cstride=2, antialiased=True, alpha=0.9)
            ax3.plot([star[0]], [star[1]], [low], '*', ms=20, color='white', mec='black',
                     zorder=10)
            ax3.set_xlim(*PIC_XLIM)
            ax3.set_ylim(*PIC_YLIM)
            ax3.set_zlim(0, SURFACE_TOP)
            ax3.set_xlabel('β₁', labelpad=10)
            ax3.set_ylabel('β₂', labelpad=10)
            ax3.set_zlabel('loss', labelpad=10)
            ax3.view_init(elev=SURFACE_VIEW[0], azim=SURFACE_VIEW[1])
            ax3.set_title(f'The loss surface (cut off at {SURFACE_TOP})')

            loss = rss + lam * pen
            ax.contourf(bowls.B1, bowls.B2, np.minimum(loss, SURFACE_TOP),
                        levels=np.linspace(0, SURFACE_TOP, 31), cmap=SURFACE_CMAP,
                        vmin=0, vmax=SURFACE_TOP)
            levels = np.unique(np.round(low + SURFACE_STEPS))
            lines = ax.contour(bowls.B1, bowls.B2, loss, levels=levels, colors='black',
                               linewidths=1)
            ax.clabel(lines, fmt='%d', fontsize=11, inline=True)
            _mark_least_squares(ax, bowls)
            ax.plot(*star, '*', ms=24, color='white', mec='black', zorder=7)
            _coefficient_axes(ax, f'Seen from above, λ = {_lam_label(lam)}:\n'
                                  'numbers are loss values; star: the bottom')
            fig.tight_layout()
            return fig
        _show(out, draw, size_percent)

    lam_slider.observe(update, names='value')
    update()
    display(widgets.VBox([_flexibility_row(lam_slider), readout, out]))


def _ridge_in_budget(bowls, s):
    """Lowest-RSS point with b1^2 + b2^2 <= s, and the lambda that gives the same
    answer. Ridge's coefficients shrink steadily as lambda grows, so search lambda."""
    if bowls.ls @ bowls.ls <= s:
        return bowls.ls.copy(), 0.0
    lo, hi = -6.0, 8.0                      # log10(lambda)
    for _ in range(80):
        mid = (lo + hi) / 2
        beta = bowls.ridge(10 ** mid)
        lo, hi = (lo, mid) if beta @ beta < s else (mid, hi)
    lam = 10 ** ((lo + hi) / 2)
    return bowls.ridge(lam), lam


def _lasso_in_budget(bowls, s):
    """Lowest-RSS point with |b1| + |b2| <= s, and the matching lambda. The RSS is
    a quadratic, so check each of the diamond's four edges exactly."""
    if np.abs(bowls.ls).sum() <= s:
        return bowls.ls.copy(), 0.0
    corners = s * np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 0]], dtype=float)
    best, best_rss = None, np.inf
    for p, q in zip(corners[:-1], corners[1:]):
        d = q - p
        # RSS(p + t d) is a parabola in t; its lowest point, kept on the edge
        t = np.clip(d @ bowls.A @ (bowls.ls - p) / (d @ bowls.A @ d), 0, 1)
        point = p + t * d
        rss = bowls.rss(*point)
        if rss < best_rss:
            best, best_rss = point, rss
    best[np.abs(best) < 1e-12] = 0.0
    # The lambda with the same answer: where a coefficient isn't 0, the slope of the
    # RSS balances lambda (from the lasso's coordinate-descent update).
    j = int(np.argmax(np.abs(best)))
    lam = float(2 * (bowls.b - bowls.A @ best)[j] * np.sign(best[j]))
    return best, lam


def budget_regions_demo(X, y, size_percent=100):
    """Demo M: ridge and the lasso as a budget, side by side (ISL Figure 6.7).

    Ridge: lowest RSS with b1^2 + b2^2 <= s (a disk). Lasso: lowest RSS with
    |b1| + |b2| <= s (a diamond). The slider sets where both regions cross the axes,
    so they are the same size; each has its own budget s. Checkboxes add the RSS
    ovals, the best model in each budget, the path as the budget grows, and the point
    on ridge's circle closest to least squares. (No such point is drawn for the lasso:
    with two standardized features, the diamond's edges run along the ovals' axes, so
    the lasso's best point is always also its closest point.)"""
    bowls = _Bowls(X, y)
    rss_grid = bowls.rss(bowls.B1, bowls.B2)
    angles = np.linspace(0, 2 * np.pi, 400)
    circle = np.column_stack([np.cos(angles), np.sin(angles)])
    diamond = circle / np.abs(circle).sum(axis=1, keepdims=True)
    ridge_fits = [_ridge_in_budget(bowls, r ** 2) for r in BUDGET_REACH]
    lasso_fits = [_lasso_in_budget(bowls, r) for r in BUDGET_REACH]
    ls_norm = np.hypot(*bowls.ls)
    panels = (('Ridge', circle, ridge_fits, lambda r: r ** 2, 'β₁² + β₂²',
               lambda r: bowls.ls * min(1, r / ls_norm)),
              ('Lasso', diamond, lasso_fits, lambda r: r, '|β₁| + |β₂|',
               None))
    reach = widgets.SelectionSlider(
        options=[(f'{r:.1f}', i) for i, r in enumerate(BUDGET_REACH)],
        value=int(np.argmin(np.abs(BUDGET_REACH - BUDGET_START))),
        description='size', style={'description_width': '40px'},
        layout=widgets.Layout(width='520px'))
    boxes = {key: widgets.Checkbox(value=False, description=label, indent=False,
                                   layout=widgets.Layout(width='auto'))
             for key, label in (('rss', 'RSS ovals'),
                                ('best', 'best model in the budget'),
                                ('path', 'path as the budget grows'),
                                ('closest', 'closest point on the circle to least squares'))}
    readout = widgets.HTML()
    out = _figure_area()

    def update(change=None):
        k = reach.value
        r = BUDGET_REACH[k]
        on = {key: box.value for key, box in boxes.items()}
        parts = []
        for name, _, fits, budget, size, _ in panels:
            text = f'{name}: {size} ≤ s = <b>{budget(r):.2f}</b>'
            if on['best']:
                beta, lam = fits[k]
                text += (f' &nbsp; best: <b>β₁ = {beta[0]:.2f}, β₂ = {beta[1]:.2f}</b>'
                         f' (same as λ = {_lam_label(lam)})')
            parts.append(text)
        readout.value = _html('<br>'.join(parts))

        def draw():
            fig, axes = subplots(1, 2, figsize=(15, 7.2))
            for ax, (name, shape, fits, budget, size, closest) in zip(axes, panels):
                if on['rss']:
                    ax.contour(bowls.B1, bowls.B2, rss_grid,
                               levels=bowls.rss_min + _heights(), colors='0.6',
                               linewidths=1)
                ax.fill(*(r * shape).T, color=_light(PENALTY_COLOR, 0.25), zorder=1)
                ax.plot(*(r * shape).T, color=PENALTY_COLOR, lw=2.8, zorder=2)
                beta = fits[k][0]
                if on['path']:
                    trail = np.array([np.zeros(2)] + [fit[0] for fit in fits[:k + 1]])
                    ax.plot(*trail.T, '-', color=TOTAL_COLOR, lw=3.5, alpha=0.55, zorder=3)
                if on['best']:
                    ax.contour(bowls.B1, bowls.B2, rss_grid,
                               levels=[bowls.rss(*beta) + 1e-9], colors='0.15',
                               linewidths=2.4)
                    _star(ax, beta, size=24)
                    zero = [f'β{"₁₂"[j]} = 0' for j in range(2) if beta[j] == 0]
                    if zero:
                        ax.annotate(' and '.join(zero) + ' exactly', beta, xytext=(10, -34),
                                    textcoords='offset points', fontsize=14,
                                    fontweight='bold',
                                    bbox=dict(fc='white', ec='none', alpha=0.9))
                if on['closest'] and closest is not None:
                    point = closest(r)
                    if np.linalg.norm(point - bowls.ls) > 0.02:
                        ax.plot(*point, 'o', ms=13, mfc='white', mec='black', mew=2,
                                zorder=8)
                _mark_least_squares(ax, bowls)
                _coefficient_axes(ax, f'{name}: {size} ≤ s = {budget(r):.2f}')
            handles = [Line2D([], [], color=PENALTY_COLOR, lw=8, alpha=0.4,
                              label='every model within the budget')]
            if on['rss']:
                handles.append(Line2D([], [], color='0.6', lw=1, label='RSS ovals'))
            if on['best']:
                handles += [Line2D([], [], marker='*', ls='', ms=16, color=TOTAL_COLOR,
                                   mec='black', label='best model in the budget'),
                            Line2D([], [], color='0.15', lw=2.4,
                                   label='the oval through it')]
            if on['path']:
                handles.append(Line2D([], [], color=TOTAL_COLOR, lw=3.5, alpha=0.55,
                                      label='path as the budget grows'))
            if on['closest']:
                handles.append(Line2D([], [], marker='o', ls='', ms=10, mfc='white',
                                      mec='black', mew=2,
                                      label='closest point on the circle\nto least squares'))
            axes[0].legend(handles=handles, loc='upper right', fontsize=11.5,
                           framealpha=0.95)
            fig.tight_layout()
            return fig
        _show(out, draw, size_percent)

    reach.observe(update, names='value')
    for box in boxes.values():
        box.observe(update, names='value')
    update()
    checks = widgets.GridBox(list(boxes.values()),
                             layout=widgets.Layout(grid_template_columns='repeat(2, auto)'))
    display(widgets.VBox([_flexibility_row(reach, left='Smaller budget', right='Bigger budget'),
                          checks, readout, out]))


# --- Demo I: choosing lambda by cross-validation -----------

def plot_cv_lambda(curves, ols_mse, mean_mse):
    """Demo I: 10-fold CV MSE against lambda, one panel per method.

    curves maps a method name to (lambdas, cv_mse). Dashed: least squares (no
    penalty). Dotted: predicting the average salary for everyone. A star marks
    each curve's lowest point."""
    with rc_context(BIG_FONTS):
        fig, axes = subplots(1, len(curves), figsize=(8 * len(curves), 6.2), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, color, (name, (lambdas, mse)) in zip(axes, OKABE_ITO, curves.items()):
            best = int(np.argmin(mse))
            ax.plot(lambdas, mse, 'o-', color=color, lw=2.5, ms=6, label=f'{name}, 10-fold CV')
            ax.plot(lambdas[best], mse[best], '*', ms=22, color=color, mec='black',
                    label=f'lowest: λ ≈ {lambdas[best]:.2g}, MSE {mse[best]:,.0f}')
            ax.axhline(ols_mse, color='black', ls='--', lw=2,
                       label=f'least squares (λ = 0): {ols_mse:,.0f}')
            ax.axhline(mean_mse, color='black', ls=':', lw=2,
                       label=f'predict the average: {mean_mse:,.0f}')
            ax.set_xscale('log')
            ax.set_xlabel('λ (scikit-learn: alpha)')
            _flexibility_ends(ax)
            ax.set_title(name)
            ax.legend(loc='center left', fontsize=12)
        axes[0].set_ylabel('10-fold CV MSE')
        fig.tight_layout()
    return fig



# --- Sections 1-2: polynomial fits and overfitting -----------------------------------

def _poly_fit(degree, x, y):
    return make_pipeline(PolynomialFeatures(degree, include_bias=False),
                         LinearRegression()).fit(x[:, None], y)


def polynomial_error_table(data, degrees):
    """Training MSE and MSE on the 2,000 new points, for each polynomial degree."""
    rows = {}
    for degree in degrees:
        model = _poly_fit(degree, data.x, data.y)
        rows[degree] = {'training MSE': np.mean((model.predict(data.x[:, None]) - data.y)**2),
                        'MSE on 2,000 new points': np.mean((model.predict(data.x_new[:, None])
                                                            - data.y_new)**2),
                        'largest coefficient': np.abs(model[-1].coef_).max()}
    return rows


def plot_polynomial_fits(data, degrees=(1, 3, 12)):
    """One panel per degree: the 20 training points, the true curve, and the fitted curve."""
    curve_x = np.linspace(0, 1, 400)
    with rc_context(BIG_FONTS):
        fig, axes = subplots(1, len(degrees), figsize=(5.6 * len(degrees), 5.4), sharey=True)
        for ax, degree in zip(np.atleast_1d(axes), degrees):
            model = _poly_fit(degree, data.x, data.y)
            ax.plot(curve_x, _wiggle_truth(curve_x), 'k--', lw=2, label='true curve')
            ax.plot(curve_x, model.predict(curve_x[:, None]), color=PENALTY_COLOR, lw=3,
                    label='fitted curve')
            ax.plot(data.x, data.y, 'o', ms=8, color=CLASS_COLORS[1], mec='black',
                    label='training points')
            ax.set_ylim(-2, 2.5)
            ax.set_xlabel('x')
            ax.set_title(f'Degree {degree}')
        np.atleast_1d(axes)[0].set_ylabel('y')
        np.atleast_1d(axes)[0].legend(loc='upper right', fontsize=11)
        fig.tight_layout()
    return fig


def plot_resampled_fits(degrees=(3, 12), n_samples=8, first_seed=100):
    """Fit each degree to several fresh samples of 20 points, and overlay the curves.

    A steady model gives curves that nearly coincide; a high-variance one gives
    curves that disagree, most of all at the ends."""
    curve_x = np.linspace(0, 1, 400)
    with rc_context(BIG_FONTS):
        fig, axes = subplots(1, len(degrees), figsize=(7.5 * len(degrees), 5.8), sharey=True)
        for ax, degree in zip(np.atleast_1d(axes), degrees):
            for k in range(n_samples):
                sample = wiggle_data(seed=first_seed + k)
                model = _poly_fit(degree, sample.x, sample.y)
                ax.plot(curve_x, model.predict(curve_x[:, None]), color=OKABE_ITO[k % 7],
                        lw=2, alpha=0.9)
            ax.plot(curve_x, _wiggle_truth(curve_x), 'k--', lw=3, label='true curve')
            ax.set_ylim(-2, 2.5)
            ax.set_xlabel('x')
            ax.set_title(f'Degree {degree}: {n_samples} fits, {n_samples} different samples')
        np.atleast_1d(axes)[0].set_ylabel('y')
        np.atleast_1d(axes)[0].legend(loc='upper right')
        fig.tight_layout()
    return fig



# --- Linear algebra refresher (notebook 01b): vectors, Demos N and O --------------------

LA_LIM = 3.6                      # the plane is drawn from -LA_LIM to LA_LIM
W_COLOR = '#D55E00'               # w (Okabe-Ito vermillion)
X_COLOR = '#0072B2'               # x (blue)
SUM_COLOR = '#009E73'             # a sum, or a positive shadow (green)
NEG_COLOR = '#CC79A7'             # a negative shadow (reddish purple)
POS_SIDE = '#E69F00'              # the side where w.x + b > 0 (orange)
NEG_SIDE = '#56B4E9'              # the side where w.x + b < 0 (sky blue)
DOT_W = 2 * np.array([np.cos(np.radians(30)), np.sin(np.radians(30))])   # Demo N's w
LINE_POINTS = {'A': (2.0, 1.0), 'B': (-1.5, 0.5), 'C': (0.5, -2.0)}       # Demo O


def _arrow(ax, start, end, color, label=None, lw=3.5, ls='-', offset=(6, 6)):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, ls=ls,
                                mutation_scale=24, shrinkA=0, shrinkB=0))
    if label:
        ax.annotate(label, end, xytext=offset, textcoords='offset points', color=color,
                    fontsize=17, fontweight='bold')


def _plane(ax, xlim=(-LA_LIM, LA_LIM), ylim=(-LA_LIM, LA_LIM)):
    ax.axhline(0, color='black', lw=0.8)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.grid(True, color='0.9')
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')


def _angle_options(step=15):
    return [(f'{a}°', a) for a in range(0, 360, step)]


def plot_vector_arithmetic():
    """Left: w + x, tip to tail. Right: scaling a vector u by 2, 0.5, and -1."""
    w, x = np.array([2.0, 1.0]), np.array([-1.0, 3.0])
    u = np.array([1.5, 1.0])
    with rc_context(BIG_FONTS):
        fig, (left, right) = subplots(1, 2, figsize=(14, 6.6))
        _arrow(left, (0, 0), w, W_COLOR, 'w = (2, 1)', offset=(-10, -28))
        _arrow(left, (0, 0), x, X_COLOR, 'x = (-1, 3)', offset=(-40, 8))
        _arrow(left, w, w + x, X_COLOR, lw=2, ls='--')
        _arrow(left, (0, 0), w + x, SUM_COLOR, 'w + x = (1, 4)', offset=(8, -4))
        _plane(left, xlim=(-2, 3.6), ylim=(-0.6, 4.6))
        left.set_title('Adding: tip to tail\n(dashed: x, moved to start at the tip of w)')
        for k, color, label, off in ((2, SUM_COLOR, '2u = (3, 2)', (6, 4)),
                                     (1, X_COLOR, 'u = (1.5, 1)', (8, -14)),
                                     (0.5, '0.45', '0.5u', (-44, 4)),
                                     (-1, NEG_COLOR, '−u = (−1.5, −1)', (8, -26))):
            _arrow(right, (0, 0), k * u, color, label, lw=4.5 if k == 2 else 3.5, offset=off)
        _plane(right, xlim=(-2.4, 3.6), ylim=(-1.9, 2.8))
        right.set_title('Scaling: stretch, shrink, or flip')
        fig.tight_layout()
    return fig


def dot_product_demo(size_percent=100):
    """Demo N: w is fixed; sliders set x's direction and length. The shadow of x on
    w's line has signed length w.x / |w|. The readout computes w.x both ways."""
    w = DOT_W
    w_len = np.linalg.norm(w)
    unit = w / w_len
    angle = widgets.SelectionSlider(options=_angle_options(), value=75,
                                    description='direction of x',
                                    style={'description_width': '110px'},
                                    layout=widgets.Layout(width='560px'))
    length = widgets.SelectionSlider(options=[(f'{v:g}', v) for v in (0.5, 1, 1.5, 2, 2.5, 3)],
                                     value=2, description='length of x',
                                     style={'description_width': '110px'},
                                     layout=widgets.Layout(width='560px'))
    readout = widgets.HTML()
    out = _figure_area()

    def update(change=None):
        a, L = angle.value, length.value
        x = L * np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])
        dot = float(w @ x)
        if abs(dot) < 1e-9:
            dot = 0.0
        theta = (a - 30) % 360
        theta = 360 - theta if theta > 180 else theta
        shadow = dot / w_len
        kind = ('positive: the angle is under 90°' if dot > 0 else
                'zero: perpendicular' if dot == 0 else 'negative: the angle is over 90°')
        readout.value = _html(
            f'w = ({w[0]:.2f}, {w[1]:.2f}), &nbsp; x = ({x[0]:.2f}, {x[1]:.2f}), &nbsp; '
            f'angle between them: <b>{theta}°</b><br>'
            f'Recipe: w₁x₁ + w₂x₂ = ({w[0]:.2f})({x[0]:.2f}) + ({w[1]:.2f})({x[1]:.2f}) '
            f'= <b>{dot:.2f}</b><br>'
            f'Picture: ‖w‖ ‖x‖ cos θ = ({w_len:.0f})({L:g}) cos {theta}° = <b>{dot:.2f}</b>'
            f' &nbsp; Shadow: w·x / ‖w‖ = <b>{shadow:.2f}</b>')

        def draw():
            from matplotlib.patches import Arc
            fig, ax = subplots(figsize=(8.6, 8.6))
            t = np.array([-2 * LA_LIM, 2 * LA_LIM])
            ax.plot(t * unit[0], t * unit[1], color='0.6', ls='--', lw=1.2)
            foot = shadow * unit
            ax.plot([x[0], foot[0]], [x[1], foot[1]], color='0.3', ls=':', lw=2)
            if dot != 0:
                side = 0.14 * np.array([unit[1], -unit[0]])     # just beside w's line
                ax.plot([side[0], foot[0] + side[0]], [side[1], foot[1] + side[1]],
                        color=SUM_COLOR if dot > 0 else NEG_COLOR, lw=9,
                        solid_capstyle='butt', zorder=2)
            turn = (a - 30) % 360            # counterclockwise from w to x
            low, high = (30, 30 + turn) if turn <= 180 else (a, a + 360 - turn)
            ax.add_patch(Arc((0, 0), 1.3, 1.3, theta1=low, theta2=high, color='0.25', lw=1.8))
            mid = np.radians((low + high) / 2)
            ax.text(0.95 * np.cos(mid), 0.95 * np.sin(mid), 'θ', fontsize=17, ha='center',
                    va='center')
            _arrow(ax, (0, 0), w, W_COLOR, 'w')
            _arrow(ax, (0, 0), x, X_COLOR, 'x')
            _plane(ax)
            ax.set_title(f'w·x = {dot:.2f}: {kind}\n'
                         'Thick segment beside w’s line: the shadow of x '
                         '(green ahead, pink behind)', fontsize=15)
            fig.tight_layout()
            return fig
        _show(out, draw, size_percent)

    for s in (angle, length):
        s.observe(update, names='value')
    update()
    display(widgets.VBox([angle, length, readout, out]))


def line_from_dot_product_demo(size_percent=100):
    """Demo O: the line w.x + b = 0. Sliders: w's direction, w's length, and b.
    Shading shows the sign of the score w.x + b; dashed lines are scores of
    +-1, +-2, ...; the readout gives the scores of three fixed points."""
    g = np.linspace(-LA_LIM, LA_LIM, 301)
    G1, G2 = np.meshgrid(g, g)
    angle = widgets.SelectionSlider(options=_angle_options(), value=45,
                                    description='direction of w',
                                    style={'description_width': '110px'},
                                    layout=widgets.Layout(width='560px'))
    length = widgets.SelectionSlider(options=[(f'{v:g}', v) for v in (0.5, 1, 1.5, 2, 3)],
                                     value=1, description='length of w',
                                     style={'description_width': '110px'},
                                     layout=widgets.Layout(width='560px'))
    bias = widgets.SelectionSlider(options=[(f'{v:g}', v) for v in np.arange(-4, 4.01, 0.5)],
                                   value=-1.0, description='b',
                                   style={'description_width': '110px'},
                                   layout=widgets.Layout(width='560px'))
    readout = widgets.HTML()
    out = _figure_area()

    def update(change=None):
        a, L, b = angle.value, length.value, float(bias.value)
        w = L * np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])
        w[np.abs(w) < 1e-12] = 0.0
        score = w[0] * G1 + w[1] * G2 + b
        parts = []
        for name, p in LINE_POINTS.items():
            s = float(w @ np.array(p) + b)
            s = 0.0 if abs(s) < 1e-9 else s
            side = 'orange side' if s > 0 else 'blue side' if s < 0 else 'on the line'
            parts.append(f'{name}{p}: <b>{s:+.2f}</b> ({side})')
        readout.value = _html(
            f'w = ({w[0]:.2f}, {w[1]:.2f}), b = {b:g}. &nbsp; Line: '
            f'{w[0]:.2f}x₁ + {w[1]:.2f}x₂ + {b:g} = 0, at distance |b| / ‖w‖ = '
            f'<b>{abs(b) / L:.2f}</b> from the origin<br>Scores w·x + b: &nbsp; '
            + ' &nbsp; '.join(parts))

        def draw():
            fig, ax = subplots(figsize=(8.6, 8.6))
            ax.contourf(G1, G2, score, levels=[-1e9, 0, 1e9],
                        colors=[_light(NEG_SIDE, 0.35), _light(POS_SIDE, 0.35)])
            top = int(np.ceil(np.abs(score).max()))
            levels = [k for k in range(-top, top + 1) if k != 0]
            if levels:
                lines = ax.contour(G1, G2, score, levels=levels, colors='0.35',
                                   linestyles='--', linewidths=1.1)
                ax.clabel(lines, fmt=lambda v: f'{v:+.0f}', fontsize=13, inline=True)
            ax.contour(G1, G2, score, levels=[0], colors='black', linewidths=3)
            foot = -b * w / L ** 2
            _arrow(ax, foot, foot + w, W_COLOR, 'w')
            ax.plot(0, 0, 'o', ms=7, color='black')
            for name, p in LINE_POINTS.items():
                ax.plot(*p, 's', ms=11, color='white', mec='black', mew=2, zorder=5)
                ax.annotate(name, p, xytext=(8, 6), textcoords='offset points', fontsize=16,
                            fontweight='bold')
            _plane(ax)
            ax.grid(False)
            ax.set_title('Orange: w·x + b > 0.  Blue: w·x + b < 0.\n'
                         'Black line: w·x + b = 0.  Dashed: scores of ±1, ±2, …',
                         fontsize=15)
            fig.tight_layout()
            return fig
        _show(out, draw, size_percent)

    for s in (angle, length, bias):
        s.observe(update, names='value')
    update()
    display(widgets.VBox([angle, length, bias, readout, out]))


# --- Notebook 02: from one neuron to a network (Demos P, Q, R, S, T) ---------------------

PLUS_COLOR = '#0072B2'            # class 1: blue circles, as in the Chapter 5 and 6/8 demos
MINUS_COLOR = '#E69F00'           # class 0 (or -1): orange triangles
ONE_SIDE = '#56B4E9'              # shading where the model predicts class 1 (z > 0)
ZERO_SIDE = '#E69F00'             # shading where it predicts class 0
UNIT_COLORS = ('#009E73', '#CC79A7')   # v1 draft only: hidden units 1 and 2 (green, reddish purple)
HIDDEN_LINE = (dict(color='0.15', ls='-'), dict(color='0.15', ls=(0, (6, 4))))   # v2: unit 1, unit 2
HIDDEN_LINE_LABELS = (r'$h_1$ switches here ($z^{(1)}_1 = 0$)', r'$h_2$ switches here ($z^{(1)}_2 = 0$)')
HIDDEN_SHARPNESS = 6.0            # Demo S: the length of each hidden unit's w
POLYGON_RADIUS = 1.25             # Demo T: each unit's line sits this far from the origin
POLYGON_SHARPNESS = 4.0           # Demo T: the length of each hidden unit's w
POLYGON_START = 0.5               # Demo T: the output is POLYGON_START - (sum of hidden outputs)


def _sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -60, 60)))


def _relu(z):
    return np.maximum(0, z)


def _direction(angle_deg):
    a = np.radians(angle_deg)
    d = np.array([np.cos(a), np.sin(a)])
    d[np.abs(d) < 1e-12] = 0.0
    return d


def blobs_data(n=20, seed=7, negative=0):
    """Two clouds of n points each in the plane, labeled 1 and `negative` (0, or -1 for the
    perceptron), that a line can separate."""
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal([1.0, 0.9], 0.65, (n, 2)), rng.normal([-1.0, -0.9], 0.65, (n, 2))])
    y = np.r_[np.ones(n), np.full(n, float(negative))]
    order = rng.permutation(2 * n)
    return X[order], y[order]


def xor_data(n=25, spread=0.12, seed=0, negative=0):
    """n points around each corner of the unit square. Label 1 when exactly one of
    x1, x2 is 'on' (the corners (1, 0) and (0, 1)), and `negative` otherwise: XOR."""
    rng = np.random.default_rng(seed)
    corners = np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=float)
    labels = np.array([negative, negative, 1, 1], dtype=float)
    X = np.vstack([c + rng.normal(0, spread, (n, 2)) for c in corners])
    y = np.repeat(labels, n)
    order = rng.permutation(len(y))
    return X[order], y[order]


def rings_data(n_inner=60, n_outer=100, seed=0):
    """A disk of +1 points (radius under 1) inside a ring of -1 points (radius 1.5 to 2.2)."""
    rng = np.random.default_rng(seed)
    r = np.r_[np.sqrt(rng.uniform(0, 1, n_inner)), rng.uniform(1.5, 2.2, n_outer)]
    t = rng.uniform(0, 2 * np.pi, n_inner + n_outer)
    X = np.column_stack([r * np.cos(t), r * np.sin(t)])
    y = np.r_[np.ones(n_inner), -np.ones(n_outer)]
    return X, y


def _limits(X, pad=0.6):
    low, high = X.min(axis=0) - pad, X.max(axis=0) + pad
    return (low[0], high[0]), (low[1], high[1])


def _points(ax, X, y, wrong=None, size=70):
    for positive, color, marker in ((True, PLUS_COLOR, 'o'), (False, MINUS_COLOR, '^')):
        keep = (y == 1) == positive
        ax.scatter(X[keep, 0], X[keep, 1], s=size, color=color, marker=marker,
                   edgecolor='black', linewidth=0.8, zorder=4)
    if wrong is not None and wrong.any():
        ax.scatter(X[wrong, 0], X[wrong, 1], s=size * 3.2, facecolor='none', edgecolor='black',
                   linewidth=2.2, zorder=5)


def _data_axes(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')


def _grid(xlim, ylim, steps=300):
    g1, g2 = np.meshgrid(np.linspace(*xlim, steps), np.linspace(*ylim, steps))
    return g1, g2, np.column_stack([g1.ravel(), g2.ravel()])


def _two_sides(ax, g1, g2, score):
    ax.contourf(g1, g2, score, levels=[-1e9, 0, 1e9],
                colors=[_light(ZERO_SIDE, 0.35), _light(ONE_SIDE, 0.35)])


def _probability_shading(ax, g1, g2, p):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        'blue_white_orange', [_light(ZERO_SIDE, 0.75), (1, 1, 1), _light(ONE_SIDE, 0.75)])
    ax.contourf(g1, g2, p, levels=np.linspace(0, 1, 21), cmap=cmap, vmin=0, vmax=1)
    lines = ax.contour(g1, g2, p, levels=[0.1, 0.9], colors='0.35', linestyles='--',
                       linewidths=1.2)
    ax.clabel(lines, fmt=lambda v: f'p = {v:.1f}', fontsize=12, inline=True)


NODE_R = 0.5                      # radius of a node in the network diagrams


def _draw_network(ax, nodes, edges, radius=NODE_R):
    """Draw a network diagram. nodes: name -> (x, y, text inside, text below, kind), where
    kind is 'input', 'bias', or 'unit'. edges: (source, target, label, where along the edge
    the label sits, from 0 at the source to 1 at the target[, color])."""
    from matplotlib.patches import Circle, FancyArrowPatch
    style = {'input': (X_COLOR, 0.25), 'bias': ('0.45', 0.15), 'unit': (SUM_COLOR, 0.3)}
    for a, b, label, frac, *rest in edges:
        edge_color, label_color = (rest[0], rest[0]) if rest else ('0.35', W_COLOR)
        (xa, ya), (xb, yb) = nodes[a][:2], nodes[b][:2]
        d = np.array([xb - xa, yb - ya])
        u = d / np.linalg.norm(d)
        start, end = np.array([xa, ya]) + radius * u, np.array([xb, yb]) - radius * u
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=18, lw=1.8,
                                     color=edge_color, zorder=1))
        spot = start + frac * (end - start)
        ax.text(*spot, label, ha='center', va='center', fontsize=17, color=label_color,
                fontweight='bold', zorder=3,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none'))
    for name, (x, y, inside, below, kind) in nodes.items():
        color, amount = style[kind]
        ax.add_patch(Circle((x, y), radius, facecolor=_light(color, amount), edgecolor=color,
                            lw=2.2, zorder=2))
        ax.text(x, y, inside, ha='center', va='center', fontsize=17, zorder=4)
        if below:
            ax.text(x, y - radius - 0.15, below, ha='center', va='top', fontsize=14, zorder=4)
    ax.set_aspect('equal')
    ax.axis('off')


def plot_neuron_diagram():
    """One neuron: inputs x1, x2 and a constant 1, edges labeled w1, w2, b, and one unit
    that computes z = w.x + b and outputs g(z)."""
    nodes = {'x1': (0.8, 3.4, r'$x_1$', '', 'input'), 'x2': (0.8, 1.9, r'$x_2$', '', 'input'),
             'one': (0.8, 0.4, '1', '', 'bias'),
             'unit': (5.0, 1.9, r'$g(z)$', r'$z = w_1x_1 + w_2x_2 + b$', 'unit')}
    edges = [('x1', 'unit', r'$w_1$', 0.45), ('x2', 'unit', r'$w_2$', 0.45),
             ('one', 'unit', r'$b$', 0.45)]
    with rc_context(BIG_FONTS):
        fig, ax = subplots(figsize=(10, 4.6))
        _draw_network(ax, nodes, edges)
        ax.annotate('', xy=(7.3, 1.9), xytext=(5.0 + NODE_R, 1.9),
                    arrowprops=dict(arrowstyle='-|>', lw=1.8, color='0.35', mutation_scale=18))
        ax.text(7.45, 1.9, 'output', va='center', fontsize=16)
        ax.set_xlim(0.1, 8.8)
        ax.set_ylim(-0.6, 4.1)
        fig.tight_layout()
    return fig


EXAMPLE_W = np.array([1.0, 1.0])  # Section 2's running example: w = (1, 1), b = -1
EXAMPLE_B = -1.0


def plot_feature_vs_weight_space():
    """Section 2: the same model in two different planes. Left: feature space, where
    every dot is a data point and the model is a line. Right: weight space, where the
    whole model is a single dot."""
    X, y = blobs_data()
    w, b = EXAMPLE_W, EXAMPLE_B
    with rc_context(BIG_FONTS):
        fig, (left, right) = subplots(1, 2, figsize=(15, 6.8))
        _points(left, X, y, size=60)
        t = np.linspace(-3, 3, 2)
        left.plot(t, (-b - w[0] * t) / w[1], color='black', lw=3)
        blue = X[np.argmin(np.abs(X[:, 0] + 1.4) + np.abs(X[:, 1] - 0.4))]
        left.annotate('one data point:\nx = (x₁, x₂)', blue, xytext=(-2.45, 1.25),
                      fontsize=14, arrowprops=dict(arrowstyle='->', lw=1.5))
        left.text(0.55, 2.3, 'the model’s decision\nboundary:\nx₁ + x₂ − 1 = 0', fontsize=14)
        _data_axes(left, (-2.6, 2.8), (-2.6, 3.4))
        left.axhline(0, color='0.6', lw=0.8)
        left.axvline(0, color='0.6', lw=0.8)
        left.set_title('Feature space (this notebook)\naxes: the features.  Each dot: one data point',
                       fontsize=15)
        right.plot(*w, 'o', ms=16, color=W_COLOR, mec='black', zorder=5)
        right.annotate('the whole model:\nw = (1, 1)\n(b = −1 isn’t shown)', w, xytext=(1.3, 1.85),
                       fontsize=14, arrowprops=dict(arrowstyle='->', lw=1.5))
        right.axhline(0, color='black', lw=0.8)
        right.axvline(0, color='black', lw=0.8)
        right.set_xlim(-1, 3)
        right.set_ylim(-1, 3)
        right.set_aspect('equal')
        right.grid(True, color='0.9')
        right.set_xlabel('w₁')
        right.set_ylabel('w₂')
        right.set_title('Weight space (like ridge’s β₁, β₂ plots)\naxes: the weights.  One dot: one model',
                        fontsize=15)
        fig.tight_layout()
    return fig


def plot_step_function():
    """Section 2: the step activation, prediction against the score z."""
    z = np.linspace(-4, 4, 801)
    with rc_context(BIG_FONTS):
        fig, ax = subplots(figsize=(8, 3.8))
        ax.plot(z[z <= 0], np.zeros((z <= 0).sum()), color=W_COLOR, lw=3.5)
        ax.plot(z[z > 0], np.ones((z > 0).sum()), color=W_COLOR, lw=3.5)
        ax.plot([0], [0], 'o', ms=9, color=W_COLOR)
        ax.plot([0], [1], 'o', ms=9, mfc='white', mec=W_COLOR, mew=2.5)
        ax.axvline(0, color='0.5', lw=1, ls='--')
        ax.set_yticks([0, 1])
        ax.set_ylim(-0.3, 1.3)
        ax.set_xlabel('score z = w·x + b')
        ax.set_ylabel('prediction')
        ax.set_title('The step: 1 if z > 0, 0 otherwise')
        ax.grid(True, color='0.9')
        fig.tight_layout()
    return fig


SURFACE_LIM = (-2.0, 3.0)         # Section 2's 3D figure: the patch of feature space shown
SCORE_VIEW = (22, -70)            # elevation and azimuth, the same in all three panels


def _step_cliff(ax, w, b, cmap, lim):
    """Draw the step surface as two flat floors joined by a wall along w.x + b = 0
    (a gridded surface would make the wall jagged)."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    corners = np.array([[lim[0], lim[0]], [lim[1], lim[0]], [lim[1], lim[1]], [lim[0], lim[1]]])
    t = np.linspace(lim[0], lim[1], 400)
    edge = np.array([(x1, (-b - w[0] * x1) / w[1]) for x1 in t])
    edge = edge[(edge[:, 1] >= lim[0]) & (edge[:, 1] <= lim[1])]
    a, c = edge[0], edge[-1]
    for sign in (1, -1):
        side = [p for p in corners if sign * (p @ w + b) > 0]
        pts = np.array([a, c] + side)
        center = pts.mean(axis=0)
        pts = pts[np.argsort(np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0]))]
        poly = [(x1, x2, 1.0 if sign > 0 else 0.0) for x1, x2 in pts]
        ax.add_collection3d(Poly3DCollection([poly], facecolor=cmap(1.0 if sign > 0 else 0.0),
                                             alpha=0.85, edgecolor='none'))
    wall = [(a[0], a[1], 0), (c[0], c[1], 0), (c[0], c[1], 1), (a[0], a[1], 1)]
    ax.add_collection3d(Poly3DCollection([wall], facecolor='0.75', alpha=0.6, edgecolor='0.4'))


def plot_score_surfaces():
    """Section 2: for w = (1, 1) and b = -1, three surfaces over feature space, seen from
    the same angle. (1) The score z = w.x + b, a tilted plane. (2) p = g(z) with g the
    sigmoid. (3) The step. The black line on the floor is z = 0 in all three, and the
    surface's height changes only along the direction of w, so every level is a line."""
    from matplotlib.colors import LinearSegmentedColormap
    w, b = EXAMPLE_W, EXAMPLE_B
    g = np.linspace(*SURFACE_LIM, 121)
    G1, G2 = np.meshgrid(g, g)
    Z = w[0] * G1 + w[1] * G2 + b
    cmap = LinearSegmentedColormap.from_list('blue_orange', [ZERO_SIDE, '#f2f2f2', ONE_SIDE])
    x1 = np.linspace(*SURFACE_LIM, 50)
    x2 = (-b - w[0] * x1) / w[1]
    inside = (x2 >= SURFACE_LIM[0]) & (x2 <= SURFACE_LIM[1])
    panels = (('1. The score: a tilted plane\nz = x₁ + x₂ − 1', Z, 0.0, 'z', (-7, 5)),
              ('2. Sigmoid: p = g(z)\nan S-shaped sheet', _sigmoid(Z), 0.5, 'p', (-0.7, 1.05)),
              ('3. Step: a cliff\n1 if z > 0, 0 otherwise', None, 0.5, 'prediction',
               (-0.7, 1.05)))
    from matplotlib.gridspec import GridSpec
    with rc_context(BIG_FONTS):
        fig = figure(figsize=(18, 8.4))
        gs = GridSpec(2, 3, height_ratios=[1, 3.2], figure=fig)
        _thumb_row(fig, [gs[0, 0], gs[0, 1], gs[0, 2]], 'neuron', ('sum', 'act', 'act'),
                   ('the weighted sum $z$', 'the activation: sigmoid', 'the activation: step'))
        for k, (title, H, level, zlabel, zlim) in enumerate(panels, 1):
            ax = fig.add_subplot(gs[1, k - 1], projection='3d')
            floor = zlim[0]
            if H is None:
                _step_cliff(ax, w, b, cmap, SURFACE_LIM)
            else:
                ax.plot_surface(G1, G2, H, cmap=cmap, vmin=H.min(), vmax=H.max(), alpha=0.85,
                                rstride=2, cstride=2, linewidth=0, antialiased=True)
            ax.contour(G1, G2, Z, levels=[v for v in range(-5, 7) if v], zdir='z',
                       offset=floor, colors='0.6', linewidths=1)
            ax.plot(x1[inside], x2[inside], floor, color='black', lw=3.5, zorder=10)
            ax.plot(x1[inside], x2[inside], level, color='black', lw=2.5, ls='--', zorder=10)
            ax.set_xlim(*SURFACE_LIM)
            ax.set_ylim(*SURFACE_LIM)
            ax.set_zlim(*zlim)
            ax.set_xticks([-2, 0, 2])
            ax.set_yticks([-2, 0, 2])
            ax.set_zticks({1: [-4, 0, 4], 2: [0, 0.5, 1], 3: [0, 1]}[k])
            ax.set_xlabel('x₁')
            ax.set_ylabel('x₂')
            ax.set_zlabel(zlabel)
            ax.view_init(*SCORE_VIEW)
            ax.set_title(title, fontsize=15)
        fig.text(0.5, 0.02, 'Floor: feature space, with gray contour lines of z.  '
                 'Black line on the floor: z = 0.  Dashed: the same line, up on the surface.',
                 ha='center', fontsize=14)
        fig.subplots_adjust(left=0.0, right=0.97, wspace=0.05, hspace=0.22, bottom=0.07)
    return fig


def plot_scores_and_boundary():
    """Section 2: for w = (1, 1) and b = -1. Left: every point in feature space has a
    score z, written at grid points. Right: the points with z = 0 form a line, which
    splits the plane into a +1 half and a -1 half."""
    w, b = EXAMPLE_W, EXAMPLE_B
    lim = (-1.5, 3.5)
    g1, g2, grid = _grid(lim, lim)
    score = (grid @ w + b).reshape(g1.shape)
    with rc_context(BIG_FONTS):
        fig, (left, right) = subplots(1, 2, figsize=(15, 7))
        for v1 in range(-1, 4):
            for v2 in range(-1, 4):
                z = w[0] * v1 + w[1] * v2 + b
                color = ONE_SIDE if z > 0 else ZERO_SIDE if z < 0 else 'black'
                left.plot(v1, v2, 's', ms=7, color=color)
                left.annotate(f'{z:+.0f}' if z else '0', (v1, v2), xytext=(6, 6),
                              textcoords='offset points', fontsize=14, fontweight='bold',
                              color='black' if z == 0 else color)
        left.set_title('Every point has a score z = x₁ + x₂ − 1\n'
                       '(blue: z > 0, orange: z < 0, black: z = 0)', fontsize=15)
        _two_sides(right, g1, g2, score)
        right.contour(g1, g2, score, levels=[0], colors='black', linewidths=3)
        for pt, label, off in (((2, 2), '(2, 2): z = 3 → 1', (-70, 14)),
                               ((0, 0), '(0, 0): z = −1 → 0', (8, -22))):
            right.plot(*pt, 's', ms=11, color='white', mec='black', mew=2, zorder=5)
            right.annotate(label, pt, xytext=off, textcoords='offset points', fontsize=14)
        foot = -b * w / (w @ w)
        right.plot([0, foot[0]], [0, foot[1]], color='0.3', lw=2, ls=':')
        right.annotate('distance |b| / ‖w‖ ≈ 0.71', foot / 2, xytext=(-150, 40),
                       textcoords='offset points', fontsize=13,
                       arrowprops=dict(arrowstyle='->', lw=1.2))
        _arrow(right, foot, foot + w, W_COLOR, 'w = (1, 1)', offset=(4, 4))
        right.annotate('z = 0: the decision boundary', (-0.7, 1.7), xytext=(-1.35, 3.0),
                       fontsize=14, arrowprops=dict(arrowstyle='->', lw=1.5))
        right.set_title('The points with z = 0 form a line\n'
                        'blue half: predicted 1.  Orange half: predicted 0', fontsize=15)
        for ax in (left, right):
            _data_axes(ax, lim, lim)
            ax.axhline(0, color='0.5', lw=0.8)
            ax.axvline(0, color='0.5', lw=0.8)
        fig.tight_layout()
    return fig


def neuron_demo(X, y, activation='step', size_percent=100, start=(30, 1.0, 0.0)):
    """Demos P and R: one neuron over 2D data. Sliders set the direction of w, the
    length of w, and how far the line w.x + b = 0 sits from the origin (b is computed
    from that distance). activation='step' shades the two sides; 'sigmoid' shades the
    probability. The right panel shows the output against signed distance from the line."""
    xlim, ylim = _limits(X)
    g1, g2, grid = _grid(xlim, ylim)
    style = {'description_width': '200px'}
    angle = widgets.SelectionSlider(options=_angle_options(), value=start[0],
                                    description='direction of w', style=style,
                                    layout=widgets.Layout(width='640px'))
    length = widgets.SelectionSlider(options=[(f'{v:g}', v) for v in (0.5, 1, 2, 4, 8)],
                                     value=start[1], description='length of w',
                                     style=style, layout=widgets.Layout(width='640px'))
    dist = widgets.SelectionSlider(
        options=[(f'{v:+.1f}', v) for v in np.round(np.arange(-2, 2.01, 0.1), 1)],
        value=start[2], description='line’s distance from origin', style=style,
        layout=widgets.Layout(width='640px'))
    readout = widgets.HTML()
    out = _figure_area()

    def update(change=None):
        u, L, d = _direction(angle.value), length.value, dist.value
        w, b = L * u, -L * d + 0.0               # + 0.0 turns -0.0 into 0.0
        z = X @ w + b
        if activation == 'step':
            wrong = (z > 0) != (y == 1)
        else:
            wrong = (z >= 0) != (y == 1)
        readout.value = _html(
            f'w = ({w[0]:.2f}, {w[1]:.2f}), &nbsp; b = {b:.2f} &nbsp; (b = −‖w‖ × distance)'
            f'<br>Mistakes: <b>{int(wrong.sum())} of {len(y)}</b> points'
            + ('' if activation == 'step' else ' (counting p ≥ 0.5 as a vote for class 1)'))

        def draw():
            fig, (left, right) = subplots(1, 2, figsize=(15, 6.6),
                                          gridspec_kw={'width_ratios': [1.05, 1]})
            score = (grid @ w + b).reshape(g1.shape)
            if activation == 'step':
                _two_sides(left, g1, g2, score)
            else:
                _probability_shading(left, g1, g2, _sigmoid(score))
            left.contour(g1, g2, score, levels=[0], colors='black', linewidths=3)
            foot = d * u
            _arrow(left, foot, foot + 0.9 * u, W_COLOR, 'w', lw=3)
            _points(left, X, y, wrong)
            _data_axes(left, xlim, ylim)
            left.set_title('Black line: w·x + b = 0.  Circled: mistakes' if activation == 'step'
                           else 'Shading: p = g(w·x + b).  Black line: p = 0.5', fontsize=15)
            s = z / L                                     # signed distance from the line
            t = np.linspace(-3, 3, 601)
            if activation == 'step':
                right.plot(t, np.where(t > 0, 1, 0), color='black', lw=3)
                level = (y == 1).astype(float)
                right.set_yticks([0, 1])
                right.set_ylim(-0.3, 1.3)
                right.set_ylabel('output: 1 if z > 0, else 0')
                right.set_title('Step: only the side matters', fontsize=15)
            else:
                right.plot(t, _sigmoid(L * t), color='black', lw=3)
                level = (y == 1).astype(float)
                right.set_ylim(-0.15, 1.15)
                right.set_ylabel('output p')
                right.set_title(f'Sigmoid: p = g(‖w‖ × distance), ‖w‖ = {L:g}', fontsize=15)
            jitter = np.random.default_rng(0).uniform(-0.05, 0.05, len(y))
            for positive, color, marker in ((True, PLUS_COLOR, 'o'), (False, MINUS_COLOR, '^')):
                keep = (y == 1) == positive
                right.scatter(s[keep], level[keep] + jitter[keep], s=60, color=color,
                              marker=marker, edgecolor='black', linewidth=0.8, zorder=4)
            right.axvline(0, color='0.5', lw=1)
            right.set_xlim(-3, 3)
            right.set_xlabel('signed distance from the line (positive: the side w points to)')
            right.grid(True, color='0.9')
            fig.tight_layout()
            return fig
        _show(out, draw, size_percent)

    for s in (angle, length, dist):
        s.observe(update, names='value')
    update()
    display(widgets.VBox([angle, length, dist, readout, out]))


def perceptron_updates(X, y, eta=1.0, max_epochs=50):
    """Run the perceptron rule from w = 0, b = 0, passing through the points in order.
    Returns one record per update."""
    w, b = np.zeros(X.shape[1]), 0.0
    updates = []
    for epoch in range(max_epochs):
        changed = False
        for i, (x_i, y_i) in enumerate(zip(X, y)):
            z = w @ x_i + b
            if y_i * z <= 0:
                new_w, new_b = w + eta * y_i * x_i, b + eta * y_i
                updates.append(SimpleNamespace(i=i, epoch=epoch + 1, z=z, w=w, b=b,
                                               new_w=new_w, new_b=new_b))
                w, b, changed = new_w, new_b, True
        if not changed:
            break
    return updates


def plot_perceptron_loss():
    """The perceptron loss for one point, against y times its score z."""
    m = np.linspace(-3, 3, 601)
    with rc_context(BIG_FONTS):
        fig, ax = subplots(figsize=(9, 4.6))
        ax.plot(m, np.maximum(0, -m), color=PENALTY_COLOR, lw=3.5)
        ax.axvline(0, color='0.5', lw=1)
        ax.text(-1.5, 2.2, 'wrong side:\nloss = −y·z', ha='center', fontsize=15)
        ax.text(1.5, 0.35, 'right side: loss = 0', ha='center', fontsize=15)
        ax.set_xlabel('y × z   (positive: on the correct side)')
        ax.set_ylabel('perceptron loss')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-0.2, 3.2)
        ax.grid(True, color='0.9')
        fig.tight_layout()
    return fig


def perceptron_steps_demo(X, y, eta=1.0, size_percent=100):
    """Demo Q: step through the perceptron's updates. Left: the data, the line before
    (dashed) and after (solid), and the point that caused the update. Right: the same
    update in the (w1, w2) plane: w + eta * y * x, tip to tail."""
    updates = perceptron_updates(X, y, eta)
    xlim, ylim = _limits(X)
    g1, g2, grid = _grid(xlim, ylim)
    path = np.array([np.zeros(2)] + [u.new_w for u in updates])
    wlim = np.abs(path).max() * 1.25 + 0.5
    step = widgets.IntSlider(value=1, min=1, max=len(updates), description='update',
                             style={'description_width': '80px'},
                             layout=widgets.Layout(width='640px'))
    readout = widgets.HTML()
    out = _figure_area()

    def update(change=None):
        k = step.value
        u = updates[k - 1]
        x_i, y_i = X[u.i], y[u.i]
        wrong_after = y * (X @ u.new_w + u.new_b) <= 0
        readout.value = _html(
            f'Update {k} of {len(updates)} (pass {u.epoch} through the data). '
            f'Point {u.i} is class <b>{y_i:+.0f}</b> with score z = {u.z:.2f}: '
            + ('on the line' if u.z == 0 else 'wrong side') + '.<br>'
            f'w ← w + η·y·x = ({u.w[0]:.2f}, {u.w[1]:.2f}) + {eta:g}·({y_i:+.0f})·'
            f'({x_i[0]:.2f}, {x_i[1]:.2f}) = <b>({u.new_w[0]:.2f}, {u.new_w[1]:.2f})</b>; '
            f'&nbsp; b ← b + η·y = {u.b:.0f} + ({eta * y_i:+.0f}) = <b>{u.new_b:.0f}</b><br>'
            f'Mistakes on all {len(y)} points after this update: <b>{int(wrong_after.sum())}</b>')

        def draw():
            fig, (left, right) = subplots(1, 2, figsize=(15, 6.8))
            after = (grid @ u.new_w + u.new_b).reshape(g1.shape)
            _two_sides(left, g1, g2, after)
            if np.any(u.w != 0):
                before = (grid @ u.w + u.b).reshape(g1.shape)
                left.contour(g1, g2, before, levels=[0], colors='0.35', linewidths=2,
                             linestyles='--')
            left.contour(g1, g2, after, levels=[0], colors='black', linewidths=3)
            _points(left, X, y, wrong_after)
            left.scatter(*x_i, s=900, facecolor='none', edgecolor=SUM_COLOR, linewidth=3.5,
                         zorder=6)
            _data_axes(left, xlim, ylim)
            left.set_title('Dashed: line before.  Solid: line after.\n'
                           'Green ring: the point that caused the update', fontsize=15)
            right.plot(path[:k + 1, 0], path[:k + 1, 1], 'o-', color='0.7', ms=5, zorder=1)
            if np.any(u.w != 0):
                _arrow(right, (0, 0), u.w, '0.55', lw=2.5, ls='--')
            _arrow(right, u.w, u.new_w, SUM_COLOR, lw=3)
            _arrow(right, (0, 0), u.new_w, W_COLOR, lw=3.5)
            handles = [Line2D([], [], color=W_COLOR, lw=3.5, label='w after'),
                       Line2D([], [], color=SUM_COLOR, lw=3, label='η·y·x, the step'),
                       Line2D([], [], color='0.55', lw=2.5, ls='--', label='w before')]
            right.legend(handles=handles, loc='lower left', fontsize=14)
            right.axhline(0, color='black', lw=0.8)
            right.axvline(0, color='black', lw=0.8)
            right.set_xlim(-wlim, wlim)
            right.set_ylim(-wlim, wlim)
            right.set_aspect('equal')
            right.grid(True, color='0.9')
            right.set_xlabel('w₁')
            right.set_ylabel('w₂')
            right.set_title('The same update in the (w₁, w₂) plane\n'
                            '(gray: the path of w so far)', fontsize=15)
            fig.tight_layout()
            return fig
        _show(out, draw, size_percent)

    step.observe(update, names='value')
    update()
    display(widgets.VBox([step, readout, out]))


def _output_unit(H, y):
    """Fit the output neuron on the hidden features H by logistic regression (nearly no
    penalty). Returns its w, b, and the number of mistakes."""
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(C=1e6, max_iter=20_000).fit(H, y)
    w, b = model.coef_[0], model.intercept_[0]
    return w, b, int((model.predict(H) != y).sum())


def hidden_units_demo(X, y, size_percent=100, start=((0, 0.5), (90, 0.5))):
    """Demo S: two hidden neurons, h_k = g(w_k.x + b_k) with g the sigmoid. Each has a
    direction slider and a distance-from-origin slider. Left: the two lines in the input
    plane. Right: every point at its new coordinates (h1, h2), with the line that the
    output neuron (logistic regression on h1, h2) draws there."""
    xlim, ylim = (-0.55, 1.55), (-0.55, 1.55)
    g1, g2, grid = _grid(xlim, ylim)
    style = {'description_width': '230px'}
    sliders = []
    for k, (a0, d0) in enumerate(start, 1):
        sliders.append(widgets.SelectionSlider(
            options=_angle_options(), value=a0, description=f'unit {k}: direction of w',
            style=style, layout=widgets.Layout(width='640px')))
        sliders.append(widgets.SelectionSlider(
            options=[(f'{v:+.2f}', v) for v in np.round(np.arange(-1.0, 2.01, 0.05), 2)],
            value=d0, description=f'unit {k}: line’s distance', style=style,
            layout=widgets.Layout(width='640px')))
    readout = widgets.HTML()
    out = _figure_area()

    def update(change=None):
        params = []
        for k in range(2):
            u, d = _direction(sliders[2 * k].value), sliders[2 * k + 1].value
            params.append((HIDDEN_SHARPNESS * u, -HIDDEN_SHARPNESS * d, u, d))
        H = np.column_stack([_sigmoid(X @ w + b) for w, b, _, _ in params])
        w_out, b_out, mistakes = _output_unit(H, y)
        parts = [f'h{k + 1} = g({w[0]:.2f}x₁ + {w[1]:.2f}x₂ {"−" if b < 0 else "+"} {abs(b):.2f})'
                 for k, (w, b, _, _) in enumerate(params)]
        readout.value = _html(' &nbsp; &nbsp; '.join(parts) + '<br>'
                              f'The output neuron’s line makes <b>{mistakes}</b> mistakes '
                              f'out of {len(y)}' + (' — separable!' if mistakes == 0 else ''))

        def draw():
            fig, (left, right) = subplots(1, 2, figsize=(15, 7))
            Hg = np.column_stack([_sigmoid(grid @ w + b) for w, b, _, _ in params])
            final = (Hg @ w_out + b_out).reshape(g1.shape)
            _two_sides(left, g1, g2, final)
            for (w, b, u, d), color, name in zip(params, UNIT_COLORS, ('unit 1', 'unit 2')):
                left.contour(g1, g2, (grid @ w + b).reshape(g1.shape), levels=[0],
                             colors=[color], linewidths=3.5)
                _arrow(left, d * u, d * u + 0.35 * u, color, name, lw=3)
            _points(left, X, y, size=45)
            _data_axes(left, xlim, ylim)
            left.set_title('Input space: each hidden unit draws one line\n'
                           'Shading: the network’s final answer', fontsize=15)
            hx = np.linspace(-0.05, 1.05, 200)
            hg1, hg2 = np.meshgrid(hx, hx)
            hscore = w_out[0] * hg1 + w_out[1] * hg2 + b_out
            _two_sides(right, hg1, hg2, hscore)
            right.contour(hg1, hg2, hscore, levels=[0], colors='black', linewidths=2.5,
                          linestyles='--')
            _points(right, H, y, size=45)
            right.set_xlim(-0.05, 1.05)
            right.set_ylim(-0.05, 1.05)
            right.set_aspect('equal')
            right.set_xlabel('h₁ (unit 1’s output)', color=UNIT_COLORS[0])
            right.set_ylabel('h₂ (unit 2’s output)', color=UNIT_COLORS[1])
            right.set_title('Hidden space: each point at (h₁, h₂)\n'
                            'Dashed: the output neuron’s line', fontsize=15)
            fig.tight_layout()
            return fig
        _show(out, draw, size_percent)

    for s in sliders:
        s.observe(update, names='value')
    update()
    display(widgets.VBox(sliders + [readout, out]))


def plot_network_diagram():
    """The one-hidden-layer network, drawn like plot_neuron_diagram: every edge labeled with
    its weight, each layer's bias drawn as an edge from a constant 1, and a superscript (l)
    marking the layer."""
    R = 0.68
    nodes = {'x1': (0.8, 5.0, r'$x_1$', '', 'input'), 'x2': (0.8, 2.6, r'$x_2$', '', 'input'),
             'one_a': (0.8, 0.2, '1', '', 'bias'),
             'h1': (6.4, 5.0, r'$g(z^{(1)}_1)$', '', 'unit'),
             'h2': (6.4, 1.2, r'$g(z^{(1)}_2)$', '', 'unit'),
             'one_b': (6.4, -1.4, '1', '', 'bias'),
             'out': (11.6, 3.1, r'$g(z^{(2)})$', '', 'unit')}
    edges = [('x1', 'h1', r'$w^{(1)}_{11}$', 0.24), ('x2', 'h1', r'$w^{(1)}_{21}$', 0.3),
             ('one_a', 'h1', r'$b^{(1)}_1$', 0.24),
             ('x1', 'h2', r'$w^{(1)}_{12}$', 0.17), ('x2', 'h2', r'$w^{(1)}_{22}$', 0.32),
             ('one_a', 'h2', r'$b^{(1)}_2$', 0.24),
             ('h1', 'out', r'$w^{(2)}_1$', 0.5), ('h2', 'out', r'$w^{(2)}_2$', 0.5),
             ('one_b', 'out', r'$b^{(2)}$', 0.5)]
    with rc_context(BIG_FONTS):
        fig, ax = subplots(figsize=(14, 8.4))
        _draw_network(ax, nodes, edges, radius=R)
        ax.annotate('', xy=(13.9, 3.1), xytext=(11.6 + R, 3.1),
                    arrowprops=dict(arrowstyle='-|>', lw=1.8, color='0.35', mutation_scale=18))
        ax.text(14.05, 3.1, 'output', va='center', fontsize=16)
        formulas = ((6.4, 5.0 + R + 0.15, 'bottom',
                     r'$z^{(1)}_1 = w^{(1)}_{11}x_1 + w^{(1)}_{21}x_2 + b^{(1)}_1$'),
                    (6.4, 1.2 - R - 0.15, 'top',
                     r'$z^{(1)}_2 = w^{(1)}_{12}x_1 + w^{(1)}_{22}x_2 + b^{(1)}_2$'),
                    (11.6, 3.1 + R + 0.15, 'bottom',
                     r'$z^{(2)} = w^{(2)}_1 h_1 + w^{(2)}_2 h_2 + b^{(2)}$'))
        for x0, y0, va, text in formulas:
            ax.text(x0, y0, text, ha='center', va=va, fontsize=15,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none'))
        for x0, text in ((0.8, 'input\n(the features)'), (6.4, 'hidden layer\n(layer 1)'),
                         (11.6, 'output layer\n(layer 2)')):
            ax.text(x0, 6.75, text, ha='center', fontsize=15, fontweight='bold')
        for name, label, dy in (('h1', r'$h_1$', 0.25), ('h2', r'$h_2$', -0.75)):
            x0, y0 = nodes[name][:2]
            ax.text(x0 + R + 0.15, y0 + dy, label, fontsize=19, color=SUM_COLOR)
        ax.text(6.4, -2.6, r'$w^{(\ell)}_{jk}$: in layer $\ell$, the weight from input $j$ '
                r'into unit $k$.     $h_k = g(z^{(1)}_k)$: hidden unit $k$’s output',
                ha='center', fontsize=14, color='0.3')
        ax.set_xlim(0.0, 15.4)
        ax.set_ylim(-3.0, 7.5)
        fig.tight_layout()
    return fig


def plot_activations():
    """Three activations side by side: sigmoid, step, ReLU."""
    z = np.linspace(-4, 4, 801)
    with rc_context(BIG_FONTS):
        fig, axes = subplots(1, 3, figsize=(15, 4.4))
        for ax, (name, values) in zip(axes, (('Sigmoid: 1 / (1 + e^(−z))', _sigmoid(z)),
                                             ('Step: 1 if z > 0, else 0', np.where(z > 0, 1, 0)),
                                             ('ReLU: max(0, z)', _relu(z)))):
            ax.plot(z, values, color=W_COLOR, lw=3.5)
            ax.axhline(0, color='black', lw=0.8)
            ax.axvline(0, color='black', lw=0.8)
            ax.set_title(name)
            ax.set_xlabel('z = w·x + b')
            ax.grid(True, color='0.9')
        fig.tight_layout()
    return fig


def polygon_units(k):
    """Demo T's hand-set network with k hidden ReLU units. Unit j points in direction
    360j/k degrees, and its line sits POLYGON_RADIUS from the origin. Returns W, b, w_out, b_out."""
    angles = 360 * np.arange(k) / k
    W = POLYGON_SHARPNESS * np.array([_direction(a) for a in angles])
    b = -POLYGON_SHARPNESS * POLYGON_RADIUS * np.ones(k)
    return W, b, -np.ones(k), POLYGON_START


def polygon_units_demo(X, y, size_percent=100, start=3):
    """Demo T: a hand-set network with k hidden ReLU units on the rings data. Each unit
    is 0 inside its line and grows outside it, so the output, POLYGON_START minus their sum,
    is positive only inside the polygon the lines make."""
    xlim, ylim = (-2.6, 2.6), (-2.6, 2.6)
    g1, g2, grid = _grid(xlim, ylim)
    units = widgets.IntSlider(value=start, min=3, max=12, description='hidden units',
                              style={'description_width': '110px'},
                              layout=widgets.Layout(width='560px'))
    readout = widgets.HTML()
    out = _figure_area()

    def update(change=None):
        k = units.value
        W, b, w_out, b_out = polygon_units(k)
        score = _relu(X @ W.T + b) @ w_out + b_out
        wrong = y * score <= 0
        readout.value = _html(
            f'{k} hidden units. Parameters: {k}×2 weights + {k} biases in the hidden layer, '
            f'plus {k} weights + 1 bias in the output = <b>{4 * k + 1}</b>.<br>'
            f'Mistakes: <b>{int(wrong.sum())} of {len(y)}</b>')

        def draw():
            fig, ax = subplots(figsize=(8, 8))
            final = (_relu(grid @ W.T + b) @ w_out + b_out).reshape(g1.shape)
            _two_sides(ax, g1, g2, final)
            for j in range(k):
                ax.contour(g1, g2, (grid @ W[j] + b[j]).reshape(g1.shape), levels=[0],
                           colors='0.45', linewidths=1.2, linestyles='--')
            ax.contour(g1, g2, final, levels=[0], colors='black', linewidths=3)
            _points(ax, X, y, wrong, size=45)
            _data_axes(ax, xlim, ylim)
            ax.set_title(f'{k} hidden units, {k} lines (dashed)\n'
                         'Black: the network’s boundary', fontsize=15)
            fig.tight_layout()
            return fig
        _show(out, draw, size_percent)

    units.observe(update, names='value')
    update()
    display(widgets.VBox([units, readout, out]))


# --- Notebook 02 (v2): why combining neurons bends the boundary -----------------------------

XOR_LIM = (-0.5, 1.5)             # the patch of feature space around the unit square
XOR_UNITS = ((np.array([1.0, 1.0]), -0.5),    # unit 1: x1 + x2 - 0.5 > 0, "at least one on" (OR)
             (np.array([1.0, 1.0]), -1.5))    # unit 2: x1 + x2 - 1.5 > 0, "both on" (AND)
XOR_OUTPUT = (np.array([1.0, -1.0]), -0.5)    # output score: h1 - h2 - 0.5
TERRACE_VIEW = (24, -60)


def _clip(poly, w, b, keep_positive):
    """Clip a convex polygon (list of 2D points) to one side of the line w.x + b = 0."""
    out = []
    side = (lambda p: p @ w + b >= 0) if keep_positive else (lambda p: p @ w + b <= 0)
    for k in range(len(poly)):
        p, q = poly[k], poly[(k + 1) % len(poly)]
        if side(p):
            out.append(p)
        if side(p) != side(q):
            t = (p @ w + b) / ((p - q) @ w)
            out.append(p + t * (q - p))
    return out


def _regions(units, lim):
    """Split the square lim x lim into the regions cut out by the units' lines. Returns a
    list of (polygon, on), where on[k] is 1 if unit k's score is positive there."""
    square = [np.array(p, dtype=float) for p in
              ((lim[0], lim[0]), (lim[1], lim[0]), (lim[1], lim[1]), (lim[0], lim[1]))]
    pieces = [(square, ())]
    for w, b in units:
        new = []
        for poly, on in pieces:
            for flag in (1, 0):
                clipped = _clip(poly, w, b, keep_positive=bool(flag))
                if len(clipped) >= 3 and abs(_area(clipped)) > 1e-9:
                    new.append((clipped, on + (flag,)))
        pieces = new
    return pieces


def _area(poly):
    x, y = np.array(poly).T
    return 0.5 * (x @ np.roll(y, -1) - y @ np.roll(x, -1))


def _flat_pieces(ax, pieces, heights, cmap, vmin, vmax):
    """Draw each region as a flat polygon at its height, plus vertical walls where
    neighboring regions sit at different heights (walls are drawn along region edges)."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    for (poly, on), h in zip(pieces, heights):
        color = cmap((h - vmin) / (vmax - vmin) if vmax > vmin else 0.5)
        ax.add_collection3d(Poly3DCollection([[(p[0], p[1], h) for p in poly]],
                                             facecolor=color, alpha=0.9, edgecolor='none'))
    for i, ((pa, _), ha) in enumerate(zip(pieces, heights)):
        for j, ((pb, _), hb) in enumerate(zip(pieces, heights)):
            if j <= i or abs(ha - hb) < 1e-9:
                continue
            shared = [p for p in pa if any(np.allclose(p, q) for q in pb)]
            if len(shared) == 2:
                a, c = shared
                ax.add_collection3d(Poly3DCollection(
                    [[(a[0], a[1], ha), (c[0], c[1], ha), (c[0], c[1], hb), (a[0], a[1], hb)]],
                    facecolor='0.8', alpha=0.55, edgecolor='0.45', linewidth=0.6))


def _surface_axes(ax, lim, zlim, zlabel, title, zticks=None):
    if zticks is not None:
        ax.set_zticks(zticks)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_zlim(*zlim)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_zlabel(zlabel)
    ax.view_init(*TERRACE_VIEW)
    ax.set_title(title, fontsize=15)


XOR_SHARPNESS = 6.0               # Appendix A: sigmoid hidden units use 6 x (step unit's w and b)
H_LIM = (-0.12, 1.12)             # the patch of h-space shown


def _xor_hidden(P, activation):
    """Hidden outputs (h1, h2) of the XOR network at the points P (one row per point)."""
    Z = np.column_stack([P @ w + b for w, b in XOR_UNITS])
    if activation == 'step':
        return (Z > 0).astype(float)
    return _sigmoid(XOR_SHARPNESS * Z)


def _xor_output(H):
    w_out, b_out = XOR_OUTPUT
    return H @ w_out + b_out


def _blue_orange():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list('orange_blue', [ZERO_SIDE, '#f2f2f2', ONE_SIDE])


def _points3d(ax, X, y, Z, size=28):
    for positive, color, marker in ((True, PLUS_COLOR, 'o'), (False, MINUS_COLOR, '^')):
        keep = (y == 1) == positive
        ax.scatter(X[keep, 0], X[keep, 1], Z[keep], s=size, color=color, marker=marker,
                   edgecolor='black', linewidth=0.6, depthshade=False, zorder=10)


def _hidden_lines():
    t = np.linspace(*XOR_LIM, 200)
    lines = []
    for w, b in XOR_UNITS:
        lx, ly = t, (-b - w[0] * t) / w[1]
        keep = (ly >= XOR_LIM[0]) & (ly <= XOR_LIM[1])
        lines.append((lx[keep], ly[keep]))
    return lines


XOR_SPOTS = (((0, 0), 'A', 'below both lines'), ((1, 0), 'B', 'between the lines'),
             ((1, 1), 'C', 'above both lines'))
REGION_SHADES = ('0.93', '0.80', '0.67')


def _hspace_axes(ax):
    ax.set_xlim(*H_LIM)
    ax.set_ylim(*H_LIM)
    ax.set_aspect('equal')
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel('$h_1$ (hidden unit 1’s output)')
    ax.set_ylabel('$h_2$ (hidden unit 2’s output)')


def _jittered_hidden(X):
    H = _xor_hidden(X, 'step')
    return H + np.random.default_rng(1).normal(0, 0.025, H.shape)


# --- Thumbnails: which part of the network a figure shows ------------------------------------
# Convention (option b): a highlighted unit OUTLINE plus its highlighted incoming edges means
# "this figure shows the unit's weighted sum z"; a highlighted unit FILL plus its outgoing
# arrow means "this figure shows the activation g(z)". Everything else is faded.

FADED = '0.86'

MINI_LAYOUTS = {
    'neuron': {'nodes': {'x1': (0, 2.0, 'input'), 'x2': (0, 1.0, 'input'), 'one': (0, 0.0, 'bias'),
                         'out': (2.4, 1.0, 'unit')},
               'labels': {'x1': r'$x_1$', 'x2': r'$x_2$', 'one': '1', 'out': ''},
               'edges': [('x1', 'out'), ('x2', 'out'), ('one', 'out')],
               'arrow': ('out', (3.5, 1.0)), 'lim': ((-0.6, 4.1), (-0.6, 2.6))},
    'xor': {'nodes': {'x1': (0, 2.4, 'input'), 'x2': (0, 1.2, 'input'), 'one_a': (0, 0.0, 'bias'),
                      'h1': (2.4, 2.4, 'unit'), 'h2': (2.4, 0.9, 'unit'),
                      'one_b': (2.4, -0.4, 'bias'), 'out': (4.6, 1.5, 'unit')},
            'labels': {'x1': r'$x_1$', 'x2': r'$x_2$', 'one_a': '1', 'h1': r'$h_1$',
                       'h2': r'$h_2$', 'one_b': '1', 'out': ''},
            'edges': [(a, h) for h in ('h1', 'h2') for a in ('x1', 'x2', 'one_a')]
                     + [(a, 'out') for a in ('h1', 'h2', 'one_b')],
            'arrow': ('out', (5.7, 1.5)), 'lim': ((-0.6, 6.3), (-1.0, 3.0))},
}


def _mini_preset(layout, name):
    """(full nodes, outline-only nodes, highlighted edges, arrow on?) for a named view."""
    edges = MINI_LAYOUTS[layout]['edges']
    into = lambda *targets: {e for e in edges if e[1] in targets}
    if layout == 'neuron':
        return {'sum': ({'x1', 'x2', 'one'}, {'out'}, into('out'), False),
                'act': ({'out'}, set(), set(), True)}[name]
    inputs = {'x1', 'x2', 'one_a'}
    return {'h1': (inputs | {'h1'}, set(), into('h1'), False),
            'h2': (inputs | {'h2'}, set(), into('h2'), False),
            'hidden': (inputs | {'h1', 'h2'}, set(), into('h1', 'h2'), False),
            'hidden_sum': (inputs, {'h1', 'h2'}, into('h1', 'h2'), False),
            'hidden_act': ({'h1', 'h2'}, set(), {e for e in edges if e[0] in ('h1', 'h2')}, False),
            'sum': ({'h1', 'h2', 'one_b'}, {'out'}, into('out'), False),
            'act': ({'out'}, set(), set(), True),
            'path_z': (inputs | {'h1', 'h2', 'one_b'}, {'out'}, set(edges), False),
            'all': (set(MINI_LAYOUTS[layout]['nodes']), set(), set(edges), True)}[name]


def _mini_net(ax, layout, preset, caption='', labels=None):
    """A small network diagram with one part highlighted (see the convention above)."""
    from matplotlib.patches import Circle, FancyArrowPatch
    spec = MINI_LAYOUTS[layout]
    full, outline, hot_edges, arrow_on = _mini_preset(layout, preset)
    r = 0.36
    colors = {'input': X_COLOR, 'bias': '0.45', 'unit': SUM_COLOR}
    for a, b in spec['edges']:
        (xa, ya, _), (xb, yb, _) = spec['nodes'][a], spec['nodes'][b]
        u = np.array([xb - xa, yb - ya]) / np.hypot(xb - xa, yb - ya)
        hot = (a, b) in hot_edges
        ax.add_patch(FancyArrowPatch(np.array([xa, ya]) + r * u, np.array([xb, yb]) - r * u,
                                     arrowstyle='-|>', mutation_scale=10,
                                     lw=2.2 if hot else 1.0, color='0.2' if hot else FADED,
                                     zorder=1))
    for name, (x, y, kind) in spec['nodes'].items():
        color = colors[kind]
        fill = _light(color, 0.35 if kind != 'unit' else 0.55) if name in full else '0.97'
        edge = color if (name in full or name in outline) else FADED
        lw = 3.0 if name in outline else 2.0 if name in full else 1.2
        ax.add_patch(Circle((x, y), r, facecolor=fill, edgecolor=edge, lw=lw, zorder=2))
        hot = name in full or name in outline
        text = (labels or {}).get(name, spec['labels'][name])
        ax.text(x, y, text, ha='center', va='center', fontsize=9 if (text.startswith('$z') or len(text) > 2) else 11,
                color='black' if hot else '0.65', zorder=3)
    src, end = spec['arrow']
    x, y, _ = spec['nodes'][src]
    ax.add_patch(FancyArrowPatch((x + r, y), end, arrowstyle='-|>', mutation_scale=10,
                                 lw=2.2 if arrow_on else 1.0, color='0.2' if arrow_on else FADED))
    ax.set_xlim(*spec['lim'][0])
    ax.set_ylim(*spec['lim'][1])
    ax.set_aspect('equal')
    ax.axis('off')
    if caption:
        ax.set_title(caption, fontsize=12.5, pad=2)


def _xor_surface(ax, which, activation):
    """One 3D surface of the XOR network over feature space: 'h1', 'h2', or 'z2' (with the
    data lifted to their scores and a gray plane at height 0)."""
    cmap = _blue_orange()
    lines = _hidden_lines()
    k = {'h1': 0, 'h2': 1, 'z2': 2}[which]
    zlim, zticks, zlabel = (((-0.3, 1.2), [0, 1], ('$h_1$', '$h_2$')[k]) if k < 2
                            else ((-0.9, 0.8), [-0.5, 0, 0.5], '$z^{(2)}$'))
    if activation == 'step':
        pieces = _regions([XOR_UNITS[k]], XOR_LIM) if k < 2 else _regions(XOR_UNITS, XOR_LIM)
        heights = ([on[0] for _, on in pieces] if k < 2 else
                   [_xor_output(np.array([on], dtype=float))[0] for _, on in pieces])
        _flat_pieces(ax, pieces, heights, cmap, min(heights), max(heights))
    else:
        g = np.linspace(*XOR_LIM, 81)
        G1, G2 = np.meshgrid(g, g)
        Hg = _xor_hidden(np.column_stack([G1.ravel(), G2.ravel()]), activation)
        Sk = (Hg[:, k] if k < 2 else _xor_output(Hg)).reshape(G1.shape)
        ax.plot_surface(G1, G2, Sk, cmap=cmap, vmin=Sk.min(), vmax=Sk.max(),
                        alpha=0.8 if k < 2 else 0.5, rstride=2, cstride=2, linewidth=0)
    for j, (lx, ly) in enumerate(lines):
        if k == 2 or k == j:
            ax.plot(lx, ly, zlim[0], lw=3, **HIDDEN_LINE[j])
    if k == 2:
        X, y = xor_data()
        g = np.linspace(*XOR_LIM, 2)
        G1, G2 = np.meshgrid(g, g)
        ax.plot_surface(G1, G2, np.zeros_like(G1), color='0.5', alpha=0.18, linewidth=0)
        _points3d(ax, X, y, _xor_output(_xor_hidden(X, activation)), size=40)
    _surface_axes(ax, XOR_LIM, zlim, zlabel, '', zticks)


def _thumb_row(fig, gs_row, layout, presets, captions):
    axes = []
    for k, (preset, caption) in enumerate(zip(presets, captions)):
        ax = fig.add_subplot(gs_row[k])
        _mini_net(ax, layout, preset, caption)
        axes.append(ax)
    return axes


def plot_hidden_cliffs(activation='step'):
    """Each hidden unit's output over feature space (panels 1 and 2), and the output unit's
    score z2 = h1 - h2 - 0.5 that combines them (panel 3), with each point drawn at its score
    and a gray plane at height 0, where the output unit cuts."""
    from matplotlib.gridspec import GridSpec
    name = 'step' if activation == 'step' else r'\sigma'
    with rc_context(BIG_FONTS):
        fig = figure(figsize=(20, 7.6))
        gs = GridSpec(2, 3, height_ratios=[1, 3.2], figure=fig)
        _thumb_row(fig, [gs[0, 0], gs[0, 1], gs[0, 2]], 'xor', ('h1', 'h2', 'path_z'),
                   (rf'hidden unit 1: $h_1 = {name}(z^{{(1)}}_1)$',
                    rf'hidden unit 2: $h_2 = {name}(z^{{(1)}}_2)$',
                    r'output score: $z^{(2)} = h_1 - h_2 - 0.5$'))
        for k, (which, title) in enumerate((('h1', '1. “at least one is on”'),
                                            ('h2', '2. “both are on”'),
                                            ('z2', '3. the two cliffs combined'))):
            ax = fig.add_subplot(gs[1, k], projection='3d')
            _xor_surface(ax, which, activation)
            ax.set_title(title, fontsize=15)
        fig.subplots_adjust(left=0.02, right=0.97, wspace=0.1, hspace=0.15)
    return fig


def plot_hidden_bridge():
    """Stage 1, both units together: where the hidden layer (step units) sends each point.
    Left: feature space, cut by the two hidden lines into regions A, B, C. Right: h-space,
    where every point in a region lands on one spot."""
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import ConnectionPatch
    X, y = xor_data()
    g1, g2, grid = _grid(XOR_LIM, XOR_LIM)
    region = _xor_hidden(grid, 'step').sum(axis=1).reshape(g1.shape)     # 0, 1, or 2
    with rc_context(BIG_FONTS):
        fig = figure(figsize=(19, 7.2))
        gs = GridSpec(1, 3, width_ratios=[0.75, 2, 2], figure=fig)
        _mini_net(fig.add_subplot(gs[0]), 'xor', 'hidden', 'the hidden layer')
        left, right = fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
        left.contourf(g1, g2, region, levels=[-0.5, 0.5, 1.5, 2.5], colors=REGION_SHADES)
        for j, (lx, ly) in enumerate(_hidden_lines()):
            left.plot(lx, ly, lw=3, label=HIDDEN_LINE_LABELS[j], **HIDDEN_LINE[j])
        _points(left, X, y, size=40)
        anchors = {'A': (-0.3, -0.3), 'B': (0.5, 0.5), 'C': (1.3, 1.3)}
        for (h, name, where) in XOR_SPOTS:
            left.text(*anchors[name], name, fontsize=24, fontweight='bold', ha='center',
                      va='center', bbox=dict(boxstyle='circle', facecolor='white'))
        _data_axes(left, XOR_LIM, XOR_LIM)
        left.set_xticks([0, 0.5, 1])
        left.set_yticks([0, 0.5, 1])
        left.legend(loc='upper left', fontsize=12)
        left.set_title('Feature space: the hidden lines cut it into regions', fontsize=15)
        _points(right, _jittered_hidden(X), y, size=40)
        spot_label = {'A': (0.17, 0.17), 'B': (0.83, 0.17), 'C': (0.83, 0.83)}
        for (h, name, where) in XOR_SPOTS:
            right.text(*spot_label[name], name, fontsize=24, fontweight='bold', ha='center',
                       va='center', bbox=dict(boxstyle='circle', facecolor='white'))
            right.annotate(f'h = {h}', h, xytext=(0, -30), textcoords='offset points',
                           ha='center', fontsize=14)
        right.text(0, 1, '(0, 1): no point\nlands here', ha='left', va='top', fontsize=13,
                   color='0.4')
        _hspace_axes(right)
        right.grid(True, color='0.9')
        right.set_title('h-space: every point in a region lands on one spot', fontsize=15)
        for (h, name, where) in XOR_SPOTS:
            fig.add_artist(ConnectionPatch(xyA=anchors[name], coordsA=left.transData,
                                           xyB=spot_label[name], coordsB=right.transData,
                                           arrowstyle='-|>', mutation_scale=22, lw=1.8,
                                           color='0.3', shrinkA=18, shrinkB=18,
                                           connectionstyle='arc3,rad=-0.12'))
        fig.subplots_adjust(left=0.02, right=0.98, wspace=0.3)
    return fig


def _hspace_labeled(ax):
    """The step network's h-space: three spots labeled with z2, and the line z2 = 0."""
    X, y = xor_data()
    hg = np.linspace(*H_LIM, 200)
    HG1, HG2 = np.meshgrid(hg, hg)
    Zg = _xor_output(np.column_stack([HG1.ravel(), HG2.ravel()])).reshape(HG1.shape)
    t = np.array(H_LIM)
    _two_sides(ax, HG1, HG2, Zg)
    ax.plot(t, t - 0.5, color='black', lw=3)
    _points(ax, _jittered_hidden(X), y, size=45)
    offsets = {(0, 0): (15, 80), (1, 0): (-120, 70), (1, 1): (-215, -110)}
    for (h, name, where) in XOR_SPOTS:
        z = _xor_output(np.array([h], dtype=float))[0]
        ax.annotate(f'{name}: h = {h}\n$z^{{(2)}}$ = {h[0]} − {h[1]} − 0.5 = {z:+.1f}', h,
                    xytext=offsets[h], textcoords='offset points', fontsize=14,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                    arrowprops=dict(arrowstyle='->', lw=1.2))
    ax.text(-0.08, 1.06, '(0, 1): $z^{(2)}$ would be −1.5, but no point lands here',
            fontsize=12, color='0.35')
    ax.text(0.70, 0.47, '$z^{(2)} = 0$', fontsize=15, rotation=45)
    _hspace_axes(ax)


def _hspace_plane(ax, activation):
    """The output unit's score formula over h-space (a tilted plane), with the data on it."""
    X, y = xor_data()
    H = _xor_hidden(X, activation)
    hg = np.linspace(*H_LIM, 60)
    HG1, HG2 = np.meshgrid(hg, hg)
    Zg = _xor_output(np.column_stack([HG1.ravel(), HG2.ravel()])).reshape(HG1.shape)
    ax.plot_surface(HG1, HG2, Zg, cmap=_blue_orange(), vmin=-1.6, vmax=1.6, alpha=0.75,
                    linewidth=0)
    t1 = np.array([H_LIM[0] + 0.5, H_LIM[1]])
    ax.plot(t1, t1 - 0.5, -1.7, color='black', lw=3.5)
    _points3d(ax, H, y, _xor_output(H), size=40)
    ax.set_xlim(*H_LIM)
    ax.set_ylim(*H_LIM)
    ax.set_zlim(-1.7, 1.0)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_zticks([-1, 0, 1])
    ax.set_xlabel('$h_1$')
    ax.set_ylabel('$h_2$')
    ax.set_zlabel('$z^{(2)}$')
    ax.view_init(22, -150)
    ax.set_title('The formula $z^{(2)} = h_1 - h_2 - 0.5$ over h-space:\n'
                 'a tilted plane, with the data lying on it', fontsize=15)


def plot_hspace(activation='step'):
    """Stage 2: the output unit's weighted sum, which works in h-space."""
    from matplotlib.gridspec import GridSpec
    X, y = xor_data()
    with rc_context(BIG_FONTS):
        if activation == 'step':
            fig = figure(figsize=(12.5, 7.6))
            gs = GridSpec(1, 2, width_ratios=[0.8, 2.2], figure=fig)
            _mini_net(fig.add_subplot(gs[0]), 'xor', 'sum', 'the output unit’s weighted sum')
            ax = fig.add_subplot(gs[1])
            _hspace_labeled(ax)
            ax.set_title('What the output unit sees: three spots,\n'
                         'and one straight line separates them', fontsize=15)
        else:
            fig = figure(figsize=(20, 7.2))
            gs = GridSpec(1, 3, width_ratios=[0.8, 2.2, 2], figure=fig)
            _mini_net(fig.add_subplot(gs[0]), 'xor', 'sum', 'the output unit’s weighted sum')
            _hspace_plane(fig.add_subplot(gs[1], projection='3d'), activation)
            ax = fig.add_subplot(gs[2])
            H = _xor_hidden(X, activation)
            hg = np.linspace(*H_LIM, 200)
            HG1, HG2 = np.meshgrid(hg, hg)
            Zg = _xor_output(np.column_stack([HG1.ravel(), HG2.ravel()])).reshape(HG1.shape)
            _two_sides(ax, HG1, HG2, Zg)
            t = np.array(H_LIM)
            ax.plot(t, t - 0.5, color='black', lw=3)
            _points(ax, H, y, size=45)
            _hspace_axes(ax)
            ax.set_title('From above: one straight line\nseparates the classes', fontsize=15)
        fig.subplots_adjust(left=0.02, right=0.98, wspace=0.15)
    return fig


def _number_line(ax, activation, cut=True, size=45):
    """Stage 3: every point placed at its output score z2, and (if cut) the step's cut at 0."""
    X, y = xor_data()
    z = _xor_output(_xor_hidden(X, activation))
    rng = np.random.default_rng(2)
    zj = z + (rng.normal(0, 0.02, len(z)) if activation == 'step' else 0)
    level = np.where(y == 1, 0.25, -0.25) + rng.uniform(-0.12, 0.12, len(z))
    if cut:
        ax.axvspan(-1, 0, color=_light(ZERO_SIDE, 0.35))
        ax.axvspan(0, 1, color=_light(ONE_SIDE, 0.35))
        ax.axvline(0, color='black', lw=3)
        ax.text(-0.4, 0.5, 'predict 0', ha='center', fontsize=15, fontweight='bold')
        ax.text(0.4, 0.5, 'predict 1', ha='center', fontsize=15, fontweight='bold')
    else:
        ax.axvline(0, color='0.6', lw=1)
    _points(ax, np.column_stack([zj, level]), y, size=size)
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.set_xlabel('output score $z^{(2)}$')


def plot_number_line(activation='step'):
    """Stage 3: the output unit's activation, a cut on the number line of z2."""
    from matplotlib.gridspec import GridSpec
    with rc_context(BIG_FONTS):
        fig = figure(figsize=(13, 4.2))
        gs = GridSpec(1, 2, width_ratios=[0.8, 2.6], figure=fig)
        _mini_net(fig.add_subplot(gs[0]), 'xor', 'act', 'the output unit’s activation')
        ax = fig.add_subplot(gs[1])
        _number_line(ax, activation)
        ax.set_title('Each point at its output score.  The step cuts at 0', fontsize=15)
        fig.subplots_adjust(left=0.02, right=0.98, wspace=0.12, bottom=0.18)
    return fig


def _strip_arrow(fig, gs_thumb, gs_arrow, preset, caption, labels=None):
    _mini_net(fig.add_subplot(gs_thumb), 'xor', preset, labels=labels)
    ax = fig.add_subplot(gs_arrow)
    ax.annotate('', xy=(0.8, 0.5), xytext=(0.08, 0.5), xycoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>', lw=3, color='0.3', mutation_scale=30))
    ax.text(0.44, 0.62, caption, ha='center', va='bottom', fontsize=14, transform=ax.transAxes)
    ax.axis('off')


def plot_forward_strip():
    """The whole forward pass, one row per layer. Panels are spaces; between them, arrows with
    a network thumbnail above each show which part of the network turns one space into the
    next. Row 1, the hidden layer: feature space -> z1-space -> h-space. Row 2, the output
    unit: h-space -> the number line of z2 -> the predictions."""
    from matplotlib.gridspec import GridSpec
    X, y = xor_data()
    Z1 = np.column_stack([X @ w + b for w, b in XOR_UNITS])
    H = _jittered_hidden(X)
    with rc_context(BIG_FONTS):
        fig = figure(figsize=(21, 13.5))
        gs = GridSpec(4, 5, height_ratios=[1.2, 2.5, 1.2, 2.5], width_ratios=[3, 1.9, 3, 1.9, 3],
                      figure=fig)
        for row, text in ((0, 'Row 1:\nthe hidden layer'), (2, 'Row 2:\nthe output unit')):
            label = fig.add_subplot(gs[row, 0])
            label.text(0.0, 0.5, text, fontsize=19, fontweight='bold', va='center',
                       transform=label.transAxes)
            label.axis('off')
        _strip_arrow(fig, gs[0, 1], gs[1, 1], 'hidden_sum', 'weighted sums:\n$z^{(1)}_1$ and $z^{(1)}_2$',
                     labels={'h1': r'$z^{(1)}_1$', 'h2': r'$z^{(1)}_2$'})
        _strip_arrow(fig, gs[0, 3], gs[1, 3], 'hidden_act', 'activations:\n$h_k = $ step$(z^{(1)}_k)$')
        _strip_arrow(fig, gs[2, 1], gs[3, 1], 'sum', 'weighted sum:\n$z^{(2)}$',
                     labels={'out': r'$z^{(2)}$'})
        _strip_arrow(fig, gs[2, 3], gs[3, 3], 'act', 'activation:\ncut at 0',
                     labels={'out': 'out'})

        a = fig.add_subplot(gs[1, 0])
        _points(a, X, y, size=30)
        _data_axes(a, XOR_LIM, XOR_LIM)
        a.set_xticks([0, 1])
        a.set_yticks([0, 1])
        a.set_title('feature space: $(x_1, x_2)$', fontsize=15)

        b = fig.add_subplot(gs[1, 2])
        b.axvline(0, lw=2, **HIDDEN_LINE[0])
        b.axhline(0, lw=2, **HIDDEN_LINE[1])
        _points(b, Z1, y, size=30)
        b.set_xlim(-1.2, 2.2)
        b.set_ylim(-2.2, 1.2)
        b.set_aspect('equal')
        b.set_xlabel('$z^{(1)}_1$')
        b.set_ylabel('$z^{(1)}_2$')
        b.text(0.06, -2.1, '$h_1$ switches', fontsize=12, color='0.2')
        b.text(1.15, 0.08, '$h_2$ switches', fontsize=12, color='0.2')
        b.set_title('$z^{(1)}$-space: $(z^{(1)}_1, z^{(1)}_2)$', fontsize=15)

        for k, spot in enumerate((gs[1, 4], gs[3, 0])):
            c = fig.add_subplot(spot)
            _points(c, H, y, size=30)
            _hspace_axes(c)
            c.set_xlabel('$h_1$')
            c.set_ylabel('$h_2$')
            c.set_title('h-space: $(h_1, h_2)$', fontsize=15)
            if k == 1:                             # row 2: the output unit's line
                t = np.array(H_LIM)
                c.plot(t, t - 0.5, color='black', lw=2.5)

        d = fig.add_subplot(gs[3, 2])
        _number_line(d, 'step', cut=False, size=30)
        d.set_title('number line: $z^{(2)}$', fontsize=15)

        e = fig.add_subplot(gs[3, 4])
        pred = (_xor_output(_xor_hidden(X, 'step')) > 0).astype(float)
        rng = np.random.default_rng(3)
        e.axhspan(-0.5, 0.5, color=_light(ZERO_SIDE, 0.35))
        e.axhspan(0.5, 1.5, color=_light(ONE_SIDE, 0.35))
        _points(e, np.column_stack([rng.uniform(-0.8, 0.8, len(y)),
                                    pred + rng.uniform(-0.25, 0.25, len(y))]), y, size=25)
        e.set_xlim(-1, 1)
        e.set_ylim(-0.5, 1.5)
        e.set_xticks([])
        e.set_yticks([0, 1])
        e.set_ylabel('predicted class')
        e.set_title('prediction', fontsize=15)
        fig.subplots_adjust(left=0.04, right=0.99, top=0.97, bottom=0.04, wspace=0.08, hspace=0.3)
    return fig


def plot_plateau(activation='step'):
    """Stage 4, an aid: the output score over feature space (the whole path to z2)."""
    from matplotlib.gridspec import GridSpec
    with rc_context(BIG_FONTS):
        fig = figure(figsize=(13, 6.6))
        gs = GridSpec(1, 2, width_ratios=[0.8, 2.4], figure=fig)
        _mini_net(fig.add_subplot(gs[0]), 'xor', 'path_z', 'everything up to $z^{(2)}$')
        ax = fig.add_subplot(gs[1], projection='3d')
        _xor_surface(ax, 'z2', activation)
        ax.set_title('The output score $z^{(2)}$ over feature space,\n'
                     'with each point drawn at its score', fontsize=15)
        fig.subplots_adjust(left=0.02, right=0.98, wspace=0.05)
    return fig


def plot_xor_boundary(activation='step'):
    """Stage 4: the network's decision boundary back in feature space (the whole network)."""
    from matplotlib.gridspec import GridSpec
    X, y = xor_data()
    g1, g2, grid = _grid(XOR_LIM, XOR_LIM)
    final = _xor_output(_xor_hidden(grid, activation)).reshape(g1.shape)
    with rc_context(BIG_FONTS):
        fig = figure(figsize=(11.5, 6.8))
        gs = GridSpec(1, 2, width_ratios=[0.8, 2], figure=fig)
        _mini_net(fig.add_subplot(gs[0]), 'xor', 'all', 'the whole network')
        ax = fig.add_subplot(gs[1])
        _two_sides(ax, g1, g2, final)
        ax.contour(g1, g2, final, levels=[0], colors='black', linewidths=3)
        _points(ax, X, y, size=40)
        _data_axes(ax, XOR_LIM, XOR_LIM)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_title('The network’s decision boundary in feature space\n'
                     r'black: $z^{(2)} = 0$.  Blue: predicted class 1', fontsize=15)
        fig.subplots_adjust(left=0.02, right=0.98, wspace=0.1)
    return fig


TWO_LINES = ((np.array([1.0, 0.0]), -0.5),    # unit 1: x1 > 0.5
             (np.array([0.0, 1.0]), -0.5))    # unit 2: x2 > 0.5
TWO_LINE_OUTPUTS = ((( 1.0,  1.0), -0.5, 'h₁ + h₂ − 0.5'),
                    (( 1.0,  1.0), -1.5, 'h₁ + h₂ − 1.5'),
                    (( 1.0, -1.0), -0.5, 'h₁ − h₂ − 0.5'),
                    ((-1.0, -1.0),  0.5, '−h₁ − h₂ + 0.5'))


def plot_two_line_combinations(show_scores=True):
    """Section 4: the same two hidden units (the lines x1 = 0.5 and x2 = 0.5, step
    activations), with four different output neurons. Each panel writes the output score
    in each of the four regions and shades score > 0 blue."""
    lim = XOR_LIM
    g1, g2, grid = _grid(lim, lim)
    H = np.column_stack([(grid @ w + b > 0).astype(float) for w, b in TWO_LINES])
    centers = {(0, 0): (0.0, 0.0), (1, 0): (1.0, 0.0), (0, 1): (0.0, 1.0), (1, 1): (1.0, 1.0)}
    with rc_context(BIG_FONTS):
        fig, axes = subplots(1, 4, figsize=(20, 5.6))
        for ax, (w_out, b_out, formula) in zip(axes, TWO_LINE_OUTPUTS):
            score = (H @ np.array(w_out) + b_out).reshape(g1.shape)
            _two_sides(ax, g1, g2, score)
            ax.axvline(0.5, lw=3, label=HIDDEN_LINE_LABELS[0], **HIDDEN_LINE[0])
            ax.axhline(0.5, lw=3, label=HIDDEN_LINE_LABELS[1], **HIDDEN_LINE[1])
            for (a, c), (px, py) in centers.items():
                label = f'h = ({a}, {c})'
                if show_scores:
                    value = w_out[0] * a + w_out[1] * c + b_out
                    label += f'\n$z^{{(2)}}$ = {value:+.1f}'
                ax.text(px, py, label, ha='center', va='center', fontsize=13,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
            _data_axes(ax, lim, lim)
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])
            ax.set_title(f'$z^{{(2)}}$ = {formula}' if show_scores else '?', fontsize=15)
        fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2,
                   fontsize=13, frameon=False)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


# --- Notebook 02 (v2): what a layer of weighted sums does to the plane (Demo Q) ---------------

LAYER_PRESETS = {
    'Our XOR hidden layer: both units point the same way': (np.array([[1.0, 1.0], [1.0, 1.0]]),
                                                             np.array([-0.5, -1.5])),
    'Two units pointing different ways': (np.array([[1.0, 1.0], [-1.0, 1.0]]), np.array([-1.0, 0.0])),
    'Stretch and slant': (np.array([[2.0, 0.5], [0.0, 1.0]]), np.array([0.0, 0.0])),
    'Bias only: a shift': (np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([-0.5, -0.5])),
}
LAYER_ACTIVATIONS = ('none: weighted sums only', 'step', 'sigmoid')
GRID_VALUES = np.linspace(-0.5, 1.5, 9)
GRID_COLORS = ('0.2', '0.62')           # lines of constant x1 (dark gray), constant x2 (light gray)


def _apply_layer(P, W, b, activation):
    Z = P @ W.T + b
    if activation == 'step':
        return (Z > 0).astype(float)
    if activation == 'sigmoid':
        return _sigmoid(2 * Z)
    return Z


def _grid_lines():
    t = np.linspace(-0.5, 1.5, 120)
    vertical = [np.column_stack([np.full_like(t, c), t]) for c in GRID_VALUES]
    horizontal = [np.column_stack([t, np.full_like(t, c)]) for c in GRID_VALUES]
    return vertical, horizontal


def _draw_grid(ax, transform=None, dots=False):
    vertical, horizontal = _grid_lines()
    for lines, color in zip((vertical, horizontal), GRID_COLORS):
        for L in lines:
            P = transform(L) if transform else L
            if dots:
                ax.plot(P[::12, 0], P[::12, 1], 'o', ms=3, color=color, alpha=0.5)
            else:
                ax.plot(P[:, 0], P[:, 1], color=color, lw=1.2, alpha=0.8)


def layer_map_demo(size_percent=100):
    """Demo Q: a square grid over feature space (with the XOR data), before and after a layer
    of two weighted sums, z = W x + b, optionally followed by an activation. Presets choose W
    and b. The readout says whether one straight line can now separate the classes."""
    X, y = xor_data()
    style = {'description_width': '120px'}
    preset = widgets.Dropdown(options=list(LAYER_PRESETS), description='weights',
                              style=style, layout=widgets.Layout(width='620px'))
    activation = widgets.ToggleButtons(options=LAYER_ACTIVATIONS, description='activation',
                                       style=style)
    readout = widgets.HTML()
    out = _figure_area()

    def update(change=None):
        W, b = LAYER_PRESETS[preset.value]
        act = {'none: weighted sums only': None}.get(activation.value, activation.value)
        F = lambda P: _apply_layer(P, W, b, act)
        Q = F(X)
        if act == 'step':
            Q = Q + np.random.default_rng(1).normal(0, 0.025, Q.shape)
        mistakes = _output_unit(Q, y)[2]
        name = 'z' if act is None else 'h'
        readout.value = _html(
            f'unit 1: w = ({W[0, 0]:g}, {W[0, 1]:g}), b = {b[0]:g} &nbsp; &nbsp; '
            f'unit 2: w = ({W[1, 0]:g}, {W[1, 1]:g}), b = {b[1]:g}'
            + ('' if act is None else f', &nbsp; then the {act}' + (' of 2z' if act == 'sigmoid' else ''))
            + '<br>Can one straight line separate the classes in the new space? '
            + ('<b>Yes.</b>' if mistakes == 0 else '<b>No.</b>'))

        def draw():
            fig, (left, right) = subplots(1, 2, figsize=(15, 7))
            _draw_grid(left)
            _points(left, X, y, size=35)
            _data_axes(left, (-0.6, 1.6), (-0.6, 1.6))
            left.set_title('Feature space, with a square grid', fontsize=15)
            _draw_grid(right, F, dots=(act == 'step'))
            _points(right, Q, y, size=35)
            corners = F(np.array([[-0.5, -0.5], [1.5, -0.5], [-0.5, 1.5], [1.5, 1.5]]))
            lo, hi = corners.min(axis=0), corners.max(axis=0)
            pad = 0.15 * max(hi - lo) + 0.1
            span = max(hi - lo) / 2 + pad
            mid = (lo + hi) / 2
            right.set_xlim(mid[0] - span, mid[0] + span)
            right.set_ylim(mid[1] - span, mid[1] + span)
            right.set_aspect('equal')
            right.axhline(0, color='0.75', lw=0.8, zorder=0)
            right.axvline(0, color='0.75', lw=0.8, zorder=0)
            right.set_xlabel(f'${name}_1$' if act else r'$z_1$')
            right.set_ylabel(f'${name}_2$' if act else r'$z_2$')
            right.set_title('After the weighted sums' + ('' if act is None else f' and the {act}'),
                            fontsize=15)
            fig.tight_layout()
            return fig
        _show(out, draw, size_percent)

    for wdg in (preset, activation):
        wdg.observe(update, names='value')
    update()
    display(widgets.VBox([preset, activation, readout, out]))


def plot_two_lines_blank():
    """Practice question: the two crossing hidden lines (x1 = 0.5 solid, x2 = 0.5 dashed) with
    the four regions labeled A-D and nothing else filled in."""
    lim = XOR_LIM
    with rc_context(BIG_FONTS):
        fig, ax = subplots(figsize=(6.4, 6.2))
        ax.axvline(0.5, lw=3, label=HIDDEN_LINE_LABELS[0], **HIDDEN_LINE[0])
        ax.axhline(0.5, lw=3, label=HIDDEN_LINE_LABELS[1], **HIDDEN_LINE[1])
        for name, (px, py) in (('A', (0.0, 0.0)), ('B', (1.0, 0.0)), ('C', (0.0, 1.0)),
                               ('D', (1.0, 1.0))):
            ax.text(px, py, name, ha='center', va='center', fontsize=26, fontweight='bold',
                    bbox=dict(boxstyle='circle', facecolor='white', edgecolor='0.4'))
        _data_axes(ax, lim, lim)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=1, fontsize=12,
                  frameon=False)
        ax.set_title('Two hidden units, four regions', fontsize=15)
        fig.tight_layout()
    return fig
