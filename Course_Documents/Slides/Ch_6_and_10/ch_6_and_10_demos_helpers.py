"""Plotting and widget helpers for this week's notebooks.

ch_6_and_10_01_ridge_and_lasso.ipynb and ch_6_and_10_01b_linear_algebra_refresher.ipynb
use this module. The notebook stays focused on
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
