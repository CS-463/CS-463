"""Simulation, plotting, and widget helpers for the Chapter 5 notebooks.

ch_5_01_knn_bayes_boundary.ipynb (KNN and the Bayes boundary) uses this module.
The notebooks stay focused on ideas; the simulation, matplotlib, and ipywidgets
details live here.

Classes are coded 1 (circles) and 2 (triangles) throughout, matching the
notebook's "class 1" and "class 2".
"""
from io import BytesIO
from types import SimpleNamespace

import ipywidgets as widgets
import numpy as np
from IPython.display import display
from matplotlib import get_backend, rc_context
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
from matplotlib.pyplot import close, cm, subplots, switch_backend
from matplotlib.ticker import PercentFormatter
from scipy.stats import multivariate_normal
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Load the backend now, outside any rc_context. Loading Jupyter's inline backend
# turns on interactive mode; if that first happens inside rc_context, exiting the
# context turns it back off, and later figures stop displaying automatically.
switch_backend(get_backend())

BIG_FONTS = {'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 15,
             'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 13}
HTML_STYLE = 'font-size: 18px; line-height: 1.5;'

# Okabe-Ito colors, which stay distinguishable under common color blindness.
CLASS_COLORS = {1: '#0072B2', 2: '#E69F00'}
CLASS_MARKERS = {1: 'o', 2: '^'}
CLASS_LABELS = {1: 'class 1 (circles)', 2: 'class 2 (triangles)'}
CLASS_NAMES = {1: ('circle', 'circles'), 2: ('triangle', 'triangles')}
QUERY_COLOR = '#D55E00'
CURVE_COLORS = {'training error': '#0072B2', 'test error': '#D55E00'}


def _light(color, amount=0.3):
    """The color mixed with white: amount = 1 is the color itself, 0 is white."""
    return tuple(amount * c + (1 - amount) for c in to_rgb(color))


REGION_CMAP = ListedColormap([_light(CLASS_COLORS[1]), _light(CLASS_COLORS[2])])
VOTE_CMAP = LinearSegmentedColormap.from_list(
    'votes', [_light(CLASS_COLORS[2], 0.6), (1, 1, 1), _light(CLASS_COLORS[1], 0.6)])


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


def _plot_classes(ax, X, y, ms=8, **kwargs):
    for k in (1, 2):
        m = y == k
        ax.plot(X[m, 0], X[m, 1], CLASS_MARKERS[k], ms=ms, color=CLASS_COLORS[k],
                mec='black', mew=0.6, label=CLASS_LABELS[k], **kwargs)


def _class_legend(ax, extra=(), **kwargs):
    """Legend with full-size class markers, plus any extra (handle, label) pairs."""
    handles = [Line2D([], [], ls='none', marker=CLASS_MARKERS[k], ms=10, color=CLASS_COLORS[k],
                      mec='black', mew=0.6, label=CLASS_LABELS[k]) for k in (1, 2)]
    for handle, label in extra:
        handle.set_label(label)
        handles.append(handle)
    ax.legend(handles=handles, **kwargs)


def _tally_text(n_circles, K):
    """'3 circles, 2 triangles → predict circle (3/5 = 60% circle)'."""
    n_triangles = K - n_circles
    winner = 1 if n_circles > n_triangles else 2
    counts = [f'{n} {CLASS_NAMES[k][0] if n == 1 else CLASS_NAMES[k][1]}'
              for n, k in ((n_circles, 1), (n_triangles, 2))]
    return (f'{counts[0]}, {counts[1]} → <b>predict {CLASS_NAMES[winner][0]}</b> '
            f'({n_circles}/{K} = {n_circles / K:.0%} circle)')


# --- Demo A: how KNN votes ---------------------------------------------------

VOTE_X = np.array([(1.5, 7.5), (2.6, 8.6), (1.0, 5.4), (3.0, 6.4), (2.0, 3.8), (4.3, 7.9),
                   (3.7, 4.7), (4.0, 6.2),                                    # circles
                   (7.5, 2.5), (8.5, 4.2), (6.4, 1.4), (8.1, 6.6), (6.6, 3.9), (9.0, 1.8),
                   (6.2, 7.3), (5.2, 5.0)])                                   # triangles
VOTE_Y = np.repeat([1, 2], 8)
VOTE_PRESETS = {'Deep in circle territory': (2.2, 6.2),
                'Deep in triangle territory': (7.6, 3.4),
                'Near the boundary': (4.8, 5.5)}
VOTE_K_VALUES = (1, 3, 5, 7, 9)


def knn_vote_demo(size_percent=100):
    """Demo A: one KNN prediction on a small hand-placed data set.

    Pick a preset query point (or move it with the sliders) and a K. The plot shows
    the distance to every training point, highlights the K nearest inside a circle
    reaching the Kth nearest, and the readout gives the vote and the prediction."""
    preset = widgets.Dropdown(options=list(VOTE_PRESETS), value='Deep in circle territory',
                              description='query point', style={'description_width': '90px'},
                              layout=widgets.Layout(width='360px'))
    style = {'description_width': '90px'}
    x1 = widgets.FloatSlider(value=VOTE_PRESETS[preset.value][0], min=0, max=10, step=0.1,
                             description='x₀: X₁', style=style)
    x2 = widgets.FloatSlider(value=VOTE_PRESETS[preset.value][1], min=0, max=10, step=0.1,
                             description='x₀: X₂', style=style)
    k_slider = widgets.SelectionSlider(options=VOTE_K_VALUES, value=3, description='K',
                                       style=style)
    readout = widgets.HTML()
    out = _figure_area()

    def update(change=None):
        x0 = np.array([x1.value, x2.value])
        K = k_slider.value
        distances = np.hypot(*(VOTE_X - x0).T)
        nearest = np.argsort(distances)[:K]
        n_circles = int(np.sum(VOTE_Y[nearest] == 1))
        readout.value = _html(_tally_text(n_circles, K))

        def draw():
            fig, ax = subplots(figsize=(8, 8))
            for point in VOTE_X:
                ax.plot([x0[0], point[0]], [x0[1], point[1]], color='0.75', lw=1, zorder=1)
            for i in nearest:
                ax.plot([x0[0], VOTE_X[i, 0]], [x0[1], VOTE_X[i, 1]], color=QUERY_COLOR, lw=2.5,
                        zorder=2)
            ax.add_patch(Circle(x0, distances[nearest[-1]], fill=False, color=QUERY_COLOR, lw=2,
                                ls='--', zorder=2))
            _plot_classes(ax, VOTE_X, VOTE_Y, ms=13, alpha=0.35, zorder=3)
            _plot_classes(ax, VOTE_X[nearest], VOTE_Y[nearest], ms=15, zorder=4)
            ax.plot(*x0, '*', ms=26, color=QUERY_COLOR, mec='black', zorder=5)
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.set_aspect('equal')
            ax.set_xlabel('X₁')
            ax.set_ylabel('X₂')
            ax.set_title(f'The K = {K} nearest neighbors of x₀')
            _class_legend(ax, extra=[(Line2D([], [], ls='none', marker='*', ms=18,
                                             color=QUERY_COLOR, mec='black'), 'x₀ (query point)')],
                          loc='upper right', framealpha=0.95)
            fig.tight_layout()
            return fig
        _show(out, draw, size_percent)

    def choose(change):
        if change['new'] is not None:
            x1.value, x2.value = VOTE_PRESETS[change['new']]

    def moved(change):
        if preset.value is not None and (x1.value, x2.value) != VOTE_PRESETS[preset.value]:
            preset.value = None
        update()

    preset.observe(choose, names='value')
    for w in (x1, x2):
        w.observe(moved, names='value')
    k_slider.observe(update, names='value')
    update()
    display(widgets.VBox([widgets.HBox([preset, k_slider]), widgets.HBox([x1, x2]),
                          readout, out]))


# --- The simulated data (Demos B, C2, D, and E) -------------------------------

BUMP_CENTERS = {1: np.array([(1.00, 0.30), (0.73, -0.89), (0.55, -0.99), (1.06, 1.34),
                             (0.51, -0.62), (1.49, 0.36), (1.11, -0.93), (0.97, 0.70),
                             (-0.34, -0.46), (-0.90, -1.29)]),
                2: np.array([(-1.84, 0.76), (-1.27, 1.27), (0.16, 0.81), (-2.52, 0.46),
                             (-0.05, 1.11), (-1.53, 0.52), (-0.98, 0.19), (1.06, 0.19),
                             (-0.03, 1.88), (-0.58, 0.89)])}
BUMP_SD = np.sqrt(0.2)
K_VALUES = (1, 3, 5, 9, 15, 25, 50, 100)
# Wide enough for every point in all of Demo E's training sets.
GRID_X1 = np.linspace(-4, 3.5, 301)
GRID_X2 = np.linspace(-3, 4, 281)


def _draw_class(rng, k, n, sd):
    bumps = rng.integers(0, len(BUMP_CENTERS[k]), n)
    return BUMP_CENTERS[k][bumps] + rng.normal(0, sd, (n, 2))


def _density(points, k, sd):
    """f_k(x): the average of class k's round Gaussian bump densities."""
    return np.mean([multivariate_normal(center, sd**2 * np.eye(2)).pdf(points)
                    for center in BUMP_CENTERS[k]], axis=0)


def simulate_data(sd=BUMP_SD, n_train=100, n_test=5000, seed=463):
    """The two-class simulation: equal priors, each class a mix of ten small round bumps.

    Returns a SimpleNamespace with the training data (X, y), the test data
    (X_test, y_test), the priors, the bump spread sd, and posterior1(points), the
    true probability of class 1 at each point (equation 4.15)."""
    rng = np.random.default_rng(seed)
    priors = {1: 0.5, 2: 0.5}

    def sample(n):
        X = np.vstack([_draw_class(rng, 1, n, sd), _draw_class(rng, 2, n, sd)])
        return X, np.repeat([1, 2], n)

    X, y = sample(n_train)
    X_test, y_test = sample(n_test)

    def posterior1(points):
        top = priors[1] * _density(points, 1, sd)
        return top / (top + priors[2] * _density(points, 2, sd))

    return SimpleNamespace(X=X, y=y, X_test=X_test, y_test=y_test, priors=priors, sd=sd,
                           posterior1=posterior1)


def bayes_predict(sim, points):
    """The Bayes classifier: class 1 where the true probability of class 1 is above 0.5."""
    return np.where(sim.posterior1(points) > 0.5, 1, 2)


def bayes_error_rate(sim):
    """The share of the test set the Bayes classifier gets wrong."""
    return float(np.mean(bayes_predict(sim, sim.X_test) != sim.y_test))


# --- Demo B: the K slider, with staged reveals --------------------------------

def _grid_points():
    g1, g2 = np.meshgrid(GRID_X1, GRID_X2)
    return np.column_stack([g1.ravel(), g2.ravel()])


def knn_boundary_demo(sim, K=1, size_percent=100):
    """Demo B: KNN's predicted class at every location, for each K on the slider.

    K is the slider's starting value, one of K_VALUES.

    Stage 1 (always on) shows the training error. "Show Bayes boundary" adds the
    Bayes boundary and the Bayes error rate; "Show test error" adds the error on the
    10,000 test points. Both reveals start off. Everything the widgets need is
    computed once here, so moving the controls only redraws.

    Returns the widget, so a later cell can display the same demo again."""
    grid = _grid_points()
    shape = (len(GRID_X2), len(GRID_X1))
    fits = {}
    for k in K_VALUES:
        model = KNeighborsClassifier(n_neighbors=k).fit(sim.X, sim.y)
        fits[k] = SimpleNamespace(
            predicted=model.predict(grid).reshape(shape),
            votes1=model.predict_proba(grid)[:, 0].reshape(shape),
            train_error=1 - model.score(sim.X, sim.y),
            test_error=1 - model.score(sim.X_test, sim.y_test))
    bayes_grid = sim.posterior1(grid).reshape(shape)
    bayes_error = bayes_error_rate(sim)
    bayes_train_wrong = int(np.sum(bayes_predict(sim, sim.X) != sim.y))

    k_slider = widgets.SelectionSlider(options=K_VALUES, value=K, description='K',
                                       style={'description_width': '30px'},
                                       layout=widgets.Layout(width='420px'))
    shade = widgets.Checkbox(value=False, description='Shade by vote fraction', indent=False)
    show_bayes = widgets.ToggleButton(value=False, description='Show Bayes boundary',
                                      layout=widgets.Layout(width='180px'))
    show_test = widgets.ToggleButton(value=False, description='Show test error',
                                     layout=widgets.Layout(width='180px'))
    readout = widgets.HTML()
    out = _figure_area()

    def update(change=None):
        fit = fits[k_slider.value]
        lines = [f'Training error: <b>{fit.train_error:.1%}</b>']
        if show_bayes.value:
            lines += [f'Bayes error rate (on new data): <b>{bayes_error:.1%}</b>',
                      f'The Bayes boundary misclassifies <b>{bayes_train_wrong}</b> of the '
                      f'{len(sim.y)} training points.']
        if show_test.value:
            lines.append(f'Test error ({len(sim.y_test):,} new points): <b>{fit.test_error:.1%}</b>')
        readout.value = _html('<br>'.join(lines))

        def draw():
            fig, ax = subplots(figsize=(10, 8.5))
            extent = (GRID_X1[0], GRID_X1[-1], GRID_X2[0], GRID_X2[-1])
            if shade.value:
                image = ax.imshow(fit.votes1, origin='lower', extent=extent, cmap=VOTE_CMAP,
                                  vmin=0, vmax=1, interpolation='nearest')
                bar = fig.colorbar(image, ax=ax, shrink=0.7, format=PercentFormatter(1))
                bar.set_label('share of the K neighbors that are circles')
            else:
                ax.imshow(fit.predicted, origin='lower', extent=extent, cmap=REGION_CMAP,
                          vmin=1, vmax=2, interpolation='nearest')
            _plot_classes(ax, sim.X, sim.y, ms=8)
            extra = []
            if show_bayes.value:
                ax.contour(GRID_X1, GRID_X2, bayes_grid, levels=[0.5], colors='black',
                           linestyles='--', linewidths=2.5)
                extra.append((Line2D([], [], color='black', ls='--', lw=2.5), 'Bayes boundary'))
            ax.set_aspect('equal')
            ax.set_xlabel('X₁')
            ax.set_ylabel('X₂')
            shown = 'the vote fraction' if shade.value else 'the predicted class'
            ax.set_title(f'KNN with K = {k_slider.value}: {shown} at every location')
            _class_legend(ax, extra=extra, loc='upper left', framealpha=0.95)
            fig.tight_layout()
            return fig
        _show(out, draw, size_percent)

    for w in (k_slider, shade, show_bayes, show_test):
        w.observe(update, names='value')
    update()
    slider_row = widgets.HBox([widgets.HTML(_html('<b>More flexible</b>')), k_slider,
                               widgets.HTML(_html('<b>Less flexible</b>'))],
                              layout=widgets.Layout(align_items='center'))
    demo = widgets.VBox([slider_row, widgets.HBox([shade, show_bayes, show_test]), readout, out])
    display(demo)
    return demo


# --- Demo D: training versus test error over K --------------------------------

ERROR_PLOT_TICKS = (1, 3, 5, 9, 15, 31, 51, 101, 151)   # labeled; every K still gets a point
# The U is only a few points deep, so a 0-to-1 axis would flatten it; training error starts at 0.
ERROR_PLOT_TOP = 0.22


def plot_error_curves(errors, bayes_error, one_over_k=False):
    """Demo D: each column of `errors` (indexed by K) as a line against K on a log scale.

    K = 1 (most flexible) is on the left. A dashed line marks the Bayes error rate.
    Columns named 'training error' and 'test error' get fixed colors; any other
    column is drawn too, in the next default color.

    one_over_k=True plots against 1/K instead, as in ISL Figure 2.17, so flexibility
    increases to the right."""
    x = 1 / errors.index if one_over_k else errors.index
    with rc_context(BIG_FONTS):
        fig, ax = subplots(figsize=(11, 6))
        for name in errors.columns:
            ax.plot(x, errors[name], 'o-', lw=2.5, ms=7, label=name,
                    color=CURVE_COLORS.get(name))
        ax.axhline(bayes_error, color='black', ls='--', lw=2,
                   label='Bayes error rate (the floor)')
        ax.set_xscale('log')
        if one_over_k:
            ticks = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1)
            ax.set_xticks(ticks, [f'{t:g}' for t in ticks])
            ax.set_xlabel('1/K')
            ends = ((0, '← Less flexible', 'left'), (1, 'More flexible →', 'right'))
        else:
            labeled = [K for K in errors.index if K in ERROR_PLOT_TICKS]
            ax.set_xticks(labeled, [str(K) for K in labeled])
            ax.set_xlabel('K (number of neighbors)')
            ends = ((0, '← More flexible', 'left'), (1, 'Less flexible →', 'right'))
        ax.minorticks_off()
        for position, label, align in ends:
            ax.text(position, -0.11, label, transform=ax.transAxes, ha=align, va='top',
                    fontsize=15, fontweight='bold')
        ax.set_ylabel('error rate')
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_ylim(0, ERROR_PLOT_TOP)
        ax.set_yticks(np.arange(0, ERROR_PLOT_TOP + 0.001, 0.02))
        ax.set_title('Training and test error for every K' + (', against 1/K' if one_over_k else ''))
        ax.legend(loc='upper right' if one_over_k else 'upper left')
        fig.tight_layout()
    return fig


# --- Demo C2: what each method assumes, drawn ---------------------------------

ASSUMPTIONS = {LogisticRegression: 'Log-odds are linear: a straight boundary',
               GaussianNB: 'One bell-shaped blob per class, features independent',
               QuadraticDiscriminantAnalysis: 'One bell-shaped blob per class, any shape',
               KNeighborsClassifier: 'Nearby points have similar labels'}
BAYES_LINE = dict(colors='black', linestyles='--', linewidths=2.5)


def plot_assumption_boundaries(sim, models):
    """Demo C2: a 2 x 2 grid, one panel per fitted model in `models` (name -> model).

    Each panel shows the model's predicted class at every location, the training
    points, and the Bayes boundary. The title gives the model's test error and, below
    it, the assumption that comes with its kind of model."""
    grid = _grid_points()
    shape = (len(GRID_X2), len(GRID_X1))
    extent = (GRID_X1[0], GRID_X1[-1], GRID_X2[0], GRID_X2[-1])
    bayes_grid = sim.posterior1(grid).reshape(shape)
    with rc_context(BIG_FONTS):
        fig, axes = subplots(2, 2, figsize=(15, 13.5), sharex=True, sharey=True)
        for ax, (name, model) in zip(axes.flat, models.items()):
            test_error = 1 - model.score(sim.X_test, sim.y_test)
            ax.imshow(model.predict(grid).reshape(shape), origin='lower', extent=extent,
                      cmap=REGION_CMAP, vmin=1, vmax=2, interpolation='nearest')
            _plot_classes(ax, sim.X, sim.y, ms=6)
            ax.contour(GRID_X1, GRID_X2, bayes_grid, levels=[0.5], **BAYES_LINE)
            ax.set_aspect('equal')
            ax.set_title(f'{name}: test error {test_error:.1%}\n'
                         f'"{ASSUMPTIONS[type(model)]}"', fontsize=15)
        for ax in axes[1]:
            ax.set_xlabel('X₁')
        for ax in axes[:, 0]:
            ax.set_ylabel('X₂')
        _class_legend(axes[0, 0], extra=[(Line2D([], [], color='black', ls='--', lw=2.5),
                                          'Bayes boundary')],
                      loc='upper left', framealpha=0.95, fontsize=11)
        fig.tight_layout()
    return fig


# --- Demo E: redraw the training data -----------------------------------------

REDRAW_MODELS = {'KNN, K = 1 (very flexible)': lambda: KNeighborsClassifier(n_neighbors=1),
                 'KNN, K = 15 (balanced)': lambda: KNeighborsClassifier(n_neighbors=15),
                 'Logistic regression (rigid)': lambda: LogisticRegression(C=np.inf)}
REDRAW_SEEDS = range(463, 513)
MAX_SHOWN = 10
# One distinct color per training set on the slider (tab10 stays readable when overlaid).
REDRAW_BOUNDARY_COLORS = [cm.tab10(i) for i in range(MAX_SHOWN)]
AVERAGE_COLOR = '#009E73'


def _average_pairwise_disagreement(predicts1):
    """The share of grid points where two fitted models disagree, averaged over all
    pairs of models. predicts1 is (models, grid points), True where class 1."""
    n = len(predicts1)
    count1 = predicts1.sum(axis=0)
    return float(np.mean(2 * count1 * (n - count1) / (n * (n - 1))))


def redraw_demo(sim, size_percent=100):
    """Demo E: one method's boundary from many fresh training sets of 200 points.

    Every model in REDRAW_MODELS is fit to each of 50 training sets (seeds 463 to
    512) when the demo is built. The slider shows the first 1 to 10 boundaries; the
    average boundary and both readouts always use all 50.

    Returns the widget, so a later cell can display the same demo again."""
    grid = _grid_points()
    shape = (len(GRID_X2), len(GRID_X1))
    bayes_grid = sim.posterior1(grid).reshape(shape)
    training_sets = [simulate_data(sd=sim.sd, n_test=0, seed=seed) for seed in REDRAW_SEEDS]
    results = {}
    for label, make in REDRAW_MODELS.items():
        predicts1, probabilities1, test_errors = [], [], []
        for data in training_sets:
            model = make().fit(data.X, data.y)
            predicts1.append(model.predict(grid) == 1)
            if len(probabilities1) < MAX_SHOWN:
                probabilities1.append(model.predict_proba(grid)[:, 0].reshape(shape))
            test_errors.append(1 - model.score(sim.X_test, sim.y_test))
        predicts1 = np.array(predicts1)
        results[label] = SimpleNamespace(
            boundaries=probabilities1,
            average=predicts1.mean(axis=0).reshape(shape),
            change=_average_pairwise_disagreement(predicts1),
            test_error=float(np.mean(test_errors)))

    style = {'description_width': '140px'}
    method = widgets.Dropdown(options=list(REDRAW_MODELS), description='method', style=style,
                              layout=widgets.Layout(width='420px'))
    shown = widgets.IntSlider(value=1, min=1, max=MAX_SHOWN, description='Training sets shown',
                              style=style, layout=widgets.Layout(width='420px'))
    show_average = widgets.ToggleButton(value=False, description='Show average boundary',
                                        layout=widgets.Layout(width='200px'))
    readout = widgets.HTML()
    out = _figure_area()

    def update(change=None):
        result = results[method.value]
        readout.value = _html(
            f'<b>Variance:</b> On average, <b>{result.change:.0%}</b> of the plot changes class from one '
            f'training set to another (averaged over all pairs of the {len(training_sets)} '
            f'training sets).<br>'
            f'Average test error: <b>{result.test_error:.1%}</b> '
            f'(averaged over the {len(training_sets)} training sets).')

        def draw():
            fig, ax = subplots(figsize=(10, 8.5))
            n_shown = shown.value
            for i, probability1 in enumerate(result.boundaries[:n_shown]):
                color = REDRAW_BOUNDARY_COLORS[i]
                ax.contour(GRID_X1, GRID_X2, probability1, levels=[0.5],
                           colors=[color], linewidths=1.5, alpha=0.85)
            ax.contour(GRID_X1, GRID_X2, bayes_grid, levels=[0.5], **BAYES_LINE)
            handles = [Line2D([], [], color='black', ls='--', lw=2.5, label='Bayes boundary')]
            if n_shown == 1:
                handles.append(Line2D([], [], color=REDRAW_BOUNDARY_COLORS[0], lw=1.5,
                                      label='one training set\'s boundary'))
            else:
                handles.append(Line2D([], [], color='0.45', lw=1.5,
                                      label=f'{n_shown} training sets (each a different color)'))
            if show_average.value:
                ax.contour(GRID_X1, GRID_X2, result.average, levels=[0.5],
                           colors=AVERAGE_COLOR, linewidths=4.5)
                handles.append(Line2D([], [], color=AVERAGE_COLOR, lw=4.5,
                                      label=f'average boundary (all {len(training_sets)} sets)'))
            ax.set_xlim(GRID_X1[0], GRID_X1[-1])
            ax.set_ylim(GRID_X2[0], GRID_X2[-1])
            ax.set_aspect('equal')
            ax.set_xlabel('X₁')
            ax.set_ylabel('X₂')
            sets = 'training set' if shown.value == 1 else 'training sets'
            ax.set_title(f'{method.value}: {shown.value} {sets}')
            ax.legend(handles=handles, loc='upper left', framealpha=0.95)
            fig.tight_layout()
            return fig
        _show(out, draw, size_percent)

    for w in (method, shown, show_average):
        w.observe(update, names='value')
    update()
    demo = widgets.VBox([widgets.HBox([method, shown, show_average]), readout, out])
    display(demo)
    return demo


# --- Part 2, Demos F and G2: one validation curve per click --------------------

PART2_SEEDS = range(10)
PART2_COLORS = [cm.tab10(i) for i in range(len(PART2_SEEDS))]
PART2_TICKS = (1, 3, 5, 9, 15, 31, 71)
_part2_cache = {}


def _validation_set_curves(sim, K_values):
    """One validation-error curve per random_state: fit on a random half, score the other."""
    curves = []
    for seed in PART2_SEEDS:
        X_train, X_valid, y_train, y_valid = train_test_split(sim.X, sim.y, test_size=0.5,
                                                              random_state=seed)
        curves.append([1 - KNeighborsClassifier(n_neighbors=K).fit(X_train, y_train)
                       .score(X_valid, y_valid) for K in K_values])
    return np.array(curves)


def _cross_validation_curves(sim, K_values):
    """One 10-fold cross-validation error curve per shuffle (random_state)."""
    curves = []
    for seed in PART2_SEEDS:
        folds = KFold(n_splits=10, shuffle=True, random_state=seed)
        curves.append([1 - cross_val_score(KNeighborsClassifier(n_neighbors=K), sim.X, sim.y,
                                           cv=folds).mean() for K in K_values])
    return np.array(curves)


def _part2_curves(sim, K_values):
    """Both demos' curves and their shared y-axis range, computed once per data set."""
    key = (id(sim), tuple(K_values))
    if key not in _part2_cache:
        validation = _validation_set_curves(sim, K_values)
        cross_validation = _cross_validation_curves(sim, K_values)
        both = np.concatenate([validation.ravel(), cross_validation.ravel()])
        ylim = (np.floor(min(both.min(), bayes_error_rate(sim)) * 20) / 20,
                np.ceil(both.max() * 20) / 20)
        _part2_cache[key] = SimpleNamespace(validation=validation,
                                            cross_validation=cross_validation, ylim=ylim)
    return _part2_cache[key]


def _flexibility_axis(ax, K_values, ticks):
    ax.set_xscale('log')
    labeled = [K for K in K_values if K in ticks]
    ax.set_xticks(labeled, [str(K) for K in labeled])
    ax.minorticks_off()
    ax.set_xlabel('K (number of neighbors)')
    for position, label, align in ((0, '← More flexible', 'left'), (1, 'Less flexible →', 'right')):
        ax.text(position, -0.11, label, transform=ax.transAxes, ha=align, va='top',
                fontsize=15, fontweight='bold')


def _curve_adder_demo(sim, true_errors, curves, ylim, button_label, row_label, curve_label,
                      title, background=None, background_label='', size_percent=100):
    """Shared widget for Demos F and G2: each click adds the next precomputed curve."""
    K_values = list(true_errors.index)
    bayes_error = bayes_error_rate(sim)
    state = {'shown': 0}
    add = widgets.Button(description=button_label, button_style='info',
                         layout=widgets.Layout(width='200px'))
    reset = widgets.Button(description='Reset', layout=widgets.Layout(width='100px'))
    controls = [add, reset]
    show_background = None
    if background is not None:
        show_background = widgets.ToggleButton(
            value=False, description='Show the validation-set curves for comparison',
            layout=widgets.Layout(width='380px'))
        controls.append(show_background)
    table = widgets.HTML(layout=widgets.Layout(min_width='170px'))
    readout = widgets.HTML()
    out = _figure_area()

    def update(change=None):
        n = state['shown']
        rows = ''.join(f'<tr><td style="padding: 0 12px 0 0;">{row_label} {i + 1}</td>'
                       f'<td>K = {K_values[int(np.argmin(curves[i]))]}</td></tr>'
                       for i in range(n))
        table.value = _html(f'<table><tr><th style="text-align: left; padding: 0 12px 0 0;">'
                            f'{row_label}</th><th style="text-align: left;">chosen K</th></tr>'
                            f'{rows}</table>')
        if n >= 2:
            spread = curves[:n].std(axis=0, ddof=1).mean()
            readout.value = _html(f'Average spread between the curves: <b>{spread * 100:.1f}</b> '
                                  f'percentage points')
        else:
            readout.value = _html('&nbsp;')

        def draw():
            fig, ax = subplots(figsize=(11, 6.5))
            if show_background is not None and show_background.value:
                for curve in background:
                    ax.plot(K_values, curve, color='0.6', lw=1.2, alpha=0.5)
            for i in range(n):
                ax.plot(K_values, curves[i], color=PART2_COLORS[i], lw=1.8, alpha=0.75)
            ax.plot(K_values, true_errors, color='black', lw=4, label='true test error')
            ax.axhline(bayes_error, color='black', ls='--', lw=2,
                       label='Bayes error rate (the floor)')
            handles, labels = ax.get_legend_handles_labels()
            if n:
                handles.append(Line2D([], [], color=PART2_COLORS[0], lw=1.8))
                labels.append(curve_label)
            if show_background is not None and show_background.value:
                handles.append(Line2D([], [], color='0.6', lw=1.2, alpha=0.5))
                labels.append(background_label)
            _flexibility_axis(ax, K_values, PART2_TICKS)
            ax.set_ylim(*ylim)
            ax.set_ylabel('error rate')
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.set_title(title)
            ax.legend(handles, labels, loc='upper left')
            fig.tight_layout()
            return fig
        _show(out, draw, size_percent)

    def clicked_add(button):
        if state['shown'] < len(curves):
            state['shown'] += 1
            update()

    def clicked_reset(button):
        state['shown'] = 0
        update()

    add.on_click(clicked_add)
    reset.on_click(clicked_reset)
    if show_background is not None:
        show_background.observe(update, names='value')
    update()
    display(widgets.VBox([widgets.HBox(controls), readout, widgets.HBox([out, table])]))


def validation_set_demo(sim, true_errors, size_percent=100):
    """Demo F: each click adds the validation-error curve for the next random split
    (random_state 0 to 9) of the 200 training points into halves.

    true_errors is Demo D's test error, indexed by K; its index sets the K values.
    The table lists the K each split would choose. All ten curves are computed when
    the demo is built."""
    part2 = _part2_curves(sim, list(true_errors.index))
    _curve_adder_demo(sim, true_errors, part2.validation, part2.ylim,
                      button_label='New random split', row_label='split',
                      curve_label='validation error (one split each)',
                      title='The validation set approach: one curve per random split',
                      size_percent=size_percent)


def cross_validation_demo(sim, true_errors, size_percent=100):
    """Demo G2: each click adds the 10-fold cross-validation error curve for the next
    shuffle (random_state 0 to 9). A toggle draws Demo F's ten validation-set curves
    faintly behind them.

    true_errors is Demo D's test error, indexed by K; its index sets the K values."""
    part2 = _part2_curves(sim, list(true_errors.index))
    _curve_adder_demo(sim, true_errors, part2.cross_validation, part2.ylim,
                      button_label='Add 10-Fold Run', row_label='run',
                      curve_label='10-fold cross-validation error (one shuffle each)',
                      title='10-fold cross-validation: one curve per random shuffle',
                      background=part2.validation,
                      background_label='validation-set curves (Demo F)',
                      size_percent=size_percent)


# --- Part 2, Demo G1: how k-fold cross-validation works ------------------------

def plot_kfold_diagram(n_folds=5):
    """Demo G1: one row per round; in each, one fold is the validation set and the
    rest are training, with the validation fold moving one place right each round."""
    with rc_context(BIG_FONTS):
        fig, ax = subplots(figsize=(11, 0.9 * n_folds + 1.2))
        for row in range(n_folds):
            y = n_folds - 1 - row
            for fold in range(n_folds):
                validation = fold == row
                ax.add_patch(Rectangle((fold, y + 0.1), 0.94, 0.8, ec='black', lw=1.2,
                                       fc=QUERY_COLOR if validation else _light(CLASS_COLORS[1])))
                ax.text(fold + 0.47, y + 0.5, 'validation' if validation else 'training',
                        ha='center', va='center', fontsize=13,
                        color='white' if validation else 'black',
                        fontweight='bold' if validation else 'normal')
            ax.text(-0.15, y + 0.5, f'Round {row + 1}', ha='right', va='center', fontsize=14)
        ax.set_xlim(-1.3, n_folds)
        ax.set_ylim(0, n_folds)
        ax.axis('off')
        ax.set_title(f'{n_folds}-fold cross-validation: each fold takes one turn as the '
                     f'validation set')
        fig.tight_layout()
    return fig
