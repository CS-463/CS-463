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


# --- Bootstrap notebook (ch_5_02): Demos B and C -------------------------------

BOOT_COLORS = {'mean': '#0072B2', 'median': '#E69F00', 'band': '#009E73', 'observed': '#D55E00'}
WAITS_BIN_WIDTH = 4


def waiting_times_demo(waits, scale=10, seed=2027, size_percent=100):
    """Demo B: a histogram of the waiting times, with the true exponential curve (the
    recipe) on top. Dashed lines mark the sample's mean and median, as the dashed line marks
    the data's mean in bootstrap_by_hand_demo; solid lines mark the recipe's true mean
    (scale) and true median (scale * ln 2).

    New sample from the recipe draws a fresh sample of the same size, to show how much
    a sample can vary. It only changes the picture; the waits passed in, which the rest
    of the notebook uses, stay the same. Back to our sample shows them again.

    Returns the widget."""
    waits = np.asarray(waits, dtype=float)
    n = len(waits)
    true_mean, true_median = scale, scale * np.log(2)
    rng = np.random.default_rng(seed)
    state = {'sample': waits, 'ours': True, 'count': 0}

    new_sample = widgets.Button(description='New sample from the recipe', button_style='info',
                                layout=widgets.Layout(width='240px'))
    ours = widgets.Button(description='Back to our sample', layout=widgets.Layout(width='180px'))
    out = _figure_area()

    def figure():
        sample = state['sample']
        right = max(60, WAITS_BIN_WIDTH * np.ceil(sample.max() / WAITS_BIN_WIDTH))
        fig, ax = subplots(figsize=(11, 5.5))
        ax.hist(sample, bins=np.arange(0, right + WAITS_BIN_WIDTH, WAITS_BIN_WIDTH),
                density=True, color='0.8', edgecolor='white', label=f'the {n} waits')
        ax.plot(sample, np.zeros(n), '|', color='black', ms=18, mew=1.5)
        x = np.linspace(0, right, 400)
        ax.plot(x, np.exp(-x / scale) / scale, color='black', lw=2.5,
                label=f'the recipe: exponential, mean {scale}')
        for name, value, true in (('mean', sample.mean(), true_mean),
                                  ('median', np.median(sample), true_median)):
            ax.axvline(value, color=BOOT_COLORS[name], lw=2.5, ls='--',
                       label=f'sample {name} = {value:.1f}')
            ax.axvline(true, color=BOOT_COLORS[name], lw=3,
                       label=f'true {name} = {true:.1f}')
        ax.set_xlim(0, right)
        ax.set_ylim(bottom=-0.01)
        ax.set_xlabel('waiting time (minutes)')
        ax.set_ylabel('density')
        which = 'Our sample' if state['ours'] else f'New sample {state["count"]}'
        ax.set_title(f'{which}: {n} waiting times, with the recipe that produced them')
        ax.legend(loc='upper right')
        fig.tight_layout()
        return fig

    def on_new_sample(_):
        state.update(sample=rng.exponential(scale=scale, size=n), ours=False,
                     count=state['count'] + 1)
        _show(out, figure, size_percent)

    def on_ours(_):
        state.update(sample=waits, ours=True)
        _show(out, figure, size_percent)

    new_sample.on_click(on_new_sample)
    ours.on_click(on_ours)
    _show(out, figure, size_percent)
    demo = widgets.VBox([widgets.HBox([new_sample, ours]), out])
    display(demo)
    return demo


from matplotlib.pyplot import figure as new_figure

BOOT_SAMPLE_COLOR = '#CC79A7'   # Okabe-Ito reddish purple, distinct from the median's orange
BOOT_STAT_BIN_WIDTH = 0.25


def waiting_times_bootstrap_demo(waits, B=10_000, seed=463, size_percent=100):
    """Demo B, bootstrap version: our sample stays fixed, and each bootstrap sample is
    n draws from it, with replacement.

    Top: our waits (gray histogram, dashed lines at their mean and median) with the
    current bootstrap sample outlined on top (solid lines at its mean and median).
    Middle: one tick per wait in our sample, with a dot for each time the current
    bootstrap sample drew it. Bottom: histograms of the bootstrap means and medians
    so far, titled with their standard deviation, the bootstrap standard error.

    The draws match the notebook's loop (rng.integers(0, n, size=n), B times, from
    default_rng(seed)), so after all B samples the standard errors match its printout.

    Returns the widget."""
    waits = np.asarray(waits, dtype=float)
    n = len(waits)
    rng = np.random.default_rng(seed)
    draws = [rng.integers(0, n, size=n) for _ in range(B)]
    stats = {'mean': np.array([waits[d].mean() for d in draws]),
             'median': np.array([np.median(waits[d]) for d in draws])}
    ours = {'mean': waits.mean(), 'median': np.median(waits)}
    every = np.concatenate([stats['mean'], stats['median']])
    stat_bins = np.arange(np.floor(every.min()), np.ceil(every.max()) + BOOT_STAT_BIN_WIDTH,
                          BOOT_STAT_BIN_WIDTH)
    right = max(60, WAITS_BIN_WIDTH * np.ceil(waits.max() / WAITS_BIN_WIDTH))
    wait_bins = np.arange(0, right + WAITS_BIN_WIDTH, WAITS_BIN_WIDTH)
    # Fixed vertical scales, tall enough for every bootstrap sample, so our waits look
    # the same from one click to the next.
    top_count = max(np.histogram(waits[d], wait_bins)[0].max() for d in draws)
    most_draws = max(np.bincount(d).max() for d in draws)
    state = {'b': 0}

    one = widgets.Button(description='One more', button_style='info',
                         layout=widgets.Layout(width='130px'))
    hundred = widgets.Button(description='100 more', button_style='info',
                             layout=widgets.Layout(width='130px'))
    all_b = widgets.Button(description=f'All {B:,}', button_style='info',
                           layout=widgets.Layout(width='130px'))
    reset = widgets.Button(description='Reset', layout=widgets.Layout(width='100px'))
    out = _figure_area()

    def figure():
        b = state['b']
        fig = new_figure(figsize=(13, 10))
        outer = fig.add_gridspec(2, 1, height_ratios=[4, 3], hspace=0.35)
        top = outer[0].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
        bottom = outer[1].subgridspec(1, 2, wspace=0.2)
        ax_data = fig.add_subplot(top[0])
        ax_strip = fig.add_subplot(top[1], sharex=ax_data)
        ax_stats = [fig.add_subplot(bottom[0]), fig.add_subplot(bottom[1])]

        # Our sample, with the current bootstrap sample on top.
        ax_data.hist(waits, bins=wait_bins, color='0.8', edgecolor='white',
                     label=f'our {n} waits')
        for name in ('mean', 'median'):
            ax_data.axvline(ours[name], color=BOOT_COLORS[name], lw=2.5, ls='--',
                            label=f'our {name} = {ours[name]:.1f}')
        counts = np.zeros(n, dtype=int)
        if b:
            draw = draws[b - 1]
            counts = np.bincount(draw, minlength=n)
            sample = waits[draw]
            ax_data.hist(sample, bins=wait_bins, histtype='step', color=BOOT_SAMPLE_COLOR,
                         lw=3, label=f'bootstrap sample {b:,}')
            for name in ('mean', 'median'):
                ax_data.axvline(stats[name][b - 1], color=BOOT_COLORS[name], lw=3,
                                label=f'bootstrap {name} = {stats[name][b - 1]:.1f}')
        ax_data.set_xlim(0, right)
        ax_data.tick_params(labelbottom=False)
        ax_data.set_ylim(0, top_count + 1)
        ax_data.set_ylabel('number of waits')
        ax_data.legend(loc='upper right', fontsize=12)
        title = f'Our {n} waits: this sample stays fixed, and every bootstrap sample is drawn from it'
        if b:
            title = (f'Bootstrap sample b = {b:,}: {n} draws from our {n} waits, with '
                     f'replacement\n({np.count_nonzero(counts)} different waits drawn, '
                     f'{n - np.count_nonzero(counts)} left out)')
        ax_data.set_title(title)

        # One tick per wait we have; a dot for each time this bootstrap sample drew it.
        ax_strip.plot(waits, np.zeros(n), '|', color='black', ms=18, mew=1.5)
        for x, c in zip(waits, counts):
            ax_strip.plot(np.full(c, x), 1 + np.arange(c), 'o', ms=7,
                          color=BOOT_SAMPLE_COLOR, mec='black', mew=0.5)
        ax_strip.set_ylim(-0.8, most_draws + 1)
        ax_strip.set_yticks([])
        ax_strip.set_xlabel('waiting time (minutes): ticks are our waits, dots are draws')

        # The bootstrap means and medians so far.
        for ax, name in zip(ax_stats, ('mean', 'median')):
            values = stats[name][:b]
            ax.hist(values, bins=stat_bins, color=BOOT_COLORS[name], edgecolor='white',
                    linewidth=0.5)
            ax.axvline(ours[name], color='black', lw=2, ls='--', zorder=3,
                       label=f'our {name} = {ours[name]:.1f}')
            se = (f'standard error = {np.std(values, ddof=1):.2f}' if b > 1
                  else 'standard error: needs 2 or more')
            ax.set_title(f'{b:,} bootstrap {name}{"" if b == 1 else "s"}\n{se}')
            ax.legend(loc='upper left', fontsize=12)
            ax.set_xlim(stat_bins[0], stat_bins[-1])
            ax.set_ylim(0, max(1, ax.get_ylim()[1]))
            ax.set_xlabel(f'{name} of a bootstrap sample (minutes)')
        ax_stats[0].set_ylabel('number of bootstrap samples')
        return fig

    def render():
        for button in (one, hundred, all_b):
            button.disabled = state['b'] == B
        _show(out, figure, size_percent)

    def add(k):
        state['b'] = min(B, state['b'] + k)
        render()

    one.on_click(lambda _: add(1))
    hundred.on_click(lambda _: add(100))
    all_b.on_click(lambda _: add(B))
    reset.on_click(lambda _: (state.update(b=0), render()))
    render()
    demo = widgets.VBox([widgets.HBox([one, hundred, all_b, reset]), out])
    display(demo)
    return demo


def plot_bootstrap_stats(boot_means, boot_medians, se_mean, se_median):
    """Demo B: histograms of the bootstrap means and medians, on a shared x axis."""
    with rc_context(BIG_FONTS):
        fig, axes = subplots(1, 2, figsize=(14, 4.8), sharex=True, sharey=True)
        for ax, values, name, se in ((axes[0], boot_means, 'mean', se_mean),
                                     (axes[1], boot_medians, 'median', se_median)):
            ax.hist(values, bins=50, color=BOOT_COLORS[name], edgecolor='white', linewidth=0.5)
            ax.set_title(f'{len(values):,} bootstrap {name}s\nstandard error = {se:.2f}')
            ax.set_xlabel(f'{name} of a bootstrap sample')
        axes[0].set_ylabel('number of bootstrap samples')
        fig.tight_layout()
    return fig


def plot_bootstrap_difference(diffs, observed, interval):
    """Demo C: histogram of the bootstrap accuracy differences (percentage points), with
    the observed difference, the 95% interval, and a dashed line at zero."""
    diffs, observed = 100 * np.asarray(diffs), 100 * observed
    low, high = 100 * np.asarray(interval)
    with rc_context(BIG_FONTS):
        fig, ax = subplots(figsize=(11, 5))
        ax.hist(diffs, bins=np.arange(diffs.min() - 0.5, diffs.max() + 1.5, 1.0),
                color='0.75', edgecolor='white', linewidth=0.5)
        ax.axvspan(low, high, color=BOOT_COLORS['band'], alpha=0.25,
                   label=f'95% interval: {low:+.0f} to {high:+.0f}')
        ax.axvline(observed, color=BOOT_COLORS['observed'], lw=3,
                   label=f'observed difference: {observed:+.0f}')
        ax.axvline(0, color='black', ls='--', lw=2.5, label='no difference')
        ax.set_xlabel('KNN accuracy minus logistic regression accuracy (percentage points)')
        ax.set_ylabel('number of bootstrap samples')
        ax.set_title(f'{len(diffs):,} bootstrap differences')
        ax.legend(loc='upper left')
        fig.tight_layout()
    return fig


# --- Bootstrap notebook (ch_5_02): bootstrapping by hand -----------------------

from time import sleep

from matplotlib.patches import ConnectionPatch

HAND_SCORES = (-4, -2, -1, 1, 2, 3, 4, 5)   # n = 8 people
HAND_B = 20                                 # B = 20 bootstrap samples
HAND_DOT_SIZE = 24
HAND_STEP_SECONDS = 0.05
HAND_TITLE_BOX = {'facecolor': 'white', 'edgecolor': 'none', 'pad': 2}   # hides the arrow behind titles
HAND_MEAN_GAP = 0.3       # means closer than this (in score units) stack instead of overlapping


def _number_line(ax, half_width, label=True):
    """A horizontal number line at y = 0 with integer ticks, and no other axes."""
    ax.axhline(0, color='black', lw=1.5, zorder=1)
    ax.set_xlim(-half_width, half_width)
    ax.set_xticks(range(-int(half_width) + 1, int(half_width)))
    ax.tick_params(axis='x', labelbottom=True, length=6)
    ax.set_yticks([])
    for side in ('left', 'right', 'top'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_position(('data', 0))
    if label:
        ax.text(0.0, -0.30, '← harmed', transform=ax.transAxes, ha='left', va='top', fontsize=13)
        ax.text(1.0, -0.30, 'improved →', transform=ax.transAxes, ha='right', va='top', fontsize=13)


def _stack_heights(values):
    """For each value, how many earlier values equal it (its height in a dot stack)."""
    seen, heights = {}, []
    for v in values:
        heights.append(seen.get(v, 0))
        seen[v] = seen.get(v, 0) + 1
    return heights


def _swarm_heights(values, gap):
    """Stack levels for a dot plot that keeps each dot at its exact x: each value takes
    the lowest level with no earlier dot within gap of it."""
    levels = []
    for i, v in enumerate(values):
        taken = {levels[j] for j in range(i) if abs(values[j] - v) < gap}
        level = 0
        while level in taken:
            level += 1
        levels.append(level)
    return levels


def bootstrap_by_hand_demo(scores=HAND_SCORES, B=HAND_B, seed=463, size_percent=100):
    """Bootstrapping, one draw at a time: B bootstrap samples of n = len(scores) points.

    Top line: the data, one numbered dot per person. Middle line: the current bootstrap
    sample. Each click of Draw one picks a person at random (with replacement), draws an
    arrow from their dot, and copies it to the middle line. After n draws the sample is
    complete: its mean and median go in the table, and its mean is added to the bottom
    line, which collects one mean per bootstrap sample. Full sample finishes the current
    sample (or does a whole new one), animated. All B samples finishes every remaining
    sample without animating each draw. After B samples, only Reset works. Reset starts
    over with the same random draws.

    Returns the widget."""
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    colors = [cm.tab10(i % 10) for i in range(n)]
    half_width = np.ceil(np.abs(scores).max()) + 1.5
    data_mean, data_median = scores.mean(), np.median(scores)
    state = {}

    draw_one = widgets.Button(description='Draw one', button_style='info',
                              layout=widgets.Layout(width='140px'))
    full_sample = widgets.Button(description='Full sample', button_style='info',
                                layout=widgets.Layout(width='140px'))
    all_samples = widgets.Button(description=f'All {B} samples', button_style='info',
                                 layout=widgets.Layout(width='160px'))
    reset = widgets.Button(description='Reset', layout=widgets.Layout(width='100px'))
    table = widgets.HTML(layout=widgets.Layout(min_width='230px', max_height='640px',
                                               overflow='auto'))
    out = _figure_area()

    def start_over():
        state.update(rng=np.random.default_rng(seed), draws=[], done=False,
                     means=[], medians=[])

    def new_sample_if_needed():
        if state['done']:
            state.update(draws=[], done=False)

    def draw_next():
        state['draws'].append(int(state['rng'].integers(0, n)))
        if len(state['draws']) == n:
            sample = scores[state['draws']]
            state['means'].append(sample.mean())
            state['medians'].append(np.median(sample))
            state['done'] = True

    def figure():
        fig, (ax_data, ax_sample, ax_means) = subplots(
            3, 1, figsize=(11, 9.5), gridspec_kw={'height_ratios': [1.3, 2.2, 2.2]})
        for ax in (ax_data, ax_sample, ax_means):
            _number_line(ax, half_width, label=ax is not ax_means)
            ax.patch.set_visible(False)   # so the arrow, drawn behind the axes, shows through

        # The data: one numbered dot per person.
        data_heights = _stack_heights(scores)
        data_y = [0.55 + 0.9 * h for h in data_heights]
        for i, (x, y) in enumerate(zip(scores, data_y)):
            ax_data.plot(x, y, 'o', ms=HAND_DOT_SIZE, color=colors[i], mec='black', zorder=3)
            ax_data.text(x, y, str(i + 1), ha='center', va='center', color='white',
                         fontsize=12, fontweight='bold', zorder=4)
        ax_data.set_ylim(0, max(data_y) + 0.7)
        ax_data.axvline(data_mean, color='black', ls='--', lw=1.5, zorder=0)
        ax_data.set_title(f'The data: n = {n} people (mean = {data_mean:.1f}, '
                          f'median = {data_median:.1f})', loc='left', bbox=HAND_TITLE_BOX)

        # The current bootstrap sample, stacked where a person is drawn more than once.
        draws = state['draws']
        sample = scores[draws]
        sample_y = [0.55 + 0.9 * h for h in _stack_heights(sample)]
        for i, x, y in zip(draws, sample, sample_y):
            ax_sample.plot(x, y, 'o', ms=HAND_DOT_SIZE, color=colors[i], mec='black', zorder=3)
            ax_sample.text(x, y, str(i + 1), ha='center', va='center', color='white',
                           fontsize=12, fontweight='bold', zorder=4)
        ax_sample.set_ylim(0, max([4.0] + [y + 0.7 for y in sample_y]))
        b = len(state['means']) + (0 if state['done'] else 1)
        title = f'Bootstrap sample b = {b} of B = {B}: {len(draws)} of n = {n} draws'
        if state['done']:
            ax_sample.axvline(sample.mean(), color=QUERY_COLOR, lw=3, zorder=2)
            title += f' (mean = {sample.mean():.1f})'
        ax_sample.set_title(title, loc='left', bbox=HAND_TITLE_BOX)
        if draws:
            i, x_to, y_to = draws[-1], sample[-1], sample_y[-1]
            arrow = ConnectionPatch(xyA=(scores[i], data_y[i] - 0.35), coordsA=ax_data.transData,
                                    xyB=(x_to, y_to + 0.35), coordsB=ax_sample.transData,
                                    arrowstyle='-|>', mutation_scale=22, lw=2.5,
                                    color=colors[i], zorder=-1)   # behind titles and dots
            fig.add_artist(arrow)

        # One mean per completed bootstrap sample, as a dot plot.
        means = state['means']
        heights = _swarm_heights(means, HAND_MEAN_GAP)
        tallest = max(heights, default=0) + 1
        step = min(1.0, 6 / tallest)
        for x, h in zip(means, heights):
            ax_means.plot(x, 0.4 + step * h, 'o', ms=max(5, 14 * step), color=QUERY_COLOR,
                          mec='black', mew=0.6, zorder=3)
        ax_means.set_ylim(0, 7)
        ax_means.axvline(data_mean, color='black', ls='--', lw=1.5, zorder=0)
        ax_means.set_title(f'Means of the bootstrap samples: {len(means)} of B = {B} '
                           f'(dashed line: mean of the data)', loc='left', bbox=HAND_TITLE_BOX)
        fig.tight_layout(h_pad=2.5)
        return fig

    def render():
        rows = ''.join(f'<tr><td style="padding: 0 14px 0 0;">{r + 1}</td>'
                       f'<td style="padding: 0 14px 0 0; text-align: right;">{m:.1f}</td>'
                       f'<td style="text-align: right;">{md:.1f}</td></tr>'
                       for r, (m, md) in enumerate(zip(state['means'], state['medians'])))
        table.value = _html(
            '<table><tr><th style="padding: 0 14px 0 0;">Sample <i>b</i></th>'
            '<th style="padding: 0 14px 0 0;">Mean</th><th>Median</th></tr>'
            f'<tr style="border-bottom: 1px solid #999;"><td style="padding: 0 14px 0 0;"><i>data</i></td>'
            f'<td style="padding: 0 14px 0 0; text-align: right;"><i>{data_mean:.1f}</i></td>'
            f'<td style="text-align: right;"><i>{data_median:.1f}</i></td></tr>'
            f'{rows}</table>')
        _show(out, figure, size_percent)

    def finished():
        return state['done'] and len(state['means']) == B

    def set_busy(busy):
        for button in (draw_one, full_sample, all_samples):
            button.disabled = busy or finished()
        reset.disabled = busy

    def on_draw_one(_):
        if finished():
            return
        new_sample_if_needed()
        draw_next()
        render()
        set_busy(False)

    def on_full_sample(_):
        if finished():
            return
        set_busy(True)
        try:
            new_sample_if_needed()
            while not state['done']:
                draw_next()
                render()
                sleep(HAND_STEP_SECONDS)
        finally:
            set_busy(False)

    def on_all_samples(_):
        set_busy(True)
        try:
            while not finished():
                new_sample_if_needed()
                while not state['done']:
                    draw_next()
                render()
        finally:
            set_busy(False)

    def on_reset(_):
        start_over()
        render()
        set_busy(False)

    draw_one.on_click(on_draw_one)
    full_sample.on_click(on_full_sample)
    all_samples.on_click(on_all_samples)
    reset.on_click(on_reset)
    start_over()
    render()
    demo = widgets.VBox([widgets.HBox([draw_one, full_sample, all_samples, reset]),
                         widgets.HBox([out, table])])
    display(demo)
    return demo
