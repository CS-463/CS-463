"""Simulation, plotting, and widget helpers for the Chapters 6 and 8 notebooks.

ch_6_and_8_01_trees_and_regularization.ipynb, ch_6_and_8_02_ridge_lasso_ensembles.ipynb,
and ch_6_and_8_03_preparing_real_data.ipynb use this module. The notebooks stay
focused on ideas; the simulation, matplotlib, and ipywidgets details live here.

The simulated data, its Bayes boundary, the colors, and the figure sizing are
copied from ch_5_demos_helpers.py (not imported), so this folder works on its own,
for example on Colab. simulate_data() returns the same 200 training points as
Chapter 5.

Classes are coded 1 (circles) and 2 (triangles) throughout, matching the
notebooks' "class 1" and "class 2".
"""
import warnings
from io import BytesIO
from types import SimpleNamespace

import ipywidgets as widgets
import numpy as np
from IPython.display import display
from matplotlib import get_backend, rc_context
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
from matplotlib.pyplot import close, subplots, switch_backend
from matplotlib.ticker import PercentFormatter
from scipy.stats import multivariate_normal
from sklearn.ensemble import BaggingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

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
CURVE_COLORS = {'training error': '#0072B2', 'test error': '#D55E00',
                'cross-validation error': '#009E73'}
OKABE_ITO = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00', '#F0E442',
             '#000000']
PENALTY_COLOR = '#D55E00'
BAYES_LINE = dict(colors='black', linestyles='--', linewidths=2.5)


def _light(color, amount=0.3):
    """The color mixed with white: amount = 1 is the color itself, 0 is white."""
    return tuple(amount * c + (1 - amount) for c in to_rgb(color))


REGION_CMAP = ListedColormap([_light(CLASS_COLORS[1]), _light(CLASS_COLORS[2])])


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


# --- The simulated data (copied from ch_5_demos_helpers.py) -------------------

BUMP_CENTERS = {1: np.array([(1.00, 0.30), (0.73, -0.89), (0.55, -0.99), (1.06, 1.34),
                             (0.51, -0.62), (1.49, 0.36), (1.11, -0.93), (0.97, 0.70),
                             (-0.34, -0.46), (-0.90, -1.29)]),
                2: np.array([(-1.84, 0.76), (-1.27, 1.27), (0.16, 0.81), (-2.52, 0.46),
                             (-0.05, 1.11), (-1.53, 0.52), (-0.98, 0.19), (1.06, 0.19),
                             (-0.03, 1.88), (-0.58, 0.89)])}
BUMP_SD = np.sqrt(0.2)
GRID_X1 = np.linspace(-4, 3.5, 301)
GRID_X2 = np.linspace(-3, 4, 281)
GRID_SHAPE = (len(GRID_X2), len(GRID_X1))
EXTENT = (GRID_X1[0], GRID_X1[-1], GRID_X2[0], GRID_X2[-1])


def _draw_class(rng, k, n, sd):
    bumps = rng.integers(0, len(BUMP_CENTERS[k]), n)
    return BUMP_CENTERS[k][bumps] + rng.normal(0, sd, (n, 2))


def _density(points, k, sd):
    """f_k(x): the average of class k's round Gaussian bump densities."""
    return np.mean([multivariate_normal(center, sd**2 * np.eye(2)).pdf(points)
                    for center in BUMP_CENTERS[k]], axis=0)


def simulate_data(sd=BUMP_SD, n_train=100, n_test=5000, seed=463):
    """The Chapter 5 two-class simulation: equal priors, each class a mix of ten small
    round bumps.

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


def _grid_points():
    g1, g2 = np.meshgrid(GRID_X1, GRID_X2)
    return np.column_stack([g1.ravel(), g2.ravel()])


_bayes_cache = {}


def _bayes_grid(sim):
    if id(sim) not in _bayes_cache:
        _bayes_cache[id(sim)] = sim.posterior1(_grid_points()).reshape(GRID_SHAPE)
    return _bayes_cache[id(sim)]


def _fit_summary(model, sim):
    """Everything a boundary demo needs from one fitted classifier, computed once."""
    summary = SimpleNamespace(
        predicted=model.predict(_grid_points()).reshape(GRID_SHAPE),
        train_error=1 - model.score(sim.X, sim.y),
        test_error=1 - model.score(sim.X_test, sim.y_test))
    if hasattr(model, 'get_n_leaves'):
        summary.leaves = int(model.get_n_leaves())
        summary.depth = int(model.get_depth())
    return summary


def _draw_regions(ax, sim, predicted, show_bayes=False, title='', ms=8, legend=True):
    """Predicted-class shading, the training points, and optionally the Bayes boundary."""
    ax.imshow(predicted, origin='lower', extent=EXTENT, cmap=REGION_CMAP, vmin=1, vmax=2,
              interpolation='nearest')
    _plot_classes(ax, sim.X, sim.y, ms=ms)
    extra = []
    if show_bayes:
        ax.contour(GRID_X1, GRID_X2, _bayes_grid(sim), levels=[0.5], **BAYES_LINE)
        extra.append((Line2D([], [], color='black', ls='--', lw=2.5), 'Bayes boundary'))
    ax.set_xlim(GRID_X1[0], GRID_X1[-1])
    ax.set_ylim(GRID_X2[0], GRID_X2[-1])
    ax.set_aspect('equal')
    ax.set_xlabel('X₁')
    ax.set_ylabel('X₂')
    ax.set_title(title)
    if legend:
        _class_legend(ax, extra=extra, loc='upper left', framealpha=0.95)


# --- NB1 Demo A: how a tree predicts ------------------------------------------

def _leaf_boxes(model, lows, highs):
    """{leaf node id: (lows, highs)}: the box each leaf covers, within the given limits."""
    tree = model.tree_
    boxes = {}

    def visit(node, lo, hi):
        if tree.children_left[node] == -1:
            boxes[node] = (lo, hi)
            return
        j, s = tree.feature[node], tree.threshold[node]
        left_hi, right_lo = list(hi), list(lo)
        left_hi[j], right_lo[j] = min(hi[j], s), max(lo[j], s)
        visit(tree.children_left[node], lo, left_hi)
        visit(tree.children_right[node], right_lo, hi)

    visit(0, list(lows), list(highs))
    return boxes


LEAF_COLORS = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#F0E442']


def _preorder(tree, node=0):
    """Node numbers in the order plot_tree draws them: a node, then its left
    subtree, then its right subtree."""
    if tree.children_left[node] == -1:
        return [node]
    return ([node] + _preorder(tree, tree.children_left[node])
            + _preorder(tree, tree.children_right[node]))


def _growth_colors(model):
    """{node: color} for a tree grown best-first, replaying its splits in order:
    the root gets the first color, and the two children of each split get the
    first colors no current leaf is using. Every smaller tree in the sequence uses
    the same node numbers, so a box keeps its color until it is split, and no two
    leaves share a color while there are at most len(LEAF_COLORS) of them."""
    tree = model.tree_
    colors = {0: LEAF_COLORS[0]}
    leaves = {0}
    for child in range(1, tree.node_count, 2):   # children come in pairs, in split order
        parent = int(np.flatnonzero(tree.children_left == child)[0])
        leaves.discard(parent)
        for node in (child, child + 1):
            used = {colors[leaf] for leaf in leaves}
            free = [c for c in LEAF_COLORS if c not in used] or LEAF_COLORS
            colors[node] = free[0]
            leaves.add(node)
    return colors


def _draw_tree_and_boxes(model, X, colors, new_node=None):
    """A fitted regression tree on two features, drawn two ways.

    Left: the tree diagram (scikit-learn's plot_tree). Right: the plane of the two
    features with the training points and one box per leaf, labeled with the
    leaf's prediction. Matching leaves and boxes share a color from colors
    ({node: color}). y is log salary (in thousands of dollars), so each label
    also gives the salary it stands for. new_node, if given, is the node split
    most recently: it gets a thick outline in the tree, and its split is drawn
    as a thick line in the plane."""
    names = list(X.columns)
    values = X.to_numpy(float)
    pad = 0.04 * (values.max(axis=0) - values.min(axis=0))
    lows, highs = values.min(axis=0) - pad, values.max(axis=0) + pad
    span = highs - lows
    tree = model.tree_
    boxes = _leaf_boxes(model, lows, highs)
    fig, (ax_tree, ax_box) = subplots(1, 2, figsize=(17, 7.5),
                                      gridspec_kw={'width_ratios': [1, 1.1]})
    annotations = plot_tree(model, feature_names=names, impurity=False, filled=False,
                            rounded=True, precision=2, fontsize=14, ax=ax_tree)
    # plot_tree also returns the 'True'/'False' arrow labels; the node boxes are
    # the annotations with a 'samples' line, in preorder.
    nodes = [a for a in annotations if 'samples' in a.get_text()]
    for node, annotation in zip(_preorder(tree), nodes):
        if node in boxes:
            annotation.set_bbox(dict(boxstyle='round', fc=_light(colors[node], 0.45),
                                     ec='black'))
        elif node == new_node:
            annotation.set_bbox(dict(boxstyle='round', fc='white', ec=PENALTY_COLOR, lw=3.5))
    ax_tree.set_title('The tree: follow the splits down to a leaf\n'
                      '(left branch = condition is true; value = mean log salary)')
    for leaf, (lo, hi) in boxes.items():
        ax_box.add_patch(Rectangle(lo, hi[0] - lo[0], hi[1] - lo[1],
                                   fc=_light(colors[leaf], 0.45), ec='black', lw=1.5))
        prediction = tree.value[leaf].item()
        # Two lines fit a roomy box; a short box gets one line, a narrow one
        # gets one line turned sideways.
        width, height = (np.array(hi) - np.array(lo)) / span
        narrow, short = width < 0.12, height < 0.15
        joiner = ' ' if narrow or short else '\n'
        ax_box.text((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2,
                    f'{prediction:.2f}{joiner}≈ ${np.exp(prediction):,.0f}K',
                    ha='center', va='center', fontweight='bold', zorder=4,
                    fontsize=12 if narrow or short else 15, rotation=90 if narrow else 0,
                    bbox=dict(fc='white', ec='none', alpha=0.85, pad=2))
    if new_node is not None:
        # The new split line runs across the box its two children share.
        left_lo, left_hi = boxes[tree.children_left[new_node]]
        right_lo, right_hi = boxes[tree.children_right[new_node]]
        lo, hi = np.minimum(left_lo, right_lo), np.maximum(left_hi, right_hi)
        s = tree.threshold[new_node]
        if tree.feature[new_node] == 0:
            ax_box.plot([s, s], [lo[1], hi[1]], color=PENALTY_COLOR, lw=4, zorder=5)
        else:
            ax_box.plot([lo[0], hi[0]], [s, s], color=PENALTY_COLOR, lw=4, zorder=5)
    ax_box.scatter(values[:, 0], values[:, 1], s=18, color='0.25', alpha=0.6, zorder=3)
    ax_box.set_xlim(lows[0], highs[0])
    ax_box.set_ylim(lows[1], highs[1])
    ax_box.set_xlabel(names[0])
    ax_box.set_ylabel(names[1])
    ax_box.set_title('The same tree as boxes: one box per leaf\n'
                     '(label: mean log salary, and the salary it stands for)')
    fig.tight_layout()
    return fig


def _total_rss(model):
    """The training RSS of a regression tree: each leaf's impurity (its mean squared
    error) times its number of points, added over the leaves."""
    tree = model.tree_
    leaves = tree.children_left == -1
    return float((tree.impurity[leaves] * tree.n_node_samples[leaves]).sum())


def grow_tree_demo(X, y, max_leaves=6, size_percent=100):
    """Demo A: grow a regression tree one split at a time.

    Each click of 'Next split' refits with max_leaf_nodes one larger. With that
    setting scikit-learn grows best-first: of all the current leaves, it splits the
    one whose best split lowers the RSS the most. So each tree is the previous one
    plus exactly one split, and node numbers carry over. The newest split is
    outlined in the tree and drawn as a thick line among the boxes."""
    names = list(X.columns)
    # max_leaf_nodes must be at least 2, so the one-leaf tree is a tree that
    # isn't allowed to split at all.
    trees = [DecisionTreeRegressor(min_samples_split=len(y) + 1).fit(X, y)]
    trees += [DecisionTreeRegressor(max_leaf_nodes=k, random_state=0).fit(X, y)
              for k in range(2, max_leaves + 1)]
    colors = _growth_colors(trees[-1])
    step =widgets.IntText(value=0, layout=widgets.Layout(display='none'))
    back = widgets.Button(description='◀ Back', layout=widgets.Layout(width='120px'))
    forward = widgets.Button(description='Next split ▶', button_style='primary',
                             layout=widgets.Layout(width='150px'))
    restart = widgets.Button(description='Start over', layout=widgets.Layout(width='120px'))
    readout = widgets.HTML()
    out = _figure_area()

    def update(change=None):
        k = step.value
        model = trees[k]
        tree = model.tree_
        back.disabled = restart.disabled = k == 0
        forward.disabled = k == max_leaves - 1
        if k == 0:
            mean = tree.value[0].item()
            readout.value = _html(
                f'<b>No splits yet.</b> The tree is a single leaf: it predicts the mean log '
                f'salary of all {tree.n_node_samples[0]} players, <b>{mean:.2f}</b> '
                f'(≈ ${np.exp(mean):,.0f}K), for everyone. Training RSS = '
                f'<b>{_total_rss(model):.1f}</b>.')
            new_node = None
        else:
            # The newest node numbers are the two children of the newest split.
            new_node = int(np.flatnonzero(tree.children_left == tree.node_count - 2)[0])
            name, s = names[tree.feature[new_node]], tree.threshold[new_node]
            left, right = tree.children_left[new_node], tree.children_right[new_node]
            readout.value = _html(
                f'<b>Split {k}</b> ({k + 1} leaves): the box of '
                f'<b>{tree.n_node_samples[new_node]}</b> players splits on <b>{name} ≤ {s:g}</b>, '
                f'into {tree.n_node_samples[left]} and {tree.n_node_samples[right]} players. '
                f'Training RSS drops from <b>{_total_rss(trees[k - 1]):.1f}</b> to '
                f'<b>{_total_rss(model):.1f}</b>.')
        _show(out, lambda: _draw_tree_and_boxes(model, X, colors, new_node), size_percent)

    def move(by):
        step.value = min(max(step.value + by, 0), max_leaves - 1)

    back.on_click(lambda button: move(-1))
    forward.on_click(lambda button: move(1))
    restart.on_click(lambda button: move(-max_leaves))
    step.observe(update, names='value')
    update()
    display(widgets.VBox([widgets.HBox([back, forward, restart]), readout, out]))


# --- NB1 Demo B: finding the best split ----------------------------------------

def _split_rss(x, y):
    """For each candidate split point s (midpoints between sorted distinct values of x),
    the total RSS of the two sides when each side predicts its own mean."""
    distinct = np.unique(x)
    points = (distinct[:-1] + distinct[1:]) / 2
    rss = []
    for s in points:
        left, right = y[x < s], y[x >= s]
        rss.append(((left - left.mean())**2).sum() + ((right - right.mean())**2).sum())
    return points, np.array(rss)


def best_split_demo(X, y, size_percent=100):
    """Demo B: try every split point on one feature and keep the best.

    Top: the points against the chosen feature, the split line, and each side's
    mean as a flat segment. Bottom: the total RSS of the two sides for every
    possible split point, with the current one marked and the best one starred."""
    y = np.asarray(y, float)
    candidates = {name: _split_rss(X[name].to_numpy(float), y) for name in X.columns}
    feature = widgets.Dropdown(options=list(X.columns), description='feature',
                               style={'description_width': '70px'},
                               layout=widgets.Layout(width='240px'))
    split = widgets.SelectionSlider(options=[0.0], description='split point s',
                                    style={'description_width': '100px'},
                                    layout=widgets.Layout(width='520px'))
    readout = widgets.HTML()
    out = _figure_area()

    def set_options(change=None):
        points, _ = candidates[feature.value]
        options = [(f'{s:g}', s) for s in points]
        split.options = options
        split.value = options[len(options) // 5][1]   # start well away from the best split

    def update(change=None):
        name = feature.value
        x = X[name].to_numpy(float)
        points, rss = candidates[name]
        s = split.value
        best = int(np.argmin(rss))
        here = int(np.argmin(np.abs(points - s)))
        left, right = y[x < s], y[x >= s]
        readout.value = _html(
            f'Left ({name} &lt; {s:g}): <b>{len(left)}</b> players, mean log salary '
            f'<b>{left.mean():.2f}</b>. Right ({name} ≥ {s:g}): <b>{len(right)}</b> players, mean '
            f'<b>{right.mean():.2f}</b>.<br>RSS = <b>{rss[here]:.1f}</b>. '
            f'Best split on {name}: {name} &lt; {points[best]:g}, RSS = <b>{rss[best]:.1f}</b>.')

        def draw():
            fig, (top, bottom) = subplots(2, 1, figsize=(11, 9.5), sharex=True,
                                          gridspec_kw={'height_ratios': [1.3, 1]})
            top.scatter(x, y, s=22, color='0.35', alpha=0.6)
            top.axvline(s, color=PENALTY_COLOR, lw=2.5)
            lo, hi = x.min() - 0.5, x.max() + 0.5
            top.plot([lo, s], [left.mean()] * 2, color=CLASS_COLORS[1], lw=4,
                     label=f'left mean {left.mean():.2f}')
            top.plot([s, hi], [right.mean()] * 2, color=CLASS_COLORS[2], lw=4,
                     label=f'right mean {right.mean():.2f}')
            top.set_ylabel('log salary')
            top.set_title(f'Split at {name} = {s:g}: each side predicts its own mean')
            top.legend(loc='lower right')
            bottom.plot(points, rss, color='0.3', lw=2)
            bottom.plot(points[here], rss[here], 'o', ms=12, color=PENALTY_COLOR,
                        label='current split')
            bottom.plot(points[best], rss[best], '*', ms=22, color='#009E73', mec='black',
                        label='best split')
            bottom.set_xlim(lo, hi)
            bottom.set_xlabel(name)
            bottom.set_ylabel('total RSS')
            bottom.set_title('Total RSS of the two sides, for every split point')
            bottom.legend(loc='upper right')
            fig.tight_layout()
            return fig
        _show(out, draw, size_percent)

    feature.observe(set_options, names='value')
    feature.observe(update, names='value')
    set_options()
    split.observe(update, names='value')
    update()
    display(widgets.VBox([widgets.HBox([feature, split]), readout, out]))


# --- NB1 Gini demo: moving shapes between boxes --------------------------------

BOX_COLORS = {'left': '#009E73', 'right': '#CC79A7'}


def _gini(circles, triangles):
    n = circles + triangles
    return 2 * (circles / n) * (triangles / n) if n else np.nan


def _draw_shape_box(ax, x0, circles, triangles, side):
    """One box of circles and triangles, 5 shapes per row, labeled with its Gini index."""
    ax.add_patch(Rectangle((x0, 0), 5.4, 2.6, fill=False, lw=3, ec=BOX_COLORS[side]))
    shapes = [1] * circles + [2] * triangles
    for i, k in enumerate(shapes):
        ax.plot(x0 + 0.7 + (i % 5), 1.9 - 1.2 * (i // 5), CLASS_MARKERS[k], ms=26,
                color=CLASS_COLORS[k], mec='black', mew=0.8)
    n = circles + triangles
    if n:
        label = (f'{circles} ○, {triangles} △\n'
                 f'$\\hat p$ = {circles}/{n} = {circles / n:.2f}\nG = {_gini(circles, triangles):.3f}')
    else:
        label = 'empty'
    ax.text(x0 + 2.7, -0.3, label, ha='center', va='top', fontsize=15, color=BOX_COLORS[side],
            fontweight='bold')
    ax.text(x0 + 2.7, 2.85, f'{side.capitalize()} box', ha='center', va='bottom', fontsize=16,
            color=BOX_COLORS[side], fontweight='bold')


def gini_split_demo(circles=5, triangles=5, size_percent=100):
    """Split a box of circles and triangles into two boxes and score the split.

    Left: the two boxes and each one's Gini index. Right: the two-class Gini curve
    2p(1 - p), with a dot for each box (sized by its share of the points), the
    parent box on the curve, and the weighted split score below it. A checkbox adds
    the line between the two boxes' dots; the split score always lies on it."""
    left_c = widgets.IntSlider(value=4, min=0, max=circles, description='circles left',
                               style={'description_width': '100px'},
                               layout=widgets.Layout(width='380px'))
    left_t = widgets.IntSlider(value=1, min=0, max=triangles, description='triangles left',
                               style={'description_width': '100px'},
                               layout=widgets.Layout(width='380px'))
    show_line = widgets.Checkbox(value=False, description='Show the line between the dots',
                                 layout=widgets.Layout(width='320px'))
    out = _figure_area()
    n = circles + triangles
    p_parent, g_parent = circles / n, _gini(circles, triangles)
    p = np.linspace(0, 1, 201)

    def update(change=None):
        boxes = {'left': (left_c.value, left_t.value),
                 'right': (circles - left_c.value, triangles - left_t.value)}
        sizes = {side: sum(counts) for side, counts in boxes.items()}
        score = sum(sizes[side] / n * _gini(*boxes[side])
                    for side in boxes if sizes[side])

        def draw():
            fig, (shapes, curve) = subplots(1, 2, figsize=(18, 7.5),
                                            gridspec_kw={'width_ratios': [1, 1.05]})
            _draw_shape_box(shapes, 0, *boxes['left'], 'left')
            _draw_shape_box(shapes, 6.2, *boxes['right'], 'right')
            shapes.set_xlim(-0.3, 11.9)
            shapes.set_ylim(-2.6, 3.6)
            shapes.set_aspect('equal')
            shapes.axis('off')
            parts = ' + '.join(f'$\\frac{{{sizes[s]}}}{{{n}}}$({_gini(*boxes[s]):.3f})'
                               for s in boxes if sizes[s])
            shapes.set_title(f'Before the split: {circles} ○, {triangles} △, G = {g_parent:.3f}\n'
                             f'Split score = {parts} = {score:.3f}\n'
                             f'Improvement = {g_parent:.3f} − {score:.3f} = {g_parent - score:.3f}',
                             fontsize=17)

            curve.plot(p, 2 * p * (1 - p), color='0.3', lw=2.5,
                       label='Gini index $G = 2\\hat p(1 - \\hat p)$')
            dots = [side for side in boxes if sizes[side]]
            if show_line.value and len(dots) == 2:
                xs = [boxes[s][0] / sizes[s] for s in dots]
                curve.plot(xs, [_gini(*boxes[s]) for s in dots], '--', color='0.45', lw=2,
                           label='line between the two dots')
            for side in dots:
                c = boxes[side][0]
                curve.scatter(c / sizes[side], _gini(*boxes[side]), s=2500 * sizes[side] / n,
                              color=BOX_COLORS[side], edgecolor='black', zorder=4,
                              label=f'{side} box ({sizes[side]} of {n} points)')
            curve.plot(p_parent, g_parent, 's', ms=14, color='white', mec='black', mew=2,
                       zorder=5, label=f'before the split (G = {g_parent:.3f})')
            curve.plot(p_parent, score, 'D', ms=14, color=PENALTY_COLOR, mec='black', zorder=5,
                       label=f'split score ({score:.3f})')
            if g_parent - score > 0.02:
                curve.annotate('', xy=(p_parent, score), xytext=(p_parent, g_parent),
                               arrowprops=dict(arrowstyle='->', lw=2.5, color=PENALTY_COLOR))
            curve.set_xlim(-0.03, 1.03)
            curve.set_ylim(-0.03, 0.8)
            curve.set_xlabel('$\\hat p$ (share of circles in the box)')
            curve.set_ylabel('Gini index')
            curve.set_title('Each box on the Gini curve')
            legend = curve.legend(loc='upper center', ncol=2, fontsize=12, labelspacing=0.8)
            for handle in legend.legend_handles:
                if hasattr(handle, 'set_sizes'):    # same-size legend dots for both boxes
                    handle.set_sizes([180])
            fig.tight_layout()
            return fig
        _show(out, draw, size_percent)

    for control in (left_c, left_t, show_line):
        control.observe(update, names='value')
    update()
    display(widgets.VBox([widgets.HBox([left_c, left_t, show_line]), out]))


# --- NB1 Demo C: grow it until it's perfect ------------------------------------

def _depth_options(full_depth):
    return [(str(d) if d < full_depth else f'{d} (full)', d) for d in range(1, full_depth + 1)]


SHOWN_LEVELS = 4    # deeper levels of the tree diagram collapse to '⋯'


def _node_depths(tree):
    depths = np.zeros(tree.node_count, int)
    for node in range(tree.node_count):     # a parent always has a smaller number
        for child in (tree.children_left[node], tree.children_right[node]):
            if child != -1:
                depths[child] = depths[node] + 1
    return depths


def _draw_class_tree(ax, model, new_level=None, levels=SHOWN_LEVELS):
    """A compact diagram of a two-class tree, drawn by hand so it stays readable.

    Internal nodes show their question; leaves are shaded by predicted class and
    show how many training points of each class they hold. If new_level is given,
    the splits on level new_level - 1 (the newest ones) are outlined in orange.
    Below `levels` levels, a subtree is collapsed to '⋯'."""
    tree = model.tree_
    depths = _node_depths(tree)
    x = {}

    def place(node):
        """In-order x positions: shown leaves get 0, 1, 2, ...; a parent sits
        midway between its children."""
        left, right = tree.children_left[node], tree.children_right[node]
        if left == -1 or depths[node] == levels:
            x[node] = len([n for n in x if n in shown_ends])
            shown_ends.add(node)
            return
        place(left)
        place(right)
        x[node] = (x[left] + x[right]) / 2

    shown_ends = set()
    place(0)
    n_ends = len(shown_ends)
    fontsize = 14 if n_ends <= 4 else 12 if n_ends <= 8 else 9.5
    for node in x:
        for child in (tree.children_left[node], tree.children_right[node]):
            if child in x:
                ax.plot([x[node], x[child]], [-depths[node], -depths[child]],
                        color='0.4', lw=1.2, zorder=1)
    for node, position in x.items():
        y = -depths[node]
        is_leaf = tree.children_left[node] == -1
        if is_leaf:
            counts = np.rint(tree.value[node, 0] * tree.n_node_samples[node]).astype(int)
            k = model.classes_[np.argmax(counts)]
            text = f'{counts[0]} ○\n{counts[1]} △'
            box = dict(boxstyle='round', fc=_light(CLASS_COLORS[k], 0.45), ec='black')
        elif node in shown_ends:
            text, box = '⋯', dict(boxstyle='round', fc='0.92', ec='0.5')
        else:
            text = f'X{"₁₂"[tree.feature[node]]} ≤ {tree.threshold[node]:.2f}'
            new = new_level is not None and depths[node] == new_level - 1
            box = dict(boxstyle='round', fc='white', ec=PENALTY_COLOR if new else 'black',
                       lw=3 if new else 1)
        ax.text(position, y, text, ha='center', va='center', fontsize=fontsize,
                bbox=box, zorder=2)
    ax.set_xlim(-0.7, n_ends - 0.3)
    ax.set_ylim(-min(depths.max(), levels) - 0.6, 0.6)
    ax.axis('off')


def _draw_class_boxes(ax, sim, model, new_level=None, show_bayes=False):
    """Every leaf as its own outlined box, shaded by predicted class, so splits
    between two boxes of the same class still show. If new_level is given, the
    newest splits (made on level new_level - 1) are drawn as thick orange lines."""
    tree = model.tree_
    lows, highs = (GRID_X1[0], GRID_X2[0]), (GRID_X1[-1], GRID_X2[-1])
    boxes = _leaf_boxes(model, lows, highs)
    for leaf, (lo, hi) in boxes.items():
        k = model.classes_[np.argmax(tree.value[leaf, 0])]
        ax.add_patch(Rectangle(lo, hi[0] - lo[0], hi[1] - lo[1], fc=_light(CLASS_COLORS[k]),
                               ec='black', lw=1, zorder=0))
    depths = _node_depths(tree)
    newest = [] if new_level is None else np.flatnonzero((depths == new_level - 1)
                                                         & (tree.children_left != -1))
    for node in newest:
        left_lo, left_hi = _subtree_box(tree, boxes, tree.children_left[node])
        right_lo, right_hi = _subtree_box(tree, boxes, tree.children_right[node])
        lo, hi = np.minimum(left_lo, right_lo), np.maximum(left_hi, right_hi)
        s = tree.threshold[node]
        if tree.feature[node] == 0:
            ax.plot([s, s], [lo[1], hi[1]], color=PENALTY_COLOR, lw=3.5, zorder=3)
        else:
            ax.plot([lo[0], hi[0]], [s, s], color=PENALTY_COLOR, lw=3.5, zorder=3)
    _plot_classes(ax, sim.X, sim.y, zorder=4)
    extra = [] if new_level is None else [(Line2D([], [], color=PENALTY_COLOR, lw=3.5),
                                           'newest splits')]
    if show_bayes:
        ax.contour(GRID_X1, GRID_X2, _bayes_grid(sim), levels=[0.5], zorder=5, **BAYES_LINE)
        extra.append((Line2D([], [], color='black', ls='--', lw=2.5), 'Bayes boundary'))
    ax.set_xlim(GRID_X1[0], GRID_X1[-1])
    ax.set_ylim(GRID_X2[0], GRID_X2[-1])
    ax.set_aspect('equal')
    ax.set_xlabel('X₁')
    ax.set_ylabel('X₂')
    _class_legend(ax, extra=extra, loc='upper left', framealpha=0.95)


def _subtree_box(tree, boxes, node):
    """The box a node covers: the smallest box holding all of its leaves' boxes."""
    if tree.children_left[node] == -1:
        return boxes[node]
    (a_lo, a_hi), (b_lo, b_hi) = (_subtree_box(tree, boxes, tree.children_left[node]),
                                  _subtree_box(tree, boxes, tree.children_right[node]))
    return np.minimum(a_lo, b_lo), np.maximum(a_hi, b_hi)


def _truncate(model, depth):
    """A fitted tree cut off at the given depth: its nodes on that level become
    leaves. Node numbers stay the full tree's, and the deeper nodes are unreachable."""
    tree = model.tree_
    cut = _node_depths(tree) >= depth
    return SimpleNamespace(classes_=model.classes_, tree_=SimpleNamespace(
        children_left=np.where(cut, -1, tree.children_left),
        children_right=np.where(cut, -1, tree.children_right),
        feature=tree.feature, threshold=tree.threshold, value=tree.value,
        n_node_samples=tree.n_node_samples, node_count=tree.node_count))


def _truncated_predict(model, depth, X):
    """What the tree cut off at the given depth predicts: the majority class of the
    deepest node on each point's path that is no deeper than depth. A child always
    has a larger node number than its parent, so that's the largest number kept."""
    tree = model.tree_
    keep = _node_depths(tree) <= depth
    numbers = (np.arange(tree.node_count) + 1) * keep
    deepest = model.decision_path(X).multiply(numbers).max(axis=1).toarray().ravel() - 1
    return model.classes_[np.argmax(tree.value[deepest.astype(int), 0], axis=1)]


def tree_depth_demo(sim, depth=1, size_percent=100):
    """Demo C: a classification tree for each max_depth, up to the full tree, drawn
    the way Demo A draws its tree.

    Left: the tree diagram (its top SHOWN_LEVELS levels). Right: one outlined box
    per leaf, shaded by predicted class. Moving the slider up one adds a level of
    splits, which are outlined in orange in both panels. "Show test error" and
    "Show Bayes boundary" start off.

    Each step is the full tree cut off at that depth, not a refit with max_depth.
    A refit can break ties between equally good splits differently and change
    splits already made (here, at depths 4, 6, and 7), which would make the tree
    seem to rearrange itself as the slider moves. The cut-off trees match the
    refits in leaves and training error, and in test error to within 0.3%.

    Returns the widget, so a later cell can display the same demo again."""
    full = DecisionTreeClassifier(random_state=0).fit(sim.X, sim.y)
    full_depth = full.get_depth()
    models = {d: _truncate(full, d) for d in range(1, full_depth + 1)}
    fits = {d: SimpleNamespace(
        leaves=len(_leaf_boxes(model, (GRID_X1[0], GRID_X2[0]), (GRID_X1[-1], GRID_X2[-1]))),
        train_error=float(np.mean(_truncated_predict(full, d, sim.X) != sim.y)),
        test_error=float(np.mean(_truncated_predict(full, d, sim.X_test) != sim.y_test)))
        for d, model in models.items()}
    bayes_error = bayes_error_rate(sim)
    depth_slider = widgets.SelectionSlider(options=_depth_options(full_depth), value=depth,
                                           description='max_depth',
                                           style={'description_width': '80px'},
                                           layout=widgets.Layout(width='460px'))
    show_test = widgets.ToggleButton(value=False, description='Show test error',
                                     layout=widgets.Layout(width='180px'))
    show_bayes = widgets.ToggleButton(value=False, description='Show Bayes boundary',
                                      layout=widgets.Layout(width='180px'))
    readout = widgets.HTML()
    out = _figure_area()

    def update(change=None):
        fit = fits[depth_slider.value]
        lines = [f'Leaves: <b>{fit.leaves}</b> &nbsp; Training error: <b>{fit.train_error:.1%}</b>']
        if show_test.value:
            lines.append(f'Test error ({len(sim.y_test):,} new points): <b>{fit.test_error:.1%}</b>')
        if show_bayes.value:
            lines.append(f'Bayes error rate (on new data): <b>{bayes_error:.1%}</b>')
        readout.value = _html('<br>'.join(lines))

        def draw():
            d = depth_slider.value
            fig, (ax_tree, ax_box) = subplots(1, 2, figsize=(19, 8.5),
                                              gridspec_kw={'width_ratios': [1.15, 1]})
            _draw_class_tree(ax_tree, models[d], d)
            shown = (f'top {SHOWN_LEVELS} levels shown; ⋯ = more splits below'
                     if d > SHOWN_LEVELS else 'left branch = condition is true')
            ax_tree.set_title(f'The tree, max_depth = {d}\n({shown})')
            _draw_class_boxes(ax_box, sim, models[d], d, show_bayes.value)
            ax_box.set_title(f'The same tree as boxes: {fit.leaves} leaves\n'
                             '(shading = predicted class)')
            fig.tight_layout()
            return fig
        _show(out, draw, size_percent)

    for w in (depth_slider, show_test, show_bayes):
        w.observe(update, names='value')
    update()
    demo = widgets.VBox([_flexibility_row(depth_slider, 'Less flexible', 'More flexible'),
                         widgets.HBox([show_test, show_bayes]), readout, out])
    display(demo)
    return demo


# --- NB1 Section 6: same size, different shape -------------------------------

def plot_tree_pair(sim, models):
    """Two classification trees side by side ({title: fitted tree}): the tree
    diagrams on top, the boxes below, and each tree's leaves, depth, training
    error, and test error in its title. For comparing trees of the same size
    that spend their leaves in different places."""
    with rc_context(BIG_FONTS):
        fig, axes = subplots(2, 2, figsize=(18, 15), gridspec_kw={'height_ratios': [1, 1.25]})
        for (title, model), (ax_tree, ax_box) in zip(models.items(), axes.T):
            _draw_class_tree(ax_tree, model, levels=model.get_depth())
            ax_tree.set_title(f'{title}\n{model.get_n_leaves()} leaves, depth {model.get_depth()}')
            _draw_class_boxes(ax_box, sim, model)
            ax_box.set_title(f'Training error {1 - model.score(sim.X, sim.y):.1%}, '
                             f'test error {1 - model.score(sim.X_test, sim.y_test):.1%}')
        fig.tight_layout()
    return fig


# --- NB1 Demo D: limits --------------------------------------------------------

LEAF_SIZES = (1, 2, 5, 10, 20, 40, 80)


def limits_demo(sim, size_percent=100):
    """Demo D: two tabs, one per limit setting (max_depth, min_samples_leaf).

    Each tab has its own slider, the tree's boxes with the Bayes boundary, the
    tree diagram beside them (its top SHOWN_LEVELS levels), and readouts for
    leaves, training error, and test error."""
    full_depth = DecisionTreeClassifier(random_state=0).fit(sim.X, sim.y).get_depth()
    settings = {
        'max_depth': dict(options=_depth_options(full_depth), value=3,
                          ends=('Less flexible', 'More flexible')),
        'min_samples_leaf': dict(options=[(str(m), m) for m in LEAF_SIZES], value=1,
                                 ends=('More flexible', 'Less flexible'))}
    bayes_error = bayes_error_rate(sim)
    panels = []
    for name, setting in settings.items():
        models = {value: DecisionTreeClassifier(**{name: value}, random_state=0)
                  .fit(sim.X, sim.y) for _, value in setting['options']}
        fits = {value: _fit_summary(model, sim) for value, model in models.items()}
        slider = widgets.SelectionSlider(options=setting['options'], value=setting['value'],
                                         description=name,
                                         style={'description_width': '130px'},
                                         layout=widgets.Layout(width='480px'))
        readout = widgets.HTML()
        out = _figure_area()

        def update(change=None, name=name, models=models, fits=fits, slider=slider,
                   readout=readout, out=out):
            fit = fits[slider.value]
            readout.value = _html(
                f'Leaves: <b>{fit.leaves}</b> &nbsp; Training error: <b>{fit.train_error:.1%}</b> '
                f'&nbsp; Test error: <b>{fit.test_error:.1%}</b> &nbsp; '
                f'(Bayes error rate: {bayes_error:.1%})')

            def draw():
                fig, (ax, ax_tree) = subplots(1, 2, figsize=(19, 8.5),
                                              gridspec_kw={'width_ratios': [1, 1.15]})
                _draw_regions(ax, sim, fit.predicted, True,
                              f'Tree with {name} = {slider.value}: {fit.leaves} leaves')
                _draw_class_tree(ax_tree, models[slider.value])
                shown = (f'top {SHOWN_LEVELS} levels shown; ⋯ = more splits below'
                         if fit.depth > SHOWN_LEVELS else 'left branch = condition is true')
                ax_tree.set_title(f'The same tree, depth {fit.depth}\n({shown})')
                fig.tight_layout()
                return fig
            _show(out, draw, size_percent)

        slider.observe(update, names='value')
        update()
        panels.append(widgets.VBox([_flexibility_row(slider, *setting['ends']), readout, out]))
    tabs = widgets.Tab(children=panels)
    for i, name in enumerate(settings):
        tabs.set_title(i, name)
    display(tabs)


def plot_depth_error_curves(errors, bayes_error):
    """Demo D: training and test error (columns of `errors`, indexed by max_depth)
    against max_depth, with the Bayes error rate as a dashed line. The deepest
    value is labeled as the full tree. More flexible is on the right."""
    depths = list(errors.index)
    with rc_context(BIG_FONTS):
        fig, ax = subplots(figsize=(11, 6))
        for name in errors.columns:
            ax.plot(depths, errors[name], 'o-', lw=2.5, ms=7, label=name,
                    color=CURVE_COLORS.get(name))
        ax.axhline(bayes_error, color='black', ls='--', lw=2, label='Bayes error rate (the floor)')
        ax.set_xticks(depths, [str(d) for d in depths[:-1]] + [f'{depths[-1]}\n(full)'])
        ax.set_xlabel('max_depth')
        _flexibility_ends(ax, '← Less flexible', 'More flexible →', height=-0.17)
        ax.set_ylabel('error rate')
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_ylim(0, 0.26)
        ax.set_title('Training and test error for every max_depth')
        ax.legend(loc='upper right')
        fig.tight_layout()
    return fig


# --- NB1 Demo E: cost-complexity pruning ----------------------------------------

def _leaf_impurity(model):
    """scikit-learn's training-error measure for pruning: each leaf's Gini index,
    weighted by its share of the training points, added up over the leaves."""
    tree = model.tree_
    leaves = tree.children_left == -1
    weights = tree.weighted_n_node_samples[leaves] / tree.weighted_n_node_samples[0]
    return float(np.sum(weights * tree.impurity[leaves]))


def _collapse_alphas(model):
    """{node: alpha}: the alpha at which each node of a fitted tree becomes a leaf
    under weakest-link pruning.

    Repeatedly finds the internal node(s) with the smallest
    g(t) = (R(t) - R(T_t)) / (|T_t| - 1), where R is the Gini index weighted by
    the node's share of the training points (scikit-learn's convention), collapses
    them, and recomputes, since g changes as branches below a node are pruned.
    The successive smallest values are cost_complexity_pruning_path's alphas.
    Nodes pruned away inside a larger branch never become leaves themselves, so
    they aren't in the result."""
    tree = model.tree_
    R = tree.weighted_n_node_samples / tree.weighted_n_node_samples[0] * tree.impurity
    left, right = tree.children_left.copy(), tree.children_right.copy()

    def leaves(node):
        return [node] if left[node] == -1 else leaves(left[node]) + leaves(right[node])

    def internal(node):
        return [] if left[node] == -1 else [node] + internal(left[node]) + internal(right[node])

    collapse = {}
    while left[0] != -1:
        g = {}
        for node in internal(0):
            below = leaves(node)
            g[node] = (R[node] - R[below].sum()) / (len(below) - 1)
        weakest = min(g.values())
        for node in internal(0):     # preorder: a tied ancestor collapses before its descendants
            if left[node] != -1 and node in internal(0) and np.isclose(g[node], weakest):
                collapse[node] = weakest
                left[node] = right[node] = -1
    return collapse


def _draw_pruning_tree(ax, model, collapse, alpha):
    """The full tree, always in the same layout, pruned at alpha.

    A node is pruned away (ghosted) if one of its ancestors has collapsed, that
    is, has a collapse alpha at or below the current alpha. A collapsed node is
    drawn as a leaf, shaded by its majority class like the boundary regions.
    Nodes that collapse on their own are labeled with their collapse alpha, and
    the node that collapses at exactly the current alpha (the most recent
    collapse) is outlined in orange."""
    tree = model.tree_
    left, right = tree.children_left, tree.children_right
    depths = _node_depths(tree)
    x, parent = {}, {}

    def place(node):
        if left[node] == -1:
            x[node] = sum(left[n] == -1 for n in x)
            return
        for child in (left[node], right[node]):
            parent[child] = node
            place(child)
        x[node] = (x[left[node]] + x[right[node]]) / 2

    place(0)

    def collapsed(node):
        # isclose: alpha is one of scikit-learn's breakpoints, which can differ
        # from the collapse alpha computed here in the last few digits.
        return node in collapse and (collapse[node] < alpha or np.isclose(collapse[node], alpha))

    def ghosted(node):
        while node in parent:
            node = parent[node]
            if collapsed(node):
                return True
        return False

    def majority_color(node):
        return _light(CLASS_COLORS[model.classes_[np.argmax(tree.value[node, 0])]], 0.45)

    for node in x:
        for child in (left[node], right[node]):
            if child != -1:
                faded = ghosted(child)
                ax.plot([x[node], x[child]], [-depths[node], -depths[child]], zorder=1,
                        color='0.85' if faded else '0.4', lw=1 if faded else 1.3)
    for node, position in x.items():
        y = -depths[node]
        faded = ghosted(node)
        acts_as_leaf = left[node] == -1 or collapsed(node)
        newest = node in collapse and np.isclose(collapse[node], alpha)
        if faded:
            face, edge, ink = '0.93', '0.8', '0.7'
        else:
            face, edge, ink = (majority_color(node) if acts_as_leaf else 'white'), 'black', 'black'
        style = dict(boxstyle='round', fc=face, ec=PENALTY_COLOR if newest else edge,
                     lw=3 if newest else 1)
        if node in collapse:
            ax.text(position, y, f'{collapse[node]:.4f}', ha='center', va='center',
                    fontsize=10.5, color=ink, bbox=style, zorder=3)
        else:
            ax.text(position, y, ' ' if left[node] == -1 else '', ha='center', va='center',
                    fontsize=7, bbox=style, zorder=2)
    ax.set_xlim(-0.8, max(x.values()) + 0.8)
    ax.set_ylim(-depths.max() - 0.6, 0.6)
    ax.axis('off')


def pruning_demo(sim, alphas, size_percent=100):
    """Demo E: the pruned tree for each alpha on scikit-learn's pruning path.

    alphas are the values from cost_complexity_pruning_path. Left: the tree's
    boxes, or (with the toggle) the full tree with the pruned branches ghosted.
    Right: a stacked bar of the training-error measure plus the penalty, alpha
    times the number of leaves. "Show test error" starts off."""
    full = DecisionTreeClassifier(random_state=0).fit(sim.X, sim.y)
    collapse = _collapse_alphas(full)
    fits = []
    for alpha in alphas:
        model = DecisionTreeClassifier(ccp_alpha=alpha, random_state=0).fit(sim.X, sim.y)
        fit = _fit_summary(model, sim)
        fit.impurity = _leaf_impurity(model)
        fit.penalty = alpha * fit.leaves
        fits.append(fit)
    top = max(f.impurity + f.penalty for f in fits) * 1.08
    alpha_slider = widgets.SelectionSlider(
        options=[(f'{a:.4f}', i) for i, a in enumerate(alphas)], value=0, description='α',
        style={'description_width': '30px'}, layout=widgets.Layout(width='520px'))
    show_test = widgets.ToggleButton(value=False, description='Show test error',
                                     layout=widgets.Layout(width='180px'))
    view = widgets.ToggleButtons(options=['Decision boundary', 'Tree'], value='Decision boundary')
    readout = widgets.HTML()
    out = _figure_area()

    def update(change=None):
        i = alpha_slider.value
        fit = fits[i]
        lines = [f'Leaves |T| = <b>{fit.leaves}</b> &nbsp; Training error: '
                 f'<b>{fit.train_error:.1%}</b>']
        if show_test.value:
            lines.append(f'Test error ({len(sim.y_test):,} new points): <b>{fit.test_error:.1%}</b>')
        readout.value = _html('<br>'.join(lines))

        def draw():
            fig, (ax, bar) = subplots(1, 2, figsize=(15, 8),
                                      gridspec_kw={'width_ratios': [3, 1]})
            if view.value == 'Tree':
                _draw_pruning_tree(ax, full, collapse, alphas[i])
                ax.set_title(f'Full tree ({full.get_n_leaves()} leaves), pruned at α = '
                             f'{alphas[i]:.4f}: {fit.leaves} leaves\n(number = the α at which '
                             'that node becomes a leaf; grey = pruned away)')
            else:
                _draw_regions(ax, sim, fit.predicted, False,
                              f'Pruned tree, α = {alphas[i]:.4f}: {fit.leaves} leaves')
            bar.bar([0], [fit.impurity], width=0.6, color=CLASS_COLORS[1],
                    label='training error measure\n(leaf impurity)')
            bar.bar([0], [fit.penalty], width=0.6, bottom=[fit.impurity], color=PENALTY_COLOR,
                    label=f'penalty: α × |T|\n= {alphas[i]:.4f} × {fit.leaves}')
            total = fit.impurity + fit.penalty
            bar.text(0, total + top * 0.015, f'total\n{total:.3f}', ha='center', va='bottom',
                     fontsize=14)
            bar.set_xlim(-0.6, 0.6)
            bar.set_ylim(0, top * 1.25)
            bar.set_xticks([])
            bar.set_title('Total cost')
            bar.legend(loc='upper center', fontsize=12, framealpha=0.95)
            fig.tight_layout()
            return fig
        _show(out, draw, size_percent)

    for w in (alpha_slider, show_test, view):
        w.observe(update, names='value')
    update()
    display(widgets.VBox([_flexibility_row(alpha_slider), widgets.HBox([view, show_test]),
                          readout, out]))


# --- NB2 Demo F: shrinking a wiggly fit ------------------------------------------

WIGGLE_DEGREE = 12
WIGGLE_LAMBDAS = np.logspace(-6, 3, 19)


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
                              StandardScaler(), Ridge(alpha=lam))
        model.fit(data.x[:, None], data.y)
        fits.append(SimpleNamespace(
            curve=model.predict(curve_x[:, None]), coef=model[-1].coef_,
            train_mse=np.mean((model.predict(data.x[:, None]) - data.y)**2),
            new_mse=np.mean((model.predict(data.x_new[:, None]) - data.y_new)**2)))
    biggest = max(np.abs(f.coef).max() for f in fits)
    lam_slider = widgets.SelectionSlider(
        options=[(f'{lam:.0e}' if lam < 0.01 else f'{lam:g}', i)
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


# --- NB2 Demo G: ridge vs. lasso coefficient paths --------------------------------

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


# --- NB2 Demo H (optional): why lasso hits zero -----------------------------------

CORNER_CENTER = np.array([2.0, 0.4])          # the least squares coefficients
CORNER_SHAPE = np.array([[1.0, 0.3], [0.3, 1.4]])
CORNER_BUDGET = 1.0


def _corner_rss(b1, b2):
    d1, d2 = b1 - CORNER_CENTER[0], b2 - CORNER_CENTER[1]
    a = CORNER_SHAPE
    return a[0, 0] * d1**2 + 2 * a[0, 1] * d1 * d2 + a[1, 1] * d2**2


def plot_lasso_corner():
    """Demo H: RSS contours for two coefficients, with ridge's circle and lasso's diamond.

    Each panel marks the best point inside the region (the lowest RSS contour that
    touches it). For the diamond it lands on a corner, where one coefficient is 0."""
    angles = np.linspace(0, 2 * np.pi, 20_001)
    circle = CORNER_BUDGET * np.column_stack([np.cos(angles), np.sin(angles)])
    diamond = circle / np.abs(circle).sum(axis=1, keepdims=True) * CORNER_BUDGET
    g = np.linspace(-1.6, 3.2, 300)
    b1, b2 = np.meshgrid(g, g)
    rss = _corner_rss(b1, b2)
    with rc_context(BIG_FONTS):
        fig, axes = subplots(1, 2, figsize=(15, 7.2))
        for ax, (name, edge, fill) in zip(axes, (('Ridge', circle, 'circle'),
                                                ('Lasso', diamond, 'diamond'))):
            values = _corner_rss(edge[:, 0], edge[:, 1])
            best = edge[np.argmin(values)]
            levels = values.min() * np.array([0.25, 0.5, 1, 1.6, 2.4, 3.4])
            ax.contour(b1, b2, rss, levels=np.sort(levels), colors='0.45', linewidths=1.3)
            ax.contour(b1, b2, rss, levels=[values.min()], colors=PENALTY_COLOR, linewidths=2.5)
            ax.add_patch(Polygon(edge[::50], closed=True, fc=_light(CLASS_COLORS[1], 0.4),
                                 ec=CLASS_COLORS[1], lw=2.5))
            ax.plot(*CORNER_CENTER, 'o', ms=11, color='black')
            ax.annotate('least squares', CORNER_CENTER, xytext=(8, 8), textcoords='offset points',
                        fontsize=14)
            ax.plot(*best, '*', ms=24, color=PENALTY_COLOR, mec='black', zorder=5)
            ax.annotate(f'best inside the {fill}\nβ₁ = {best[0]:.2f}, β₂ = {abs(best[1]):.2f}',
                        best, xytext=(14, -40), textcoords='offset points', fontsize=14,
                        bbox=dict(fc='white', ec='none', alpha=0.9))
            ax.axhline(0, color='black', lw=0.8)
            ax.axvline(0, color='black', lw=0.8)
            ax.set_xlim(g[0], g[-1])
            ax.set_ylim(g[0], g[-1])
            ax.set_aspect('equal')
            ax.set_xlabel('β₁')
            ax.set_ylabel('β₂')
            ax.set_title(f'{name}: coefficients must stay inside the {fill}')
        axes[0].legend(handles=[Line2D([], [], color='0.45', lw=1.3, label='RSS contours'),
                                Line2D([], [], color=PENALTY_COLOR, lw=2.5,
                                       label='lowest contour that touches')],
                       loc='upper left', framealpha=0.95)
        fig.tight_layout()
    return fig


# --- NB2 Demo I: choosing lambda by cross-validation -------------------------------

def plot_cv_lambda(curves, ols_mse, mean_mse):
    """Demo I: 10-fold CV MSE against lambda, one panel per method.

    curves maps a method name to (lambdas, cv_mse). Dashed: least squares (no
    penalty). Dotted: predicting the average salary for everyone. A star marks
    each curve's lowest point."""
    with rc_context(BIG_FONTS):
        fig, axes = subplots(1, len(curves), figsize=(16, 6.2), sharey=True)
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


# --- NB2 Demo J: ensembles ----------------------------------------------------------

def plot_ensemble_boundaries(sim, models):
    """Demo J: a 2 x 2 grid, one panel per fitted classifier in `models` (name -> model),
    with the Bayes boundary, titled with training and test error."""
    with rc_context(BIG_FONTS):
        fig, axes = subplots(2, 2, figsize=(15, 13.5), sharex=True, sharey=True)
        for i, (ax, (name, model)) in enumerate(zip(axes.flat, models.items())):
            fit = _fit_summary(model, sim)
            _draw_regions(ax, sim, fit.predicted, True,
                          f'{name}\ntraining error {fit.train_error:.1%}, '
                          f'test error {fit.test_error:.1%}', ms=6, legend=i == 0)
            ax.set_xlabel('X₁' if i >= 2 else '')
            ax.set_ylabel('X₂' if i % 2 == 0 else '')
        fig.tight_layout()
    return fig


BAG_SIZES = (1, 5, 25, 100, 300)


def bagging_demo(sim, size_percent=100):
    """Demo J control: the boundary of a bag of B trees, for B on the slider.

    One BaggingClassifier with 300 trees is fit once; a bag of B trees is its first
    B trees, which is itself a bag of B bootstrap trees. Predictions average the
    trees' class shares, as BaggingClassifier does."""
    bag = BaggingClassifier(DecisionTreeClassifier(), n_estimators=max(BAG_SIZES),
                            random_state=0).fit(sim.X, sim.y)
    grid = _grid_points()

    def running_share(points):
        shares = np.array([tree.predict_proba(points)[:, 0] for tree in bag.estimators_])
        return np.cumsum(shares, axis=0) / np.arange(1, len(shares) + 1)[:, None]

    grid_share, test_share = running_share(grid), running_share(sim.X_test)
    fits = {}
    for B in BAG_SIZES:
        fits[B] = SimpleNamespace(
            predicted=np.where(grid_share[B - 1] >= 0.5, 1, 2).reshape(GRID_SHAPE),
            test_error=float(np.mean(np.where(test_share[B - 1] >= 0.5, 1, 2) != sim.y_test)))
    b_slider = widgets.SelectionSlider(options=BAG_SIZES, value=1, description='B (trees)',
                                       style={'description_width': '80px'},
                                       layout=widgets.Layout(width='420px'))
    readout = widgets.HTML()
    out = _figure_area()

    def update(change=None):
        fit = fits[b_slider.value]
        readout.value = _html(f'Bagging with B = {b_slider.value}: test error '
                              f'<b>{fit.test_error:.1%}</b>')

        def draw():
            fig, ax = subplots(figsize=(10, 8.5))
            trees = 'tree' if b_slider.value == 1 else 'trees'
            _draw_regions(ax, sim, fit.predicted, True,
                          f'Bagging: {b_slider.value} bootstrap {trees} vote')
            fig.tight_layout()
            return fig
        _show(out, draw, size_percent)

    b_slider.observe(update, names='value')
    update()
    display(widgets.VBox([b_slider, readout, out]))


# --- NB3: preparing real data ---------------------------------------------------------

def plot_capital_gain(values):
    """NB3 §3: a histogram of the nonzero capital-gain values, with 99,999 marked."""
    nonzero = values[values > 0]
    with rc_context(BIG_FONTS):
        fig, ax = subplots(figsize=(12, 5.5))
        ax.hist(nonzero, bins=np.arange(0, 102_000, 2_000), color=CLASS_COLORS[1], ec='white')
        ax.set_yscale('log')
        n_code = int(np.sum(values == 99_999))
        ax.annotate(f'{n_code} people at exactly 99,999', (99_999, n_code),
                    xytext=(-260, 40), textcoords='offset points', fontsize=14,
                    arrowprops=dict(arrowstyle='->', lw=1.5))
        ax.set_xlabel('capital-gain (dollars)')
        ax.set_ylabel('number of people (log scale)')
        ax.set_title(f'capital-gain for the {len(nonzero):,} people with a nonzero value')
        fig.tight_layout()
    return fig


def plot_threshold_tradeoff(y_true, probability, marked=(0.5,)):
    """NB3 §7: recall and precision of the positive class for every decision threshold,
    from predicted probabilities. Vertical lines mark the thresholds in `marked`."""
    thresholds = np.linspace(0.05, 0.95, 91)
    recall = [recall_score(y_true, probability >= t) for t in thresholds]
    precision = [precision_score(y_true, probability >= t, zero_division=0) for t in thresholds]
    with rc_context(BIG_FONTS):
        fig, ax = subplots(figsize=(11, 6))
        ax.plot(thresholds, recall, lw=3, color=CURVE_COLORS['training error'], label='recall')
        ax.plot(thresholds, precision, lw=3, color=CURVE_COLORS['test error'], label='precision')
        for t in marked:
            ax.axvline(t, color='black', ls='--', lw=1.5)
            ax.text(t + 0.01, 0.03, f'threshold {t:g}', ha='left', va='bottom', fontsize=13)
        ax.set_xlabel('decision threshold: predict ">50K" when the probability is at least this')
        ax.set_ylabel('share')
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_ylim(0, 1.08)
        ax.legend(loc='center left')
        ax.set_title('Moving the threshold trades recall for precision')
        fig.tight_layout()
    return fig


def plot_importances(importances, names, top=10):
    """NB3 §8: the `top` largest feature importances as a horizontal bar chart."""
    order = np.argsort(importances)[::-1][:top][::-1]
    with rc_context(BIG_FONTS):
        fig, ax = subplots(figsize=(11, 0.5 * top + 1.5))
        ax.barh([names[i] for i in order], importances[order], color=CLASS_COLORS[1])
        ax.set_xlabel('importance (share of the total drop in Gini impurity)')
        ax.set_title(f'The tree\'s {top} most important features')
        fig.tight_layout()
    return fig
