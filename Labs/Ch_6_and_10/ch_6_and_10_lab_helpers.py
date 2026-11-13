"""Plotting and checking helpers for Lab 6 (ch_6_and_10_lab.ipynb).

You don't need to read this file. It holds the matplotlib and ipywidgets details so the
notebook can stay focused on the math. Every plot the lab asks you to *make* is written
out in the notebook itself.

On Colab: upload this file next to the notebook (Files panel, then Upload).
"""
from io import BytesIO

import ipywidgets as widgets
import numpy as np
from IPython.display import display
from matplotlib import get_backend, rc_context
from matplotlib.colors import to_rgb
from matplotlib.pyplot import close, subplots, switch_backend, figure

# Load the backend now, outside any rc_context (see the demos' helpers for why).
switch_backend(get_backend())

BIG_FONTS = {'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 15,
             'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 13}
HTML_STYLE = 'font-size: 18px; line-height: 1.5;'

# Okabe-Ito colors, which stay distinguishable under common color blindness.
W_COLOR = '#D55E00'       # the weight vector w (vermillion)
X_COLOR = '#0072B2'       # a data vector x (blue)
SUM_COLOR = '#009E73'     # a sum, or a positive shadow (green)
NEG_COLOR = '#CC79A7'     # a negative shadow (reddish purple)
POS_SIDE = '#56B4E9'      # shading where the score is positive (blue: class 1)
NEG_SIDE = '#E69F00'      # shading where the score is negative (orange: class 0)
CLASS1 = '#0072B2'        # class 1 points: blue circles
CLASS0 = '#E69F00'        # class 0 points: orange triangles
LA_LIM = 3.6              # the plane is drawn from -LA_LIM to LA_LIM
DOT_W = 2 * np.array([np.cos(np.radians(30)), np.sin(np.radians(30))])
LINE_POINTS = {'A': (2.0, 1.0), 'B': (-1.5, 0.5), 'C': (0.5, -2.0)}
OKABE_PTS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9']   # one color per point


# --- checking ---------------------------------------------------------------------------

def check(what, yours, numpy_says, tol=1e-9):
    """Compare a hand-computed answer with NumPy's, and say whether they agree."""
    a = np.asarray(yours, dtype=float)
    b = np.asarray(numpy_says, dtype=float)
    if a.shape == b.shape and np.allclose(a, b, atol=tol):
        print(f'✓ {what}: your answer {np.round(a, 4).tolist()} matches NumPy.')
    else:
        print(f'✗ {what}: your answer is {np.round(a, 4).tolist()}, but NumPy says '
              f'{np.round(b, 4).tolist()}. Find which one is wrong.')


# --- shared drawing pieces --------------------------------------------------------------

def _light(color, amount=0.3):
    """The color mixed with white: amount = 1 is the color itself, 0 is white."""
    return tuple(amount * c + (1 - amount) for c in to_rgb(color))


def _html(text):
    return f'<div style="{HTML_STYLE}">{text}</div>'


def _figure_area():
    return widgets.Image(format='png', layout=widgets.Layout(max_width='100%'))


def _show(out, draw, size_percent=100):
    """Redraw a matplotlib figure into an Image widget (avoids doubled output in VS Code)."""
    with rc_context(BIG_FONTS):
        fig = draw()
    buffer = BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    close(fig)
    png = buffer.getvalue()
    if size_percent != 100:
        width = int.from_bytes(png[16:20], 'big')
        out.layout.width = f'{round(width * size_percent / 100)}px'
    out.value = png


def _arrow(ax, start, end, color, label=None, lw=3.5, ls='-', offset=(6, 6), fontsize=17):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, ls=ls,
                                mutation_scale=24, shrinkA=0, shrinkB=0))
    if label:
        ax.annotate(label, end, xytext=offset, textcoords='offset points', color=color,
                    fontsize=fontsize, fontweight='bold')


def _plane(ax, xlim=(-LA_LIM, LA_LIM), ylim=(-LA_LIM, LA_LIM), grid=True):
    ax.axhline(0, color='black', lw=0.8)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    if grid:
        ax.grid(True, color='0.9')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')


def _line_segment(w, c, lim):
    """Two far-apart points on the line w . x = c, for drawing."""
    w = np.asarray(w, dtype=float)
    foot = c * w / (w @ w)
    along = np.array([-w[1], w[0]]) / np.linalg.norm(w)
    return foot - 3 * lim * along, foot + 3 * lim * along


def _angle_options(step=15):
    return [(f'{a}°', a) for a in range(0, 360, step)]


# --- Part 1 -----------------------------------------------------------------------------

def plot_vector_arithmetic():
    """Adding (tip to tail) and scaling."""
    w, x = np.array([2.0, 1.0]), np.array([-1.0, 3.0])
    u = np.array([1.5, 1.0])
    with rc_context(BIG_FONTS):
        fig, (left, right) = subplots(1, 2, figsize=(13, 6.4))
        _arrow(left, (0, 0), w, W_COLOR, 'w = (2, 1)', offset=(-10, -28))
        _arrow(left, (0, 0), x, X_COLOR, 'x = (−1, 3)', offset=(-40, 8))
        _arrow(left, w, w + x, X_COLOR, lw=2, ls='--')
        left.annotate('x', (w + w + x) / 2, xytext=(10, 0), textcoords='offset points',
                      color=X_COLOR, fontsize=17, fontweight='bold')
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


def plot_vector_subtraction():
    """Negating; subtracting as adding the negative; and q - p as the vector from p to q."""
    p, q = np.array([1.0, 1.0]), np.array([3.0, 2.0])
    with rc_context(BIG_FONTS):
        fig, (left, mid, right) = subplots(1, 3, figsize=(19, 6.2))
        _arrow(left, (0, 0), p, X_COLOR, 'p = (1, 1)', offset=(8, -6))
        _arrow(left, (0, 0), -p, NEG_COLOR, '−p = (−1, −1)', offset=(-20, -28))
        left.set_title('Negating: −p = (−1)p\n(same length, opposite direction)')
        _arrow(mid, (0, 0), q, '0.45', 'q = (3, 2)', offset=(-20, 10))
        _arrow(mid, q, q - p, NEG_COLOR, lw=2, ls='--')
        mid.annotate('−p', (q + q - p) / 2, xytext=(10, -6), textcoords='offset points',
                     color=NEG_COLOR, fontsize=17, fontweight='bold')
        _arrow(mid, (0, 0), q - p, SUM_COLOR, 'q − p = (2, 1)', offset=(-10, -30))
        mid.set_title('Subtracting: q − p = q + (−p)\n(dashed: −p, moved to start at the tip of q)')
        _arrow(right, (0, 0), p, X_COLOR, 'p = (1, 1)', offset=(-100, 4))
        _arrow(right, (0, 0), q, '0.45', 'q = (3, 2)', offset=(-20, 10))
        _arrow(right, p, q, SUM_COLOR, lw=2, ls='--')
        right.annotate('q − p', (p + q) / 2, xytext=(-52, 12), textcoords='offset points',
                       color=SUM_COLOR, fontsize=17, fontweight='bold')
        right.set_title('q − p is the vector from p to q\n(dashed: q − p, moved to start at the tip of p)')
        for ax in (left, mid, right):          # same limits, so the same scale in every panel
            _plane(ax, xlim=(-1.6, 3.7), ylim=(-1.7, 2.7))
        fig.tight_layout()
    return fig


def plot_three_points(A, B, C, AB=None, AC=None):
    """Points A, B, C, and (once computed) the vectors from A to B and from A to C."""
    A, B, C = (np.asarray(v, dtype=float) for v in (A, B, C))
    with rc_context(BIG_FONTS):
        fig, ax = subplots(figsize=(8, 5.5))
        # AB is drawn first and thick; AC is drawn thinner, on top of it, so AC is visible
        # all the way from A to C (it covers AB's whole length, then continues).
        if AB is not None:
            _arrow(ax, A, A + np.asarray(AB), W_COLOR, 'AB', lw=9, offset=(-14, 14))
        if AC is not None:
            _arrow(ax, A, A + np.asarray(AC), SUM_COLOR, 'AC', lw=2.5, offset=(-30, 8))
        for name, v in (('A', A), ('B', B), ('C', C)):
            ax.plot(*v, 'o', ms=11, color='black', zorder=5)
            ax.annotate(f'{name} = ({v[0]:g}, {v[1]:g})', v, xytext=(10, -18),
                        textcoords='offset points', fontsize=15)
        _plane(ax, xlim=(-1, 7.5), ylim=(-0.5, 5.5))
        ax.set_title('Three points on one line')
        fig.tight_layout()
    return fig


# --- Part 2 -----------------------------------------------------------------------------

def dot_product_demo(size_percent=100):
    """w is fixed; sliders set x's direction and length. Left: w, x, the angle, and the
    shadow of x on w's line (signed length w.x / |w|). Right: w.x itself as one bar on a
    fixed scale, with dashed lines at +-|w||x|, its largest and smallest possible values
    for the current length of x. The readout computes w.x both ways."""
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
        kind = ('positive (under 90°)' if dot > 0 else
                'zero (perpendicular)' if dot == 0 else 'negative (over 90°)')
        readout.value = _html(
            f'w = ({w[0]:.2f}, {w[1]:.2f}), &nbsp; x = ({x[0]:.2f}, {x[1]:.2f}), &nbsp; '
            f'angle between them: <b>{theta}°</b><br>'
            f'Algebraic definition: w₁x₁ + w₂x₂ = ({w[0]:.2f})({x[0]:.2f}) + ({w[1]:.2f})({x[1]:.2f}) '
            f'= <b>{dot:.2f}</b><br>'
            f'Geometric formula: ‖w‖ ‖x‖ cos θ = ({w_len:.0f})({L:g}) cos {theta}° = <b>{dot:.2f}</b>'
            f' &nbsp; Shadow: w·x / ‖w‖ = <b>{shadow:.2f}</b>')

        def draw():
            from matplotlib.patches import Arc
            fig, (ax, bar) = subplots(1, 2, figsize=(12.4, 8.6),
                                      gridspec_kw={'width_ratios': [4, 1.35]})
            t = np.array([-2 * LA_LIM, 2 * LA_LIM])
            ax.plot(t * unit[0], t * unit[1], color='0.6', ls='--', lw=1.2)
            foot = shadow * unit
            ax.plot([x[0], foot[0]], [x[1], foot[1]], color='0.3', ls=':', lw=2)
            if dot != 0:
                side = 0.14 * np.array([unit[1], -unit[0]])
                ax.plot([side[0], foot[0] + side[0]], [side[1], foot[1] + side[1]],
                        color=SUM_COLOR if dot > 0 else NEG_COLOR, lw=9,
                        solid_capstyle='butt', zorder=2)
            turn = (a - 30) % 360
            low, high = (30, 30 + turn) if turn <= 180 else (a, a + 360 - turn)
            ax.add_patch(Arc((0, 0), 1.3, 1.3, theta1=low, theta2=high, color='0.25', lw=1.8))
            mid = np.radians((low + high) / 2)
            ax.text(0.95 * np.cos(mid), 0.95 * np.sin(mid), 'θ', fontsize=17, ha='center',
                    va='center')
            _arrow(ax, (0, 0), w, W_COLOR, 'w')
            _arrow(ax, (0, 0), x, X_COLOR, 'x')
            _plane(ax)
            ax.set_title(f'Angle between w and x: {theta}°\n'
                         'Thick segment: the shadow of x on w’s line\n'
                         '(green ahead of the origin, pink behind)', fontsize=15)

            # Right panel: the dot product itself, as one bar on a fixed scale.
            most = w_len * L                    # |w·x| can't exceed |w| |x| (x parallel to w)
            top = w_len * 3                     # the largest |w| |x| any slider setting allows
            # Neutral gray, so the bar isn't mistaken for the shadow (green/pink, left panel).
            # Its sign is shown by which way it extends from 0.
            bar.bar([0], [dot], width=0.6, color='0.55', edgecolor='black', lw=1.2, zorder=3)
            bar.axhline(0, color='black', lw=1.2)
            for level, words in ((most, 'largest: x points\nthe same way as w'),
                                 (-most, 'smallest: x points\nthe opposite way')):
                bar.axhline(level, color='0.35', ls='--', lw=1.4)
                bar.text(0.36, level, words, fontsize=11, ha='right',
                         va='bottom' if level > 0 else 'top', color='0.3')
            bar.text(0, dot + (0.25 if dot >= 0 else -0.25), f'{dot:.2f}', ha='center',
                     va='bottom' if dot >= 0 else 'top', fontsize=16, fontweight='bold')
            bar.set_xlim(-0.5, 0.5)
            bar.set_ylim(-top - 0.8, top + 0.8)
            bar.set_xticks([])
            bar.set_ylabel('w · x')
            bar.grid(True, axis='y', color='0.9')
            bar.set_title(f'w · x = {dot:.2f}\n{kind}', fontsize=15)
            fig.tight_layout()
            return fig
        _show(out, draw, size_percent)

    for s in (angle, length):
        s.observe(update, names='value')
    update()
    display(widgets.VBox([angle, length, readout, out]))


# --- Parts 3 and 4 ----------------------------------------------------------------------

def plot_points_and_w(points, w, title='Points with w · x = 0'):
    """Scatter the given points, and draw w as an arrow from the origin."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    with rc_context(BIG_FONTS):
        fig, ax = subplots(figsize=(7, 7))
        _arrow(ax, (0, 0), w, W_COLOR, f'w = ({w[0]:g}, {w[1]:g})', offset=(4, 6))
        ax.plot(points[:, 0], points[:, 1], 'o', ms=11, color='black', zorder=5)
        for v in points:
            ax.annotate(f'({v[0]:g}, {v[1]:g})', v, xytext=(10, -16),
                        textcoords='offset points', fontsize=14)
        _plane(ax)
        ax.set_title(title)
        fig.tight_layout()
    return fig


def plot_perpendicular_set():
    """Left: w = (2, 1), several solutions of w . x = 0 drawn as vectors (all on one line,
    perpendicular to w), and two faded non-solutions labeled with their dot products.
    Right: in 3D, the solutions of w . x = 0 fill a plane perpendicular to w."""
    from matplotlib.patches import Polygon
    w = np.array([2.0, 1.0])
    solutions = [(1, -2), (-1, 2), (1.5, -3), (-0.5, 1)]
    others = [((1, 1), 'w·x = 3'), ((-1, 0), 'w·x = −2')]
    with rc_context(BIG_FONTS):
        fig = figure(figsize=(17, 7.6))
        ax = fig.add_subplot(1, 2, 1)
        t = np.array([-2 * LA_LIM, 2 * LA_LIM])
        ax.plot(t * 1, t * -2, color=X_COLOR, lw=8, alpha=0.15, solid_capstyle='butt')
        ax.annotate('the line w·x = 0', (-1.45, 2.9), fontsize=14, color=X_COLOR)
        for v, label in others:
            _arrow(ax, (0, 0), v, '0.72', label, lw=2.5, fontsize=13,
                   offset=(6, 4) if v[0] > 0 else (-70, 8))
        for v in solutions:
            _arrow(ax, (0, 0), v, X_COLOR, lw=3)
            ax.annotate(f'({v[0]:g}, {v[1]:g})', v, xytext=(8, -4), textcoords='offset points',
                        fontsize=13, color=X_COLOR)
        # right-angle marker between w and the line
        u = w / np.linalg.norm(w) * 0.35
        d = np.array([1.0, -2.0]) / np.sqrt(5) * 0.35
        ax.add_patch(Polygon([(0, 0), u, u + d, d], closed=True, fill=False, ec='black', lw=1.5))
        _arrow(ax, (0, 0), w, W_COLOR, 'w = (2, 1)', offset=(-40, -30))
        _plane(ax)
        ax.set_title('Solutions of w · x = 0 (blue), drawn as vectors\n'
                     'all lie on one line, perpendicular to w')

        ax3 = fig.add_subplot(1, 2, 2, projection='3d')
        w3 = np.array([0.5, 0.4, 1.6])
        a = np.cross(w3, [1.0, 0.0, 0.0])
        a /= np.linalg.norm(a)
        b = np.cross(w3, a)
        b /= np.linalg.norm(b)
        g = np.linspace(-1.6, 1.6, 2)
        S, T = np.meshgrid(g, g)
        P = S[..., None] * a + T[..., None] * b
        ax3.plot_surface(P[..., 0], P[..., 1], P[..., 2], color=X_COLOR, alpha=0.18,
                         linewidth=0)
        for k, (cs, ct) in enumerate(((1.4, 0), (0, 1.4), (-1, 0.9), (0.9, -1.1))):
            v = cs * a + ct * b
            ax3.quiver(0, 0, 0, *v, color=X_COLOR, lw=2.5, arrow_length_ratio=0.12)
        ax3.quiver(0, 0, 0, *w3, color=W_COLOR, lw=3.5, arrow_length_ratio=0.12)
        ax3.text(*(w3 * 1.08), 'w', color=W_COLOR, fontsize=17, fontweight='bold')
        lim = 1.6
        ax3.set_xlim(-lim, lim)
        ax3.set_ylim(-lim, lim)
        ax3.set_zlim(-lim, lim)
        ax3.set_box_aspect((1, 1, 1))
        ax3.set_xlabel('$x_1$')
        ax3.set_ylabel('$x_2$')
        ax3.set_zlabel('$x_3$')
        ax3.view_init(elev=18, azim=-60)
        ax3.set_title('With three features, the solutions of w · x = 0\n'
                      'fill a plane perpendicular to w')
        fig.tight_layout()
    return fig


def plot_level_lines(w, values, kind='c', mark_crossings=True, lim=LA_LIM):
    """The lines w . x = c for each c in values (kind='c'), or the lines w . x + b = 0
    for each b in values (kind='b'). w is drawn from the origin; each line's crossing
    with w's direction is marked."""
    w = np.asarray(w, dtype=float)
    cs = [v if kind == 'c' else -v for v in values]
    shades = np.linspace(0.15, 0.6, len(values))
    with rc_context(BIG_FONTS):
        fig, ax = subplots(figsize=(8.4, 8.4))
        t = np.array([-3 * lim, 3 * lim]) / np.linalg.norm(w)
        ax.plot(t * w[0], t * w[1], color=W_COLOR, ls=':', lw=1.3)       # w's direction
        for v, c, shade in zip(values, cs, shades):
            p1, p2 = _line_segment(w, c, lim)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=str(shade), lw=2.4)
            cross = c * w / (w @ w)
            label = f'w·x = {v:g}' if kind == 'c' else f'b = {v:g}'
            if mark_crossings:
                ax.plot(*cross, 'o', ms=8, color=str(shade), zorder=5)
            along = np.array([-w[1], w[0]]) / np.linalg.norm(w)
            spot = cross + 1.15 * along
            if max(abs(spot)) > lim - 0.3:
                spot = cross - 1.15 * along
            ax.annotate(label, spot, fontsize=14, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))
        _arrow(ax, (0, 0), w, W_COLOR, f'w = ({w[0]:g}, {w[1]:g})', offset=(6, -20))
        _plane(ax, xlim=(-lim, lim), ylim=(-lim, lim))
        what = 'w · x = c' if kind == 'c' else 'w · x + b = 0'
        ax.set_title(f'The lines {what}\n(dots: where each line crosses w’s direction)')
        fig.tight_layout()
    return fig


def plot_score_sides():
    """Why w points toward the positive side. Left (b = 0): the line w . x = 0 through the
    origin; a point on w's side is less than 90 degrees from w (score > 0), a point on the
    other side is more than 90 degrees away (score < 0). Right (b != 0): the same picture,
    with angles measured from a point x0 on the line w . x + b = 0."""
    from matplotlib.patches import Arc
    w = np.array([2.0, 1.0])
    offsets = {'pos': np.array([1.2, 1.6]), 'neg': np.array([-1.4, 0.4])}
    half = 2.6
    # Two different points per panel, so they get different names: x_A and x_B (not x_1 and
    # x_2, which are the axes' feature coordinates).
    panels = ((np.zeros(2), 'b = 0: the line w · x = 0 passes through the origin,\n'
               'and each point is drawn as a vector from the origin', 'w · x = 0', 'origin',
               r'$\mathbf{x}_{%s}$'),
              (np.array([2.0, 1.0]), 'b ≠ 0: the same picture, with vectors\n'
               'drawn from a point x₀ on the line', 'w · x + b = 0',
               'x₀ (a point on the line)', r'$\mathbf{x}_{%s} - \mathbf{x}_0$'))
    with rc_context(BIG_FONTS):
        fig, axes = subplots(1, 2, figsize=(16, 8))
        for ax, (base, title, line_label, base_label, vec_label) in zip(axes, panels):
            lo, hi = base - half, base + half
            g1, g2 = np.meshgrid(np.linspace(lo[0], hi[0], 200), np.linspace(lo[1], hi[1], 200))
            score = w[0] * (g1 - base[0]) + w[1] * (g2 - base[1])
            ax.contourf(g1, g2, score, levels=[-1e9, 0, 1e9],
                        colors=[_light(NEG_SIDE, 0.3), _light(POS_SIDE, 0.3)])
            ax.contour(g1, g2, score, levels=[0], colors='black', linewidths=2.5)
            along = np.array([-w[1], w[0]]) / np.linalg.norm(w)
            ax.annotate(line_label, base + 2.0 * along, xytext=(10, 0), textcoords='offset points',
                        fontsize=14, va='center')
            w_angle = np.degrees(np.arctan2(w[1], w[0]))
            for key, name, color, words in (('pos', 'A', CLASS1, 'score > 0'),
                                            ('neg', 'B', CLASS0, 'score < 0')):
                v = offsets[key]
                tip = base + v
                _arrow(ax, base, tip, '0.25', lw=2.5)
                # label the vector at its midpoint, just off to its left side
                perp = np.array([-v[1], v[0]]) / np.linalg.norm(v)
                # x_A's vector points up and right: put its label just left of the midpoint
                xytext, ha = ((-12, 12), 'right') if key == 'pos' else (tuple(20 * perp), 'center')
                ax.annotate(vec_label % name, base + v / 2, xytext=xytext, textcoords='offset points',
                            fontsize=17, fontweight='bold', color='0.2', ha=ha, va='center')
                ax.plot(*tip, 'o', ms=13, color=color, mec='black', mew=1, zorder=6)
                ax.annotate(words, tip, xytext=(10, 8), textcoords='offset points', fontsize=15,
                            fontweight='bold', color=color)
                v_angle = np.degrees(np.arctan2(v[1], v[0]))
                low, high = sorted((w_angle, v_angle))
                radius = 0.55 if key == 'pos' else 0.42
                ax.add_patch(Arc(base, 2 * radius, 2 * radius, theta1=low, theta2=high,
                                 color='0.2', lw=1.8))
                mid = np.radians((low + high) / 2)
                ax.annotate('< 90°' if key == 'pos' else '> 90°',
                            base + (radius + 0.28) * np.array([np.cos(mid), np.sin(mid)]),
                            fontsize=13, ha='center', va='center')
            _arrow(ax, base, base + w, W_COLOR, 'w', offset=(6, -6))
            ax.plot(*base, 'o', ms=7, color='black', zorder=6)
            ax.annotate(base_label, base, xytext=(-10, -44), textcoords='offset points',
                        fontsize=14, ha='right',
                        arrowprops=dict(arrowstyle='-', color='0.3', lw=1, shrinkA=2, shrinkB=4))
            ax.axhline(0, color='0.6', lw=0.8)
            ax.axvline(0, color='0.6', lw=0.8)
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])
            ax.set_aspect('equal')
            ax.set_xlabel('$x_1$')
            ax.set_ylabel('$x_2$')
            ax.set_title(title)
        fig.text(0.5, 0.01, 'Blue side: score > 0, the side w points toward.   '
                 'Orange side: score < 0.   Black line: score = 0.',
                 ha='center', fontsize=15)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def line_from_dot_product_demo(size_percent=100):
    """The line w.x + b = 0. Sliders: w's direction, w's length, and b. Blue shading:
    score > 0; orange: score < 0. Dashed lines are scores of +-1, +-2, ...; the readout
    gives the scores of three fixed points."""
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
            side = 'blue side' if s > 0 else 'orange side' if s < 0 else 'on the line'
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
            _plane(ax, grid=False)
            ax.set_title('Blue: w·x + b > 0.  Orange: w·x + b < 0.\n'
                         'Black line: w·x + b = 0.  Dashed: scores of ±1, ±2, …',
                         fontsize=15)
            fig.tight_layout()
            return fig
        _show(out, draw, size_percent)

    for s in (angle, length, bias):
        s.observe(update, names='value')
    update()
    display(widgets.VBox([angle, length, bias, readout, out]))


def two_class_data(n=60, seed=0):
    """Two overlapping groups of points: class 1 up and to the right of class 0."""
    rng = np.random.default_rng(seed)
    X0 = rng.normal(loc=(-0.6, -0.4), scale=0.9, size=(n, 2))
    X1 = rng.normal(loc=(1.0, 0.9), scale=0.9, size=(n, 2))
    X = np.vstack([X0, X1])
    y = np.r_[np.zeros(n, dtype=int), np.ones(n, dtype=int)]
    return X, y


def _class_points(ax, X, y):
    ax.scatter(*X[y == 1].T, marker='o', s=55, color=CLASS1, edgecolor='white', lw=0.6,
               label='class 1', zorder=4)
    ax.scatter(*X[y == 0].T, marker='^', s=60, color=CLASS0, edgecolor='white', lw=0.6,
               label='class 0', zorder=4)


def plot_logistic_boundary(X, y, model):
    """A fitted two-feature logistic regression: data, the p = 0.5 boundary, shading by
    predicted class, and beta drawn as an arrow from the boundary point closest to the
    data's center."""
    beta, beta0 = model.coef_[0], model.intercept_[0]
    lim = 3.6
    g = np.linspace(-lim, lim, 301)
    G1, G2 = np.meshgrid(g, g)
    Z = beta[0] * G1 + beta[1] * G2 + beta0
    center = X.mean(axis=0)
    foot = center - (beta @ center + beta0) / (beta @ beta) * beta
    with rc_context(BIG_FONTS):
        fig, ax = subplots(figsize=(8.4, 8.4))
        ax.contourf(G1, G2, Z, levels=[-1e9, 0, 1e9],
                    colors=[_light(NEG_SIDE, 0.3), _light(POS_SIDE, 0.3)])
        ax.contour(G1, G2, Z, levels=[0], colors='black', linewidths=3)
        _class_points(ax, X, y)
        unit = beta / np.linalg.norm(beta)
        _arrow(ax, foot, foot + 1.6 * unit, W_COLOR, 'β (direction)', offset=(6, 4))
        _plane(ax, xlim=(-lim, lim), ylim=(-lim, lim), grid=False)
        ax.legend(loc='lower right')
        ax.set_title('Logistic regression: the boundary (log-odds 0, p = 0.5)\n'
                     'and the direction of β, perpendicular to it')
        fig.tight_layout()
    return fig


# --- Part 5 -----------------------------------------------------------------------------

def plot_surface_and_contours(w, b):
    """The score w.x + b as a tilted plane, and the same score seen from above as a
    contour map, with w drawn from the line where the score is 0."""
    w = np.asarray(w, dtype=float)
    lim = 3
    g = np.linspace(-lim, lim, 61)
    G1, G2 = np.meshgrid(g, g)
    Z = w[0] * G1 + w[1] * G2 + b
    with rc_context(BIG_FONTS):
        fig = figure(figsize=(14.5, 6.6))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.plot_surface(G1, G2, Z, cmap='coolwarm_r', alpha=0.85, linewidth=0)
        ax1.contour(G1, G2, Z, levels=[0], colors='black', linewidths=3, offset=0)
        ax1.set_xlabel('$x_1$')
        ax1.set_ylabel('$x_2$')
        ax1.set_zlabel('score')
        ax1.set_title(f'The score w·x + b as a height\n(w = ({w[0]:g}, {w[1]:g}), b = {b:g})')
        ax1.view_init(elev=24, azim=-120)

        ax2 = fig.add_subplot(1, 2, 2)
        filled = ax2.contourf(G1, G2, Z, levels=np.arange(-12, 12.1, 1), cmap='coolwarm_r')
        lines = ax2.contour(G1, G2, Z, levels=np.arange(-8, 8.1, 2), colors='0.25',
                            linewidths=1)
        ax2.clabel(lines, fmt='%+.0f', fontsize=12)
        ax2.contour(G1, G2, Z, levels=[0], colors='black', linewidths=3)
        foot = -b * w / (w @ w)
        _arrow(ax2, foot, foot + w, W_COLOR, 'w', offset=(6, 4))
        ax2.set_aspect('equal')
        ax2.set_xlabel('$x_1$')
        ax2.set_ylabel('$x_2$')
        ax2.set_title('The same score, seen from above\n(level lines; w points uphill)')
        fig.colorbar(filled, ax=ax2, shrink=0.8, label='score')

        fig.tight_layout()
    return fig


def mystery_contours():
    """A labeled contour map of a score w.x + b whose w and b aren't shown."""
    w, b = np.array([1.0, -1.0]), 0.5
    g = np.linspace(-3, 3, 241)
    G1, G2 = np.meshgrid(g, g)
    Z = w[0] * G1 + w[1] * G2 + b
    with rc_context(BIG_FONTS):
        fig, ax = subplots(figsize=(7.4, 7))
        filled = ax.contourf(G1, G2, Z, levels=np.arange(-6.5, 7.6, 1), cmap='coolwarm_r')
        lines = ax.contour(G1, G2, Z, levels=np.arange(-5, 6.1, 1), colors='0.25', linewidths=1)
        ax.clabel(lines, fmt='%+.0f', fontsize=12)
        ax.set_aspect('equal')
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_title('A mystery score w·x + b, seen from above')
        fig.colorbar(filled, ax=ax, shrink=0.8, label='score')
        fig.tight_layout()
    return fig


# --- Part 7 -----------------------------------------------------------------------------

def plot_plane_to_line(w, b, X, names=None):
    """Left: feature space shaded by the score z = w . x + b (Part 5's colors), with the
    level lines through the points. Right: the same points on a number line at their
    scores, with the line colored on the same scale. Points on the same level line land
    on the same spot."""
    from matplotlib.colors import Normalize
    w = np.asarray(w, dtype=float)
    X = np.asarray(X, dtype=float)
    z = X @ w + b
    names = names or [chr(ord('A') + i) for i in range(len(X))]
    lim = 3.2
    cmap, norm = 'coolwarm_r', Normalize(vmin=-6, vmax=6)     # same colors on both panels
    marker = dict(ms=12, color='white', mec='black', mew=2, zorder=5)
    with rc_context(BIG_FONTS):
        fig, (left, right) = subplots(1, 2, figsize=(16, 6.6),
                                      gridspec_kw={'width_ratios': [1, 1.25]})
        g = np.linspace(-lim, lim, 241)
        G1, G2 = np.meshgrid(g, g)
        left.contourf(G1, G2, w[0] * G1 + w[1] * G2 + b, levels=np.arange(-13, 13.1, 1),
                      cmap=cmap, norm=norm, extend='both')
        for value in sorted(set(np.round(z, 9))):
            p1, p2 = _line_segment(w, value - b, lim)
            left.plot([p1[0], p2[0]], [p1[1], p2[1]], color='0.15', lw=1.4, ls='--')
        for name, p in zip(names, X):
            left.plot(*p, 'o', **marker)
            left.annotate(name, p, xytext=(8, 6), textcoords='offset points', fontsize=16,
                          fontweight='bold')
        foot = -b * w / (w @ w)
        _arrow(left, foot, foot + w, 'black', 'w', offset=(6, -4))
        _plane(left, xlim=(-lim, lim), ylim=(-lim, lim), grid=False)
        left.set_title('Feature space: points $(x_1, x_2)$, shaded by score\n'
                       '(dashed: level lines through the points)')

        lo, hi = np.floor(z.min()) - 1, np.ceil(z.max()) + 1
        strip = np.linspace(lo, hi, 400)[None, :]
        right.imshow(strip, extent=(lo, hi, -0.03, 0.03), aspect='auto', cmap=cmap, norm=norm,
                     zorder=1)
        right.plot([lo, hi], [0, 0], color='black', lw=0.8, zorder=2)
        for t in np.arange(lo, hi + 1):
            right.plot([t, t], [-0.045, 0.045], color='black', lw=1.2, zorder=3)
            right.annotate(f'{t:g}', (t, -0.07), ha='center', va='top', fontsize=13)
        seen = {}
        for name, value in zip(names, z):
            k = seen.get(round(value, 9), 0)
            seen[round(value, 9)] = k + 1
            right.plot(value, 0, 'o', **marker)
            right.annotate(name, (value, 0), xytext=(0, 22 + 22 * k), textcoords='offset points',
                           ha='center', fontsize=16, fontweight='bold')
        right.set_xlim(lo - 0.3, hi + 0.3)
        right.set_ylim(-0.35, 0.45)
        right.annotate('z', (hi + 0.25, 0), xytext=(0, 10), textcoords='offset points',
                       fontsize=16, ha='center')
        right.axis('off')
        right.set_title(f'z-space: a number line, $z = \\mathbf{{w}}\\cdot\\mathbf{{x}} + b$, '
                        f'colored on the same scale\n(w = ({w[0]:g}, {w[1]:g}), b = {b:g})')
        fig.tight_layout()
    return fig


def plot_plane_to_plane(W, b, X, names=None):
    """Left: points in feature space. Right: the same points at W x + b in z-space. Each
    point keeps its color and letter, so you can see where it went. If W's rows are
    parallel, the image is a line, drawn faintly."""
    W = np.asarray(W, dtype=float)
    b = np.asarray(b, dtype=float)
    X = np.asarray(X, dtype=float)
    Z = np.array([W @ x + b for x in X])
    names = names or [chr(ord('A') + i) for i in range(len(X))]
    colors = OKABE_PTS[:len(X)]
    with rc_context(BIG_FONTS):
        fig, (left, right) = subplots(1, 2, figsize=(15, 7))
        if np.linalg.matrix_rank(W) == 1:
            d = W[:, 0] if np.linalg.norm(W[:, 0]) > 0 else W[:, 1]
            t = np.array([-20, 20])
            right.plot(b[0] + t * d[0], b[1] + t * d[1], color='0.75', lw=6, alpha=0.5,
                       solid_capstyle='butt', zorder=1)
        for ax, P in ((left, X), (right, Z)):
            stacked = {}
            for name, p, c in zip(names, P, colors):
                key = tuple(np.round(p, 9))
                k = stacked.get(key, 0)
                stacked[key] = k + 1
                ax.plot(*p, 'o', ms=14 - 4 * k, color=c, mec='black', mew=1.5, zorder=5 + k)
                ax.annotate(name, p, xytext=(10, 6 + 18 * k), textcoords='offset points',
                            fontsize=16, fontweight='bold', color=c)
            ax.axhline(0, color='black', lw=0.8)
            ax.axvline(0, color='black', lw=0.8)
            ax.set_aspect('equal')
            ax.grid(True, color='0.9')
        left.set_xlim(-3.5, 3.5)
        left.set_ylim(-3.5, 3.5)
        left.set_xlabel('$x_1$')
        left.set_ylabel('$x_2$')
        left.set_title('Feature space: points $\\mathbf{x} = (x_1, x_2)$')
        zl = max(4.5, np.abs(Z).max() + 1)
        right.set_xlim(-zl, zl)
        right.set_ylim(-zl, zl)
        right.set_xlabel('$z_1$')
        right.set_ylabel('$z_2$')
        right.set_title('z-space: the same points at $\\mathbf{z} = W\\mathbf{x} + \\mathbf{b}$')
        fig.tight_layout()
    return fig


def plot_points_to_scores(X, W, b, names=None):
    """Left: points and the two lines (rows of W, entries of b). Right: the same points
    at their new coordinates (z1, z2) = W x + b."""
    X = np.asarray(X, dtype=float)
    W = np.asarray(W, dtype=float)
    b = np.asarray(b, dtype=float)
    Z = X @ W.T + b
    names = names or [chr(ord('A') + i) for i in range(len(X))]
    styles = ({'color': '0.15', 'ls': '-'}, {'color': '0.15', 'ls': (0, (6, 4))})
    with rc_context(BIG_FONTS):
        fig, (left, right) = subplots(1, 2, figsize=(15, 7))
        lim = 3.6
        for k in range(2):
            p1, p2 = _line_segment(W[k], -b[k], lim)
            left.plot([p1[0], p2[0]], [p1[1], p2[1]], lw=2.4, **styles[k],
                      label=f'row {k + 1}\'s line: $z_{k + 1}$ = 0')
        for name, p, z in zip(names, X, Z):
            for ax, v in ((left, p), (right, z)):
                ax.plot(*v, 'o', ms=11, color=X_COLOR, zorder=5)
                ax.annotate(name, v, xytext=(8, 6), textcoords='offset points', fontsize=16,
                            fontweight='bold')
        _plane(left, xlim=(-lim, lim), ylim=(-lim, lim))
        left.legend(loc='lower left', fontsize=12)
        left.set_title('Feature space: points $(x_1, x_2)$')
        zlim = max(3.6, np.abs(Z).max() + 0.8)
        right.axhline(0, **styles[1], lw=2.4)
        right.axvline(0, **styles[0], lw=2.4)
        right.set_xlim(-zlim, zlim)
        right.set_ylim(-zlim, zlim)
        right.set_aspect('equal')
        right.grid(True, color='0.9')
        right.set_xlabel("$z_1$ (row 1's score)")
        right.set_ylabel("$z_2$ (row 2's score)")
        right.set_title('z-space: the same points at $(z_1, z_2) = W\\mathbf{x} + \\mathbf{b}$')
        fig.tight_layout()
    return fig


# --- Parts 9 and 10 ---------------------------------------------------------------------

def plot_column_space(matrices, names=None, n=300, seed=0):
    """For each matrix: where a cloud of random points lands after multiplying by it."""
    rng = np.random.default_rng(seed)
    cloud = rng.uniform(-1, 1, size=(n, 2))
    names = names or [f'matrix {i + 1}' for i in range(len(matrices))]
    with rc_context(BIG_FONTS):
        fig, axes = subplots(1, len(matrices) + 1, figsize=(5.2 * (len(matrices) + 1), 5.4))
        axes[0].scatter(*cloud.T, s=10, color='0.4')
        axes[0].set_title('Before: random points')
        for ax, M, name in zip(axes[1:], matrices, names):
            M = np.asarray(M, dtype=float)
            out = cloud @ M.T
            ax.scatter(*out.T, s=10, color=X_COLOR)
            _arrow(ax, (0, 0), M[:, 0], W_COLOR, 'col 1', lw=2.5, fontsize=13)
            _arrow(ax, (0, 0), M[:, 1], SUM_COLOR, 'col 2', lw=2.5, offset=(-40, 8),
                   fontsize=13)
            ax.set_title(f'After {name}')
        for ax in axes:
            ax.axhline(0, color='black', lw=0.8)
            ax.axvline(0, color='black', lw=0.8)
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            ax.set_aspect('equal')
            ax.grid(True, color='0.9')
            ax.set_xlabel('first coordinate')
            ax.set_ylabel('second coordinate')
        fig.tight_layout()
    return fig


def plot_level_line_images(points, mapped):
    """Points colored by x1 + x2, before and after a map."""
    points, mapped = np.asarray(points, dtype=float), np.asarray(mapped, dtype=float)
    s = points.sum(axis=1)
    with rc_context(BIG_FONTS):
        fig, (left, right) = subplots(1, 2, figsize=(14, 6.4))
        for ax, P, title in ((left, points, 'Before: colored by $x_1 + x_2$'),
                             (right, mapped, 'After multiplying by W')):
            sc = ax.scatter(*P.T, c=s, cmap='viridis', s=60, edgecolor='white', lw=0.5,
                            zorder=4)
            ax.axhline(0, color='black', lw=0.8)
            ax.axvline(0, color='black', lw=0.8)
            ax.set_aspect('equal')
            ax.grid(True, color='0.9')
            ax.set_title(title)
        left.set_xlabel('$x_1$')
        left.set_ylabel('$x_2$')
        right.set_xlabel('first coordinate')
        right.set_ylabel('second coordinate')
        right.set_xlim(-4.5, 4.5)
        right.set_ylim(-4.5, 4.5)
        fig.colorbar(sc, ax=[left, right], shrink=0.8, label='$x_1 + x_2$')
    return fig


def _grid_lines(lim=2, step=0.5, pts=81):
    """Lines of a square grid: (constant-x1 lines, constant-x2 lines), each an (n, pts, 2) array."""
    ticks = np.arange(-lim, lim + 1e-9, step)
    t = np.linspace(-lim, lim, pts)
    v = np.array([np.column_stack([np.full_like(t, c), t]) for c in ticks])
    h = np.array([np.column_stack([t, np.full_like(t, c)]) for c in ticks])
    return v, h


def _draw_grid(ax, W, b, lim=2, short=False):
    W = np.eye(2) if W is None else np.asarray(W, dtype=float)
    b = np.zeros(2) if b is None else np.asarray(b, dtype=float)
    v, h = _grid_lines(lim)
    for lines, color, lw in ((h, '0.72', 1.4), (v, '0.25', 1.4)):
        for line in lines:
            out = line @ W.T + b
            ax.plot(out[:, 0], out[:, 1], color=color, lw=lw)
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) @ W.T + b
    ax.fill(square[:, 0], square[:, 1], color=SUM_COLOR, alpha=0.35, zorder=3)
    labels = ('(1, 0)', '(0, 1)') if short else ('where (1, 0) goes', 'where (0, 1) goes')
    for col, color, label, default in ((W[:, 0], W_COLOR, labels[0], (4, -16)),
                                       (W[:, 1], X_COLOR, labels[1], (4, 6))):
        n = np.linalg.norm(col)
        # Put the label just beyond the arrow's tip, in the direction the arrow points.
        offset = default if n < 1e-9 or not short else \
            (col[0] / n * 22 - 18, col[1] / n * 16 - 6)
        _arrow(ax, b, b + col, color, label, lw=3, fontsize=12, offset=offset)
    ax.axhline(0, color='black', lw=0.6)
    ax.axvline(0, color='black', lw=0.6)
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    ax.set_aspect('equal')


def plot_grid_map(W, b=None, title=None):
    """A square grid before and after x -> W x + b. Dark lines: constant x1. Light lines:
    constant x2. Green: the unit square. Arrows: where (1, 0) and (0, 1) go."""
    with rc_context(BIG_FONTS):
        fig, (left, right) = subplots(1, 2, figsize=(14, 7))
        _draw_grid(left, None, None)
        left.set_title('Before')
        left.set_xlabel('$x_1$')
        left.set_ylabel('$x_2$')
        _draw_grid(right, W, b)
        W_ = np.asarray(W, dtype=float)
        default = f'After x ↦ Wx + b, W = {np.round(W_, 2).tolist()}'
        if b is not None:
            default += f', b = {np.round(np.asarray(b, dtype=float), 2).tolist()}'
        right.set_title(title or default, fontsize=14)
        right.set_xlabel('first coordinate')
        right.set_ylabel('second coordinate')
        fig.tight_layout()
    return fig


QUIZ_MATRICES = {'M1': [[2, 0], [0, 1]], 'M2': [[0, -1], [1, 0]], 'M3': [[1, 1], [0, 1]],
                 'M4': [[-1, 0], [0, 1]], 'M5': [[1, 0], [0, 0]]}
QUIZ_ORDER = ['M3', 'M5', 'M2', 'M4', 'M1']          # panel A shows M3, panel B shows M5, ...


def grid_map_quiz():
    """Five after-pictures, labeled A to E, of the matrices in QUIZ_MATRICES, shuffled."""
    with rc_context(BIG_FONTS):
        fig, axes = subplots(1, 5, figsize=(26, 5.8))
        for ax, letter, key in zip(axes, 'ABCDE', QUIZ_ORDER):
            _draw_grid(ax, QUIZ_MATRICES[key], None, short=True)
            ax.set_title(f'Picture {letter}', fontsize=18, fontweight='bold')
            ax.set_xlim(-4.5, 4.5)
            ax.set_xlabel('first coordinate')
            ax.set_ylabel('second coordinate')
        fig.tight_layout()
        fig.text(0.5, -0.02, 'Orange arrow: where (1, 0) lands.  Blue arrow: where (0, 1) lands.  '
                 'Green: where the unit square lands.', ha='center', fontsize=15)
    return fig
