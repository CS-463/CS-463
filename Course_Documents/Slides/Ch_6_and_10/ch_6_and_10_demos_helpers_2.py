"""Plotting and widget helpers for ch_6_and_10_04_softmax_and_cross_entropy.ipynb and
ch_6_and_10_05_deeper_networks.ipynb.

Kept separate from ch_6_and_10_demos_helpers.py so the two files can be edited
independently; it reuses that file's styling pieces (fonts, colors, the
figure-redraw helper), so both files must sit next to the notebook.
"""
import numpy as np
import ipywidgets as widgets
from IPython.display import display
from matplotlib import rc_context
from matplotlib.pyplot import subplots

from ch_6_and_10_demos_helpers import BIG_FONTS, OKABE_ITO, _figure_area, _html, _light, _show

# The running example of Sections 3 and 4: tomorrow's weather.
WEATHER = ('sunny', 'cloudy', 'rain', 'snow')
WEATHER_P = np.array([1 / 2, 1 / 4, 1 / 8, 1 / 8])
WEATHER_CODE = ('0', '10', '110', '111')
WEATHER_COLORS = (OKABE_ITO[1], '#999999', OKABE_ITO[0], OKABE_ITO[4])

TRUTH_COLOR = '#009E73'      # the truth p (green)
MODEL_COLOR = '#CC79A7'      # the model's beliefs q (reddish purple)
GAP_COLOR = '#D55E00'        # the KL gap (vermillion)
CE_COLOR = '#0072B2'         # cross-entropy (blue)
SE_COLOR = '#E69F00'         # squared error (orange)


def _bits(q):
    return -np.log2(q)


# ---------------------------------------------------------------------------
# Demo S1: softmax, three scores in, three probabilities out
# ---------------------------------------------------------------------------

def softmax_demo(size_percent=100):
    """Demo S1: three score sliders. Panels: the scores, e to each score, and the
    probabilities (each e^z divided by their total)."""
    options = [(f'{v:g}', v) for v in np.round(np.arange(-4, 4.01, 0.5), 1)]
    style = {'description_width': '40px'}
    sliders = [widgets.SelectionSlider(options=options, value=v, description=f'z{k}',
                                       style=style, layout=widgets.Layout(width='520px'))
               for k, v in zip('₁₂₃', (2.0, 1.0, 0.0))]
    readout = widgets.HTML()
    out = _figure_area()
    colors = OKABE_ITO[:3]
    names = ['class 1', 'class 2', 'class 3']

    def update(change=None):
        z = np.array([s.value for s in sliders])
        e = np.exp(z)
        p = e / e.sum()
        readout.value = _html(
            'e<sup>z</sup>: ' + ', '.join(f'{v:.2f}' for v in e)
            + f' &nbsp; total: <b>{e.sum():.2f}</b><br>'
            'probabilities: ' + ', '.join(f'<b>{v:.3f}</b>' for v in p)
            + f' &nbsp; sum: <b>{p.sum():.3f}</b>')

        def draw():
            fig, axes = subplots(1, 3, figsize=(15, 4.6))
            panels = [(z, 'scores z (any sign)', (-4.5, 4.5)),
                      (e, 'step 1: e^z (all positive)', (0, max(np.exp(4), 1) * 1.05)),
                      (p, 'step 2: divide by the total', (0, 1.05))]
            for ax, (vals, title, ylim) in zip(axes, panels):
                bars = ax.bar(names, vals, color=colors, edgecolor='black')
                ax.set_ylim(*ylim)
                ax.axhline(0, color='black', lw=1)
                ax.set_title(title)
                for bar, v in zip(bars, vals):
                    y = max(v, 0) + 0.02 * (ylim[1] - ylim[0])
                    ax.text(bar.get_x() + bar.get_width() / 2, y, f'{v:.2f}',
                            ha='center', va='bottom', fontsize=14)
            axes[2].set_ylabel('probability')
            fig.tight_layout()
            return fig
        _show(out, draw, size_percent)

    for s in sliders:
        s.observe(update, names='value')
    update()
    display(widgets.VBox(sliders + [readout, out]))


# ---------------------------------------------------------------------------
# Demo S2: the surprise curve
# ---------------------------------------------------------------------------

def plot_surprise_curve():
    """Demo S2: surprise -log2(q) against q, with a fair coin, one card of 8, and a
    1-in-1,000 outcome marked."""
    q = np.linspace(0.001, 1, 2000)
    marks = [(1 / 2, 'fair coin lands heads\nq = 1/2: 1 bit', (25, 35)),
             (1 / 8, 'one card out of 8\nq = 1/8: 3 bits', (30, 25)),
             (1 / 1000, 'a 1-in-1,000 event\nq = 0.001: 9.97 bits', (35, -5)),
             (1.0, 'certain\nq = 1: 0 bits', (-60, 40))]
    with rc_context(BIG_FONTS):
        fig, ax = subplots(figsize=(10, 5.6))
        ax.plot(q, _bits(q), color=CE_COLOR, lw=3)
        for qq, text, offset in marks:
            ax.plot(qq, _bits(qq), 'o', color='black', ms=9, zorder=3)
            ax.annotate(text, (qq, _bits(qq)), xytext=offset, textcoords='offset points',
                        fontsize=14, arrowprops=dict(arrowstyle='-', color='0.4'))
        ax.set_xlim(-0.02, 1.05)
        ax.set_ylim(-0.4, 11)
        ax.set_xlabel('q(x): the probability you gave the outcome')
        ax.set_ylabel('surprise, in bits')
        ax.set_title('Surprise = −log₂ q(x): rare outcomes are surprising')
        ax.grid(alpha=0.3)
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Demo S3: the weather distribution and its code
# ---------------------------------------------------------------------------

def plot_weather_code():
    """Demo S3: left, the weather distribution with each outcome's probability,
    surprise, and code word; right, the code as a tree of yes/no questions."""
    from matplotlib.patches import Circle
    p = WEATHER_P
    with rc_context(BIG_FONTS):
        fig, (ax, tree) = subplots(1, 2, figsize=(15, 5.8), gridspec_kw={'width_ratios': [1.1, 1]})
        bars = ax.bar(WEATHER, p, color=WEATHER_COLORS, edgecolor='black')
        for bar, pp, code in zip(bars, p, WEATHER_CODE):
            ax.text(bar.get_x() + bar.get_width() / 2, pp + 0.015,
                    f'p = {pp:g}\nsurprise {_bits(pp):g} bit' + ('' if _bits(pp) == 1 else 's') + f'\ncode {code}',
                    ha='center', va='bottom', fontsize=13)
        ax.set_ylim(0, 0.8)
        ax.set_ylabel('p(x)')
        ax.set_title(f'Tomorrow’s weather: H(p) = {-(p * np.log2(p)).sum():.2f} bits')

        # The tree: each internal node is one yes/no question; the path is the code word.
        nodes = {'q1': (0, 3), 'q2': (1, 2), 'q3': (2, 1),
                 'sunny': (-1, 2), 'cloudy': (0, 1), 'rain': (1, 0), 'snow': (3, 0)}
        questions = {'q1': 'sunny?', 'q2': 'cloudy?', 'q3': 'rain?'}
        edges = [('q1', 'sunny', '0'), ('q1', 'q2', '1'), ('q2', 'cloudy', '0'),
                 ('q2', 'q3', '1'), ('q3', 'rain', '0'), ('q3', 'snow', '1')]
        for a, b, bit in edges:
            (xa, ya), (xb, yb) = nodes[a], nodes[b]
            tree.plot([xa, xb], [ya, yb], color='0.35', lw=2, zorder=1)
            tree.text((xa + xb) / 2 + (0.42 if xb > xa else -0.42), (ya + yb) / 2 + 0.12,
                      'yes → ' + bit if bit == '0' else 'no → ' + bit, ha='center', fontsize=13,
                      color='0.2')
        for name, (x, y) in nodes.items():
            if name in questions:
                tree.add_patch(Circle((x, y), 0.36, facecolor='white', edgecolor='black',
                                      lw=2, zorder=2))
                tree.text(x, y, questions[name], ha='center', va='center', fontsize=12, zorder=3)
            else:
                k = WEATHER.index(name)
                tree.add_patch(Circle((x, y), 0.36, facecolor=_light(WEATHER_COLORS[k], 0.5),
                                      edgecolor=WEATHER_COLORS[k], lw=2, zorder=2))
                tree.text(x, y, name, ha='center', va='center', fontsize=13, zorder=3)
                tree.text(x, y - 0.5, WEATHER_CODE[k], ha='center', va='top', fontsize=15,
                          fontweight='bold', family='monospace')
        tree.set_xlim(-1.6, 3.6)
        tree.set_ylim(-0.9, 3.6)
        tree.set_aspect('equal')
        tree.axis('off')
        tree.set_title('The code as yes/no questions\n(each answer is one bit)')
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Demo S4: believing the wrong distribution
# ---------------------------------------------------------------------------

Q_PRESETS = {
    'q = p': WEATHER_P,
    'uniform': np.array([0.25, 0.25, 0.25, 0.25]),
    'almost right': np.array([0.4, 0.3, 0.15, 0.15]),
    'confident and wrong': np.array([0.01, 0.01, 0.01, 0.97]),
}


def weather_entropies(q, p=WEATHER_P):
    """H(p), H(p, q), and D_KL(p || q), in bits, for the weather truth p."""
    q = np.asarray(q, dtype=float)
    q = q / q.sum()
    h = float(-(p * np.log2(p)).sum())
    ce = float(-(p * np.log2(q)).sum())
    kl = float((p * np.log2(p / q)).sum())
    return h, ce, (0.0 if abs(kl) < 1e-12 else kl)


def _nice(v):
    """A probability as a fraction when it's a simple one (1/2, 3/8, ...), else 2 decimals."""
    for den in (1, 2, 4, 8, 16, 32):
        num = round(v * den)
        if num and abs(v - num / den) < 1e-9:
            return '1' if num == den else f'{num}/{den}'
    return f'{v:.2f}'


def _cross_entropy_figure(q, hide_q=False, hide_beliefs=False):
    """Demo S4's three panels for beliefs q: the truth p, your beliefs q, and the stacked
    pieces p(x) * surprise for H(p) and H(p, q). hide_q hides the numbers on the
    "believe q" bar (its pieces, H(p, q), and the KL gap), for the practice questions;
    hide_beliefs also leaves out the beliefs panel."""
    p = WEATHER_P
    h, ce, kl = weather_entropies(q)
    if hide_beliefs:
        fig, axes = subplots(1, 2, figsize=(12, 6.2), gridspec_kw={'width_ratios': [1, 1.05]})
        panels = ((axes[0], p, 'the truth p (fixed)', TRUTH_COLOR),)
    else:
        fig, axes = subplots(1, 3, figsize=(16, 5.4), gridspec_kw={'width_ratios': [1, 1, 1.05]})
        panels = ((axes[0], p, 'the truth p (fixed)', TRUTH_COLOR),
                  (axes[1], q, 'your beliefs q', MODEL_COLOR))
    for ax, vals, title, color in panels:
        bars = ax.bar(WEATHER, vals, color=_light(color, 0.6), edgecolor=color, lw=2)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, _nice(v),
                    ha='center', va='bottom', fontsize=13)
        ax.set_ylim(0, 1.08)
        ax.set_title(title)
        ax.tick_params(axis='x', labelsize=13)
    axes[0].set_ylabel('probability')

    ax = axes[-1]
    for col, dist in ((0, p), (1, q)):
        bottom = 0.0
        for k, w in enumerate(WEATHER):
            piece = p[k] * _bits(dist[k])
            ax.bar(col, piece, bottom=bottom, width=0.6, color=WEATHER_COLORS[k],
                   edgecolor='black', label=w if col == 0 else None)
            if piece > 0.3:
                text = w if (hide_q and col == 1) else f'{w} {piece:.3g}'
                ax.text(col, bottom + piece / 2, text, ha='center',
                        va='center', fontsize=12, zorder=5,
                        bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.75,
                                  edgecolor='none'))
            bottom += piece
    ax.axhline(h, color='black', ls='--', lw=1.5)
    if kl > 1e-9:
        ax.bar(1, kl, bottom=h, width=0.72, color='none', edgecolor=GAP_COLOR, lw=3,
               hatch='//')
        y = max(h + kl / 2, h + 0.45)
        gap = 'KL gap\n= ?' if hide_q else f'KL gap\n{kl:.4g} bits'
        ax.text(1.42, y, gap, fontsize=13, color=GAP_COLOR,
                va='center', fontweight='bold')
    ce_text = '?' if hide_q else f'{ce:.4g}'
    ax.set_xticks([0, 1], [f'believe p\nH(p) = {h:.3g}', f'believe q\nH(p, q) = {ce_text}'])
    ax.set_xlim(-0.5, 2.3)
    ax.set_ylim(0, max(2.6, ce * 1.25) if hide_beliefs else max(6.2, ce * 1.08))
    ax.set_ylabel('average surprise, bits')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left', fontsize=12)
    ax.set_title('each piece: p(x) × surprise')
    fig.tight_layout()
    return fig


def plot_cross_entropy_snapshot(q, mystery=False):
    """A static copy of Demo S4's figure for fixed beliefs q (sunny, cloudy, rain, snow),
    with the "believe q" bar's numbers hidden: the Practice Questions ask for them.
    mystery=True also hides the beliefs panel, so q must be read off the stacked bars."""
    q = np.asarray(q, dtype=float)
    with rc_context(BIG_FONTS):
        return _cross_entropy_figure(q / q.sum(), hide_q=True, hide_beliefs=mystery)


def cross_entropy_demo(size_percent=100):
    """Demo S4: the truth p is fixed; presets and sliders set the model's q (renormalized
    to sum to 1). The right panel stacks each outcome's contribution p(x) * surprise, once
    with your surprise measured by p (Shannon entropy) and once by q (cross-entropy); the
    difference in heights is the KL divergence."""
    p = WEATHER_P
    preset = widgets.ToggleButtons(options=list(Q_PRESETS) + ['(sliders)'], value='uniform',
                                   style={'button_width': '150px'})
    style = {'description_width': '110px'}
    sliders = [widgets.FloatSlider(value=0.25, min=0.01, max=1.0, step=0.005,
                                   description=f'belief: {w}', style=style, readout_format='.3f',
                                   continuous_update=False, layout=widgets.Layout(width='520px'))
               for w in WEATHER]
    readout = widgets.HTML()
    out = _figure_area()
    busy = {'flag': False}

    def update(change=None):
        raw = np.array([s.value for s in sliders])
        q = raw / raw.sum()
        h, ce, kl = weather_entropies(q)
        readout.value = _html(
            'your beliefs q (rescaled to sum to 1): '
            + ', '.join(f'{w} {v:.3f}' for w, v in zip(WEATHER, q)) + '<br>'
            f'Shannon entropy H(p) = <b>{h:.3f}</b> bits &nbsp; '
            f'cross-entropy H(p, q) = <b>{ce:.3f}</b> bits &nbsp; '
            f'KL divergence = H(p, q) − H(p) = <b style="color:{GAP_COLOR}">{kl:.3f}</b> bits')

        _show(out, lambda: _cross_entropy_figure(q), size_percent)

    def on_preset(change):
        if change['new'] == '(sliders)':
            return
        busy['flag'] = True
        for s, v in zip(sliders, Q_PRESETS[change['new']]):
            s.value = float(v)
        busy['flag'] = False
        update()

    def on_slider(change):
        if busy['flag']:
            return
        if preset.value != '(sliders)':
            preset.unobserve(on_preset, names='value')
            preset.value = '(sliders)'
            preset.observe(on_preset, names='value')
        update()

    preset.observe(on_preset, names='value')
    for s in sliders:
        s.observe(on_slider, names='value')
    update()
    display(widgets.VBox([preset] + sliders + [readout, out]))


# ---------------------------------------------------------------------------
# Demo S5: cross-entropy versus squared error
# ---------------------------------------------------------------------------

def plot_ce_vs_squared_error():
    """Demo S5: left, each loss against q, the probability given to what happened;
    right, each loss's slope with respect to the score z behind q = sigmoid(z)."""
    q = np.linspace(0.001, 1, 2000)
    with rc_context(BIG_FONTS):
        fig, (left, right) = subplots(1, 2, figsize=(15, 5.4))
        left.plot(q, -np.log(q), color=CE_COLOR, lw=3, label='cross-entropy  −ln q')
        left.plot(q, (1 - q) ** 2, color=SE_COLOR, lw=3, label='squared error  (1 − q)²')
        left.set_ylim(-0.2, 7.2)
        left.set_title('The loss')
        left.set_ylabel('loss')

        right.plot(q, q - 1, color=CE_COLOR, lw=3, label='cross-entropy  q − 1')
        right.plot(q, 2 * (q - 1) * q * (1 - q), color=SE_COLOR, lw=3,
                   label='squared error  2(q − 1) q (1 − q)')
        right.axhline(0, color='black', lw=1)
        right.set_ylim(-1.08, 0.3)
        right.set_title('Its slope ∂L/∂z: how hard it pushes on the score z')
        right.set_ylabel('slope')
        right.annotate('squared error’s slope is near 0\nwhen the model is confidently wrong',
                       (0.02, 2 * (0.02 - 1) * 0.02 * 0.98), xytext=(0.1, 0.12), fontsize=13,
                       arrowprops=dict(arrowstyle='->', color='0.3'))
        for ax in (left, right):
            ax.set_xlabel('q: the probability the model gave to what happened')
            ax.set_xlim(0, 1.02)
            ax.grid(alpha=0.3)
            ax.legend(loc='upper right' if ax is left else 'lower right')
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Output-layer diagram: last hidden layer -> 10 scores -> softmax -> 10 probabilities
# ---------------------------------------------------------------------------

OUTPUT_EXAMPLE_Z = np.array([-1.0, 0.5, 1.2, 3.5, -0.5, 0.8, -2.0, 0.3, 2.0, 0.6])


def plot_output_layer():
    """The output layer of a 10-class network: the last hidden layer's units feed 10
    output units, each computing a score z_k; softmax turns the 10 scores into 10
    probabilities that sum to 1 (example values, an image of a 3)."""
    from matplotlib.patches import Ellipse, FancyBboxPatch
    rx, ry = 0.22, 0.4             # drawn as circles: the axes are wider per unit than tall
    z = OUTPUT_EXAMPLE_Z
    p = np.exp(z) / np.exp(z).sum()
    ys = np.linspace(9, 0, 10)
    hidden_y = [7.2, 5.4, 3.6]
    unit_color = '#009E73'
    with rc_context(BIG_FONTS):
        fig, ax = subplots(figsize=(13, 7.6))
        for hy in hidden_y:
            for oy in ys:
                ax.plot([rx, 3 - rx], [hy, oy], color='0.75', lw=0.8, zorder=1)
        for hy, label in zip(hidden_y, ('h₁', 'h₂', 'h₃')):
            ax.add_patch(Ellipse((0, hy), 2 * rx, 2 * ry, facecolor=_light(unit_color, 0.3),
                                edgecolor=unit_color, lw=2, zorder=2))
            ax.text(0, hy, label, ha='center', va='center', fontsize=15, zorder=3)
        ax.text(0, 2.5, '⋮', ha='center', va='center', fontsize=22)
        ax.text(0, 9.9, 'last hidden\nlayer', ha='center', va='bottom', fontsize=14)
        for k, oy in enumerate(ys):
            ax.add_patch(Ellipse((3, oy), 2 * rx, 2 * ry, facecolor='white', edgecolor='black', lw=2, zorder=2))
            ax.text(3, oy, f'z{chr(0x2080 + k)}', ha='center', va='center', fontsize=14, zorder=3)
            ax.text(3.45, oy, f'{z[k]:+.1f}', ha='left', va='center', fontsize=13, color='0.3')
            ax.annotate('', (5.0, oy), (4.15, oy), arrowprops=dict(arrowstyle='-|>', color='0.4'))
            ax.annotate('', (6.55, oy), (6.05, oy), arrowprops=dict(arrowstyle='-|>', color='0.4'))
            ax.barh(oy, 2.6 * p[k], left=7.0, height=0.6, color=_light(CE_COLOR, 0.75),
                    edgecolor=CE_COLOR)
            ax.text(7.05 + 2.6 * p[k], oy, f' p{chr(0x2080 + k)} = {p[k]:.2f}', ha='left',
                    va='center', fontsize=13)
            ax.text(6.78, oy, str(k), ha='center', va='center', fontsize=13, fontweight='bold')
        ax.text(3, 9.9, 'output layer:\n10 scores, one per digit', ha='center', va='bottom',
                fontsize=14)
        ax.add_patch(FancyBboxPatch((5.1, -0.4), 0.85, 9.8, boxstyle='round,pad=0.1',
                                    facecolor=_light(MODEL_COLOR, 0.35), edgecolor=MODEL_COLOR,
                                    lw=2.5))
        ax.text(5.52, 4.5, 'softmax', rotation=90, ha='center', va='center', fontsize=20,
                fontweight='bold')
        ax.text(8.2, 9.9, f'10 probabilities\n(sum = {p.sum():.2f})', ha='center', va='bottom',
                fontsize=14)
        ax.set_xlim(-0.6, 10.6)
        ax.set_ylim(-0.8, 11.2)
        ax.axis('off')
        fig.tight_layout()
    return fig



# ===========================================================================
# ch_6_and_10_05_deeper_networks.ipynb
# ===========================================================================

def plot_two_hidden_layers():
    """Demo D1: a 2-3-3-1 network, two hidden layers, every unit labeled."""
    from ch_6_and_10_demos_helpers import _draw_network
    xs = (0, 3.2, 6.4, 9.6)
    nodes = {'x1': (xs[0], 3.4, '$x_1$', '', 'input'), 'x2': (xs[0], 1.6, '$x_2$', '', 'input')}
    for layer, x0 in ((1, xs[1]), (2, xs[2])):
        for k, y0 in zip((1, 2, 3), (4.3, 2.5, 0.7)):
            nodes[f'h{layer}{k}'] = (x0, y0, f'$h^{{({layer})}}_{k}$', '', 'unit')
    nodes['out'] = (xs[3], 2.5, '$p$', '', 'unit')
    edges = [(a, f'h1{k}', '', 0.5) for a in ('x1', 'x2') for k in (1, 2, 3)]
    edges += [(f'h1{j}', f'h2{k}', '', 0.5) for j in (1, 2, 3) for k in (1, 2, 3)]
    edges += [(f'h2{k}', 'out', '', 0.5) for k in (1, 2, 3)]
    with rc_context(BIG_FONTS):
        fig, ax = subplots(figsize=(13, 6.2))
        _draw_network(ax, nodes, edges, radius=0.55)
        for x0, top, bottom in ((xs[0], 'inputs', '$\\mathbf{x}$: 2 features'),
                                (xs[1], 'hidden layer 1', '$\\mathbf{h}^{(1)}$: 3 units'),
                                (xs[2], 'hidden layer 2', '$\\mathbf{h}^{(2)}$: 3 units'),
                                (xs[3], 'output', '$p$: 1 unit')):
            ax.text(x0, 5.4, top, ha='center', fontsize=16, fontweight='bold')
            ax.text(x0, -0.35, bottom, ha='center', va='top', fontsize=15)
        for x0, name, shape in ((1.6, '$W^{(1)}$', '3 × 2'), (4.8, '$W^{(2)}$', '3 × 3'),
                                (8.0, '$W^{(3)}$', '1 × 3')):
            ax.text(x0, -1.25, f'{name}: {shape}', ha='center', va='top', fontsize=15, color='#D55E00')
        ax.set_xlim(-1.2, 11.0)
        ax.set_ylim(-2.0, 6.0)
        fig.tight_layout()
    return fig


def plot_rings_boundaries(net2, net3, X, y):
    """Demo D2: each network's probability over the plane, its p = 0.5 boundary (solid),
    and each hidden unit's line, where that unit's score is 0 (dashed)."""
    from ch_6_and_10_demos_helpers import _class_cmap, _data_axes, _grid, _points
    lim = (-2.8, 2.8)
    g1, g2, G = _grid(lim, lim, steps=250)
    t = np.linspace(*lim, 50)
    with rc_context(BIG_FONTS):
        fig, axes = subplots(1, 2, figsize=(14, 7))
        for ax, net in zip(axes, (net2, net3)):
            p = net.predict_proba(G)[:, 1].reshape(g1.shape)
            ax.contourf(g1, g2, p, levels=np.linspace(0, 1, 11), cmap=_class_cmap(), vmin=0, vmax=1)
            ax.contour(g1, g2, p, levels=[0.5], colors='black', linewidths=2.5)
            W, b = net.coefs_[0], net.intercepts_[0]
            for k in range(W.shape[1]):                  # hidden unit k's line: w . x + b = 0
                w1, w2 = W[:, k]
                if abs(w2) > abs(w1):
                    ax.plot(t, -(w1 * t + b[k]) / w2, color='0.25', ls='--', lw=1.6)
                else:
                    ax.plot(-(w2 * t + b[k]) / w1, t, color='0.25', ls='--', lw=1.6)
            wrong = net.predict(X) != y
            _points(ax, X, y, wrong=wrong, size=40)
            _data_axes(ax, lim, lim)
            h = W.shape[1]
            ax.set_title(f'{h} hidden units: {np.mean(~wrong):.0%} correct\n'
                         f'(dashed: the {h} hidden units\' lines)')
        fig.tight_layout()
    return fig


def plot_rings_h_space(net, X, y):
    """Demo D2, 3-D: each point at its three hidden outputs (h1, h2, h3), with the output
    unit's plane, where p = 0.5. A plotly figure, so it can be rotated."""
    import plotly.graph_objects as go
    from ch_6_and_10_demos_helpers import MINUS_COLOR, PLUS_COLOR
    W, b = net.coefs_[0], net.intercepts_[0]
    H = 1 / (1 + np.exp(-(X @ W + b)))                   # sigmoid hidden outputs, shape (n, 3)
    w, c = net.coefs_[1][:, 0], net.intercepts_[1][0]
    fig = go.Figure()
    for label, color, symbol, name in ((1, PLUS_COLOR, 'circle', 'disk (class 1)'),
                                       (0, MINUS_COLOR, 'diamond', 'ring (class 0)')):
        keep = y == label
        fig.add_trace(go.Scatter3d(x=H[keep, 0], y=H[keep, 1], z=H[keep, 2], mode='markers',
                                   name=name, marker=dict(size=4, color=color, symbol=symbol,
                                                          line=dict(width=0.5, color='black'))))
    k = int(np.argmax(np.abs(w)))                        # solve the plane for its steepest axis
    others = [i for i in range(3) if i != k]
    a1, a2 = np.meshgrid(np.linspace(0, 1, 30), np.linspace(0, 1, 30))
    solved = -(w[others[0]] * a1 + w[others[1]] * a2 + c) / w[k]
    solved = np.where((solved >= 0) & (solved <= 1), solved, np.nan)
    coords = [None, None, None]
    coords[others[0]], coords[others[1]], coords[k] = a1, a2, solved
    fig.add_trace(go.Surface(x=coords[0], y=coords[1], z=coords[2], showscale=False, opacity=0.45,
                             colorscale=[[0, '#666666'], [1, '#666666']],
                             name='output unit: p = 0.5'))
    fig.update_layout(width=820, height=680, margin=dict(l=0, r=0, t=50, b=0),
                      title='h-space: the 3 hidden outputs of each point (drag to rotate)',
                      scene=dict(xaxis_title='h₁', yaxis_title='h₂', zaxis_title='h₃',
                                 xaxis_range=[0, 1], yaxis_range=[0, 1], zaxis_range=[0, 1]),
                      legend=dict(x=0.02, y=0.95, font=dict(size=14)))
    return fig


def plot_bottlenecks(H_first, H_after, y, titles):
    """Demo D3: two 2-unit layers' outputs for the same images, side by side, colored by digit."""
    from matplotlib.pyplot import get_cmap
    cmap = get_cmap('tab10')
    with rc_context(BIG_FONTS):
        fig, axes = subplots(1, 2, figsize=(15, 7))
        for ax, H, title in zip(axes, (H_first, H_after), titles):
            for d in range(10):
                keep = y == d
                ax.scatter(H[keep, 0], H[keep, 1], s=3, color=cmap(d), alpha=0.5, label=str(d),
                           rasterized=True)
            for d in range(10):                          # each digit's label at its middle
                keep = y == d
                mx, my = np.median(H[keep, 0]), np.median(H[keep, 1])
                ax.text(mx, my, str(d), fontsize=18, fontweight='bold', ha='center', va='center',
                        color='black', bbox=dict(boxstyle='circle,pad=0.2', facecolor=cmap(d),
                                                 alpha=0.85, edgecolor='black'))
            ax.set_xlabel('unit 1 of the 2-unit layer')
            ax.set_ylabel('unit 2 of the 2-unit layer')
            ax.set_title(title)
        fig.tight_layout()
    return fig


def plot_transformer_ffn(d_model=768, d_ff=3072):
    """Demo D4: one transformer layer's feed-forward network, for one token's vector:
    up-projection to d_ff, an activation, down-projection back to d_model."""
    from matplotlib.patches import FancyBboxPatch
    up, down = '#0072B2', '#D55E00'
    with rc_context(BIG_FONTS):
        fig, ax = subplots(figsize=(13, 6.2))
        scale = 4.6 / d_ff
        boxes = ((0.0, d_model, f'{d_model:,}', 'token vector in'),
                 (4.2, d_ff, f'{d_ff:,}', 'after the activation'),
                 (8.4, d_model, f'{d_model:,}', 'token vector out'))
        for x0, n, label, below in boxes:
            h = n * scale
            ax.add_patch(FancyBboxPatch((x0, -h / 2), 0.9, h, boxstyle='round,pad=0.04',
                                        facecolor=_light('#009E73', 0.35), edgecolor='#009E73', lw=2))
            ax.text(x0 + 0.45, 0, label, rotation=90, ha='center', va='center', fontsize=17,
                    fontweight='bold')
            ax.text(x0 + 0.45, -2.75, below, ha='center', va='top', fontsize=14)
        for x0, x1, color, name, shape in ((1.0, 4.1, up, 'up-projection', f'{d_ff:,} × {d_model:,}'),
                                           (5.2, 8.3, down, 'down-projection', f'{d_model:,} × {d_ff:,}')):
            for y0 in (-0.6, 0.0, 0.6):
                ax.annotate('', (x1, y0 * (1.6 if color == up else 1)), (x0, y0 * (1 if color == up else 1.6)),
                            arrowprops=dict(arrowstyle='-|>', color=color, lw=2))
            ax.text((x0 + x1) / 2, 1.45, name, ha='center', fontsize=16, color=color, fontweight='bold')
            ax.text((x0 + x1) / 2, 1.05, f'weights {shape}', ha='center', fontsize=14, color=color)
        ax.text(4.65, 2.75, 'widen 4×, activation, shrink back', ha='center', fontsize=17)
        ax.set_xlim(-0.6, 9.9)
        ax.set_ylim(-3.4, 3.3)
        ax.axis('off')
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Demo S6: estimating a cross-entropy from samples of the truth
# ---------------------------------------------------------------------------

SAMPLE_Q = np.array([0.1, 0.2, 0.35, 0.35])   # beliefs that underrate sunny (no practice question uses them)
SAMPLE_COLORS = (OKABE_ITO[0], OKABE_ITO[1], OKABE_ITO[3])


def plot_sampled_cross_entropy(q=SAMPLE_Q, p=WEATHER_P, n_days=365, seed=0):
    """Demo S6: left, the running average of your surprise over three sampled years,
    settling on H(p, q); right, the estimates from many samples of 10 days and of
    100 days, both centered on H(p, q), the 10-day ones far more spread out."""
    rng = np.random.default_rng(seed)
    q = np.asarray(q, float)
    exact = float(-(p * np.log2(q)).sum())
    surprise = _bits(q)
    days = np.arange(1, n_days + 1)
    with rc_context(BIG_FONTS):
        fig, (left, right) = subplots(1, 2, figsize=(15, 5.4))
        for k, color in enumerate(SAMPLE_COLORS):
            s = surprise[rng.choice(len(p), size=n_days, p=p)]
            left.plot(days, np.cumsum(s) / days, color=color, lw=2, label=f'year {k + 1}')
        left.axhline(exact, color='black', ls='--', lw=2)
        left.text(n_days, exact + 0.03, f'exact H(p, q) = {exact:.3f}', ha='right', va='bottom',
                  fontsize=14)
        left.set_xlim(0, n_days)
        left.set_ylim(exact - 1.0, exact + 1.0)
        left.set_xlabel('days so far')
        left.set_ylabel('average surprise so far, in bits')
        left.set_title('One sampled year: the average settles down')
        left.legend(loc='lower right')

        bins = np.linspace(exact - 1.0, exact + 1.0, 26)
        for n, color in ((10, SAMPLE_COLORS[1]), (100, SAMPLE_COLORS[0])):
            est = surprise[rng.choice(len(p), size=(2000, n), p=p)].mean(axis=1)
            right.hist(est, bins=bins, color=color, alpha=0.6, edgecolor='white', lw=1,
                       label=f'samples of {n} days (std {est.std():.2f})')
        right.axvline(exact, color='black', ls='--', lw=2)
        right.set_xlabel('estimate of H(p, q), in bits')
        right.set_ylabel('number of samples (of 2,000)')
        right.set_title('Many samples: small ones are noisier')
        right.legend(loc='upper right', fontsize=13)
        for ax in (left, right):
            ax.grid(alpha=0.3)
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# NB05 practice question: two unlabeled bottlenecks
# ---------------------------------------------------------------------------

def _relu_hidden_outputs(net, X):
    """Every hidden layer's outputs of a ReLU MLPClassifier (coefs_ are (inputs, units))."""
    H, out = X, []
    for W, b in zip(net.coefs_[:-1], net.intercepts_[:-1]):
        H = np.maximum(0, H @ W + b)
        out.append(H)
    return out


def plot_bottleneck_question(mnist_nets, X, y):
    """NB05 practice question: the 2-unit layers of (64, 2) and (2, 64), titled only
    'Network A' and 'Network B', so the code doesn't give away which is which."""
    a = _relu_hidden_outputs(mnist_nets[(64, 2)], X)[1]
    b = _relu_hidden_outputs(mnist_nets[(2, 64)], X)[0]
    return plot_bottlenecks(a, b, y, titles=['Network A', 'Network B'])
