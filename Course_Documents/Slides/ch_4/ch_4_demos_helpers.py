"""Plotting and widget helpers for the Chapter 4 notebooks.

ch_4_demos.ipynb (log-odds and the logistic model) and
logistic_regression_metrics.ipynb (fitting and classification metrics) both use
this module. The notebooks stay focused on ideas; the matplotlib, plotly, and
ipywidgets details live here. The Default data is used throughout, with
`default` recoded as 0/1 in a column named `default01`.
"""
from fractions import Fraction
from io import BytesIO
from types import SimpleNamespace

import ipywidgets as widgets
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.formula.api as smf
from IPython.display import display
from ISLP import load_data
from matplotlib import get_backend, rc_context
from matplotlib.patches import Rectangle
from matplotlib.pyplot import close, figure, subplots, switch_backend
from matplotlib.ticker import PercentFormatter
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

# Load the backend now, outside any rc_context. Loading Jupyter's inline backend
# turns on interactive mode; if that first happens inside rc_context, exiting the
# context turns it back off, and later figures stop displaying automatically.
switch_backend(get_backend())

DEFAULT_COLOR = 'C3'
NO_DEFAULT_COLOR = 'C0'
DEFAULT_HEX = '#d62728'      # matplotlib's C3, for plotly figures
NO_DEFAULT_HEX = '#1f77b4'   # matplotlib's C0


def load_default():
    """Return ISLP's Default data with an added 0/1 column `default01`."""
    df = load_data('Default')
    df['default01'] = (df['default'] == 'Yes').astype(int)
    return df


def plot_default_overview(df, n_no=1000, seed=463):
    """Recreate ISL Figure 4.1: balance vs income, plus boxplots by default status.

    Every defaulter is shown, but only n_no randomly chosen non-defaulters, so the
    rare defaulters aren't buried.
    """
    yes = df[df['default'] == 'Yes']
    no = df[df['default'] == 'No'].sample(n_no, random_state=seed)

    fig, (ax_scatter, ax_balance, ax_income) = subplots(
        1, 3, figsize=(13, 4.5), gridspec_kw={'width_ratios': [2, 1, 1]})

    ax_scatter.scatter(no['balance'], no['income'], s=12, marker='o',
                       facecolors='none', edgecolors=NO_DEFAULT_COLOR, alpha=0.6,
                       label=f'No ({n_no} of {(df["default"] == "No").sum():,} shown)')
    ax_scatter.scatter(yes['balance'], yes['income'], s=14, marker='+',
                       color=DEFAULT_COLOR, label=f'Yes (all {len(yes)})')
    ax_scatter.set_xlabel('balance')
    ax_scatter.set_ylabel('income')
    ax_scatter.legend(title='default')

    for ax, column in [(ax_balance, 'balance'), (ax_income, 'income')]:
        groups = [df.loc[df['default'] == label, column] for label in ('No', 'Yes')]
        boxes = ax.boxplot(groups, tick_labels=['No', 'Yes'], patch_artist=True, widths=0.6)
        for patch, color in zip(boxes['boxes'], (NO_DEFAULT_COLOR, DEFAULT_COLOR)):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        ax.set_xlabel('default')
        ax.set_ylabel(column)

    fig.tight_layout()
    return fig


def _jittered_labels(df, jitter, seed):
    rng = np.random.default_rng(seed)
    return df['default01'] + rng.uniform(-jitter, jitter, len(df))


def _scatter_labels(ax, df, jitter=0.03, seed=463):
    """Scatter the 0/1 labels against balance with a little vertical jitter."""
    y = _jittered_labels(df, jitter, seed)
    colors = np.where(df['default01'] == 1, DEFAULT_COLOR, NO_DEFAULT_COLOR)
    ax.scatter(df['balance'], y, s=6, c=colors, alpha=0.25, linewidths=0)
    ax.set_xlabel('balance')
    ax.set_ylabel('default (No = 0, Yes = 1)')


def _shade_impossible(ax):
    """Shade the regions below 0 and above 1, where no probability can be."""
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.axhline(1, color='gray', linewidth=0.8)
    low, high = ax.get_ylim()
    ax.axhspan(low, 0, color='gray', alpha=0.15)
    ax.axhspan(1, high, color='gray', alpha=0.15)
    ax.set_ylim(low, high)


def _line_predictions(model, balances):
    return model.predict(pd.DataFrame({'balance': balances}))


def plot_linear_fit(df, model, ax=None, jitter=0.03, seed=463):
    """Plot the jittered 0/1 labels with the least squares line through them.

    The shaded bands mark values that can't be probabilities.
    """
    if ax is None:
        _, ax = subplots(figsize=(8, 5))
    _scatter_labels(ax, df, jitter, seed)

    grid = np.linspace(0, df['balance'].max(), 200)
    ax.plot(grid, _line_predictions(model, grid), color='black', linewidth=2,
            label='least squares line')
    ax.set_ylim(-0.2, 1.2)
    _shade_impossible(ax)

    share_negative = (model.predict(df[['balance']]) < 0).mean()
    ax.text(0.02, 0.06, f'{share_negative:.0%} of customers get a negative prediction',
            transform=ax.transAxes, fontsize=10)
    ax.legend(loc='upper left')
    return ax


def linear_prediction_table(model, balances=(0, 250, 500, 1000, 1500, 2000, 2500)):
    """Return the line's prediction at each balance, as a DataFrame."""
    balances = np.asarray(balances, dtype=float)
    return pd.DataFrame({'balance': balances,
                         'predicted value': _line_predictions(model, balances).round(3)})


def binned_default_rate(df, width=200):
    """Return the share of defaulters within each balance bin of the given width."""
    edges = np.arange(0, df['balance'].max() + width, width)
    centers = edges[:-1] + width / 2
    bins = pd.cut(df['balance'], edges, labels=centers, include_lowest=True)
    table = df.groupby(bins, observed=True)['default01'].agg(['mean', 'size'])
    table.index = table.index.astype(float)
    table.index.name = 'balance (bin center)'
    return table.rename(columns={'mean': 'default rate', 'size': 'customers'})


def plot_binned_vs_line(df, model, width=200, ax=None, threshold=0.5):
    """Compare the line with the actual default rate in each balance bin.

    Each dot is a bin; its area grows with the number of customers in it. The
    dashed line is the cutoff for classifying a customer as a defaulter.
    """
    if ax is None:
        _, ax = subplots(figsize=(8, 5))
    rates = binned_default_rate(df, width)

    ax.scatter(rates.index, rates['default rate'], s=10 + rates['customers'] / 5,
               color=DEFAULT_COLOR, alpha=0.7, zorder=3,
               label=f'actual default rate (bins of ${width:,})')
    grid = np.linspace(0, df['balance'].max(), 200)
    ax.plot(grid, _line_predictions(model, grid), color='black', linewidth=2,
            label='least squares line')
    ax.axhline(threshold, color='black', linestyle='--', linewidth=1,
               label=f'cutoff: predict "default" above {threshold}')
    ax.set_xlabel('balance')
    ax.set_ylabel('share of customers who defaulted')
    ax.set_ylim(-0.2, 1.2)
    _shade_impossible(ax)
    ax.legend(loc='upper left')
    return ax


def threshold_summary(df, model, threshold=0.5, high_balance=2000):
    """Compare the line's classifications with what actually happened.

    Returns a one-column DataFrame of counts and rates, for all customers and for
    customers with balance above high_balance.
    """
    flagged = model.predict(df[['balance']]) > threshold
    high = df['balance'] > high_balance
    rows = {
        'customers': len(df),
        'actual defaulters': df['default01'].sum(),
        f'flagged by the line (prediction > {threshold})': flagged.sum(),
        f'customers with balance > ${high_balance:,}': high.sum(),
        f'  ...who actually defaulted': df.loc[high, 'default01'].sum(),
        f'  ...flagged by the line': flagged[high].sum(),
    }
    return pd.DataFrame({'count': rows})


def fit_logistic(df):
    """Fit the logistic regression of default01 on balance; return the statsmodels result."""
    return smf.logit('default01 ~ balance', data=df).fit(disp=0)


def plot_linear_vs_logistic(df, linear_model, jitter=0.03, seed=463):
    """Recreate ISL Figure 4.2: the line vs the logistic curve on the same data."""
    logistic = fit_logistic(df)
    grid = np.linspace(0, df['balance'].max(), 200)
    curves = [('Linear regression', _line_predictions(linear_model, grid)),
              ('Logistic regression (coming up)', logistic.predict(pd.DataFrame({'balance': grid})))]

    fig, axes = subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, (title, values) in zip(axes, curves):
        _scatter_labels(ax, df, jitter, seed)
        ax.plot(grid, values, color='black', linewidth=2)
        ax.set_ylim(-0.2, 1.2)
        _shade_impossible(ax)
        ax.set_title(title)
    axes[1].set_ylabel('')
    fig.tight_layout()
    return fig


# --- Part 2: odds, evidence, and log-odds -------------------------------------

PRACTICE_P = (0.01, 0.10, 0.20, 0.50, 0.75, 0.80, 0.99)

# The invented bank: (clue, % of defaulters with it, % of non-defaulters with it,
# short label for having it, short label for not having it).
CLUES = [('missed a payment', 60, 20, 'missed', "didn't"),
         ('high balance', 50, 25, 'high', 'not high'),
         ('long-time customer', 20, 40, 'long-time', 'newer')]
START = (100, 900)


def odds(p):
    return p / (1 - p)


def log_odds(p):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log(p / (1 - p))


def logistic(z):
    return 1 / (1 + np.exp(-z))


def _odds_text(o):
    """Format odds as 'a : 1' when at least even, otherwise '1 : b'."""
    def number(v):
        return f'{v:,.0f}' if v >= 100 else f'{v:.3g}'
    return f'{number(o)} : 1' if o >= 1 else f'1 : {number(1 / o)}'


def _pct_text(p):
    if p < 0.001:
        return 'under 0.1%'
    if p > 0.999:
        return 'over 99.9%'
    return f'{p:.1%}'


def _ratio_text(yes, no):
    """Reduce a count ratio like 60 : 180 to lowest terms, '1 : 3'."""
    r = Fraction(yes) / Fraction(no)
    return f'{r.numerator} : {r.denominator}'


def _count_text(x):
    x = Fraction(x)
    return f'{x.numerator:,}' if x.denominator == 1 else f'{float(x):,.1f}'


def odds_explorer(p_values=None, y_max=20):
    """Interactive plot of odds against probability, with a slider for p.

    The practice-table probabilities are marked. Odds above y_max run off the top
    of the chart, but the readout still reports them.
    """
    if p_values is None:
        p_values = np.round(np.arange(0.01, 1.0, 0.01), 2)
    grid = np.linspace(0, 0.995, 400)
    practice = np.array(PRACTICE_P)

    def readout(p):
        o = odds(p)
        return f'p = {p:.0%}   →   odds = {_odds_text(o)} = {o:.3g}'

    def guide_xy(p):
        o = odds(p)
        return [0, p, p], [o, o, 0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid, y=odds(grid), mode='lines', hoverinfo='skip',
                             line=dict(color='black', width=2)))
    fig.add_trace(go.Scatter(x=practice, y=odds(practice), mode='markers+text',
                             text=[f'{p:.0%}' for p in practice],
                             textposition=['top right' if p < 0.05 else 'top left' for p in practice],
                             marker=dict(size=8, color='gray'), hoverinfo='skip'))
    gx, gy = guide_xy(p_values[0])
    fig.add_trace(go.Scatter(x=gx, y=gy, mode='lines', hoverinfo='skip',
                             line=dict(color=DEFAULT_HEX, dash='dot')))
    fig.add_trace(go.Scatter(x=[p_values[0]], y=[odds(p_values[0])], mode='markers',
                             marker=dict(size=13, color=DEFAULT_HEX), hoverinfo='skip'))

    steps = []
    for p in p_values:
        gx, gy = guide_xy(p)
        steps.append(dict(method='update', label=f'{p:.0%}',
                          args=[{'x': [gx, [p]], 'y': [gy, [odds(p)]]},
                                {'title.text': readout(p)}, [2, 3]]))

    fig.update_layout(
        height=500, showlegend=False, title=readout(p_values[0]),
        xaxis=dict(title='probability p', range=[-0.02, 1], tickformat='.0%'),
        yaxis=dict(title='odds = p / (1 − p)', range=[0, y_max]),
        annotations=[dict(x=0.93, y=y_max * 0.95, text='99% is 99 : 1,<br>far off the top',
                          showarrow=False, xanchor='right', align='right')],
        sliders=[dict(active=0, steps=steps, currentvalue=dict(prefix='p = '), pad=dict(t=50))],
    )
    return fig


def clue_counts(n_clues=3, start=START, clues=CLUES):
    """Return [(label, defaulters, non-defaulters)] at the start and after each clue.

    Counts are Fractions, so ratios reduce exactly.
    """
    d, n = Fraction(start[0]), Fraction(start[1])
    rows = [('start', d, n)]
    for name, pct_d, pct_n, *_ in clues[:n_clues]:
        d, n = d * pct_d / 100, n * pct_n / 100
        rows.append((name, d, n))
    return rows


def plot_customer_tree(n_clues=1, ax=None):
    """Draw the counting tree for the invented bank, through the first n_clues clues.

    At each clue, both groups split. The branch matching our customer continues
    straight down; the other branch is grayed out and dropped.
    """
    rows = clue_counts(n_clues)
    if ax is None:
        _, ax = subplots(figsize=(12, 1.9 + 1.5 * n_clues))
    ax.set_axis_off()
    x_d, x_n, out = -2.2, 2.2, 1.6

    def node(x, y, text, color=None):
        faded = color is None
        ax.text(x, y, text, ha='center', va='center', fontsize=10,
                color='gray' if faded else 'black',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white' if faded else color,
                          edgecolor='lightgray' if faded else 'none'))

    def edge(x0, y0, x1, y1, label='', faded=False):
        ax.plot([x0, x1], [y0, y1], color='lightgray' if faded else 'black', linewidth=1, zorder=0)
        if label:
            ax.text((x0 + x1) / 2 + 0.08, (y0 + y1) / 2, label, fontsize=9,
                    color='gray' if faded else 'black', ha='left', va='center')

    red, blue = '#f4b6b6', '#b8d4ec'
    total = START[0] + START[1]
    node(0, 0, f'{total:,} customers', 'whitesmoke')
    edge(0, 0, x_d, -1)
    edge(0, 0, x_n, -1)
    node(x_d, -1, f'{START[0]:,} defaulters', red)
    node(x_n, -1, f'{START[1]:,} non-defaulters', blue)
    ax.text(0, -1, f'odds {_ratio_text(*START)}\n(p = {START[0] / total:.0%})',
            ha='center', va='center', fontsize=10)

    for k in range(1, n_clues + 1):
        name, pct_d, pct_n, has, lacks = CLUES[k - 1]
        _, d_prev, n_prev = rows[k - 1]
        _, d, n = rows[k]
        y = -1 - k
        for x, side, prev, kept, pct, color in ((x_d, -1, d_prev, d, pct_d, red),
                                                (x_n, 1, n_prev, n, pct_n, blue)):
            edge(x, y + 1, x, y, f'{pct}%')
            edge(x, y + 1, x + side * out, y, f'{100 - pct}%', faded=True)
            node(x, y, f'{_count_text(kept)} {has}', color)
            node(x + side * out, y, f'{_count_text(prev - kept)} {lacks}')
        ax.text(0, y, f'{_count_text(d)} : {_count_text(n)} = {_ratio_text(d, n)}\n'
                      f'(p = {float(d / (d + n)):.0%})', ha='center', va='center', fontsize=10,
                fontweight='bold')
        ax.text(-6.3, y, f'clue {k}:\n{name}', ha='left', va='center', fontsize=10, style='italic')

    ax.set_xlim(-6.4, 4.6)
    ax.set_ylim(-1.5 - n_clues, 0.4)
    return ax


def plot_clue_bars():
    """Stacked bars of who's left after each clue: counts on the left, shares on the right."""
    rows = clue_counts(3)
    labels = ['start'] + [f'clue {k}:\n{name}' for k, (name, *_) in enumerate(CLUES, start=1)]
    d = np.array([float(r[1]) for r in rows])
    n = np.array([float(r[2]) for r in rows])
    x = np.arange(len(rows))

    fig, (ax_count, ax_share) = subplots(1, 2, figsize=(13, 4.8))
    ax_count.bar(x, d, color=DEFAULT_COLOR, label='defaulters')
    ax_count.bar(x, n, bottom=d, color=NO_DEFAULT_COLOR, alpha=0.7, label='non-defaulters')
    for xi, (_, di, ni) in zip(x, rows):
        ax_count.text(xi, float(di + ni) + 20, f'{_count_text(di)} : {_count_text(ni)}\n'
                                              f'= {_ratio_text(di, ni)}',
                      ha='center', va='bottom', fontsize=10)
    ax_count.set_ylim(0, 1250)
    ax_count.set_ylabel('customers still in the running')
    ax_count.set_title('Counts: the bars shrink')
    ax_count.legend(loc='upper right')

    share = d / (d + n)
    ax_share.bar(x, share, color=DEFAULT_COLOR)
    ax_share.bar(x, 1 - share, bottom=share, color=NO_DEFAULT_COLOR, alpha=0.7)
    for xi, s in zip(x, share):
        ax_share.text(xi, s + 0.02, f'{s:.0%}', ha='center', va='bottom', fontsize=11,
                      fontweight='bold', color='white')
    ax_share.yaxis.set_major_formatter(PercentFormatter(1))
    ax_share.set_ylabel('share of those still in the running')
    ax_share.set_title('Shares: only the ratio matters')

    for ax in (ax_count, ax_share):
        ax.set_xticks(x, labels)
    fig.tight_layout()
    return fig


def one_clue_explorer(strengths=(0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 10),
                      customers=(0.01, 0.10, 0.50, 0.90), start=3):
    """Interactive: one clue applied to several customers, on two scales.

    Left: new probability vs starting probability. Right: the same on log-odds.
    The slider sets the clue's strength; arrows run from each customer's starting
    value to their new value.
    """
    customers = np.array(customers)
    grid = np.linspace(0.001, 0.999, 300)
    z_grid = np.linspace(-5, 5, 2)
    z_customers = log_odds(customers)

    def new_p(p, s):
        o = odds(p) * s
        return o / (1 + o)

    def arrows(xs, y0s, y1s):
        ax_, ay_ = [], []
        for x, y0, y1 in zip(xs, y0s, y1s):
            ax_ += [x, x, None]
            ay_ += [y0, y1, None]
        return ax_, ay_

    def changing(s):
        """x, y, and text for traces 1-3 (probability) and 5-7 (log-odds)."""
        p1 = new_p(customers, s)
        shift = np.log(s)
        la_x, la_y = arrows(customers, customers, p1)
        ra_x, ra_y = arrows(z_customers, z_customers, z_customers + shift)
        xs = [grid, la_x, customers, z_grid, ra_x, z_customers]
        ys = [new_p(grid, s), la_y, p1, z_grid + shift, ra_y, z_customers + shift]
        texts = [None, None, [f'{(b - a) * 100:+.1f} pts' for a, b in zip(customers, p1)],
                 None, None, [f'{shift:+.2f}'] * len(customers)]
        return xs, ys, texts

    def readout(s):
        return (f'Strength {s:g}: probability moves by different amounts, '
                f'but log-odds always moves by ln {s:g} = {np.log(s):+.2f}')

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                        subplot_titles=('Probability scale', 'Log-odds scale'))
    s0 = strengths[list(strengths).index(start)] if start in strengths else strengths[0]
    xs, ys, texts = changing(s0)
    arrow_line = dict(color=DEFAULT_HEX, width=3)
    label_style = dict(mode='markers+text', textposition='middle right',
                       marker=dict(size=9, color=DEFAULT_HEX), hoverinfo='skip')

    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', hoverinfo='skip',
                             line=dict(color='gray', dash='dash')), 1, 1)
    fig.add_trace(go.Scatter(x=xs[0], y=ys[0], mode='lines', hoverinfo='skip',
                             line=dict(color='black', width=2)), 1, 1)
    fig.add_trace(go.Scatter(x=xs[1], y=ys[1], mode='lines', hoverinfo='skip', line=arrow_line), 1, 1)
    fig.add_trace(go.Scatter(x=xs[2], y=ys[2], text=texts[2], **label_style), 1, 1)
    fig.add_trace(go.Scatter(x=[-5, 5], y=[-5, 5], mode='lines', hoverinfo='skip',
                             line=dict(color='gray', dash='dash')), 1, 2)
    fig.add_trace(go.Scatter(x=xs[3], y=ys[3], mode='lines', hoverinfo='skip',
                             line=dict(color='black', width=2)), 1, 2)
    fig.add_trace(go.Scatter(x=xs[4], y=ys[4], mode='lines', hoverinfo='skip', line=arrow_line), 1, 2)
    fig.add_trace(go.Scatter(x=xs[5], y=ys[5], text=texts[5], **label_style), 1, 2)

    moving = [1, 2, 3, 5, 6, 7]
    steps = []
    for s in strengths:
        xs, ys, texts = changing(s)
        steps.append(dict(method='update', label=f'{s:g}',
                          args=[{'x': xs, 'y': ys, 'text': texts}, {'title.text': readout(s)},
                                moving]))

    fig.update_layout(
        height=520, showlegend=False, title=readout(s0),
        xaxis=dict(title='starting probability', range=[0, 1.12], tickformat='.0%'),
        yaxis=dict(title='new probability', range=[0, 1.05], tickformat='.0%'),
        xaxis2=dict(title='starting log-odds', range=[-5, 6.2]),
        yaxis2=dict(title='new log-odds', range=[-6.5, 7.5]),
        sliders=[dict(active=list(strengths).index(s0), steps=steps,
                      currentvalue=dict(prefix='strength = '), pad=dict(t=50))],
    )
    return fig


def plot_evidence_scoreboard(ax=None):
    """Waterfall chart of the log-odds tally for the three clues.

    The left axis is log-odds; the right axis labels the same heights as probabilities.
    """
    if ax is None:
        _, ax = subplots(figsize=(10, 5.5))
    start = float(np.log(START[0] / START[1]))
    steps = [(name, float(np.log(pct_d / pct_n)), pct_d / pct_n) for name, pct_d, pct_n, *_ in CLUES]

    ax.bar(0, start, color='gray', width=0.6)
    ax.text(0, start - 0.05, f'{start:.2f}', ha='center', va='top', fontsize=11)
    level = start
    for k, (name, step, strength) in enumerate(steps, start=1):
        ax.plot([k - 1 + 0.3, k - 0.3], [level, level], color='gray', linestyle=':')
        ax.bar(k, step, bottom=level, width=0.6,
               color=DEFAULT_COLOR if step > 0 else NO_DEFAULT_COLOR)
        top = level + step
        ax.text(k, max(level, top) + 0.05, f'{step:+.2f}', ha='center', va='bottom', fontsize=11)
        level = top
    k_total = len(steps) + 1
    ax.plot([k_total - 1 + 0.3, k_total - 0.3], [level, level], color='gray', linestyle=':')
    ax.bar(k_total, level, color='gray', alpha=0.6, width=0.6)
    ax.text(k_total, level - 0.05, f'{level:.2f}\n(p = {logistic(level):.0%})',
            ha='center', va='top', fontsize=11)

    labels = ['baseline\n(odds 1 : 9)'] + [f'{name}\n(× {strength:g})' for name, _, strength in steps]
    ax.set_xticks(range(k_total + 1), labels + ['total'])
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylim(-2.75, 0.35)
    ax.set_ylabel('log-odds (evidence score)')

    probability = ax.secondary_yaxis('right', functions=(logistic, log_odds))
    probability.set_yticks([0.07, 0.10, 0.15, 0.25, 0.40, 0.50])
    probability.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    probability.set_ylabel('probability of default')
    ax.set_title('Each clue adds its own points to the score')
    return ax


def plot_three_scales(ps=(0.01, 0.10, 0.50, 0.90, 0.99), odds_max=100):
    """Three stacked number lines (probability, odds, log-odds) with matching points connected."""
    ps = np.array(ps)
    scales = [
        ('probability', ps, (0, 1), [0, 0.25, 0.5, 0.75, 1], lambda v: f'{v:.0%}'),
        ('odds', odds(ps), (0, odds_max), [0, 20, 40, 60, 80, 100], lambda v: f'{v:g}'),
        ('log-odds', log_odds(ps), (-5, 5), [-4, -2, 0, 2, 4], lambda v: f'{v:+.2f}' if v else '0'),
    ]
    heights = [2, 1, 0]
    colors = [f'C{i}' for i in range(len(ps))]

    _, ax = subplots(figsize=(11, 4.8))
    ax.set_axis_off()
    positions = []
    for (name, values, (lo, hi), ticks, fmt), y in zip(scales, heights):
        to_pos = lambda v, lo=lo, hi=hi: (np.asarray(v) - lo) / (hi - lo)
        positions.append(to_pos(values))
        ax.plot([0, 1], [y, y], color='black', linewidth=1)
        for t in ticks:
            ax.plot([to_pos(t)] * 2, [y - 0.04, y + 0.04], color='black', linewidth=1)
            ax.text(to_pos(t), y - 0.1, f'{t:g}' if name != 'probability' else f'{t:.0%}',
                    ha='center', va='top', fontsize=8, color='gray')
        ax.text(-0.03, y, name, ha='right', va='center', fontsize=11, fontweight='bold')
        for pos, v, c in zip(to_pos(values), values, colors):
            ax.scatter(pos, y, s=60, color=c, zorder=3)
            if name != 'odds' or v > 5:
                ax.text(pos, y + 0.1, fmt(v), ha='center', va='bottom', fontsize=9, color=c)

    for k, c in enumerate(colors):
        ax.plot([positions[0][k], positions[1][k], positions[2][k]], heights, color=c,
                alpha=0.4, linewidth=1, zorder=1)
    ax.annotate('1%, 10%, and 50% are all\nsqueezed into [0, 1] here', xy=(0.005, 1.05),
                xytext=(0.12, 1.45), fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))
    ax.set_xlim(-0.16, 1.04)
    ax.set_ylim(-0.4, 2.4)
    return ax


def _standardized_balance(df):
    b = df['balance']
    return (b - b.mean()) / b.std()


def fit_logistic_standardized(df):
    """Fit default01 on standardized balance; return (b0, b1)."""
    data = pd.DataFrame({'default01': df['default01'], 'x': _standardized_balance(df)})
    result = smf.logit('default01 ~ x', data=data).fit(disp=0)
    return result.params['Intercept'], result.params['x']


def two_views_explorer(df, vary='b0', b0=None, b1=None, marker=0.5, values=None,
                       jitter=0.03, seed=463):
    """Interactive: the same logistic model as a line in log-odds and a curve in probability.

    Balance is standardized (0 = average, 1 = one SD above average). The slider
    varies one thing while everything else stays fixed:
      vary='b0'      the intercept, with slope b1
      vary='b1'      the slope, with intercept b0
      vary='marker'  where the marker sits, with the model fixed at (b0, b1)
    b0 and b1 default to the values fit to the data.
    """
    b0_fit, b1_fit = fit_logistic_standardized(df)
    b0 = round(float(b0_fit), 2) if b0 is None else b0
    b1 = round(float(b1_fit), 2) if b1 is None else b1
    if values is None:
        values = {'b0': np.round(np.arange(-8, 2.01, 0.25), 2),
                  'b1': np.round(np.arange(-2, 6.01, 0.2), 1),
                  'marker': np.round(np.arange(-1.5, 3.01, 0.25), 2)}[vary]
    names = {'b0': 'β₀', 'b1': 'β₁', 'marker': 'marker at x'}

    x_data = _standardized_balance(df).to_numpy()
    y_data = _jittered_labels(df, jitter, seed).to_numpy()
    grid = np.linspace(x_data.min(), x_data.max(), 200)

    def settings(v):
        return {'b0': (v, b1, marker), 'b1': (b0, v, marker), 'marker': (b0, b1, v)}[vary]

    def changing(v):
        c0, c1, m = settings(v)
        mx = np.array([m, m + 1])
        z = c0 + c1 * mx
        return ([grid, mx, grid, mx],
                [c0 + c1 * grid, z, logistic(c0 + c1 * grid), logistic(z)])

    def readout(v):
        c0, c1, m = settings(v)
        z0, z1 = c0 + c1 * m, c0 + c1 * (m + 1)
        p0, p1 = logistic(z0), logistic(z1)
        return (f'β₀ = {c0:.2f}, β₁ = {c1:.2f}.   At x = {m:.2f}: log-odds {z0:.2f}, '
                f'odds {_odds_text(np.exp(z0))}, p = {_pct_text(p0)}<br>'
                f'One unit right multiplies the odds by e<sup>β₁</sup> = {np.exp(c1):.3g}, '
                f'so p goes from {_pct_text(p0)} to {_pct_text(p1)} '
                f'({(p1 - p0) * 100:+.1f} points)')

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                        subplot_titles=('Evidence view: a straight line',
                                        'Probability view: an S-curve'))
    colors = np.where(df['default01'] == 1, DEFAULT_HEX, NO_DEFAULT_HEX)
    x_range = [float(x_data.min()), float(x_data.max())]

    fig.add_trace(go.Scatter(x=x_range, y=[0, 0], mode='lines', hoverinfo='skip',
                             line=dict(color='gray', dash='dash')), 1, 1)
    fig.add_trace(go.Scattergl(x=x_data, y=y_data, mode='markers', hoverinfo='skip',
                               marker=dict(size=3, color=colors, opacity=0.3)), 1, 2)
    fig.add_trace(go.Scatter(x=x_range, y=[0.5, 0.5], mode='lines', hoverinfo='skip',
                             line=dict(color='gray', dash='dash')), 1, 2)
    current = {'b0': b0, 'b1': b1, 'marker': marker}[vary]
    active = int(np.argmin(np.abs(np.asarray(values) - current)))
    xs, ys = changing(values[active])
    marker_style = dict(mode='markers+lines', hoverinfo='skip',
                        marker=dict(size=11, color='black', symbol=['circle', 'circle-open']),
                        line=dict(color='black', dash='dot'))
    fig.add_trace(go.Scatter(x=xs[0], y=ys[0], mode='lines', hoverinfo='skip',
                             line=dict(color='black', width=3)), 1, 1)
    fig.add_trace(go.Scatter(x=xs[1], y=ys[1], **marker_style), 1, 1)
    fig.add_trace(go.Scatter(x=xs[2], y=ys[2], mode='lines', hoverinfo='skip',
                             line=dict(color='black', width=3)), 1, 2)
    fig.add_trace(go.Scatter(x=xs[3], y=ys[3], **marker_style), 1, 2)

    moving = [3, 4, 5, 6]
    steps = []
    for v in values:
        xs, ys = changing(v)
        steps.append(dict(method='update', label=f'{v:g}',
                          args=[{'x': xs, 'y': ys}, {'title.text': readout(v)}, moving]))

    x_title = 'balance (standardized: 0 = average, 1 = one SD above)'
    fig.update_layout(
        height=560, showlegend=False, title=dict(text=readout(values[active]), font=dict(size=13)),
        xaxis=dict(title=x_title, range=x_range),
        yaxis=dict(title='log-odds (evidence score)', range=[-15, 15]),
        xaxis2=dict(title=x_title, range=x_range),
        yaxis2=dict(title='probability of default', range=[-0.1, 1.1]),
        annotations=list(fig.layout.annotations) + [
            dict(x=x_range[0], y=0, xref='x', yref='y', text='even odds', showarrow=False,
                 xanchor='left', yanchor='bottom', font=dict(color='gray'))],
        sliders=[dict(active=active, steps=steps, currentvalue=dict(prefix=f'{names[vary]} = '),
                      pad=dict(t=60))],
        margin=dict(t=110),
    )
    return fig


def plot_linear_in_odds(b0=0.5, b1=1.0, x_range=(-2, 3)):
    """Show why a line in odds fails and a line in log-odds doesn't.

    Top row: odds of default as a line, and the odds of no default it implies.
    Bottom row: the same comparison on the log-odds scale.
    """
    x = np.linspace(*x_range, 1000)
    line = b0 + b1 * x
    root = -b0 / b1
    with np.errstate(divide='ignore'):
        flipped = np.where(np.abs(line) > 0.02, 1 / line, np.nan)

    fig, axes = subplots(2, 2, figsize=(12, 7.5), sharex=True)
    panels = [
        (axes[0, 0], line, 'odds of default = β₀ + β₁X\n(a line, by assumption)', 'odds', (-2, 4)),
        (axes[0, 1], flipped, 'odds of no default = 1 / (β₀ + β₁X)\n(not a line)', 'odds', (-6, 6)),
        (axes[1, 0], line, 'log-odds of default = β₀ + β₁X', 'log-odds', (-2, 4)),
        (axes[1, 1], -line, 'log-odds of no default = −β₀ − β₁X\n(still a line)', 'log-odds', (-4, 2)),
    ]
    for ax, y, title, ylabel, ylim in panels:
        ax.plot(x, y, color='black', linewidth=2)
        ax.axhline(0, color='gray', linewidth=0.8)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        if ylabel == 'odds':
            ax.axhspan(ylim[0], 0, color='gray', alpha=0.15)
            ax.text(x_range[1] - 0.1, ylim[0] * 0.85, 'negative odds: impossible',
                    fontsize=9, color='gray', ha='right')
    axes[0, 1].axvline(root, color='gray', linestyle=':')
    for ax in axes[1]:
        ax.set_xlabel('X')
    fig.tight_layout()
    return fig


def logistic_table(result):
    """Return the coefficient table of a fitted statsmodels logit, like ISL Table 4.1."""
    return pd.DataFrame({'coefficient': result.params, 'std. error': result.bse,
                         'z-statistic': result.tvalues, 'p-value': result.pvalues}).round(4)


def next_100_table(result, balances=range(1000, 2501, 250), step=100):
    """For each balance: the fitted log-odds, odds, and probability, and what one more step does."""
    b0, b1 = result.params['Intercept'], result.params['balance']
    balances = np.asarray(balances, dtype=float)
    z = b0 + b1 * balances
    p, p_next = logistic(z), logistic(z + b1 * step)
    return pd.DataFrame({
        'balance': balances.astype(int),
        'log-odds': z.round(2),
        'odds': np.exp(z).round(4),
        'probability': p.round(4),
        f'odds × (next ${step})': np.full(len(z), np.exp(b1 * step)).round(3),
        f'change in p (next ${step}, points)': ((p_next - p) * 100).round(2),
    })


def plot_next_100(result, step=100, max_balance=2700):
    """Left: the fitted probability curve. Right: the change in probability across one step.

    Each step is centered on the balance it's plotted at, so the peak lines up
    with the 50% point.
    """
    b0, b1 = result.params['Intercept'], result.params['balance']
    grid = np.linspace(0, max_balance, 400)
    p = logistic(b0 + b1 * grid)
    jump = (logistic(b0 + b1 * (grid + step / 2)) - logistic(b0 + b1 * (grid - step / 2))) * 100
    middle = -b0 / b1

    fig, (ax_p, ax_jump) = subplots(1, 2, figsize=(13, 4.5))
    ax_p.plot(grid, p, color='black', linewidth=2)
    ax_p.yaxis.set_major_formatter(PercentFormatter(1))
    ax_p.set_ylabel('fitted probability of default')
    ax_p.set_title('The fitted model')

    ax_jump.plot(grid, jump, color=DEFAULT_COLOR, linewidth=2)
    ax_jump.set_ylabel(f'change in probability across ${step} (points)')
    ax_jump.set_title(f'Every +${step} multiplies the odds by {np.exp(b1 * step):.2f}, but...')

    for ax in (ax_p, ax_jump):
        ax.axvline(middle, color='gray', linestyle=':')
        ax.set_xlabel('balance')
    ax_p.text(middle + 30, 0.05, f'p = 50% at\n${middle:,.0f}', fontsize=9, color='gray')
    fig.tight_layout()
    return fig


def simulate_overlapping_clues(n=10_000, base_rate=0.10, seed=463):
    """Simulate customers where "missed two payments" only happens to people who missed one.

    Defaulters miss one payment 60% of the time and non-defaulters 20%. Of those who
    missed one, 70% of defaulters and 25% of non-defaulters also missed a second.
    """
    rng = np.random.default_rng(seed)
    default = rng.random(n) < base_rate
    missed_one = rng.random(n) < np.where(default, 0.60, 0.20)
    missed_two = missed_one & (rng.random(n) < np.where(default, 0.70, 0.25))
    return pd.DataFrame({'default01': default.astype(int),
                         'missed_one': missed_one.astype(int),
                         'missed_two': missed_two.astype(int)})


def _clue_rates(df, clue):
    yes, no = df[df['default01'] == 1], df[df['default01'] == 0]
    return yes[clue].mean(), no[clue].mean()


def clue_strengths(df, clues=('missed_one', 'missed_two')):
    """Return each clue's rate among defaulters and non-defaulters, and its strength."""
    rows = {}
    for c in clues:
        rate_yes, rate_no = _clue_rates(df, c)
        rows[c] = {'% of defaulters': rate_yes * 100, '% of non-defaulters': rate_no * 100,
                   'strength': rate_yes / rate_no}
    return pd.DataFrame(rows).T.round(2)


def overlap_comparison(df, clues=('missed_one', 'missed_two')):
    """Three answers for the odds of default for a customer with both clues."""
    start = df['default01'].mean() / (1 - df['default01'].mean())
    naive = start
    for c in clues:
        rate_yes, rate_no = _clue_rates(df, c)
        naive *= rate_yes / rate_no
    both = df[(df[clues[0]] == 1) & (df[clues[1]] == 1)]
    fit = smf.logit(f'default01 ~ {clues[0]} + {clues[1]}', data=df).fit(disp=0)

    answers = {
        'multiply both strengths (naive)': naive,
        'count customers with both clues': both['default01'].mean() / (1 - both['default01'].mean()),
        'logistic regression on both clues': np.exp(fit.params.sum()),
    }
    return pd.DataFrame({'odds': answers,
                         'probability': {k: o / (1 + o) for k, o in answers.items()}}).round(3)


# ---------------------------------------------------------------------------
# logistic_regression_metrics.ipynb: fitting, then measuring a classifier
# ---------------------------------------------------------------------------
# These demos use matplotlib inside ipywidgets Output areas, so they work the
# same in Jupyter, VS Code, and Colab. Fonts are set per figure with
# rc_context so the ch_4_demos.ipynb figures are unaffected.

BIG_FONTS = {'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 15,
             'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 13}

# Okabe-Ito colors, which stay distinguishable under common color blindness.
CLASS_COLORS = {1: '#E69F00', 0: '#56B4E9'}
CLASS_LABELS = {1: 'defaulted (1)', 0: 'did not default (0)'}
OUTCOMES = ('TP', 'FN', 'FP', 'TN')
OUTCOME_COLORS = {'TP': '#009E73', 'FN': '#D55E00', 'FP': '#CC79A7', 'TN': '#999999'}
OUTCOME_LABELS = {'TP': 'caught defaulters', 'FN': 'missed defaulters',
                  'FP': 'false alarms', 'TN': 'correctly left alone'}

NEVER_FLAG = 1.01
BAD_GUESS = (-4.2, 1.3)
BANK_COSTS = (2000, 200)
EQUAL_COSTS = (1000, 1000)
THRESHOLDS = np.round(np.arange(0.01, 1.0, 0.01), 2)

HTML_STYLE = 'font-size: 18px; line-height: 1.5;'


def fit_balance_model(df, jitter=0.06, seed=463):
    """Fit default01 on standardized balance with scikit-learn, and precompute everything
    the demos need: x, y, fitted probabilities p, coefficients, and jittered y for plotting.

    C=np.inf turns off scikit-learn's default penalty, so this is the same
    maximum-likelihood fit as statsmodels and ISL.
    """
    x = _standardized_balance(df).to_numpy()
    y = df['default01'].to_numpy()
    model = LogisticRegression(C=np.inf).fit(x[:, None], y)
    rng = np.random.default_rng(seed)
    return SimpleNamespace(
        x=x, y=y, p=model.predict_proba(x[:, None])[:, 1],
        b0=model.intercept_[0], b1=model.coef_[0, 0],
        y_jitter=y + rng.uniform(-jitter, jitter, len(y)),
        balance_mean=df['balance'].mean(), balance_sd=df['balance'].std(),
        model=model)


def log_likelihood(b0, b1, x, y):
    """Sum of log P(what actually happened) under the curve p = logistic(b0 + b1 x)."""
    z = b0 + b1 * x
    return float(np.sum(y * z - np.logaddexp(0, z)))


def _surprise(b0, b1, x, y):
    """-log P(what actually happened), per point: 0 means no surprise at all."""
    z = b0 + b1 * x
    return np.logaddexp(0, z) - y * z


def _figure_area():
    return widgets.Image(format='png', layout=widgets.Layout(max_width='100%'))


def _show(out, draw):
    """Redraw a matplotlib figure into a _figure_area() Image widget.

    Deliberately avoids ipywidgets.Output: some notebook frontends (VS Code, Cursor)
    also echo Output-captured figures into the cell, so the demo appears twice."""
    with rc_context(BIG_FONTS):
        fig = draw()
    buffer = BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    close(fig)
    out.value = buffer.getvalue()


def _html(text):
    return f'<div style="{HTML_STYLE}">{text}</div>'


def _plot_classes(ax, fit, ms=3):
    for k in (0, 1):
        m = fit.y == k
        ax.plot(fit.x[m], fit.y_jitter[m], 'o', ms=ms, alpha=0.4, mec='none',
                color=CLASS_COLORS[k], label=CLASS_LABELS[k])
    ax.set_xlabel('balance (standardized)')
    ax.set_ylabel('P(default)')
    ax.set_ylim(-0.1, 1.1)


def _class_legend(ax):
    """Legend with the two class markers enlarged, since the plotted dots are tiny."""
    legend = ax.legend(loc='center left')
    for handle in legend.legend_handles[:2]:
        handle.set_markersize(10)
        handle.set_alpha(1)


def _x_grid(fit):
    return np.linspace(fit.x.min() - 0.1, fit.x.max() + 0.1, 300)


# --- Section 0 and 1: where the curve comes from ----------------------------

def plot_recap(fit):
    """The fitted curve over the jittered 0/1 data, with the fitted formula in the title."""
    with rc_context(BIG_FONTS):
        fig, ax = subplots(figsize=(11, 5.5))
        _plot_classes(ax, fit)
        grid = _x_grid(fit)
        ax.plot(grid, logistic(fit.b0 + fit.b1 * grid), color='black', lw=3,
                label='fitted logistic curve')
        ax.set_title(rf'$\hat p(x) = 1 / (1 + e^{{-({fit.b0:.2f} + {fit.b1:.2f}x)}})$')
        _class_legend(ax)
        fig.tight_layout()
    return fig


def which_curve_demo(fit, bad=BAD_GUESS):
    """Two unlabeled candidate curves. Reveal shows which is the fit and each curve's
    log-likelihood."""
    curves = {'A': bad, 'B': (fit.b0, fit.b1)}
    out = _figure_area()
    reveal = widgets.ToggleButton(description='Reveal', button_style='info')

    def draw():
        fig, axes = subplots(1, 2, figsize=(15, 5.5), sharey=True)
        grid = _x_grid(fit)
        for ax, (name, (b0, b1)) in zip(axes, curves.items()):
            _plot_classes(ax, fit)
            ax.plot(grid, logistic(b0 + b1 * grid), color='black', lw=3)
            title = f'Curve {name}'
            if reveal.value:
                who = 'the computer\'s fit' if name == 'B' else 'a guess'
                title += f': {who}\nlog-likelihood = {log_likelihood(b0, b1, fit.x, fit.y):,.0f}'
            ax.set_title(title)
        _class_legend(axes[0])
        axes[1].set_ylabel('')
        fig.tight_layout()
        return fig

    reveal.observe(lambda change: _show(out, draw), names='value')
    _show(out, draw)
    display(widgets.VBox([reveal, out]))


def hand_fit_demo(fit, start=(-3.0, 1.0)):
    """Fit the curve by hand with b0 and b1 sliders, scored by log-likelihood.
    Points the curve finds surprising grow larger."""
    b0 = widgets.FloatSlider(value=start[0], min=-10, max=0, step=0.1, description='β₀',
                             layout=widgets.Layout(width='500px'))
    b1 = widgets.FloatSlider(value=start[1], min=0, max=5, step=0.05, description='β₁',
                             layout=widgets.Layout(width='500px'))
    reveal = widgets.Button(description="Reveal computer's fit", button_style='info',
                            layout=widgets.Layout(width='220px'))
    readout = widgets.HTML()
    out = _figure_area()
    state = {'best': None, 'revealing': False, 'snapping': False}

    def draw():
        fig, ax = subplots(figsize=(11, 5.5))
        _plot_classes(ax, fit, ms=2)
        s = _surprise(b0.value, b1.value, fit.x, fit.y)
        big = s > 1
        ax.scatter(fit.x[big], fit.y_jitter[big], s=np.clip(12 * s[big], 12, 250),
                   c=[CLASS_COLORS[k] for k in fit.y[big]], edgecolors='black', linewidths=0.6,
                   alpha=0.8)
        grid = _x_grid(fit)
        ax.plot(grid, logistic(b0.value + b1.value * grid), color='black', lw=3)
        ax.set_title('Bigger points = more surprised (the curve gave what happened < 37%)')
        _class_legend(ax)
        fig.tight_layout()
        return fig

    def update(change=None):
        ll = log_likelihood(b0.value, b1.value, fit.x, fit.y)
        if not state['revealing'] and (state['best'] is None or ll > state['best'][0]):
            state['best'] = (ll, b0.value, b1.value)
        best_ll, best_b0, best_b1 = state['best']
        text = (f'<b>Log-likelihood: {ll:,.1f}</b> (higher, closer to 0, is better)<br>'
                f'Your best so far: {best_ll:,.1f} at β₀ = {best_b0:.1f}, β₁ = {best_b1:.2f}')
        if state['revealing']:
            text += (f"<br>Computer's fit: β₀ = {fit.b0:.2f}, β₁ = {fit.b1:.2f}, "
                     f'log-likelihood {log_likelihood(fit.b0, fit.b1, fit.x, fit.y):,.1f}')
        readout.value = _html(text)
        _show(out, draw)

    def snap(button):
        state['snapping'] = True
        b0.value, b1.value = fit.b0, fit.b1
        state['snapping'], state['revealing'] = False, True
        update()

    def moved(change):
        if not state['snapping']:
            state['revealing'] = False
            update()

    b0.observe(moved, names='value')
    b1.observe(moved, names='value')
    reveal.on_click(snap)
    update()
    display(widgets.VBox([widgets.HBox([widgets.VBox([b0, b1]), reveal]), readout, out]))


# --- Section 2 to 5: outcomes, the confusion matrix, and metrics -------------

def outcome_codes(y, p, threshold):
    """Label each customer TP, FN, FP, or TN. Flag (predict default) when p >= threshold."""
    flagged = p >= threshold
    return np.select([flagged & (y == 1), ~flagged & (y == 1), flagged & (y == 0)],
                     ['TP', 'FN', 'FP'], 'TN')


def outcome_counts(y, p, threshold):
    codes = outcome_codes(y, p, threshold)
    return {k: int((codes == k).sum()) for k in OUTCOMES}


def _ratio(a, b):
    return a / b if b else np.nan


def metric_values(counts):
    tp, fn, fp, tn = (counts[k] for k in OUTCOMES)
    return {'accuracy': _ratio(tp + tn, tp + fn + fp + tn),
            'recall': _ratio(tp, tp + fn),
            'precision': _ratio(tp, tp + fp),
            'specificity': _ratio(tn, tn + fp),
            'false positive rate': _ratio(fp, tn + fp),
            'F1': _ratio(2 * tp, 2 * tp + fp + fn)}


def _metric_text(v):
    return 'undefined' if np.isnan(v) else f'{v:.3f}'


def _draw_outcome_scatter(ax, fit, threshold, show_counts=(), legend=True):
    """x = predicted probability, y = actual outcome (jittered), split into the four
    outcome regions by a vertical threshold line."""
    line = min(threshold, 1.0)
    codes = outcome_codes(fit.y, fit.p, threshold)
    counts = {k: int((codes == k).sum()) for k in OUTCOMES}
    lo, hi = -0.02, 1.02
    regions = {'TP': (line, hi, 0.5, 1.5), 'FN': (lo, line, 0.5, 1.5),
               'FP': (line, hi, -0.5, 0.5), 'TN': (lo, line, -0.5, 0.5)}
    for k, (x0, x1, y0, y1) in regions.items():
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color=OUTCOME_COLORS[k],
                               alpha=0.12, lw=0))
        if k in show_counts:
            ax.text((x0 + x1) / 2, y1 - 0.12, f'{counts[k]:,}', ha='center', va='top',
                    fontsize=24, fontweight='bold', color=OUTCOME_COLORS[k])
    for k in ('TN', 'FP', 'FN', 'TP'):
        m = codes == k
        ax.plot(fit.p[m], fit.y_jitter[m], 'o', ms=3.5, alpha=0.6, mec='none',
                color=OUTCOME_COLORS[k], label=f'{OUTCOME_LABELS[k]} ({k})')
    ax.axvline(line, color='black', lw=2, ls='--')
    label = 'never flag' if threshold > 1 else f'threshold = {threshold:.2f}'
    ax.text(line, 1.53, label, ha='center', va='bottom', fontsize=14)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([0, 1], ['did not\ndefault', 'defaulted'])
    ax.set_xlabel('predicted probability of default')
    if legend:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2, markerscale=3,
                  frameon=False)
    return counts


def plot_outcome_scatter(fit, threshold=0.5):
    """The outcome-colored scatter on its own."""
    with rc_context(BIG_FONTS):
        fig, ax = subplots(figsize=(10, 6.5))
        _draw_outcome_scatter(ax, fit, threshold)
        fig.tight_layout()
    return fig


CELLS = {'TN': (0, 0), 'FP': (0, 1), 'FN': (1, 0), 'TP': (1, 1)}


def _draw_confusion(ax, counts, shown=OUTCOMES, numerator=None, denominator=None):
    """2x2 confusion matrix. Rows = actual, columns = predicted (scikit-learn's layout).
    With numerator/denominator given, highlight those cells and fade the rest."""
    for k, (r, c) in CELLS.items():
        if k not in shown:
            face, alpha, edge, lw, weight, text_color = 'white', 1, '#bbbbbb', 1.5, 'normal', '#999999'
        elif numerator is None:
            face, alpha, edge, lw, weight, text_color = OUTCOME_COLORS[k], 0.75, 'white', 2, 'bold', 'black'
        elif k in numerator:
            face, alpha, edge, lw, weight, text_color = OUTCOME_COLORS[k], 0.9, 'black', 4, 'bold', 'black'
        elif k in denominator:
            face, alpha, edge, lw, weight, text_color = OUTCOME_COLORS[k], 0.3, 'black', 4, 'normal', 'black'
        else:
            face, alpha, edge, lw, weight, text_color = OUTCOME_COLORS[k], 0.06, '#dddddd', 1.5, 'normal', '#bbbbbb'
        ax.add_patch(Rectangle((c, r), 1, 1, facecolor=face, alpha=alpha, lw=0))
        ax.add_patch(Rectangle((c, r), 1, 1, facecolor='none', edgecolor=edge, lw=lw))
        number = f'{counts[k]:,}' if k in shown else '?'
        ax.text(c + 0.5, r + 0.2, k, ha='center', va='center', fontsize=15, color=text_color)
        ax.text(c + 0.5, r + 0.5, number, ha='center', va='center', fontsize=30,
                fontweight=weight, color=text_color)
        ax.text(c + 0.5, r + 0.82, OUTCOME_LABELS[k].replace(' ', '\n', 1), ha='center',
                va='center', fontsize=13, color=text_color, linespacing=1.0)
    ax.text(0.5, -0.08, 'not flagged', ha='center', va='bottom', fontsize=15)
    ax.text(1.5, -0.08, 'flagged', ha='center', va='bottom', fontsize=15)
    ax.text(1.0, -0.28, 'predicted', ha='center', va='bottom', fontsize=15, fontweight='bold')
    ax.text(-0.06, 0.5, "didn't\ndefault", ha='right', va='center', fontsize=15)
    ax.text(-0.06, 1.5, 'defaulted', ha='right', va='center', fontsize=15)
    ax.text(-0.5, 1.0, 'actual', ha='right', va='center', fontsize=15, fontweight='bold',
            rotation=90)
    ax.set_xlim(-0.85, 2.02)
    ax.set_ylim(2.02, -0.42)
    ax.set_aspect('equal')
    ax.axis('off')


def confusion_build_demo(fit, threshold=0.5):
    """Count the four scatter regions into the confusion matrix, one cell per click."""
    order = ('TP', 'FN', 'FP', 'TN')
    state = {'n': 0}
    count = widgets.Button(description='Count next cell', button_style='info')
    reset = widgets.Button(description='Reset')
    which = widgets.ToggleButtons(options=[f'Model at {threshold}', 'Always "no"'])
    out = _figure_area()

    def draw():
        t = threshold if which.index == 0 else NEVER_FLAG
        shown = order[:state['n']]
        fig, (ax_s, ax_m) = subplots(1, 2, figsize=(16, 6.5), width_ratios=[1.25, 1])
        counts = _draw_outcome_scatter(ax_s, fit, t, show_counts=shown)
        _draw_confusion(ax_m, counts, shown=shown)
        fig.tight_layout()
        return fig

    def step(button):
        state['n'] = min(state['n'] + 1, 4)
        _show(out, draw)

    def restart(button):
        state['n'] = 0
        _show(out, draw)

    count.on_click(step)
    reset.on_click(restart)
    which.observe(lambda change: _show(out, draw), names='value')
    _show(out, draw)
    display(widgets.VBox([widgets.HBox([count, reset, which]), out]))


METRICS = {
    'Accuracy': ('accuracy', 'What fraction of all our decisions were right?',
                 ('TP', 'TN'), OUTCOMES),
    'Recall': ('recall', 'Of the actual defaulters, what fraction did we catch?',
               ('TP',), ('TP', 'FN')),
    'Precision': ('precision', 'Of the customers we flagged, what fraction actually defaulted?',
                  ('TP',), ('TP', 'FP')),
    'Specificity': ('specificity', 'Of the non-defaulters, what fraction did we leave alone?',
                    ('TN',), ('TN', 'FP')),
    'False positive rate': ('false positive rate',
                            'Of the non-defaulters, what fraction did we wrongly flag?',
                            ('FP',), ('TN', 'FP')),
}


def _sum_text(cells):
    joined = ' + '.join(cells)
    return f'({joined})' if len(cells) > 1 else joined


def metric_highlighter(fit, threshold=0.5):
    """Pick a metric: its numerator cells light up, its denominator cells are outlined."""
    metric = widgets.ToggleButtons(options=list(METRICS))
    which = widgets.ToggleButtons(options=[f'Model at {threshold}', 'Always "no"'])
    readout = widgets.HTML()
    out = _figure_area()

    def update(change=None):
        t = threshold if which.index == 0 else NEVER_FLAG
        counts = outcome_counts(fit.y, fit.p, t)
        key, question, num, den = METRICS[metric.value]
        top, bottom = sum(counts[k] for k in num), sum(counts[k] for k in den)
        value = metric_values(counts)[key]
        result = (f'{top:,} / {bottom:,} = <b>{value:.3f}</b>' if bottom else
                  f'{top:,} / 0 = <b>undefined</b> (nobody was flagged)')
        readout.value = _html(f'<b>{metric.value}:</b> {question}<br>'
                              f'{_sum_text(num)} / {_sum_text(den)} = {result}')

        def draw():
            fig, ax = subplots(figsize=(8, 6))
            _draw_confusion(ax, counts, numerator=num, denominator=den)
            fig.tight_layout()
            return fig
        _show(out, draw)

    metric.observe(update, names='value')
    which.observe(update, names='value')
    update()
    display(widgets.VBox([metric, which, readout, out]))


# --- Section 6 and 7: moving the threshold, and F1 ---------------------------

def _counts_by_threshold(fit, thresholds=THRESHOLDS):
    """Confusion counts at every threshold at once, as arrays."""
    flagged = fit.p[None, :] >= np.asarray(thresholds)[:, None]
    yes = fit.y == 1
    tp, fp = (flagged & yes).sum(axis=1), (flagged & ~yes).sum(axis=1)
    return {'TP': tp, 'FN': yes.sum() - tp, 'FP': fp, 'TN': (~yes).sum() - fp}


def f1_by_threshold(fit, thresholds=THRESHOLDS):
    c = _counts_by_threshold(fit, thresholds)
    return 2 * c['TP'] / (2 * c['TP'] + c['FP'] + c['FN'])


def _arrow(new, old):
    if old is None or np.isnan(new) or np.isnan(old) or abs(new - old) < 5e-4:
        return '<span style="color: #999;">–</span>'
    return '▲' if new > old else '▼'


def _threshold_controls(presets):
    slider = widgets.FloatSlider(value=0.5, min=0.01, max=0.99, step=0.01,
                                 description='threshold', readout_format='.2f',
                                 layout=widgets.Layout(width='600px'))
    state = {'never': False}
    buttons = []
    for label, value in presets:
        b = widgets.Button(description=label, layout=widgets.Layout(width='110px'))

        def click(button, value=value):
            state['never'] = value > 1
            slider.value = slider.max if value > 1 else value
            slider.description = 'never flag' if value > 1 else 'threshold'
        b.on_click(click)
        buttons.append(b)

    def threshold():
        return NEVER_FLAG if state['never'] else slider.value

    def moved(change):
        if change['owner'] is slider and state['never'] and change['new'] != slider.max:
            state['never'] = False
        slider.description = 'never flag' if state['never'] else 'threshold'

    slider.observe(moved, names='value')
    return slider, buttons, threshold, state


def threshold_demo(fit, show_f1=False):
    """Slide the threshold: the scatter, confusion matrix, and metrics update together.
    Nothing is refit; only the cutoff on the precomputed probabilities moves."""
    slider, buttons, threshold, state = _threshold_controls(
        [('0.5', 0.5), ('0.2', 0.2), ('never flag', NEVER_FLAG)])
    readout = widgets.HTML()
    out = _figure_area()
    previous = {'values': None, 'threshold': None}
    f1_curve = f1_by_threshold(fit)
    best = THRESHOLDS[np.argmax(f1_curve)]
    names = ['accuracy', 'recall', 'precision', 'specificity', 'false positive rate', 'F1']

    def update(change=None):
        t = threshold()
        if t == previous['threshold']:
            return
        previous['threshold'] = t
        counts = outcome_counts(fit.y, fit.p, t)
        values = metric_values(counts)
        old = previous['values'] or {}
        rows = []
        for name in names:
            if name == 'F1' and not show_f1:
                rows.append('<tr><td>F1</td><td colspan="2" style="color: #999;">'
                            '(coming in Section 7)</td></tr>')
                continue
            rows.append(f'<tr><td>{name}</td><td style="text-align: right; padding: 0 12px;">'
                        f'<b>{_metric_text(values[name])}</b></td>'
                        f'<td>{_arrow(values[name], old.get(name))}</td></tr>')
        readout.value = _html('<table>' + ''.join(rows) + '</table>')
        previous['values'] = values

        def draw():
            if show_f1:
                fig, (ax_s, ax_m, ax_f) = subplots(1, 3, figsize=(20, 6.5),
                                                   width_ratios=[1.2, 1, 1])
            else:
                fig, (ax_s, ax_m) = subplots(1, 2, figsize=(16, 6.5), width_ratios=[1.25, 1])
            _draw_outcome_scatter(ax_s, fit, t)
            _draw_confusion(ax_m, counts)
            if show_f1:
                ax_f.plot(THRESHOLDS, f1_curve, color='black', lw=2.5)
                ax_f.axvline(best, color=OUTCOME_COLORS['TP'], lw=2, ls=':')
                ax_f.text(best + 0.02, 0.05, f'best F1 = {f1_curve.max():.3f}\nat {best:.2f}',
                          color=OUTCOME_COLORS['TP'], fontsize=13)
                if t <= 1:
                    ax_f.plot(t, values['F1'], 'o', ms=12, color='black')
                ax_f.set_xlim(0, 1)
                ax_f.set_ylim(0, 1)
                ax_f.set_xlabel('threshold')
                ax_f.set_ylabel('F1')
                ax_f.set_title('F1 at every threshold')
            fig.tight_layout()
            return fig
        _show(out, draw)

    slider.observe(update, names='value')
    for b in buttons:
        b.on_click(update)
    update()
    display(widgets.VBox([widgets.HBox([slider] + buttons),
                          widgets.HBox([out, readout])]))


def f1_vs_average_demo():
    """Precision and recall sliders: the plain average next to F1 (their harmonic mean)."""
    style = {'description_width': '80px'}
    precision = widgets.FloatSlider(value=0.7, min=0, max=1, step=0.01, description='precision',
                                    style=style, layout=widgets.Layout(width='450px'))
    recall = widgets.FloatSlider(value=0.3, min=0, max=1, step=0.01, description='recall',
                                 style=style, layout=widgets.Layout(width='450px'))
    extreme = widgets.Button(description='extreme: 1.0 and 0.02', layout=widgets.Layout(width='200px'))
    out = _figure_area()

    def draw():
        p, r = precision.value, recall.value
        average = (p + r) / 2
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        fig, ax = subplots(figsize=(8, 5))
        bars = ax.bar(['average', 'F1'], [average, f1], color=['#bbbbbb', OUTCOME_COLORS['TP']],
                      width=0.55)
        for bar, v in zip(bars, [average, f1]):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f'{v:.2f}', ha='center',
                    fontsize=22, fontweight='bold')
        ax.set_ylim(0, 1.12)
        ax.set_title(f'precision = {p:.2f}, recall = {r:.2f}')
        fig.tight_layout()
        return fig

    def set_extreme(button):
        with precision.hold_trait_notifications(), recall.hold_trait_notifications():
            precision.value, recall.value = 1.0, 0.02

    for s in (precision, recall):
        s.observe(lambda change: _show(out, draw), names='value')
    extreme.on_click(set_extreme)
    _show(out, draw)
    display(widgets.VBox([widgets.HBox([widgets.VBox([precision, recall]), extreme]), out]))


# --- Section 8: ROC curves and AUC -------------------------------------------

def _draw_roc_axes(ax):
    ax.plot([0, 1], [0, 1], color='#888888', ls='--', lw=2)
    ax.text(0.62, 0.55, 'random guessing', rotation=45, color='#666666', fontsize=13,
            ha='center', va='center', rotation_mode='anchor', transform_rotates_text=True)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect('equal')
    ax.set_xlabel('false positive rate\n(share of non-defaulters flagged)')
    ax.set_ylabel('recall (true positive rate)\n(share of defaulters caught)')


SWEEP = np.round(np.r_[np.arange(0.95, 0.2, -0.05), np.arange(0.2, 0.005, -0.01)], 2)


def roc_tracer(fit, sweep=SWEEP):
    """Each threshold is one point on the ROC axes. Sweep traces the curve from high
    thresholds to low ones (in finer steps below 0.2, where the point moves fastest);
    Show full curve adds every threshold and shades the AUC."""
    slider, buttons, threshold, state = _threshold_controls([])
    sweep_button = widgets.Button(description='Sweep', button_style='info')
    full = widgets.ToggleButton(description='Show full curve')
    readout = widgets.HTML()
    out = _figure_area()
    trace = {'points': [], 'sweeping': False}
    fpr_all, tpr_all, _ = roc_curve(fit.y, fit.p)
    auc = roc_auc_score(fit.y, fit.p)

    def update(change=None):
        t = threshold()
        values = metric_values(outcome_counts(fit.y, fit.p, t))
        point = (values['false positive rate'], values['recall'])
        if trace['sweeping']:
            trace['points'].append(point)
        text = (f'threshold {t:.2f}: false positive rate <b>{point[0]:.3f}</b>, '
                f'recall <b>{point[1]:.3f}</b>')
        if full.value:
            text += f'<br>AUC (shaded area) = <b>{auc:.3f}</b>'
        readout.value = _html(text)

        def draw():
            fig, (ax_s, ax_r) = subplots(1, 2, figsize=(16, 6.8), width_ratios=[1.25, 1])
            _draw_outcome_scatter(ax_s, fit, t)
            _draw_roc_axes(ax_r)
            if full.value:
                ax_r.fill_between(fpr_all, tpr_all, color=OUTCOME_COLORS['TP'], alpha=0.15)
                ax_r.plot(fpr_all, tpr_all, color=OUTCOME_COLORS['TP'], lw=2.5)
                ax_r.text(0.55, 0.2, f'AUC = {auc:.3f}', fontsize=18, fontweight='bold',
                          color=OUTCOME_COLORS['TP'])
            if trace['points']:
                xs, ys = zip(*trace['points'])
                ax_r.plot(xs, ys, 'o-', ms=5, color='black', lw=1.5)
            ax_r.plot(*point, 'o', ms=15, color=OUTCOME_COLORS['FP'], mec='black', mew=1.5)
            ax_r.set_title(f'threshold = {t:.2f}')
            fig.tight_layout()
            return fig
        _show(out, draw)

    def run_sweep(button):
        trace['points'] = []
        trace['sweeping'] = True
        for t in sweep:
            slider.value = t
        trace['sweeping'] = False

    slider.observe(update, names='value')
    full.observe(update, names='value')
    sweep_button.on_click(run_sweep)
    update()
    display(widgets.VBox([widgets.HBox([slider, sweep_button, full]), readout, out]))


def roc_comparison(df, fit, seed=463):
    """ROC curves for three scorers: balance, income only, and random numbers."""
    income = ((df['income'] - df['income'].mean()) / df['income'].std()).to_numpy()[:, None]
    scores = {
        'balance model': fit.p,
        'income-only model': LogisticRegression(C=np.inf).fit(income, fit.y)
                                                           .predict_proba(income)[:, 1],
        'random scores': np.random.default_rng(seed).random(len(fit.y)),
    }
    colors = dict(zip(scores, [OUTCOME_COLORS['TP'], CLASS_COLORS[1], '#555555']))
    toggles = {name: widgets.ToggleButton(value=(name == 'balance model'), description=name,
                                          layout=widgets.Layout(width='180px'))
               for name in scores}
    out = _figure_area()

    def draw():
        fig, ax = subplots(figsize=(8, 7.5))
        _draw_roc_axes(ax)
        for name, s in scores.items():
            if toggles[name].value:
                fpr, tpr, _ = roc_curve(fit.y, s)
                ax.plot(fpr, tpr, lw=3, color=colors[name],
                        label=f'{name}: AUC = {roc_auc_score(fit.y, s):.3f}')
        if any(t.value for t in toggles.values()):
            ax.legend(loc='lower right')
        fig.tight_layout()
        return fig

    for t in toggles.values():
        t.observe(lambda change: _show(out, draw), names='value')
    _show(out, draw)
    display(widgets.VBox([widgets.HBox(list(toggles.values())), out]))


# --- Section 9: choosing a threshold with costs ------------------------------

def cost_demo(fit, costs=BANK_COSTS):
    """Total cost = cost per missed defaulter x FN + cost per false alarm x FP, at every
    threshold. The scatter and matrix show the cheapest threshold."""
    style = {'description_width': '230px'}
    fn_cost = widgets.IntSlider(value=costs[0], min=0, max=5000, step=50, style=style,
                                description='cost of a missed defaulter ($)',
                                layout=widgets.Layout(width='600px'))
    fp_cost = widgets.IntSlider(value=costs[1], min=0, max=2000, step=25, style=style,
                                description='cost of a false alarm ($)',
                                layout=widgets.Layout(width='600px'))
    bank = widgets.Button(description=f'bank: ${BANK_COSTS[0]:,} / ${BANK_COSTS[1]:,}',
                          layout=widgets.Layout(width='200px'))
    equal = widgets.Button(description=f'equal: ${EQUAL_COSTS[0]:,} / ${EQUAL_COSTS[1]:,}',
                           layout=widgets.Layout(width='200px'))
    out = _figure_area()
    counts_all = _counts_by_threshold(fit)

    def draw():
        total = fn_cost.value * counts_all['FN'] + fp_cost.value * counts_all['FP']
        i = int(np.argmin(total))
        best = THRESHOLDS[i]
        at_half = total[THRESHOLDS == 0.5][0]
        fig = figure(figsize=(16, 12))
        axes = fig.subplot_mosaic([['cost', 'cost'], ['scatter', 'matrix']],
                                  height_ratios=[1, 1.3], width_ratios=[1.25, 1])
        ax = axes['cost']
        ax.plot(THRESHOLDS, total / 1000, color='black', lw=2.5)
        ax.plot(best, total[i] / 1000, 'o', ms=14, color=OUTCOME_COLORS['TP'])
        ax.plot(0.5, at_half / 1000, 'o', ms=10, color='#888888')
        ax.annotate(f'cheapest: threshold {best:.2f}\ntotal \\${total[i]:,.0f}',
                    (best, total[i] / 1000), xytext=(15, -55), textcoords='offset points',
                    fontsize=15, color=OUTCOME_COLORS['TP'], fontweight='bold')
        ax.annotate(f'threshold 0.50: \\${at_half:,.0f}', (0.5, at_half / 1000),
                    xytext=(10, -30), textcoords='offset points', fontsize=14, color='#555555')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, max(total.max() / 1000 * 1.15, 1))
        ax.set_xlabel('threshold')
        ax.set_ylabel('total cost ($1,000s)')
        ax.set_title(f'\\${fn_cost.value:,} × missed defaulters  +  '
                     f'\\${fp_cost.value:,} × false alarms')
        counts = _draw_outcome_scatter(axes['scatter'], fit, best)
        _draw_confusion(axes['matrix'], counts)
        fig.tight_layout()
        return fig

    def preset(values):
        def click(button):
            with fn_cost.hold_trait_notifications(), fp_cost.hold_trait_notifications():
                fn_cost.value, fp_cost.value = values
        return click

    for s in (fn_cost, fp_cost):
        s.observe(lambda change: _show(out, draw), names='value')
    bank.on_click(preset(BANK_COSTS))
    equal.on_click(preset(EQUAL_COSTS))
    _show(out, draw)
    display(widgets.VBox([widgets.HBox([widgets.VBox([fn_cost, fp_cost]),
                                        widgets.VBox([bank, equal])]), out]))


# --- Appendix: likelihood and gradient descent -------------------------------

def coin_table(heads=7, flips=10, ps=(0.5, 0.7, 0.9)):
    """Likelihood of the observed flips for a few candidate values of p."""
    return pd.DataFrame({'p': ps,
                         'likelihood': [p**heads * (1 - p)**(flips - heads) for p in ps]}
                        ).set_index('p').round(6)


def coin_demo(flips=10):
    """Slide p and the number of heads: the likelihood curve peaks at heads / flips."""
    p = widgets.FloatSlider(value=0.5, min=0.01, max=0.99, step=0.01, description='p',
                            layout=widgets.Layout(width='450px'))
    heads = widgets.IntSlider(value=7, min=0, max=flips, description='heads',
                              layout=widgets.Layout(width='450px'))
    out = _figure_area()

    def draw():
        h = heads.value
        grid = np.linspace(0, 1, 401)
        like = grid**h * (1 - grid)**(flips - h)
        current = p.value**h * (1 - p.value)**(flips - h)
        fig, ax = subplots(figsize=(10, 5))
        ax.plot(grid, like, color='black', lw=2.5)
        ax.axvline(h / flips, color=OUTCOME_COLORS['TP'], ls=':', lw=2)
        ax.text(h / flips, like.max() * 1.06, f'peak at {h}/{flips} = {h / flips:.1f}',
                ha='center', color=OUTCOME_COLORS['TP'], fontsize=14,
                bbox=dict(facecolor='white', edgecolor='none'))
        ax.plot(p.value, current, 'o', ms=14, color=CLASS_COLORS[1], mec='black')
        ax.set_ylim(0, like.max() * 1.15)
        ax.set_xlabel('p (chance of heads)')
        ax.set_ylabel('likelihood')
        ax.set_title(f'{h} heads, {flips - h} tails:   L({p.value:.2f}) = {current:.6f}')
        fig.tight_layout()
        return fig

    for s in (p, heads):
        s.observe(lambda change: _show(out, draw), names='value')
    _show(out, draw)
    display(widgets.VBox([p, heads, out]))


def underflow_table(fit, sizes=(10, 100, 1000, 10000)):
    """Multiply the probabilities of what happened vs add their logs, for more and more customers."""
    p_observed = np.where(fit.y == 1, fit.p, 1 - fit.p)
    return pd.DataFrame({
        'customers': sizes,
        'product of probabilities': [np.prod(p_observed[:n]) for n in sizes],
        'sum of log probabilities': [np.log(p_observed[:n]).sum() for n in sizes],
    }).set_index('customers')


def plot_penalties():
    """Penalty for one defaulter as the predicted probability of default varies."""
    with rc_context(BIG_FONTS):
        p = np.linspace(0.001, 1, 500)
        fig, ax = subplots(figsize=(10, 5.5))
        ax.plot(p, (1 - p)**2, lw=3, color=CLASS_COLORS[0], label='squared error: $(1 - p)^2$')
        ax.plot(p, -np.log(p), lw=3, color=OUTCOME_COLORS['FN'], label=r'log loss: $-\ln p$')
        ax.set_ylim(0, 5)
        ax.set_xlabel('predicted probability of default, for someone who did default')
        ax.set_ylabel('penalty')
        ax.legend()
        fig.tight_layout()
    return fig


def _mean_nll(b0, b1, x, y):
    z = b0[..., None] + b1[..., None] * x
    return (np.logaddexp(0, z) - y * z).mean(axis=-1)


def _gd_setup(fit, scale, n=70):
    """Coordinates, contour grid, and slider settings for one version of the balance variable."""
    if scale == 'standardized':
        x, best = fit.x, (fit.b0, fit.b1)
        box = dict(b0=(-10, 0, 0.1, -2.0), b1=(-1, 5, 0.05, 0.5), lr=(-0.3, 1.8, 10.0))
    else:
        x = fit.x * fit.balance_sd + fit.balance_mean
        best = (fit.b0 - fit.b1 * fit.balance_mean / fit.balance_sd, fit.b1 / fit.balance_sd)
        box = dict(b0=(-14, 0, 0.1, -2.0), b1=(-0.002, 0.01, 0.0001, 0.0),
                   lr=(-8, -5, 1e-6))
    g0 = np.linspace(box['b0'][0], box['b0'][1], n)
    g1 = np.linspace(box['b1'][0], box['b1'][1], n)
    B0, B1 = np.meshgrid(g0, g1)
    return SimpleNamespace(x=x, best=best, box=box, B0=B0, B1=B1,
                           Z=_mean_nll(B0, B1, x, fit.y))


def gradient_descent_demo(fit, run_steps=25):
    """Gradient descent on the average negative log-likelihood, drawn on its contour map.
    The raw-balance toggle shows why standardizing first matters."""
    scale = widgets.ToggleButtons(options=['standardized', 'raw dollars'],
                                  description='balance:')
    b0 = widgets.FloatSlider(description='start β₀', layout=widgets.Layout(width='420px'))
    b1 = widgets.FloatSlider(description='start β₁', readout_format='.4f',
                             layout=widgets.Layout(width='420px'))
    lr = widgets.FloatLogSlider(description='step size', base=10, readout_format='.1e',
                                layout=widgets.Layout(width='420px'))
    step_one = widgets.Button(description='Step')
    run = widgets.Button(description=f'Run {run_steps} steps', button_style='info')
    reset = widgets.Button(description='Reset')
    readout = widgets.HTML()
    out = _figure_area()
    setups = {}
    state = {'path': [], 'configuring': False}

    def setup():
        if scale.value not in setups:
            setups[scale.value] = _gd_setup(fit, 'standardized' if scale.index == 0 else 'raw')
        return setups[scale.value]

    def configure(change=None):
        s = setup()
        state['configuring'] = True
        for slider, key in ((b0, 'b0'), (b1, 'b1')):
            lo, hi, step, start = s.box[key]
            slider.min, slider.max = min(lo, slider.min), max(hi, slider.max)
            slider.step, slider.value = step, start
            slider.min, slider.max = lo, hi
        lo, hi, start = s.box['lr']
        lr.min, lr.max = min(lo, lr.min), max(hi, lr.max)
        lr.value = start
        lr.min, lr.max = lo, hi
        state['configuring'] = False
        restart()

    def restart(change=None):
        if state['configuring']:
            return
        state['path'] = [np.array([b0.value, b1.value])]
        redraw()

    def gradient(b, x):
        r = logistic(b[0] + b[1] * x) - fit.y
        return np.array([r.mean(), (r * x).mean()])

    def take(n):
        s = setup()
        b = state['path'][-1]
        for _ in range(n):
            b = b - lr.value * gradient(b, s.x)
            state['path'].append(b)
        redraw()

    def redraw():
        s = setup()
        path = np.array(state['path'])
        b = path[-1]
        nll = _mean_nll(np.array(b[0]), np.array(b[1]), s.x, fit.y)
        readout.value = _html(
            f'step {len(path) - 1}: β₀ = {b[0]:.4g}, β₁ = {b[1]:.4g}, '
            f'average negative log-likelihood = <b>{nll:.4f}</b> '
            f'(minimum {_mean_nll(np.array(s.best[0]), np.array(s.best[1]), s.x, fit.y):.4f})')

        def draw():
            fig, ax = subplots(figsize=(10, 7))
            levels = s.Z.min() + (s.Z.max() - s.Z.min()) * np.geomspace(2e-3, 1, 22)
            ax.contour(s.B0, s.B1, s.Z, levels=levels, cmap='viridis', linewidths=1.2)
            ax.plot(*s.best, '*', ms=22, color=OUTCOME_COLORS['TP'], mec='black',
                    label='best fit')
            ax.plot(path[:, 0], path[:, 1], 'o-', ms=5, color=OUTCOME_COLORS['FN'], lw=2,
                    label='gradient descent')
            ax.plot(*path[0], 's', ms=12, color=OUTCOME_COLORS['FN'], mec='black',
                    label='start')
            ax.set_xlim(s.B0.min(), s.B0.max())
            ax.set_ylim(s.B1.min(), s.B1.max())
            ax.set_xlabel('β₀')
            ax.set_ylabel('β₁')
            ax.set_title(f'average negative log-likelihood, {scale.value} balance')
            ax.legend(loc='upper right')
            fig.tight_layout()
            return fig
        _show(out, draw)

    scale.observe(configure, names='value')
    for slider in (b0, b1, lr):
        slider.observe(restart, names='value')
    step_one.on_click(lambda button: take(1))
    run.on_click(lambda button: take(run_steps))
    reset.on_click(restart)
    configure()
    display(widgets.VBox([scale, widgets.HBox([widgets.VBox([b0, b1, lr]),
                                               widgets.VBox([step_one, run, reset])]),
                          readout, out]))
