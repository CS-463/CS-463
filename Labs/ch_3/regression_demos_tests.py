"""Checks for regression_demos_helpers.py, using the same setup as the notebook.

Run from this folder with the CS_463 environment:
    python regression_demos_tests.py

Prints the numbers the lesson relies on, asserts the claims the notebook makes,
and saves every figure to OUT so the plots can be inspected without Jupyter.
"""
import os
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from regression_demos_helpers import (fit_line, loss_curves, loss_explorer,
                                      pick_spread_points, plot_lines,
                                      plot_residuals, plot_scenario, r_squared,
                                      residuals, rss_contour, rss_grid,
                                      rss_surface_3d, make_scenario, score_table,
                                      null_r2, null_r2_explorer, plot_null_samples,
                                      make_truth, true_se, simulate_fits, slope_explorer,
                                      confidence_band, plot_tv_decision,
                                      plot_slope_histogram, plot_intervals, plot_se_factors,
                                      plot_r2_histogram, plot_sample_luck,
                                      sample_r2, SCENARIOS)

OUT = os.path.join(tempfile.gettempdir(), 'regression_demo_checks')
os.makedirs(OUT, exist_ok=True)


def save(name):
    plt.savefig(os.path.join(OUT, name), dpi=80, bbox_inches='tight')
    plt.close('all')


Advertising = pd.read_csv('Advertising.csv', index_col=0)
pts = pick_spread_points(Advertising, 'TV', n=3)
print(pts[['TV', 'sales']], '\n')
assert pts['TV'].tolist() == [25.0, 149.8, 261.3]

m_best, b_best = fit_line(pts['TV'], pts['sales'])
m_D = -0.05
lines = {
    'Line A': (0.072, 5.4),
    'Line B': (0.02, 12.0),
    'Line C': (m_best, b_best),
    'Line D': (m_D, pts['sales'].mean() - m_D * pts['TV'].mean()),
}
for label, (m, b) in lines.items():
    print(f'{label}: m = {m:.4f}, b = {b:.3f}, residuals = {residuals(pts, m, b).round(2).tolist()}')

table = score_table(pts, lines)
print('\n', table, '\n')

# The notebook's claims
assert abs(table.loc['Line C', 'sum of e']) < 0.01, 'best-fit residuals should sum to 0'
assert abs(table.loc['Line D', 'sum of e']) < 0.01, 'a line through the mean point should too'
assert table['sum of e²'].idxmin() == 'Line C', 'Line C should have the smallest RSS'
assert table.loc['Line A', 'sum of |e|'] < table.loc['Line C', 'sum of |e|'], \
    'sum of |e| should prefer Line A over Line C'

plot_lines(pts, lines)
save('lines.png')
for label in lines:
    plot_residuals(pts, lines, label)
    save(f'residuals_{label[-1]}.png')

b_values = np.linspace(b_best - 10, b_best + 10, 41)
curves = loss_curves(pts, m_best, b_values)
for name, values in curves.items():
    print(f'{name}: smallest at b = {b_values[values.argmin()]:.2f} (value {values.min():.2f})')
rss, abs_loss = curves.values()
assert np.isclose(b_values[rss.argmin()], b_best), 'RSS should be smallest at the best-fit intercept'
assert b_values[abs_loss.argmin()] != b_values[rss.argmin()], 'the two losses should pick different b'

m_values = np.linspace(m_best - 0.05, m_best + 0.05, 41)
figs = {'b': loss_explorer(pts, 'b', b_values, m=m_best),
        'm': loss_explorer(pts, 'm', m_values, b=b_best)}
for sweep, fig in figs.items():
    assert len(fig.data) == 7
    assert len(fig.layout.sliders[0].steps) == 41
    for step in fig.layout.sliders[0].steps:
        assert len(step.args[0]['x']) == len(step.args[1]) == 6
    fig.write_html(os.path.join(OUT, f'loss_explorer_{sweep}.html'))

# With the intercept fixed at the best-fit value, RSS should be smallest at the best-fit slope
slope_rss = [np.sum(residuals(pts, m, b_best) ** 2) for m in m_values]
assert np.isclose(m_values[np.argmin(slope_rss)], m_best)

# Loss surface: the grid minimum should sit at the best-fit line
m_grid = np.linspace(m_best - 0.06, m_best + 0.06, 101)
b_grid = np.linspace(b_best - 12, b_best + 12, 101)
Z = rss_grid(pts, m_grid, b_grid)
assert Z.shape == (len(b_grid), len(m_grid))
row, col = np.unravel_index(Z.argmin(), Z.shape)
print(f'\nRSS grid minimum: m = {m_grid[col]:.4f}, b = {b_grid[row]:.3f}, RSS = {Z.min():.2f}')
assert np.isclose(m_grid[col], m_best) and np.isclose(b_grid[row], b_best)
rss_surface_3d(pts, m_grid, b_grid).write_html(os.path.join(OUT, 'rss_surface.html'))
rss_contour(pts, m_grid, b_grid, lines)
save('rss_contour.png')

# Closed-form formula and three libraries should all agree
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

x, y = pts['TV'], pts['sales']
m_hat = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
b_hat = y.mean() - m_hat * x.mean()
sk_model = LinearRegression().fit(pts[['TV']], y)
sm_model = sm.OLS(y, sm.add_constant(x)).fit()
fits = {'formula': (m_hat, b_hat),
        'numpy': tuple(np.polyfit(x, y, deg=1)),
        'scikit-learn': (sk_model.coef_[0], sk_model.intercept_),
        'statsmodels': (sm_model.params['TV'], sm_model.params['const'])}
for name, (m, b) in fits.items():
    print(f'{name:13s} slope = {m:.4f}, intercept = {b:.4f}')
    assert np.allclose((m, b), (m_best, b_best))

# R²: TSS - RSS depends on units; R² doesn't
r2 = r_squared(x, y)
print(f'\nR² on the three stores: {r2:.3f}')
assert np.isclose(r2, 1 - table.loc['Line C', 'sum of e²'] / np.sum((y - y.mean()) ** 2), atol=1e-3)
assert np.isclose(r_squared(x, y * 1000), r2)
diffs = []
for scale in (1, 1000):
    ys = y * scale
    m, b = fit_line(x, ys)
    diffs.append(np.sum((ys - ys.mean()) ** 2) - np.sum((ys - (m * x + b)) ** 2))
assert np.isclose(diffs[1] / diffs[0], 1e6)

compare = {'Line C (best fit)': (m_best, b_best), 'Mean only': (0, y.mean())}
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
plot_residuals(pts, compare, 'Line C (best fit)', ax=axes[0], show_rss=True)
plot_residuals(pts, compare, 'Mean only', ax=axes[1], show_rss=True)
save('r2_compare.png')

# Scenarios: the baseline should fit well and every problem case should look bad
for name in SCENARIOS:
    sx, sy, _ = make_scenario(name)
    print(f'{name:9s} R² = {r_squared(sx, sy):.3f}')
    plot_scenario(name)
    save(f'scenario_{name}.png')
    if name == 'baseline':
        assert r_squared(sx, sy) > 0.7
    else:
        assert r_squared(sx, sy) < 0.5, f'{name} should have a low R²'

# Sampling luck: R² from 3 random stores should swing widely
_, luck = sample_r2(Advertising, n=3, draws=1000)
print(f'\nR² over 1000 samples of 3: min {luck.min():.3f}, median {np.median(luck):.3f}, max {luck.max():.3f}')
assert luck.min() < 0.1 and luck.max() > 0.95, 'three-store samples should give wildly different R²'
fig = plot_sample_luck(Advertising)
assert len(fig.axes) == 6 and all(ax.has_data() for ax in fig.axes)
save('sample_luck.png')
plot_r2_histogram(luck, reference=r_squared(Advertising['TV'], Advertising['sales']))
save('r2_histogram.png')

# p-value: "no relationship" samples of 3, compared with Line C's R²
null_samples, null_values = null_r2(Advertising, n=3, draws=1000)
assert all(len(xs) == len(ys) == 3 for xs, ys in null_samples)
p_sim = np.mean(null_values >= r2)
p_theory = 1 - 2 / np.pi * np.arcsin(np.sqrt(r2))   # R² ~ Beta(1/2, 1/2) with n = 3 and normal errors
p_sm = sm_model.pvalues['TV']
print(f'\np-value for Line C: simulated {p_sim:.3f}, statsmodels {p_sm:.3f}, Beta theory {p_theory:.3f}')
assert np.isclose(p_sm, p_theory, atol=1e-3)
from scipy.stats import linregress
assert np.isclose(linregress(x, y).pvalue, p_sm), 'the notebook uses scipy; it should match statsmodels'
assert abs(p_sim - p_sm) < 0.05, 'simulation should roughly match statsmodels'
assert 0.15 < p_sim < 0.35, 'notebook says roughly a quarter'

plot_null_samples(Advertising)
save('null_samples.png')
explorer = null_r2_explorer(Advertising)
assert len(explorer.data) == 4 and len(explorer.frames) == 16
for f in explorer.frames:
    assert sum(f.data[2].y) == int(f.name), 'histogram should count every draw so far'
explorer.write_html(os.path.join(OUT, 'null_r2_explorer.html'))
plot_r2_histogram(null_values, shade_above=r2)
save('p_value_histogram.png')

# More data: 30 real stores vs. 1000 "no relationship" samples of 30
stores_30 = Advertising.sample(30, random_state=463)
r2_30 = r_squared(stores_30['TV'], stores_30['sales'])
_, null_30 = null_r2(Advertising, n=30, draws=1000)
fit_30 = sm.OLS(stores_30['sales'], sm.add_constant(stores_30['TV'])).fit()
print(f'30 stores: R² = {r2_30:.3f}, null max = {null_30.max():.3f}, statsmodels p = {fit_30.pvalues["TV"]:.2g}')
assert np.sum(null_30 >= r2_30) == 0, 'notebook says none of the 1000 reach it'
plot_r2_histogram(null_30, shade_above=r2_30)
save('p_value_histogram_30.png')

# Standard error: simulated samples from a true line borrowed from all 200 stores
truth = make_truth(Advertising, n=30)
print(f"\ntruth: m = {truth['m']:.4f}, b = {truth['b']:.3f}, sigma = {truth['sigma']:.3f}, "
      f"true SE = {true_se(truth):.4f}")
assert np.isclose(truth['m'], 0.0475, atol=5e-5) and np.isclose(truth['sigma'], 3.26, atol=0.01)
_, fits = simulate_fits(truth, draws=1000)
sd, one_se = fits['slope'].std(), fits['stderr'][0]
print(f'slopes: mean {fits["slope"].mean():.4f}, SD {sd:.4f}; SE from sample 1: {one_se:.4f}')
assert abs(fits['slope'].mean() - truth['m']) < 0.1 * true_se(truth), 'estimates centered on the truth'
assert abs(sd / true_se(truth) - 1) < 0.1, 'SD of the estimates should be the true SE'
assert abs(one_se / sd - 1) < 0.3, "notebook says sample 1's SE is close to the SD"

low, high = fits['slope'] - 2 * fits['stderr'], fits['slope'] + 2 * fits['stderr']
covered = ((low <= truth['m']) & (truth['m'] <= high)).mean()
print(f'±2 SE coverage: {covered:.3f}')
assert 0.92 < covered < 0.97, 'notebook says about 95%'

explorer = slope_explorer(truth)
assert len(explorer.data) == 5 and len(explorer.frames) == 16
for f in explorer.frames:
    assert sum(f.data[3].y) <= int(f.name), 'histogram counts at most every sample so far'
explorer.write_html(os.path.join(OUT, 'slope_explorer.html'))
plot_slope_histogram(fits['slope'], truth['m'], one_se=one_se)
save('slope_histogram.png')
plot_intervals(fits, truth['m'])
save('intervals.png')

fig = plot_se_factors(truth)
save('se_factors.png')
xs = truth['x']
for label, ratio in [('less noise', true_se(truth, sigma=truth['sigma'] / 2) / true_se(truth)),
                     ('more data', true_se(truth, x=np.tile(xs, 4)) / true_se(truth)),
                     ('less spread', true_se(truth, x=xs.mean() + (xs - xs.mean()) / 2) / true_se(truth))]:
    print(f'{label}: SE ratio {ratio:.2f}')
assert np.isclose(true_se(truth, x=np.tile(xs, 4)) / true_se(truth), 0.5)

# The real regression table: notebook quotes these numbers
fit_all = linregress(Advertising['TV'], Advertising['sales'])
t_all = fit_all.slope / fit_all.stderr
print(f'all 200 stores: slope {fit_all.slope:.4f}, SE {fit_all.stderr:.4f}, t {t_all:.1f}, '
      f'CI ({fit_all.slope - 2 * fit_all.stderr:.4f}, {fit_all.slope + 2 * fit_all.stderr:.4f})')
assert round(fit_all.slope, 4) == 0.0475 and round(fit_all.stderr, 4) == 0.0027
assert round(t_all, 1) == 17.7

# Putting it all together: +100 (i.e. +$100k) of TV, 3 stores vs. all 200 markets
from scipy.stats import t as t_dist
answers = {}
for label, data in [('3 stores', pts), ('all 200', Advertising)]:
    fit = linregress(data['TV'], data['sales'])
    mult = t_dist.ppf(0.975, len(data) - 2)
    answers[label] = (fit, mult, 100 * (fit.slope - mult * fit.stderr), 100 * (fit.slope + mult * fit.stderr))
    print(f'{label}: increase {100 * fit.slope:.2f}, multiplier {mult:.2f}, '
          f'interval ({answers[label][2]:.2f}, {answers[label][3]:.2f}), p {fit.pvalue:.3f}')
fit3, mult3, low3, high3 = answers['3 stores']
assert np.isclose(mult3, 12.7, atol=0.01) and low3 < 0 < high3, '3-store interval should include 0'
assert np.isclose(100 * fit3.slope, 7.1, atol=0.1)
fit200, mult200, low200, high200 = answers['all 200']
assert np.isclose(mult200, 1.97, atol=0.01)
assert round(low200, 1) == 4.2 and round(high200, 1) == 5.3, 'notebook says about 4.2 to 5.3'

x_new = Advertising['TV'].max() + 100
(y_new,), (h_new,) = confidence_band(Advertising['TV'], Advertising['sales'], [x_new])
_, h_mid = confidence_band(Advertising['TV'], Advertising['sales'], [Advertising['TV'].mean()])
print(f'prediction at TV = {x_new:.0f}: {y_new:.2f} ± {h_new:.2f} (half-width at mean TV: {h_mid[0]:.2f})')
assert np.isclose(y_new, fit200.intercept + fit200.slope * x_new)
assert h_new > 2 * h_mid[0], 'band should be much wider out in the extrapolation region'
plot_tv_decision(Advertising, pts)
save('tv_decision.png')

print(f'\nAll checks passed. Figures saved to {OUT}')
