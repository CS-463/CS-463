This doc establishes the desiderata for the regression_demo_plan.md file.

# Sections:

## Problem Setup 
- Show an interactive 3-D plot of the advertising data.
- Show a projection of the data: Sales vs TV

## Linear Regression
- Restrict attention to 3 (deterministically chosen/reproducible) points, spread out along the TV axis (no longer need the 3-D interactive plotting)
- Show a few different possible lines we could try to use to model the situation (make one of them the line of best fit.)
- Show the residuals for one line at a time, and display the value of each residual on the plots.
- Discuss: Which line looks best? How can we measure/operationalize/quantify "best" using the residual values?

## Formalizing "Best" Fit
- Find a line that is obviously a terrible fit with the same sum of residuals as one of the other lines.
- Reject sum of residuals. Why did this fail? Because positive and negative residual values can cancel. What do we need? (Answer: Make sure all residuals contribute to a "poor" score, even if their raw values can cancel.)
- Options: Absolute value and Squaring
- Show table with Line A, Line B and Line C on horiz axis, and columns depicting sum of |e| and sum of e^2.
- See this is a choice. 
- Side-by-side plots: Sales vs TV; RSS vs y-intercept with other parameter fixed. <-- Nice to have: A metric picker in the right plot, so I can toggle the view of RSS vs TV and sum of |e| vs TV. 
    - Ideally, we can add points on the loss vs y-intercept plot one by one.
    - As we sweep the y-intercept values (adding one point at a time), we should see the left plot of Sales vs TV translating vertically.
    - Similar seperate plot for the slopes.
    
- Notice that we only varied one parameter at a time, with the other fixed. Where is the "global" minimal loss? 
- Show 3-D rendering of loss surface and a contour plot of the loss surface. These can be in separate cells. 
- Using calculus (skip details) it can be shown that the minimal loss occurs where (show the formulas for slope and intercept). 
    - Hardly anyone has this memorized, and you don't need to. The point is: there's a formula that specifies the "best" linear model parameters (slope and intercept). 
    - You can invoke/use it in multiple different libraries (show example invocation with numpy or whatever we have installed)

### Should we update our boss?
    - Estimate how much we expect sales to increase based on our Sales vs TV model (if we increase TV spending from a few values to a few other values. Make sure to predict 100 dollars higher "off distribution").
    - But wait... how confident should we be in our model?
## Formalizing Confidence
- Three questions:
    - R^2: How much better does the model predict the data than simply predicting the mean value (aka variance explained)? Is this enough to quantify confidence?
    - p-value: How likely is it that other random samples of data would  produce an R^2 value that big (or bigger)?
    - Standard Error: How precisely did we estimate each parameter (slope and intercept)? Can we put error bars on them, or specify the interval of values where our confidence is very high (95%)?
- R^2:
    - Show a plot with a few points and a line of best fit. Display the SSR.
    - Beside that plot, show the same points and a horizontal line with y=mean(y_i).
    - Display the red dashed residual lines on a both plots.
    - Markdown: How much better is the model, than just predicting the average?
    - Tempting to just subtract the SSR's for each model. Explain or demonstrate why we want a percent change relative to the mean.
    - Show a few examples with bad R^2 values due to (Display the linear model and the data points with the "true" function where appropriate on the same plot for each scenario below and R^2 values displayed, clearly indicate in markdown when we're not using the Sales dataset any more and put a note in code comments about where data come from):
        - Noisy Y the predictor can't caputer.
        - Wrong functional form (fit a straight line to a quadratic with a little noise added)
        - Narrow range in X
        - Outliers in Y
- p-value
    - Question: Are we done? Should we be confident in our model's prediction on the basis of a good R^2 value?
    - Show how much variation we get in R^2 from several samples (with plots).
    - Reframe: Line C (least squares through our 3 stores) has R^2 ≈ 0.85. If TV had *no relationship* to sales, how often would 3 random stores give an R^2 at least this large, just by luck?
    - "No relationship" samples: take 3 TV values from 3 random stores and 3 sales values from 3 *other* random stores, pair them up, fit a line, record R^2. (Code comment: real Advertising values, pairing deliberately broken.)
        - Show 3 or so of these side by side with their fitted lines and R^2, so students see that luck alone sometimes produces a steep line and a high R^2.
        - Why not shuffle our own 3 sales values? Only 6 orderings, so the histogram would be too coarse.
    - Build-up visualization (plotly slider + Play button, same pattern as the loss explorer):
        - Left: the current "no relationship" sample, its fitted line, and R^2 in the title.
        - Right: histogram of R^2 for all draws so far, bins fixed on [0, 1], with the current draw's bar highlighted.
        - Steps: draws 1–10 one at a time (so each draw visibly lands in a bar), then 20, 50, 100, 200, 500, 1000 (the shape emerges).
    - Tail: static histogram of all 1000 draws (reuse plot_r2_histogram), red dashed line at R^2 = 0.85, bars at or above it shaded. Title reports the fraction of draws at or above 0.85 (expect ≈ 0.25).
    - Markdown: that fraction is the **p-value**. "If there were no relationship, about 1 in 4 samples of 3 stores would give an R^2 this large." Not convincing evidence, despite the "good" R^2.
        - Cautions: the p-value is *not* the probability that the model is wrong, or that there's no relationship. It's how surprising our result would be if there were no relationship. Small (commonly < 0.05) = surprising.
    - Library check: statsmodels' sm_model.pvalues['TV'] (≈ 0.25) matches the simulation. The library uses a formula (assuming normally distributed errors) instead of simulating, so expect close, not identical.
    - Contrast: repeat with more stores (e.g. 10 or 30). The "no relationship" histogram piles up near 0 and a real sample's R^2 lands far in the tail, so p ≈ 0. More data makes the same R^2 much less likely to be luck.
    - Helpers/tests: null_r2(df, n, draws, seed), a slider explorer for the build-up, and a shading option on plot_r2_histogram. Test that the simulated p-value is within a few hundredths of 0.25 and of statsmodels.
 - Standard Error:


**1. Simulate.** Draw many samples from a known true model, fit each, and show the lines fanning out plus a histogram of the slope estimates.
- True model borrows its numbers from all 200 Advertising stores (least squares line, noise sd = residual standard error). Clearly say we're back on simulated data.
- Fixed x: the same 30 TV budgets in every sample; only the noise changes. (This is the setting the SE formula describes, so step 3's comparison works.)
- Interactive build-up (same slider + Play pattern as the p-value explorer): left shows the true line dashed, the current sample, its fitted line, and a faint fan of earlier lines; right shows the histogram of slopes so far with the true slope dashed and the running SD in the title.

**2. Say what: the definition.** "The standard error is the typical size of this bounce: the standard deviation of that histogram. It tells you how much your estimate would change if you collected a new sample." Point out that the histogram is centered on the truth: estimates are right on average, but any single one misses by about one SE.

**3. Say what: you don't need the simulation in practice.** "In real life you have one sample, but the software can estimate the bounce from that one sample. That's the `std err` column in the regression table." If the notebook can show the formula's SE from one sample next to the simulated SD, with the two matching, that's the whole payoff in one picture. No formula needs to be shown.

**4. Say what: it's used for two things.**
- **Confidence interval:** estimate ± 2 SE gives a range of plausible values for the true slope.
    - Show it working: stack the ±2 SE intervals from 50 simulated samples, red where they miss the true slope, and report the fraction of all 1000 that cover it (≈ 95%).
- **t-statistic:** estimate ÷ SE is how many "bounces" the estimate is from zero. If it's far (say, more than 2), a slope of zero is implausible. The p-value section came first, so tie back: this is how software computes that p-value.

**Finish with the real regression table** (all 200 stores, built with scipy's linregress): coef, std err, t, and ±2 SE interval for the intercept and TV. Read the TV row aloud as in the learning outcome below.

**Optional, one slide, no math:** what makes the SE smaller: less noise, more data, and more spread in x. The simulation can show each factor in a few seconds, but you can skip this if time is tight. (Built as three panels, each baseline vs. one change: noise ÷ 2, 4× the stores, TV spread squeezed by half. Squeezing avoids negative TV budgets.)

**What to skip:** the SE formulas, the σ²/n derivation, the RSE-as-estimate-of-σ connection, and the details of the t-distribution.

**Minimum learning outcome:** a student can look at a regression table and say, "The slope is 0.0475 with standard error 0.0027, so the true slope is probably between about 0.042 and 0.053, and it's clearly not zero."

- Putting it all together: Should we increase TV advertising?
    - Recap the boss's two questions from the top of the notebook: "Should we increase TV advertising by $100?" and "How confident are you?"
    - Units first: TV is in thousands of dollars and sales in thousands of units, so "+100" in the data means +$100,000, and a slope of 0.0475 means about 47.5 more units per extra $1,000.
    - Answer with all 200 markets (reuse the regression table from the SE section):
        - The increase from a change in TV spending is slope × change, so it doesn't depend on the intercept or the starting budget.
        - +$100k → about 4.75 thousand more units, with a 95% interval of 100 × (slope ± 2 SE) ≈ 4.2 to 5.3 thousand.
    - Contrast with the 3-store answer (the earlier "30% increase" cell): slope ≈ 0.071, p ≈ 0.25, a far wider interval. Same method, but only 200 points justify real confidence. Show both side by side in a small table: slope, predicted increase, ±2 SE interval, p-value.
    - One picture: all 200 markets, the fitted line with a shaded confidence band, the 3-store line faintly for comparison, and a shaded "beyond our data" region past the largest TV budget. Mark the prediction at max TV + 100 with its error bar. The band widens as we move away from the middle of the data, and it keeps widening in the extrapolation region. (Band formula hidden in a helper.)
        - Optional interactive version: a slider for the proposed TV budget that moves the prediction point and its interval, turning red once it leaves the observed range.
    - How confident, honestly? What the standard error does *not* cover:
        - Extrapolation: max TV + 100 is beyond every budget in the data. The SE assumes the straight line keeps holding there; diminishing returns would break it.
        - Association, not causation: this is observational data. Markets with big TV budgets may differ in other ways.
        - Other media: radio and newspaper also matter. That leads into multiple regression next time.
        - (Optional) The residuals fan out as TV grows, so the textbook SEs are a bit optimistic.
    - Closing exercise: students draft a 2–3 sentence answer to the boss before revealing ours in a collapsible, e.g. "Within the budgets we've seen, each extra $1,000 of TV goes with about 42–53 more units sold. An extra $100k would suggest roughly 4,200–5,300 more units, but that's beyond any budget in our data, so treat it as a rough guess, and our data show association, not proof that TV causes the sales."
    - Tests: the 200-market interval for +100, the 3-store numbers in the comparison table, and that the prediction point lies outside the observed TV range.