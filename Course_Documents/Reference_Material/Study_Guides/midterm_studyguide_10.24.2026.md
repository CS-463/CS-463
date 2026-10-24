# Midterm Study Guide

**Midterm: Wednesday, October 28, in class.** It covers Chapters 3–5: everything from the demo
notebooks through the bootstrap.

## What to expect

- **Format:** on paper, with devices put away. **A calculator is allowed**, and so is **your
  standard notecard with formulas only** (no words, diagrams, or plots).
- **17 questions, worth 100 points.** Each question shows its point value.
  - Questions 1–13: answer every question.
  - Questions 14 and 15: answer **one** of the two.
  - Questions 16 and 17: answer **one** of the two.
- **Extra credit:** four optional questions at the end. Answer any **two**. A few questions also
  have a small extra-credit part.
- **No surprises:** every question is a close sibling of a practice question or demo you've
  already seen. The best way to study is to work the practice questions listed below **on paper**,
  before you open their answers.

## How to use this guide

The exam's ideas fall into the ten topics below. Each topic lists what you should be able to do,
then a table of where to look: which notebook, which **sections** to reread, and which **practice
questions** to work.

### Finding things

All the demo notebooks are in your copy of the course repo, in `Course_Documents/Slides/`:

| Short name | File |
|---|---|
| [Regression demos](../../Slides/ch_3/regression_demos.ipynb) | `ch_3/regression_demos.ipynb` |
| [Log-odds notebook](../../Slides/ch_4/ch_4_demos_1_log_odds_and_logistic_regression.ipynb) | `ch_4/ch_4_demos_1_log_odds_and_logistic_regression.ipynb` |
| [Classification metrics notebook](../../Slides/ch_4/ch_4_demos_2_classification_metrics.ipynb) | `ch_4/ch_4_demos_2_classification_metrics.ipynb` |
| [Naive Bayes reading](../../Slides/ch_4/naive-bayes-reading/naive-bayes-reading.md) | `ch_4/naive-bayes-reading/naive-bayes-reading.md` |
| [Gradient descent demos](../../Slides/ch_4/gradient_descent_demos.ipynb) | `ch_4/gradient_descent_demos.ipynb` |
| [KNN notebook](../../Slides/ch_5/ch_5_01_knn_bayes_boundary.ipynb) | `ch_5/ch_5_01_knn_bayes_boundary.ipynb` |
| [Bootstrap notebook](../../Slides/ch_5/ch_5_demos_2_bootstrap.ipynb) | `ch_5/ch_5_demos_2_bootstrap.ipynb` |

Each notebook name links to the file in your copy of the repo. Run `git pull` first so your copy is
up to date. The file opens at the top; the section names in the tables are the notebooks' own
headings, so you can jump straight to them:

- **VS Code or Cursor:** use the **Outline** view (in the Explorer sidebar) to click a heading.
- **JupyterLab** or **Colab:** use the **Table of Contents** panel in the left sidebar.
- **Anywhere:** search the page (Ctrl+F or Cmd+F) for the heading text or the question title.

Practice questions are listed by their block heading (for example, *Learning Check Practice* or
*Practice Questions*), then by question number.

---

## 1. Fitting a line: residuals and loss

You should be able to:

- Compute a residual, $e = y - \hat{y}$, and say whether the actual value is above or below the
  prediction.
- Explain why the plain sum of residuals is a bad way to score a line.
- Say in your own words what a **loss function** is.
- Read a contour plot of RSS over $(m, b)$: each point is one line, and lower contours mean a
  better fit.

| Notebook | Reread | Practice |
|---|---|---|
| [Regression demos](../../Slides/ch_3/regression_demos.ipynb) | Residuals · Proposal: Whichever line has the lowest *sum* of residuals is the best. (ends with the loss function paragraph) · Finding the global minimum | Practice Questions › Section 1: 1, 4, 8 |

## 2. $R^2$: how much better than guessing the mean?

You should be able to:

- Match scatter plots to $R^2$ values, including why a curved relationship or outliers give a low
  $R^2$.

| Notebook | Reread | Practice |
|---|---|---|
| [Regression demos](../../Slides/ch_3/regression_demos.ipynb) | $R^2$: better than guessing the mean? · A Few Cases | Practice Questions › Section 1: 7 |

## 3. Confidence intervals and p-values

You should be able to:

- Compute a 95% confidence interval from an estimate and its standard error (SE), and use it to
  judge whether an effect is plausibly real: does it include 0?
- Compute a p-value by simulation, and draw a conclusion with a 0.05 cutoff.

| Notebook | Reread | Practice |
|---|---|---|
| [Regression demos](../../Slides/ch_3/regression_demos.ipynb) | p-Value · Standard error · Using the standard error | Practice Questions › Section 2: 15, 19 |

## 4. Judging a classifier: recall, precision, thresholds, and ROC curves

You should be able to:

- Compute **recall** and **precision** from a confusion matrix.
- Say what raising the decision threshold does to recall, to the number of false alarms, and to the
  model's predicted probabilities.
- Choose a threshold on an ROC curve when one kind of mistake matters much more than the other.

| Notebook | Reread | Practice |
|---|---|---|
| [Classification metrics notebook](../../Slides/ch_4/ch_4_demos_2_classification_metrics.ipynb) | 4 · The confusion matrix · 5 · Metrics as questions · 6 · The threshold · 8 · ROC curves and AUC | Practice Questions (near the end): 2, 3, 8 |

## 5. Logistic regression: log-odds and straight-line boundaries

You should be able to:

- Explain why logistic regression models the **log-odds** as a linear combination of the inputs,
  rather than the probability or the odds.
- Explain why its decision boundary is a straight line when there are two inputs.

| Notebook | Reread | Practice |
|---|---|---|
| [Log-odds notebook](../../Slides/ch_4/ch_4_demos_1_log_odds_and_logistic_regression.ipynb) | Evidence multiplies odds · Logs turn multiplying into adding · From clues to a model | Learning Check Prep: Question 5 |
| [Classification metrics notebook](../../Slides/ch_4/ch_4_demos_2_classification_metrics.ipynb) | 11 · More than one input | — |

## 6. Bayes' theorem and naive Bayes

You should be able to:

- Explain why rewriting $P(D \mid \text{evidence})$ with Bayes' theorem gives us a way to estimate
  it from data, and where the naive assumption comes in.
- Read ISL's equation 4.15: say what $\pi_k$, $f_k(x)$, the posterior $\Pr(Y = k \mid X = x)$, and
  the denominator mean.

| Notebook | Reread | Practice |
|---|---|---|
| [Naive Bayes reading](../../Slides/ch_4/naive-bayes-reading/naive-bayes-reading.md) | 2. Why flipping the condition helps with classification · 6. The hard part · 7. The naive assumption · 9. Translating into ISL's notation | 11. Learning Check Practice: Question 0, Question 7 |

## 7. Gradient descent

You should be able to:

- Take one gradient descent step by hand: compute the loss, the slope, the step $-\eta \, L'(\beta)$,
  and the new value.
- Explain why steps zigzag across a long, narrow valley on a contour plot, and what to change before
  running gradient descent.

| Notebook | Reread | Practice |
|---|---|---|
| [Gradient descent demos](../../Slides/ch_4/gradient_descent_demos.ipynb) | 6 · Walking downhill by hand · 9 · Descent on a contour plot, and why scaling matters | Learning Check Practice: Question 3, 7 |

## 8. KNN, flexibility, and bias versus variance

You should be able to:

- Make a KNN prediction from a point's $K$ nearest neighbors, and give KNN's probability estimate.
- Compare training and test error curves as $K$ changes, and explain why the lowest training error
  is not a good way to choose $K$.
- Describe **bias** and **variance** as properties of a method, and explain why many high-variance
  boundaries can average out close to the Bayes boundary.

| Notebook | Reread | Practice |
|---|---|---|
| [KNN notebook](../../Slides/ch_5/ch_5_01_knn_bayes_boundary.ipynb) | 1 · KNN: how one prediction is made · 2 · Decision boundaries, flexibility, and training error · 6 · New data: test error · 6b · Bias and variance: redrawing the training data | Learning Check Practice: Question 1, 8, 9 |

## 9. The validation set approach and cross-validation

You should be able to:

- Describe the validation set approach and its two weaknesses, and explain how $k$-fold
  cross-validation addresses each one.

| Notebook | Reread | Practice |
|---|---|---|
| [KNN notebook](../../Slides/ch_5/ch_5_01_knn_bayes_boundary.ipynb) | 8 · Holding data back: the validation set approach · 9 · $k$-fold cross-validation | Learning Check Practice: Question 11, 12 |

## 10. The bootstrap

You should be able to:

- Read a bootstrap percentile interval for the difference between two models, and say what it means
  when the interval includes 0: "better, or just lucky?"

| Notebook | Reread | Practice |
|---|---|---|
| [Bootstrap notebook](../../Slides/ch_5/ch_5_demos_2_bootstrap.ipynb) | 2 · The bootstrap: resample the data you have · 3 · Back to the problem: better, or just lucky? | No practice section: answer the **Ask** prompts in Section 3 yourself before opening them |

