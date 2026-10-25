
# Course Rubrics CS-463

All grading rubrics for CS-463 live here. 

## Learning Checks

Most questions are graded automatically by Canvas the moment you submit.

**Multiple choice and true/false** are all-or-nothing.

### Select-all questions

Canvas is set up to split the question's points evenly across its *correct* options. Each correct box you check earns one share; each incorrect box you check costs one share. A question never scores below zero.

share = points ÷ (number of correct options)

score = share × (correct checked − incorrect checked), minimum 0


On a 2-point question with 3 correct options, each share is 0.67:

| What you check | Score |
| --- | --- |
| All 3 correct, nothing else | 2.00 |
| 2 of the 3 correct, nothing else | 1.33 |
| All 3 correct, plus 1 wrong | 1.33 |
| Every box on the question | 0.00 |

Checking everything is never a winning strategy. If you're unsure about one
option, leaving it blank costs you less than guessing wrong.

### Paper questions

Some checks include a question that asks you to work on paper: sketch a plot, run a short calculation, write out a step. The Canvas question will say so, and you'll hand the paper in when you submit the quiz. I grade these myself.

Until I do, Canvas shows those points as zero. **Your score isn't final until the paper portion is entered** — don't panic at the number you see on the way out the door. I'll avoid assigning this style of question during remote class sessions.

## Labs
### Rubric and Guidelines

Labs are worth **15%** of your course grade. Each lab is worth **10 points** and is graded generously, on **completion and good-faith effort**.

Labs are released Friday morning and are **due the following Monday at 8:30 AM** in Canvas. Late labs are not accepted. See the syllabus for exceptions for illness and other extenuating circumstances.

---

## Rubric (10 points total)

| Criterion | Full credit | Partial credit | No credit |
|---|---|---|---|
| **Runs cleanly (2 pts)** | The notebook runs from top to bottom after **Restart & Run All**, with no errors, and all outputs are visible. | Runs, but with minor problems: a few errors, or missing outputs. | Doesn't run from a fresh kernel, or has no outputs. |
| **Completion (4 pts)** | Every task is attempted. | Most tasks are attempted. | Few or no tasks are attempted. |
| **Good-faith effort (4 pts)** | Written answers address the question using a mix of your own words and the vocab/concepts from class. Plots have titles and labeled axes. | Answers are vague, or missing in places. | Answers are missing, or don't engage with the questions. |

"Attempted" means real effort. You don't have to get a task fully working. If you're stuck, write down what you tried and where you got stuck, and that counts as an attempt.

---

## Before you submit

1. **Kernel → Restart & Run All.** This is required. A notebook that only works because of cells you ran earlier, out of order, will fail when I run it.
2. Scroll through the whole notebook. Check that every cell ran and that all outputs, plots, and written answers are there.
3. Make sure you haven't hard-coded file paths that only exist on your computer. Load data with the paths given in the lab. When in doubt, make the data file a sibling of the notebook or load data using the ISL package.
4. Submit the `.ipynb` file in Canvas.

---

## Resources

Use whatever resources help you complete the labs: documentation, tutorials, videos, the textbook, classmates, and me. If a specific source saved you a lot of time, a short note in the notebook saying so is appreciated.

---

## AI and LLMs

AI coding assistants are allowed on labs. When you use them, you're balancing two goals.

**1. Professional skill.** Employers increasingly expect engineers to produce high-quality code quickly with AI tools. Part of this is catching the LLM's mistakes. You can only do that if you, yourself understand the material deeply.

**2. Learning.** If you're not deliberate about it, an LLM will do the thinking for you, and you'll learn very little. So use it as a tutor, not a ghostwriter. For example:

- **Clarifying the question:** "Explain what this lab question is asking. Don't solve it."
- **Getting unstuck:** "What concepts do I need to understand to tackle this? Don't tell me how to do it."
- **Checking your understanding:** "Here's my explanation of why this model overfits. What's wrong or missing?"
- **Explaining errors:** "What does this error message mean?"

**Avoid asking an LLM to simply solve the lab problems for you.**

**Exception: plotting.** Crafting plotting code can be tedious and isn't the main point of the labs. If you're short on time, go ahead and get direct LLM help writing your graphs, plots, and charts. You should scrutinize this generated code and make sure you understand it.

**The test:** You'll discuss your labs in detail with me and your classmates. That's practice for your Kaggle Programming Project oral defense. If you can't explain your code, methods, or interpretations, you're probably leaning on AI too much. Ask me for advice on adjusting your approach.


## Perspectives 

Each of these assignments is graded on a 10 pt scale. Each assignment contains a writing prompt. 

Each writing prompt usually contains a list of my desiderata for the essay. 
I will typically give you a point break down for each item in that list. If I don't, just assume each item carries approximately equal point-value, summing to 10 pts.

These assignments are never graded on the basis of your opinion or your position. They *are* graded on the basis of:

- Compliance with the writing prompt (including essay-length specs).
- The accuracy of your claims about the content of the readings.
- The quality of your arguments.

**AI-Policy for Perspectives:** 

You may use LLM's to assist with your research, stress-test your ideas, explore alternatives, etc. **Do not** ask an LLM to write the essay for you. Another way LLM's can be genuinely useful here is if you encounter formulas, mathematical expressions, symbols or even technical jargon you've never seen before. It used to be tricky to learn what some mathematical symbol or expression meant. What keyword do you even search? How do you type $\psi$ on a keyboard? Now you can just screen shot the whole expression or equation, paste it into a Claude chat and ask, "what does this expression mean?". I would consider that kind of AI usage cleanly in-bounds for Perspectives writing assignments.

*Cognitive Autonomy Tip:* Tempting as it is to just ask Claude "Please summarize this article", I highly recommend that you don't to do this. If you really think reading an LLM's summary would be valuable for some particular article or research paper, try to wait *at least* until you've read the article yourself and drafted *your own* summary. Now you can compare and contrast your summaries. You will learn more this way, I promise!


## Kaggle Programming Project

The project is worth 25% of your course grade, split into four parts:

| Component | Points | Share of project |
|---|---|---|
| Checkpoint 1 — Baseline | 15 | 15% |
| Checkpoint 2 — Model Comparison | 15 | 15% |
| Submission | 50 | 50% |
| Oral Defense | 20 | 20% |
| **Total** | **100** | **100%** |

**Your leaderboard rank is never part of your grade.** What's graded is the quality of your process and how well you understand it. A thoughtful project at 78% accuracy beats a copied one at 81%.

### Rules that apply to every part

- **Restart & Run All** before you submit. Same rule as labs.
- **Relative paths only.** Put `train.csv` and `test.csv` in a `data/` folder next to your notebook.
- **Set your random seeds** (`random_state=...`) so your CV numbers are reproducible.
- **Tune with cross-validation on `train.csv`, not the public leaderboard.** The public leaderboard only sees 30% of the test rows.
- **Don't use the original Kaggle Spaceship Titanic data** to look up answers. That breaks the competition rules and the academic integrity policy.
- **Keep an experiment log** from day one: a table in your README (or a `log.md`) with one row per Kaggle submission — date, model, features, CV accuracy (mean ± std), public LB accuracy. You'll build on it at every checkpoint.

---

### Checkpoint 1 — Baseline (15 pts)

**Goal:** a clean, honest pipeline from raw data to a Kaggle submission. Fancy models aren't the point yet; a trustworthy validation setup is.

**Submit in Canvas:** `checkpoint1.ipynb` (and any `.py` helper files it imports).

**Requirements**

1. **Look at the data.** Report how many values are missing in each column. Look at the distribution of every numeric column (a histogram or `.describe()` is enough) and note anything unusual: skew, outliers, very different ranges. Make at least two plots that tell you something about `Transported` (titles, labeled axes), each with a one- or two-sentence takeaway.
2. **Clean and transform it.**
   - Handle every missing value, and say in one sentence per column why you chose that approach.
   - Encode the categorical columns. Read the Data tab carefully: some columns pack several pieces of information into one string.
   - Transform the numeric columns where it makes sense. Rescale them if your model is sensitive to scale, and consider a transformation (e.g., `log1p`) for heavily skewed columns. In one or two sentences, say which transformations your model needs and why. KNN and logistic regression don't answer that question the same way.
3. **Establish a floor.** Report the accuracy of always predicting the majority class.
4. **Fit one real model** (logistic regression or KNN) and evaluate it with 5-fold cross-validation. Report mean ± standard deviation of accuracy.
5. **Submit to Kaggle** at least once, using predictions generated by this notebook.
6. **Start your experiment log.** In one or two sentences, compare your CV accuracy to your public leaderboard score. Is the gap bigger than your CV standard deviation?

**Rubric**

| Criterion | Full credit | Partial credit | No credit |
|---|---|---|---|
| Runs cleanly (3 pts) | Runs top to bottom after Restart & Run All; all outputs visible. | Minor errors or missing outputs. | Doesn't run from a fresh kernel. |
| Data understanding & preparation (4 pts) | Missingness and numeric distributions reported; every missing value handled with a stated reason; categoricals encoded; scaling/transformations applied and justified for the chosen model; two labeled plots with takeaways. | Preparation done, but reasons, transformations, or plots are missing or thin. | Little or no EDA or preparation. |
| Baseline & cross-validation (5 pts) | Majority-class floor reported; one model evaluated with 5-fold CV, mean ± std reported. | CV done incorrectly (e.g., single train/test split) or the floor is missing. | No validated model. |
| Kaggle submission & log (3 pts) | At least one submission from this notebook; log started; CV vs. LB gap discussed. | Submission made, but log or discussion is missing. | No submission. |

*Encouraged now, required by the final submission:* put your preprocessing (imputation, encoding, scaling) inside an sklearn `Pipeline` / `ColumnTransformer` so it's fit only on the training folds. A scaler fit on all of `train.csv` before cross-validation is a small leak.

---

### Checkpoint 2 — Model Comparison (15 pts)

**Goal:** compare model families fairly, improve your features deliberately, and practice explaining your work out loud.

**Submit in Canvas:** `checkpoint2.ipynb`, your helper files, and a draft `README.md`.

**Requirements**

1. **Compare at least three model families**, one of which is a tree ensemble (random forest or gradient boosting, from Lab 5). Use the **same CV folds** for every model. Show the results in one table: model, CV accuracy mean ± std.
2. **Tune at least one hyperparameter with CV** and plot CV accuracy against it. Say which value you chose and why.
3. **Engineer at least two new features.** For each, report CV accuracy with and without it, using the same model and folds. Keep features that help; note the ones that didn't.
4. **Leakage check.** All preprocessing lives inside a `Pipeline` (or is otherwise fit only on training folds). Point to where in your code this happens.
5. **Update your experiment log** and submit your current best model to Kaggle.
6. **Draft README.md** with: a one-paragraph project summary, how to run your code, and your plan for the remaining two weeks.
7. **Mini-defense (practice, we'll discuss scheduling details).** A short  breakout-room conversation with me: walk me through your current best model and answer one follow-up question. This is graded on participation only. Its purpose is to show you what the real defense feels like.

**Rubric**

| Criterion | Full credit | Partial credit | No credit |
|---|---|---|---|
| Runs cleanly (3 pts) | Runs top to bottom after Restart & Run All; all outputs visible. | Minor errors or missing outputs. | Doesn't run from a fresh kernel. |
| Fair model comparison (5 pts) | Three families incl. a tree ensemble, same folds, one table with mean ± std; one hyperparameter tuned via CV with a plot. | Comparison present but folds differ, std missing, or tuning missing. | Fewer than two models compared. |
| Feature engineering (3 pts) | Two or more new features, each with a with/without CV comparison. | Features added without evidence that they help. | No new features. |
| Pipeline, log & README (3 pts) | No leakage; log current; README draft has summary, how-to-run, and plan. | One of the three is missing or incomplete. | Two or more missing. |
| Mini-defense (1 pt) | Showed up and talked through your model. | — | Didn't participate. |

---

### Final Submission (50 pts)

**Submit in Canvas:** a single `.zip` (or a link to a GitHub repo, with the final commit before the deadline) containing:

- `final.ipynb` — your end-to-end pipeline, from raw CSVs to `submission.csv`
- any `.py` helper modules
- `requirements.txt` (or a list of packages and versions in the README)
- `README.md`
- your experiment log

**Your notebook must regenerate one of your two selected Kaggle submissions.** I'll run it and compare the output to the file you selected for final scoring.

**README.md must include**

1. **Summary** — the problem, your final model, and your CV accuracy, in one paragraph.
2. **How to run** — exact steps from a fresh clone.
3. **Results table** — every model family you tried, with CV mean ± std, and your final public and private LB scores.
4. **Final model choice** — why this model, and not the runner-up? Refer to your CV evidence, not your leaderboard rank.
5. **What didn't work** — at least two ideas you tried and dropped, and what you learned from each.
6. **AI-use statement** — which tools you used, for what, and one example where you caught or corrected an AI mistake.

**Rubric**

| Criterion | Excellent | Proficient | Developing | Missing |
|---|---|---|---|---|
| **Reproducibility & code quality (10 pts)** | Runs from a fresh kernel; regenerates a selected submission; seeds set; relative paths; repeated logic factored into functions. (10) | Runs and reproduces, with minor rough edges. (8) | Runs only with manual fixes, or doesn't reproduce the submission. (4) | Doesn't run. (0) |
| **Data preparation & features (10 pts)** | Every cleaning, transformation, and feature decision is justified; engineered features are backed by with/without evidence; no leakage. (10) | Sound preparation; justification or evidence thin in places. (8) | Ad hoc preparation, or leakage present. (4) | Minimal preparation. (0) |
| **Modeling & validation (12 pts)** | Three or more families compared on the same folds; hyperparameters tuned via CV; final choice weighs mean *and* variability; CV vs. LB gap discussed. (12) | Solid comparison; one element (tuning, variability, gap) missing. (9) | Few models, or chosen by public LB rather than CV. (5) | No meaningful validation. (0) |
| **Analysis & interpretation (10 pts)** | Error analysis (where does the model fail?); feature importance or coefficients interpreted in plain language; honest account of what didn't work. (10) | Interpretation present but shallow in one area. (8) | Results reported with little interpretation. (4) | None. (0) |
| **README & communication (8 pts)** | All six README sections present, clear, and accurate; plots labeled. (8) | One section missing or unclear. (6) | Several sections missing. (3) | No README. (0) |

Graders may award scores between the listed levels.

---

### Oral Defense (20 pts)

A one-on-one, roughly 12-minute conversation during the Dec 4 session. Bring your laptop with your notebook open and running.

**How it goes**

1. **Walkthrough (≈4 min).** You walk me through your pipeline, from raw data to final prediction. No slides needed; just your notebook.
2. **Your choices (≈4 min).** I'll point at specific decisions in your code and ask why you made them, and what you'd expect to happen if you'd chosen differently.
3. **Course concepts (≈4 min).** Questions that connect your project to the ideas from class: bias–variance, overfitting, cross-validation, regularization, why ensembles help, how your model actually makes a prediction.

"I don't know, but here's how I'd find out" is a good answer. Guessing confidently is a worse one.

**Rubric**

| Criterion (5 pts each) | Excellent (5) | Proficient (4) | Developing (2) | Missing (0) |
|---|---|---|---|---|
| **Walkthrough** | Explains the whole pipeline fluently in their own words; can navigate to any part of the code on request. | Clear overall, with small gaps or some reliance on reading the code aloud. | Can describe what the code does only in general terms. | Can't explain their own pipeline. |
| **Justifying choices** | Gives evidence-based reasons for decisions (refers to CV results, plots, the data) and names alternatives they considered. | Reasons are sensible but sometimes vague. | Reasons are mostly "it worked" or "the AI suggested it." | No reasons given. |
| **"What if" reasoning** | Correctly predicts the effect of a proposed change (drop a feature, change *k*, remove regularization) and explains why. | Mostly correct predictions with some prompting. | Predictions are guesses without reasoning. | Can't engage. |
| **Conceptual understanding** | Connects project to course concepts accurately and precisely, using correct vocabulary. | Mostly accurate, with minor misconceptions. | Significant misconceptions, or vocabulary without understanding. | Can't connect project to course material. |

**How to prepare:** re-read your own notebook as if a classmate wrote it. For every cell, ask "why is this here, and what would break without it?" Practice explaining your final model to someone who isn't in the class.

## Midterm and Final Exams
**Coming Soon**



