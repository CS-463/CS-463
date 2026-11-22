
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

**The test:** You'll discuss your labs in detail with me and your classmates. If you can't explain your code, methods, or interpretations, you're probably leaning on AI too much. Ask me for advice on adjusting your approach.


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

The project is worth 25% of your course grade, split into two parts:

| Component | Points | Share of project |
|---|---|---|
| Checkpoint 1 — Baseline | 15 | 15% |
| Final Submission | 85 | 85% |
| **Total** | **100** | **100%** |

**Your leaderboard rank is never part of your grade.** What's graded is the quality of your process and how well you understand it. A thoughtful project at 78% accuracy beats a copied one at 81%.

### Rules that apply to every part

- **Restart & Run All** before you submit. Same rule as labs.
- **Relative paths only.** Put `train.csv` and `test.csv` in a `data/` folder next to your notebook.
- **Set your random seeds** (`random_state=...`) so your CV numbers are reproducible.
- **Tune with cross-validation on `train.csv`, not the public leaderboard.** The public leaderboard only sees 30% of the test rows.
- **Don't use the original Kaggle Spaceship Titanic data** to look up answers. That breaks the competition rules and the academic integrity policy.
- **Keep an experiment log** from day one: a table in `log.md` (the template is in `Project/Checkpoint_1/`) with one row per Kaggle submission — date, model, features, CV accuracy (mean ± std), public LB accuracy. You'll keep it current through your final submission.

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

### Final Submission (85 pts)

**Goal:** turn your Checkpoint 1 baseline into a finished project. Compare model families fairly, tune one setting, engineer a feature, and explain what you found, with cross-validation evidence for every choice.

**Start from** `Project/Final/final.ipynb`, a skeleton like Checkpoint 1's, and its reference notebook, `final_guidance.ipynb`. The guidance works through every step on a different data set, and its Section 1 has a ready-to-copy pipeline for each of five model families.

**Submit in Canvas:** `final.ipynb`, your `log.md`, and any `.py` helper files the notebook imports. Don't submit the data files.

**Requirements**

1. **Reproduce a submission.** Restart & Run All must write the `submission.csv` you selected for final scoring on Kaggle. I'll run it and compare the output to that file.
2. **Start from Checkpoint 1.** Bring in your Checkpoint 1 preparation and preprocessing, revised using its feedback, and say what you changed.
3. **Compare at least three model families**, one of which is a tree ensemble (a random forest or gradient boosting, from the *Ensembles* demo and the guidance menu). A neural network (`MLPClassifier`, from Lab 7) is a welcome choice for another. Use the **same CV folds** for every model, in one table of mean ± std and fold-by-fold accuracy (the skeleton's `compare` function makes it). Judge whether your best model's lead is real with the guidance's three questions: the gap against the spread, whether it holds in every fold, and the gap as a passenger count.
4. **Tune one setting with CV.** Try 5–7 widely spread values, plot CV accuracy against them (the skeleton provides the plot), and say which value you chose and why.
5. **Engineer at least one feature.** Report CV accuracy with and without it, on the same model and folds, and explain why it helped or didn't. A feature that didn't help still counts if you test it and explain the result.
6. **Put each step in the right place.** A feature built from a row's own values, or from `X` and `X_test` without the label, goes before cross-validation; anything that learns from the training rows (imputing, scaling, encoding) goes inside the pipeline; nothing is built from the label.
7. **Write it up.** Answer the notebook's write-up prompts: a summary, CV against the leaderboard, what didn't work, and AI use.
8. **Keep your experiment log current**, through your last Kaggle submission.

**Rubric**

| Criterion | Excellent | Proficient | Developing | Missing |
|---|---|---|---|---|
| **Reproducibility (15 pts)** | Runs from a fresh kernel and writes the selected submission; seeds set; relative paths. (15) | Runs and reproduces, with minor rough edges. (12) | Runs only with manual fixes, or doesn't reproduce the selected submission. (6) | Doesn't run. (0) |
| **Features & preprocessing (20 pts)** | Checkpoint 1 feedback addressed; at least one engineered feature, with CV accuracy with and without it on the same model and folds, and a sound explanation of the result; every step in the right place (before CV, inside the pipeline, or nowhere), with no leakage. (20) | Sound work; the explanation or evidence is thin in places. (16) | A feature without with/without evidence, feedback not addressed, or leakage present. (8) | No engineered feature. (0) |
| **Model comparison & tuning (30 pts)** | Three or more families, including a tree ensemble, on the same folds in one table; one setting tuned through CV, with the plot; choices judged with the three questions (gap against spread, fold by fold, passenger count), not the leaderboard. (30) | Solid comparison; one element (shared folds, tuning plot, or judging the gaps) missing or weak. (23) | Fewer than three families, or choices made by public leaderboard rather than CV. (12) | No meaningful validation. (0) |
| **Write-up & log (20 pts)** | Every write-up prompt answered clearly and accurately; CV compared with the leaderboard; at least one thing that didn't work and what it taught; log current; plots labeled. (20) | One prompt thin or missing, or the log is behind. (16) | Several prompts missing. (8) | No write-up. (0) |

Graders may award scores between the listed levels.

---

## Midterm and Final Exams
**Coming Soon**



