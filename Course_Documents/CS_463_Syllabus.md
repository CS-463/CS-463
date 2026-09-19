# CS 463: Artificial Intelligence — Syllabus

**Edmonds College · Fall 2026 · 5 credits**
**MWF 8:30–10:20 AM · Monday and Wednesday in person · Friday Lab - remote, synchronous**

**Text: *An Introduction to Statistical Learning with Applications in Python* (ISLP) — free at [statlearning.com](https://www.statlearning.com/) (download the *Python-based* book, *NOT* the R-based version.)**

## Instructor

**Bradley Lignoski**

- **Email:** bradley.lignoski+cs463@edmonds.edu — the `+cs463` is important. My inbox filters use it to make sure I prioritize *your* messages, so please keep it in the address.
- **In person:** catch me before or after our Monday and Wednesday sessions.
- **Response time:** I'll do my best to reply within 24 hours on weekdays. Weekends may take up to 48 hours, and official holidays can take longer.
- **For an office-hours style conversation:** email me and we'll set up a time. We can also usually jump into a breakout room during a Friday lab.

---

> **This calendar is subject to change.** Its purpose is to give you a solid but low-resolution idea of what to expect. Canvas carries all due dates, submission links, and assignment details — **always treat the assignment descriptions and deadlines in Canvas as the official source of truth.**



| Week | Dates | Topic | Reading | Lab / Activity | Major events |
|---|---|---|---|---|---|
| 1 | Sep 21, 23, 25 | ML-Landscape: Major paradigms. Notation intro. | Ch. 1; §2.1–2.2 | Lab 1 — Python and notebook fluency | Course intro; Environment setup and tooling intro Fri |
| 2 | Sep 28, 30, Oct 2 | Linear regression | §3.1–3.3, §3.5 | Lab 2 — Linear regression<br>**Perspectives 1 due Thu Oct 1** | |
| 3 | Oct 5, 7 | Classification and its metrics | §4.1–4.3 | — | **No class Fri Oct 9** |
| 4 | Oct 12, 14, 16 | Bayes; gradient descent | §4.4–4.5; §10.7 (opening) | Lab 3 — Gradient descent<br>**Perspectives 2 due Thu Oct 15** |  |
| 5 | Oct 19, 21, 23 | Cross-Validation; Bootstrap | Ch. 5 | Lab 4 — Cross-validation and the bootstrap<br>**Perspectives 3 due Thu Oct 22** |  **Kaggle Programming Project released** |
| 6 | Oct 26, 28 | Regularization; linear model selection; PCA Intro | §6.1–6.2, §6.3.1, §6.4 | — | **Midterm — Wed Oct 28**; no class Fri Oct 30 |
| 7 | Nov 2, 4, 6 | Trees and ensembles | Ch. 8 | Lab 5 — Tree-based methods<br>**Perspectives 4 due Thu Nov 5** | **Kaggle Programming Project checkpoint 1**; midterm returned by Nov 4 |
| 8 | Nov 9, 13 | Unsupervised learning: PCA and clustering | §12.1–12.2, §12.4 | Lab 6 — PCA and clustering<br>**Perspectives 5 due Thu Nov 12** | **No class Wed Nov 11** (Veterans Day) |
| 9 | Nov 16, 18, 20 | Neural networks | §10.1–10.2, §10.6–10.7 | Lab 7 — Train and diagnose a network<br>**Perspectives 6 due Thu Nov 19** |  **Kaggle Programming Project checkpoint 2** |
| 10 | Nov 23 | Embeddings, retrieval, and recommenders | §10.4, §12.3 | — | Monday only; **no class Nov 25, 27** (Thanksgiving) |
| 11 | Nov 30, Dec 2, 4 | Double descent; LLM tour | §10.8 | — | **Kaggle Programming Project due Wed Dec 2**; **project defenses Fri Dec 4** |
| — | Wed Dec 9 | **Final exam** (cumulative) | | | Finals week |

## Calendar Notes

- The midterm is returned by **Nov 4**, so you'll have graded feedback in hand well before the withdrawal deadline.
- **Week 11 Wednesday** is an exposure session on large language models. No final exam question drawn from *that session* will count against you.
- **Kaggle Programming Project defenses** are individual conversations about your own project, scheduled during the Dec 4 session.


# Grading


| Component | Weight |
|---|---|
| Learning Checks | 10% |
| Labs | 15% |
| Perspectives | 10% |
| Kaggle Programming Project | 25% |
| Midterm Exam | 20% |
| Final Exam | 20% |

I reserve the right to occasionally "throw out" a question or task that stumps almost everyone, or otherwise rescale grading on an assignment or category to the top score (not a curve, but a shift in everyone's favor).

## Grade Scale

Final grades are reported on Edmonds College's 0.0–4.0 decimal scale. Your weighted course percentage converts as follows:

- **95% and above → 4.0**
- **65% through 94%** — one tenth of a grade point per percentage point, starting at 1.0 for 65%. Formally, `decimal = 1.0 + (percentage − 65) × 0.1`, rounded to one decimal place.
- **Below 65% → 0.0.** Edmonds College does not award passing grades below 1.0, so there is no decimal grade between 0.0 and 1.0.

Reference points:

| Percentage | 95 | 90 | 85 | 80 | 75 | 70 | 65 | below 65 |
|---|---|---|---|---|---|---|---|---|
| **Decimal grade** | 4.0 | 3.5 | 3.0 | 2.5 | 2.0 | 1.5 | 1.0 | 0.0 |

Your weighted percentage is rounded to the nearest whole percent before conversion.

---

## Learning Checks — 10%

At the start of most class sessions you'll complete a short check on material from the previous session. The Checks are delivered through Canvas, so bring a laptop on Monday and Wednesday, or let me know ahead of class if you'll need to borrow one. If you borrow one, make sure you know how to log into Canvas on it. The learning checks are closed book, closed note, and no consulting LLMs, etc. 

**Each check is worth 10 points.** The format varies — sometimes ten quick one-point questions, sometimes five two-point questions, sometimes two longer five-point questions.

These are designed to take **5 to 10 minutes** and to be straightforward. They cover notation, definitions, reading a plot, interpreting a short piece of code — the basics of what we just did. **There are no gotchas and no trick questions.** If you review your notes and take the reading seriously, you will be fine.

**Your four lowest scores are dropped**, which covers illness, a rough morning, etc. Because of that, there are no make-ups. 

---

## Labs — 15%

Most Fridays we meet remotely for a lab. You'll fetch a notebook from the course GitHub repository, and we'll spend most of the session with you working through it while I'm available for questions.

**Labs are graded generously, on completion and good-faith effort.** 

One firm requirement:

> **Every submitted notebook must run correctly from a fresh kernel.** Use `Restart & Run All` before you submit.

Labs are released Friday morning and are due the following **Monday at 8:30 AM**, submitted in Canvas. 

**No late labs are accepted.** The labs are straightforward, the grading is generous, and you have three days to finish — including two weekend days.

Exceptions are made for illness and other extenuating circumstances, in accordance with the college's policies. If you anticipate missing a lab deadline, send me a message as soon as possible, ideally before the deadline passes.

AI-coding assistants are allowed for working on labs. You're going to talk about your labs in detail and often. If you find that you can't explain your code, methods, interpretations, etc, this might mean that you're leaning on AI too much. Ask me for advice on how to improve your approach. Remember, you're preparing for your Kaggle Programming Project oral defense every time you complete a lab and discuss it with me or your peers.

---

## Perspectives — 10%

**Six times during the quarter** I'll assign a short set of readings on a topic related to AI. The readings and the writing prompt go out Monday, and we discuss them for about 20 minutes on Friday before the lab. Not every week has a Perspectives assignment. See the calendar above for dates.

Submit your response by **Thursday at 11:59 PM**, the night before we discuss the readings in class. Each response is graded on a 10-point scale.

**On using LLMs:** You can use LLMs to help form your ideas, do research, find sources, or stress-test your reasoning. But the essay you submit must be your own, original writing. Submitting text you did not write as your own work is plagiarism under the college's academic integrity policy — see the Academic Integrity section at the end of this document.

**Why Perspectives Essays are part of the course:** As graduates of this course, you will have more technical knowledge about AI than the vast majority of your peers, family and fellow citizens. As such, you'll likely have an out-sized influence on the AI-relevant opinions of others. Perspectives Essays and in-class discussions are meant to ensure that your consequential views have been formed, challenged and updated before they impact your wider social circle. Equally important, you'll gain practice using sound methods to evaluate arguments both in written form, and live discussions.

---

## Kaggle Programming Project — 25%

A single project running from Week 5 through the end of the quarter, hosted as a private Kaggle competition. **This year we're working with the Cell Phone Addiction dataset.**

You'll submit predictions to Kaggle throughout the quarter and see how your models score against held-out data. **Your leaderboard rank is not part of your grade.** The leaderboard is feedback, not assessment.

### Kaggle Programming Project grade breakdown

| Component | Share of project grade |
|---|---|
| Checkpoint 1 | 10% |
| Checkpoint 2 | 10% |
| Submission | 40% |
| Oral Defense | 40% |

**Checkpoints — 20%.** Two checkpoints spread across the quarter, each released with at least two weeks of lead time. Checkpoints ensure that you're on track to finish the project on time.

**Submission — 40%.** Your completed work: notebook(s), helper functions, `README.md`, and anything else you produced. Detailed requirements are in the project instructions on Canvas.

**Oral Defense — 40%.** In the age of LLMs, it is incredibly easy to submit working code, having learned almost nothing. The oral defense is a one-on-one conversation at the end of the quarter about your project, your choices, and your analysis, scheduled during the Dec 4 session. I'll provide practice opportunities throughout the quarter and guidance on how to prepare. Don't be intimidated by the term "defense": there won't be any gotchas, but you're taking a real risk if you don't prepare.

**The defense follows the same make-up rule as the exams.** Rescheduling happens only in the case of a serious emergency, requires that you contact me before your scheduled slot, and requires the same kind of documentation. Travel plans and schedule convenience are not grounds for rescheduling.

### On AI tools

You may use any coding assistant you like on this project. 

Remember, in the oral defense, I will ask you to explain your own work, in person, out loud. I'll ask you some additional questions about your project, too. If you understand what you built, and you've practiced coding with the relevant libraries, etc, the defense will be straightforward. 

---

## Midterm — 20% · Final — 20%

Both exams are taken in person on campus, with devices put away. You may not use books, notes, internet searches, friends, internet message boards, AI assistants, Python interpreters, or any other reference or tool during the exams.

**Make-up exams will not be given except in the case of a serious emergency.** If you must miss an exam, even if you are sick or injured, you must contact me before the exam, or arrange for someone to do so on your behalf. You must show evidence that you were physically unable to take the exam, such as a clear and specific doctor's note mentioning the date, the exam, and the reason. No make-ups will be granted for personal reasons such as travel, personal hardship, leisure, or to ease your exam-week schedule.

**Midterm: Wednesday, October 28.** Covers the first half of the course. You'll get it back the following week, so you have graded feedback before the withdrawal deadline.

**Final: Wednesday, December 9.** **Cumulative.** 

**Both exams look like the Learning Checks.** Same kinds of questions, just more of them: reading and interpreting plots, explaining what a piece of code does or why it's wrong, short conceptual questions, small calculations. 

### Week 11 material

Week 11 contains two kinds of material, and the final treats them differently.

**Monday, Nov 30 — double descent (§10.8) is fair game.** This is regular course material and can appear on the final as an ordinary graded question.

**Wednesday, Dec 2 — the large language model session is exposure only.** That material arrives too late in the quarter to hold you accountable for it on an exam, so **any final exam question drawn from the Wednesday LLM session is extra credit only.** It can help your score and it cannot hurt it.

Learning Checks that week are graded normally.

---

# College Policies and Student Resources

## Academic Integrity

Edmonds College students shall demonstrate academic integrity. Instructors are required to report all violations of academic integrity (cheating and plagiarism) to the College, where the record is maintained for three years. Evidence of repeat incidents results in additional action by the Office of the Vice President for Student Services, as governed by the [Student Code of Conduct (WAC 132Y-125)](https://app.leg.wa.gov/wac/default.aspx?cite=132Y-125).

What counts as acceptable AI assistance varies by assignment in this course — see the Labs, Perspectives, and Kaggle Programming Project sections above. If you're unsure whether something is allowed, ask me before you submit.

## Services for Students with Disabilities

Services for Students with Disabilities (SSD), located in MLT 159, ensures that programs at Edmonds College are accessible and usable by students regardless of ability. If you require an accommodation for a disability, or if you have struggled with learning in the past and might benefit from a consultation, please contact [SSD](https://www.edmonds.edu/student-services/services-for-students-with-disabilities/contact.html) by phone at 425.640.1320 or by email at ssdmail@edmonds.edu. Note that accommodations must be [requested each quarter](https://www.edmonds.edu/student-services/services-for-students-with-disabilities/request-accommodations.html).

## Absence for Reasons of Faith or Conscience

Students who will be absent from course activities for reasons of faith or conscience may request reasonable accommodation so that grades are not impacted. Requests must be made within the first two weeks of the quarter and should follow the college's Absence for Reasons of Faith or Conscience policy (SS 8.01Pr).

## Non-Discrimination and Title IX

Edmonds College does not discriminate on the basis of race; color; religion; national origin; sex; disability; sexual orientation; age; citizenship, marital, or veteran status; or genetic information in its programs and activities.

As your instructor, I am required to report any incidents of gender discrimination or sexual harassment to the college's Title IX Officer — see [Title IX reporting](https://www.edmonds.edu/about-edmonds/titleix/reporting.html). If you would prefer to speak with a confidential counselor instead, contact the Counseling and Resource Center (MLT 145, 425.640.1358).

## Student Resources

Edmonds College offers tutoring, advising, counseling, food and emergency assistance, and technology help. Asking for help, whether you are struggling or not, is a normal part of being a student — browse [Student Services](https://www.edmonds.edu/student-services/) to find what you need.

## Registration and Withdrawal Dates

Refund deadlines, the last day to drop without a transcript entry, the last day to withdraw, and similar dates are set by the college, not by me. **You are responsible for keeping track of them.** They're all published on the [2026–27 Academic Calendar](https://www.edmonds.edu/calendar/academic.html).

---

*This syllabus is intended to give students guidance in what may be covered during the term and will be followed as closely as possible. However, the instructor may modify, supplement, and make changes to the course in the event of extenuating circumstances, by mutual agreement, and/or to ensure better learning.*
