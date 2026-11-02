# Notation

Symbols used in the course, grouped by topic, with how to read them aloud.

<a id="sets"></a>

## Sets

A **set** is a collection of things, called its **elements**. In this course the elements are usually numbers, data points, or points in [feature space](vocabulary.md#feature-space).

<a id="curly-braces"></a>

### Curly braces: a set

Curly braces list the elements of a set.

| Notation | Read as | Meaning |
|---|---|---|
| $\{1, 2, 3\}$ | "the set containing 1, 2, and 3" | A set with three elements. Order doesn't matter, so $\{3, 1, 2\}$ is the same set. |
| $\{\}$ or $\varnothing$ | "the empty set" | The set with no elements. |

<a id="membership"></a>

### Membership: $\in$ and $\notin$

| Notation | Read as | Meaning |
|---|---|---|
| $x \in A$ | "$x$ is in $A$" or "$x$ is an element of $A$" | $x$ is one of the elements of the set $A$. |
| $x \notin A$ | "$x$ is not in $A$" | $x$ is not one of the elements of $A$. |

Example: $2 \in \{1, 2, 3\}$ and $5 \notin \{1, 2, 3\}$.

<a id="set-builder"></a>

### Set-builder notation: the bar means "such that"

When a set is too big to list, we describe it by a rule instead. The vertical bar inside curly braces separates the elements from the rule they must satisfy:

$$\{X \mid X_j < s\}$$

Read it as **"the set of all $X$ such that $X_j < s$."** Left of the bar: what the elements look like. Right of the bar: the condition they have to meet.

Some books use a colon instead of a bar; $\{X : X_j < s\}$ means the same thing.

Example (ISL equation 8.2): a tree's split on feature $j$ at split point $s$ cuts the feature space into two boxes,

$$R_1(j, s) = \{X \mid X_j < s\} \quad\text{and}\quad R_2(j, s) = \{X \mid X_j \ge s\}.$$

$R_1(j, s)$ is "the set of all points $X$ whose $j$th feature is less than $s$."

<a id="such-that-vs-given"></a>

### Not the same bar as conditional probability

The vertical bar also appears in conditional probability, where it means something different:

| Where the bar appears | Example | Read the bar as |
|---|---|---|
| Inside curly braces $\{\ \}$ | $\{X \mid X_j < s\}$ | **"such that"**: it builds a set |
| Inside $P(\ )$ or $E[\ ]$ | $P(Y = 1 \mid X = x)$ | **"given"**: it states what we already know |

$P(Y = 1 \mid X = x)$ is "the probability that $Y = 1$, given that $X = x$." The "given" bar never appears inside curly braces, so the braces tell you which reading applies.

<a id="sum-over-a-set"></a>

### A condition under a sum

A sum's subscript can name a condition instead of a range:

$$\sum_{i:\ x_i \in R_1} (y_i - \hat{y}_{R_1})^2$$

Read it as **"the sum over every $i$ such that $x_i$ is in $R_1$"**: add up the term once for each training point that falls in the box $R_1$, and skip the rest. The colon plays the same "such that" role as the bar in set-builder notation. (This is from ISL equation 8.3, a tree split's RSS.)

<a id="vertical-bars"></a>

### Other vertical bars

Vertical bars show up in several more places. Look at what's around them:

| Notation | Read as | Meaning |
|---|---|---|
| $\lvert x \rvert$, around a number | "the absolute value of $x$" | Distance from 0: $\lvert -3 \rvert = 3$. Lasso's penalty uses $\lvert \beta_j \rvert$. |
| $\lvert A \rvert$, around a set | "the size (or *cardinality*) of $A$" | The number of elements in $A$: $\lvert \{1, 2, 3\} \rvert = 3$. Pruning's $\lvert T \rvert$ is the number of leaves in tree $T$. |
| $\{\ \cdots \mid \cdots\ \}$ | "such that" | Set-builder notation (above). |
| $P(A \mid B)$ | "given" | Conditional probability (above). |
