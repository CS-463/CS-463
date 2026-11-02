# Vocabulary

Terms used in the course, grouped by topic. For symbols, see [notation.md](notation.md).

<a id="data-and-features"></a>

## Data and features

<a id="feature-space"></a>

### Feature space

The space of all possible combinations of feature values, with one axis per feature. With $p$ features, each observation is a point in a $p$-dimensional space. (Also called the **input space**.)

- **Two features:** feature space is a plane. In the Hitters data with `Years` and `Hits`, each player is a dot at (`Years`, `Hits`), and the plane holds every combination, including ones no player has.
- **More features:** the same idea in more dimensions. We can't draw it, but the math works the same way.
- **The response is not an axis.** Feature space holds only the inputs. A plot with $y$ on the vertical axis, like log salary against `Years`, is not a picture of feature space.

Why it matters: many methods are easiest to understand as carving up feature space.

- A classifier's **decision boundary** (Chapter 5's KNN, logistic regression) is a line or surface in feature space that separates the regions predicted as each class.
- A **decision tree** cuts feature space into boxes (with more than two features, "hyper-boxes"), one per leaf. ISL calls these boxes **regions**, $R_1, \dots, R_J$. Each one is a set of points in feature space; see [set-builder notation](notation.md#set-builder).
