# Square Brackets in NumPy and pandas: A Refresher

This guide is for you if you've used NumPy before but feel rusty, and you've never used pandas. We'll rebuild the NumPy ideas first, then see that pandas uses the same ideas with a few extra features.

---

## Part 1: NumPy

### Setup

Every example in Part 1 uses this array:

```python
import numpy as np

a = np.arange(12).reshape(3, 4)
# array([[ 0,  1,  2,  3],
#        [ 4,  5,  6,  7],
#        [ 8,  9, 10, 11]])
```

It has 3 rows and 4 columns, so `a.shape` is `(3, 4)`.

### What square brackets really do

When you write `a[1]`, it can look like special syntax built into arrays. It isn't. Python translates square brackets into an ordinary method call:

```python
a[1]
# Python turns this into:
a.__getitem__(1)
```

Any Python object can support square brackets by defining a method called `__getitem__`. Lists, dictionaries, strings, and NumPy arrays all do. Whatever you put between the brackets is passed to that method as a single argument, called the **key**.

Assignment works the same way, using a different method:

```python
a[1] = 0
# Python turns this into:
a.__setitem__(1, 0)
```

### Try it yourself

You can prove this by writing a tiny class that just prints the key it receives:

```python
class Spy:
    def __getitem__(self, key):
        print("__getitem__ got:", repr(key))

    def __setitem__(self, key, value):
        print("__setitem__ got:", repr(key), "and value", repr(value))

s = Spy()
s[1]            # __getitem__ got: 1
s[[1]]          # __getitem__ got: [1]
s[1:3]          # __getitem__ got: slice(1, 3, None)
s[1, 2]         # __getitem__ got: (1, 2)
s[0] = 99       # __setitem__ got: 0 and value 99
```

Two things are worth noticing:

- `1:3` becomes a `slice` object. The colon syntax only works inside square brackets.
- `s[1, 2]` doesn't pass two arguments. The comma creates a **tuple**, `(1, 2)`, and that single tuple is the key. This is how NumPy handles one index per axis.

So square brackets are simply "index into this object." The object decides what the key means.

### The general rule

**Brackets select data. Parentheses do something. Bare attributes describe the object.**

| Brackets (selection) | Parentheses (methods and functions) | Bare attributes (no call) |
|---|---|---|
| `a[1]` | `a.sum()` | `a.shape` |
| `a[1:3, ::2]` | `a.reshape(3, 4)` | `a.ndim` |
| `a[mask]` | `a.mean(axis=0)` | `a.dtype` |
| `a[[0, 2]]` | `np.sort(a)` | `a.T` |

The third column trips people up. `a.shape` is already a stored value, not a method, so `a.shape()` raises `TypeError: 'tuple' object is not callable`.

Why use brackets at all instead of a method like `a.get(1)`? Two reasons:

1. **Slice syntax only works inside brackets.** `a[1:3]` is valid Python, but `a.get(1:3)` is a syntax error.
2. **You can assign to brackets.** `a[a < 0] = 0` is valid. You can't assign to a method call.

### The type of the key determines the shape of the result

This is the most important idea in this guide.

| Expression | Key type | Result | Shape |
|---|---|---|---|
| `a[1]` | int | row 1 | `(4,)` |
| `a[[1]]` | list | row 1, still 2-D | `(1, 4)` |
| `a[1:2]` | slice | row 1, still 2-D | `(1, 4)` |
| `a[1, 2]` | tuple of ints | the value `6` | scalar |
| `a[:, 1]` | (slice, int) | column 1 | `(3,)` |
| `a[:, [1]]` | (slice, list) | column 1, still 2-D | `(3, 1)` |

The pattern:

> **A single integer drops that dimension. A list or a slice keeps it.**

An integer means "exactly this one," so there's no reason to keep an axis of length 1. A list or slice means "these ones," which might be several items, so the axis stays even when it holds only one item.

The double brackets in `a[[1]]` aren't special syntax either. The outer brackets are the indexing operation, and `[1]` is an ordinary Python list used as the key.

### Boolean masks

A comparison on an array produces an array of `True`/`False` values of the same shape. Using that as the key selects the `True` positions:

```python
a > 5
# array([[False, False, False, False],
#        [False, False,  True,  True],
#        [ True,  True,  True,  True]])

a[a > 5]          # array([ 6,  7,  8,  9, 10, 11])
a[a > 5] = 0      # sets those elements to 0 in the original array
```

### A few NumPy details worth knowing

**Prefer `a[1, 2]` over `a[1][2]`.** Both return `6`, but `a[1][2]` indexes twice: it builds row 1, then indexes into that. The chained form also misleads with slices: `a[:][1]` is row 1, not column 1.

**Slices share memory. Lists and masks make copies.**

```python
row = a[1:2]
row[0, 0] = 99     # a changes too: the slice is a "view" of a

row = a[[1]]
row[0, 0] = -1     # a is unchanged: the list made a copy
```

Assigning directly through brackets (`a[[0, 2]] = 0`) always changes `a`, whatever the key type.

**Two lists pair up.** `a[[0, 2], [1, 3]]` does not give you a 2x2 block. It gives the elements at (0, 1) and (2, 3), which is `array([1, 11])`. For a block, use `a[np.ix_([0, 2], [1, 3])]`.

---

## Part 2: pandas

### What pandas is

pandas is a library for working with tables of data, like a spreadsheet you control with code. It's built on top of NumPy. It has two main objects:

- A **Series** is one-dimensional labeled data. It's like a 1-D NumPy array where every value also has a **label**. You'll most often see it as a single column of a DataFrame, but a single row comes back as a Series too.
- A **DataFrame** is a table. It's like a 2-D NumPy array where the rows have labels and the columns have names.

The labels are called the **index**. In NumPy, positions are the only way to refer to data. In pandas, you can use either the position or the label.

### Setup

Every example in Part 2 uses this DataFrame:

```python
import pandas as pd

df = pd.DataFrame(
    {"name":  ["Ana", "Ben", "Cy"],
     "age":   [23, 35, 41],
     "score": [88, 92, 79]},
    index=["a", "b", "c"],
)
#   name  age  score
# a  Ana   23     88
# b  Ben   35     92
# c   Cy   41     79
```

The row labels are `"a"`, `"b"`, `"c"`. The column names are `"name"`, `"age"`, `"score"`. The row positions are still 0, 1, 2 underneath.

### Same mechanism: `__getitem__` and `__setitem__`

Everything from Part 1 applies. `df["age"]` is `df.__getitem__("age")`, and `df["age"] = 0` is `df.__setitem__("age", 0)`. The same general rule holds: brackets select, parentheses act, bare attributes describe.

| Brackets (selection) | Parentheses (methods) | Bare attributes |
|---|---|---|
| `df["age"]` | `df.head()` | `df.shape` |
| `df.loc["b"]` | `df.describe()` | `df.columns` |
| `df.iloc[1]` | `df.sort_values("age")` | `df.index` |
| `df[df["age"] > 30]` | `df["age"].mean()` | `df.dtypes` |

### Two ways to index: `.iloc` and `.loc`

Because pandas has both positions and labels, it gives you two separate indexers:

- **`.iloc`** uses integer **positions**, just like NumPy. (Think "i" for integer.)
- **`.loc`** uses **labels**.

`.iloc` and `.loc` look like methods, but they aren't. They are attributes that hold an indexer object, and that object defines `__getitem__`. So `df.iloc[1]` is really `df.iloc.__getitem__(1)`. That's why they take square brackets and not parentheses.

```python
df.iloc[1]          # row at position 1
df.loc["b"]         # row with label "b"  (the same row)

df.iloc[1, 2]       # 92   (position 1, position 2)
df.loc["b", "score"]  # 92   (label "b", column "score")
```

### Same rule: the key type determines the shape

The rule from Part 1 carries over exactly. The only new detail is that a 1-D result is a Series and a 2-D result is a DataFrame.

| pandas | Result | NumPy equivalent | Result |
|---|---|---|---|
| `df.iloc[1]` | Series (1-D) | `a[1]` | shape `(4,)` |
| `df.iloc[[1]]` | DataFrame, 1 row (2-D) | `a[[1]]` | shape `(1, 4)` |
| `df.iloc[1:2]` | DataFrame, 1 row (2-D) | `a[1:2]` | shape `(1, 4)` |
| `df.iloc[1, 2]` | scalar | `a[1, 2]` | scalar |
| `df.iloc[:, 1]` | Series (1-D) | `a[:, 1]` | shape `(3,)` |
| `df.iloc[:, [1]]` | DataFrame, 1 column | `a[:, [1]]` | shape `(3, 1)` |

> **A single key drops that dimension. A list or a slice keeps it.**

The same is true for column selection with plain brackets:

```python
df["age"]       # Series
df[["age"]]     # DataFrame with one column
```

This matters in practice. Many machine learning tools, such as scikit-learn, expect a 2-D table of features. Passing `df["age"]` often fails, while `df[["age"]]` works.

### Boolean masks work the same way

```python
df["age"] > 30
# a    False
# b     True
# c     True

df[df["age"] > 30]                        # rows b and c
df.loc[df["age"] > 30, "score"] = 100     # assignment through __setitem__
```

### Where pandas differs from NumPy

**Plain brackets on a DataFrame select columns, not rows.** In NumPy, `a[1]` is a row. In pandas, `df["age"]` is a column. To select rows, use `.iloc` or `.loc`. (Plain brackets with a slice or a boolean mask do select rows, which is a historical quirk. When in doubt, use `.loc` or `.iloc` and be explicit.)

**Label slices include the end.** Position slices exclude it, as in NumPy, but label slices include it:

```python
df.iloc[0:2]      # rows a, b      (position 2 excluded)
df.loc["a":"c"]   # rows a, b, c   (label "c" included)
```

This is because labels don't have a natural "one past the end," so pandas includes the endpoint you named.

**Two lists give a block, not pairs.** `df.loc[["a", "c"], ["age", "score"]]` returns a 2x2 table. This is the behavior NumPy needs `np.ix_` for.

**Assign with one `.loc` call, not two sets of brackets.**

```python
df.loc[df["age"] > 30, "score"] = 100     # correct
df[df["age"] > 30]["score"] = 100         # doesn't change df
```

The second line selects rows (possibly making a copy), then assigns into that temporary result. The same idea from NumPy applies: chained indexing does two separate steps, and the assignment may not reach the original. Recent versions of pandas guarantee that chained assignment like this never changes `df`.

**Getting back to NumPy.** `df.to_numpy()` returns the underlying data as a plain NumPy array, and all of Part 1 applies to it.

---

## Summary

| Idea | NumPy | pandas |
|---|---|---|
| Brackets call | `a.__getitem__(key)` | `df.__getitem__(key)`, `df.iloc.__getitem__(key)`, `df.loc.__getitem__(key)` |
| Assignment calls | `__setitem__` | `__setitem__` |
| Index by position | `a[1]` | `df.iloc[1]` |
| Index by label | not available | `df.loc["b"]` |
| Single key | drops a dimension | drops a dimension (DataFrame → Series → scalar) |
| List or slice key | keeps the dimension | keeps the dimension |
| Plain brackets `x["..."]` | not used with strings | selects a column |
| Slice endpoint | excluded | excluded with `.iloc`, **included** with `.loc` |
| Boolean mask | `a[a > 5]` | `df[df["age"] > 30]` |
| Safe assignment | `a[mask] = v` | `df.loc[mask, "col"] = v` |

## Practice

Use the `a` and `df` defined above. Predict the answer before running the code.

1. What is the shape of `a[2]`? Of `a[[2]]`? Of `a[2:]`?
2. What is the shape of `a[:, 0]`? How would you get column 0 as a `(3, 1)` array?
3. Is `df.iloc[0]` a Series or a DataFrame? What about `df.iloc[[0]]`?
4. How many rows does `df.loc["a":"b"]` return? How many does `df.iloc[0:1]` return?
5. Write one line that sets `score` to 0 for everyone younger than 30.
6. Why does `a.shape()` raise an error, but `a.sum()` works?

<details>
<summary>Answers</summary>

1. `(4,)`, `(1, 4)`, `(1, 4)`. The integer drops the row axis. The list and slice keep it.
2. `(3,)`. Use `a[:, [0]]` or `a[:, 0:1]`.
3. `df.iloc[0]` is a Series. `df.iloc[[0]]` is a one-row DataFrame.
4. `df.loc["a":"b"]` returns 2 rows, because label slices include the end. `df.iloc[0:1]` returns 1 row, because position slices exclude it.
5. `df.loc[df["age"] < 30, "score"] = 0`
6. `a.shape` is a stored tuple, not a method, so there's nothing to call. `a.sum` is a method, so it needs parentheses to run.

</details>
