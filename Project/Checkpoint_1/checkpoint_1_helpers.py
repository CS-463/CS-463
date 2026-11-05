"""Helpers for checkpoint_1_guidance.ipynb.

Keep this file in the same folder as the notebook. The notebook stays focused on data
preparation; the setup for its planted-problems demo and its plotting code live here.
"""
from pathlib import Path
from tempfile import gettempdir

import numpy as np
from matplotlib import rc_context
from matplotlib.pyplot import subplots

BIG_FONTS = {'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 15,
             'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 13}
BAR_COLOR = '#0072B2'


def write_raw_adult_csv(X, folder=None):
    """Section 0: write X out as a "raw" CSV with two loading problems planted in it.

    Returns the path of the file. The two problems mimic the Kaggle project's train.csv:

    1. Missing values in the text columns are written as '?', the code the original
       census file used. pd.read_csv doesn't treat '?' as missing, so .isna() finds
       nothing until the reader passes na_values='?'.
    2. An extra column, born_in_us, is True/False, and blank wherever native-country is
       missing (857 rows). The blanks make pd.read_csv load it as object dtype holding
       Python bools and NaN, just like CryoSleep and VIP in the Kaggle data: .astype(bool)
       turns every NaN into True, and a 'missing' imputer followed by OneHotEncoder raises
       a TypeError for mixing bools and strings.

    Everything else is written exactly as it is in X. The file goes in the system's temp
    folder unless `folder` is given, so it doesn't clutter the student's notebook folder.
    """
    raw = X.copy()
    country = raw['native-country']
    raw['born_in_us'] = (country == 'United-States').astype(object)
    raw.loc[country.isna(), 'born_in_us'] = None              # written as a blank field
    for column in raw.columns:
        if column != 'born_in_us' and raw[column].isna().any():
            raw[column] = raw[column].astype(object).where(raw[column].notna(), '?')
    path = Path(folder if folder is not None else gettempdir()) / 'adult_raw.csv'
    raw.to_csv(path, index=False)
    return path


def plot_capital_gain(values):
    """Section 3: a histogram of the nonzero capital-gain values, with 99,999 marked."""
    nonzero = values[values > 0]
    with rc_context(BIG_FONTS):
        fig, ax = subplots(figsize=(12, 5.5))
        ax.hist(nonzero, bins=np.arange(0, 102_000, 2_000), color=BAR_COLOR, ec='white')
        ax.set_yscale('log')
        n_code = int(np.sum(values == 99_999))
        ax.annotate(f'{n_code} people at exactly 99,999', (99_999, n_code),
                    xytext=(-260, 40), textcoords='offset points', fontsize=14,
                    arrowprops=dict(arrowstyle='->', lw=1.5))
        ax.set_xlabel('capital-gain (dollars)')
        ax.set_ylabel('number of people (log scale)')
        ax.set_title(f'capital-gain for the {len(nonzero):,} people with a nonzero value')
        fig.tight_layout()
    return fig
