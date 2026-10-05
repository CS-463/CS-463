# Chapter 4 demos: updating your environment

The interactive demos in the second Chapter 4 notebook (`ch_4_demos_2_classification_metrics.ipynb`)
use sliders and buttons that need one package that wasn't in the course environment before:
**`ipywidgets`**. If you set up your environment before October 4, add it using the steps below for
whichever setup you use. (The first notebook's demos work without it.)

If you haven't set up an environment at all yet, follow
[First-time setup](https://github.com/CS-463/CS-463#first-time-setup) in the main README instead.
It already includes `ipywidgets`.

## Files you need

Get all three files from this folder and keep them **in the same folder** on your computer:

- `ch_4_demos_1_log_odds_and_logistic_regression.ipynb`
- `ch_4_demos_2_classification_metrics.ipynb`
- `ch_4_demos_helpers.py` (the notebooks import their demos from this file)

If you cloned the course repo, run `git pull` to get them.

## Add `ipywidgets` to your environment

Pick the section that matches how you set up your environment. If you're not sure, open a terminal
and run `conda env list`. If that command works and lists a `CS_463` environment, use the conda
instructions.

<details>
<summary><b>conda (Miniforge, Miniconda, or Anaconda)</b></summary>

Open a terminal:

- **Windows:** open the *Miniforge Prompt* or *Anaconda Prompt* from the Start menu.
- **macOS:** open *Terminal*.
- **Linux:** open your usual terminal.

**First, activate the course environment.** If you skip this, conda installs `ipywidgets` into a
different environment and the notebooks still won't work.

```bash
conda activate CS_463
```

Your prompt should now start with `(CS_463)`. If conda says the environment doesn't exist, run
`conda env list` to find its actual name (for example, `cs_463`) and activate that instead.

Then install the package:

```bash
conda install -c conda-forge ipywidgets
```

Type `y` if conda asks you to confirm. Then [check that it worked](#check-that-it-worked).

</details>

<details>
<summary><b>Regular Python virtual environment (<code>venv</code>)</b></summary>

Open a terminal and `cd` to the folder that contains your `CS_463` environment folder.

**First, activate the environment:**

macOS / Linux:

```bash
source CS_463/bin/activate
```

Windows (PowerShell):

```powershell
CS_463\Scripts\Activate.ps1
```

Windows (Command Prompt):

```bat
CS_463\Scripts\activate.bat
```

Your prompt should now start with `(CS_463)`. Then, on any OS:

```bash
pip install ipywidgets
```

Then [check that it worked](#check-that-it-worked).

</details>

<details>
<summary><b>Google Colab</b></summary>

Colab already has `ipywidgets`, so there's nothing new to install. For each notebook:

1. Open [Colab](https://colab.research.google.com/) and upload the notebook (**File > Upload
   notebook**).
2. Click the folder icon in the left sidebar and upload `ch_4_demos_helpers.py`. Without it, the
   demo cells fail with `ModuleNotFoundError: No module named 'ch_4_demos_helpers'`.
3. Add this as the first cell and run it once per session:

   ```python
   %pip install -q lifelines pygam
   %pip install -q --no-deps ISLP
   ```

Uploaded files disappear when the Colab session ends, so re-upload `ch_4_demos_helpers.py` each
time you come back, and download your notebook (**File > Download > Download .ipynb**) before you
close the tab.

</details>

## Check that it worked

With your environment still activated (conda or venv), run:

```bash
python -c "import ipywidgets, plotly; from ISLP import load_data; print('ipywidgets', ipywidgets.__version__)"
```

You should see a line like `ipywidgets 8.1.5`. The exact version doesn't matter. If you see
`ModuleNotFoundError`, the environment probably wasn't activated when you installed. Activate it
and run the install command again.

Then **restart your notebook's kernel** so it picks up the new package:

- **VS Code or Cursor:** open a Chapter 4 notebook, make sure the kernel (top right) is `CS_463`,
  and click **Restart**. If the sliders still don't appear, reload the window
  (<kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>P</kbd>, or <kbd>Cmd</kbd>+<kbd>Shift</kbd>+<kbd>P</kbd> on
  macOS, then *Developer: Reload Window*).
- **JupyterLab:** if `jupyter lab` was already running, stop it (<kbd>Ctrl</kbd>+<kbd>C</kbd> in its
  terminal) and start it again from a terminal where the environment is activated.

Open `ch_4_demos_2_classification_metrics.ipynb` and run its cells from the top, through
*Demo 1.1: Which curve explains the data?* You should see a blue **Reveal** button above two plots.
Clicking it should label which curve is the computer's fit.

## If the demos still don't show up

| What you see | What to do |
|---|---|
| `ModuleNotFoundError: No module named 'ipywidgets'` | The kernel is using a different environment. Select the `CS_463` kernel, or install into the environment the kernel is using. |
| `ModuleNotFoundError: No module named 'ch_4_demos_helpers'` | `ch_4_demos_helpers.py` isn't in the same folder as the notebook. Move it there (or upload it, on Colab). |
| Text like `VBox(children=(ToggleButton(...` instead of a button or slider | The package is installed but the notebook window hasn't caught up. Restart the kernel and reload the window (VS Code/Cursor) or restart JupyterLab. |
| `Error displaying widget` | Same as above: restart the kernel, then reload the window or restart JupyterLab. |

If nothing here fixes it, it's often fastest to rebuild your environment from scratch. See
*Rebuilding your environment* in the
[main README](https://github.com/CS-463/CS-463#check-your-setup).
