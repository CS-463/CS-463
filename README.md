# CS 463: Artificial Intelligence

Course materials for CS 463 at Edmonds College, Fall 2026. Labs are in [`Labs/`](Labs/), and the
syllabus and slides are in [`Course_Documents/`](Course_Documents/).

## First-time setup

Pick **one** of the three options below. If you're not sure, use conda.

All three install the same packages. `ISLP` is installed with `--no-deps` on purpose: without it,
`ISLP` pulls in PyTorch (several GB), which this course doesn't need.

<details>
<summary><b>Option 1: conda (Miniforge, Miniconda, or Anaconda)</b></summary>

If you don't have conda yet, install [Miniforge](https://conda-forge.org/download/).

1. Download [`cs_463_environment.yml`](cs_463_environment.yml). If you're downloading it in your
   browser, open the file on GitHub, click **Raw**, and save it. Make sure the name ends in `.yml`,
   not `.yml.txt`. If you cloned this repo, the file is already in the top folder.
2. Open a terminal (on Windows, use the *Miniforge Prompt* or *Anaconda Prompt*), `cd` to the
   folder containing the file, and run:

   ```bash
   conda env create -f cs_463_environment.yml
   conda activate CS_463
   pip install --no-deps ISLP
   ```

3. Run the [check](#check-your-setup) below.

</details>

<details>
<summary><b>Option 2: a regular Python virtual environment (<code>venv</code>)</b></summary>

You need Python 3.11 from [python.org](https://www.python.org/downloads/). In a terminal, `cd` to
the folder where you keep your course work, then:

macOS / Linux:

```bash
python3.11 -m venv CS_463
source CS_463/bin/activate
```

Windows (PowerShell):

```powershell
py -3.11 -m venv CS_463
CS_463\Scripts\Activate.ps1
```

Then, on any OS:

```bash
pip install numpy "pandas<3" scipy matplotlib scikit-learn statsmodels plotly nbformat ipykernel ipywidgets jupyter lxml joblib lifelines pygam
pip install --no-deps ISLP
```

Then run the [check](#check-your-setup) below.

</details>

<details>
<summary><b>Option 3: Google Colab</b></summary>

Nothing to install on your computer. Open [Colab](https://colab.research.google.com/), upload the
lab notebook, and also upload any files the lab uses (data files like `Auto.csv`, and helper
files like `ch_03_lab_helpers.py`) using the folder icon in the left sidebar.

Colab already has most packages. Add this as the first cell of each notebook and run it once per
session:

```python
%pip install -q lifelines pygam
%pip install -q --no-deps ISLP
```

Uploaded files disappear when the Colab session ends, so download your notebook
(**File > Download > Download .ipynb**) before you close the tab.

</details>

### Activating your environment

Activating an environment tells a terminal to use that environment's Python and packages. Each
time you open a new terminal window, activate the environment before you run a command that uses
the course packages. For example, activate it before you:

- run the setup check below
- install a package with `pip install` or `conda install`
- start JupyterLab with `jupyter lab`
- run one of your own scripts with `python`

If you skip this step, the terminal uses a different Python. Then `python` can't find the course
packages, and `pip install` or `conda install` puts new packages in the wrong place. Notebooks in
VS Code or Cursor don't need this step, because the kernel you select already points at the
environment, but the *terminal* in vs code **does** need this step.

conda (on Windows, in the Miniforge Prompt or Anaconda Prompt):

```bash
conda activate CS_463
```

venv, from the folder that contains your `CS_463` folder:

```bash
source CS_463/bin/activate          # macOS / Linux
```

```powershell
CS_463\Scripts\Activate.ps1         # Windows (PowerShell)
```

Your prompt should now start with `(CS_463)`. Run `conda deactivate` (conda) or `deactivate`
(venv) to switch back.

### Check your setup

With your environment [activated](#activating-your-environment), run:

```bash
python -c "import numpy, pandas, sklearn, statsmodels, plotly; from ISLP import load_data; print(load_data('Carseats').shape)"
```

You should see `(400, 11)`. Then open a lab notebook and select your environment as the kernel.
In VS Code or Cursor, click *Select Kernel* (top right), choose *Python Environments*, and pick
`CS_463`. In JupyterLab, run `jupyter lab` from a terminal where the
environment is activated.

<details>
<summary><b>Rebuilding your environment</b></summary>

If your kernel keeps crashing, imports fail in ways a single `install` doesn't fix, or you
installed packages into the wrong environment, it's usually faster to delete the environment and
build a fresh one than to repair it. Your notebooks and other files aren't affected: they live in
your course folder, not in the environment.

**Conda instructions:** Close VS Code, Cursor, or Jupyter first, so nothing is using the environment. Then, in
the Miniforge or Anaconda Prompt:

```bash
conda deactivate
conda env list
conda env remove -n CS_463
```

`conda env list` shows your environments. If yours has a different name (like `cs_463`), use that
name in the `remove` command. Then get the latest
[`cs_463_environment.yml`](cs_463_environment.yml) and follow Option 1 starting at step 2, above.

**Venv Instructions:** Close your editor, delete the `CS_463` folder, and follow Option 2 again.

After rebuilding, run the [check](#check-your-setup), reopen your notebook, select the kernel
again, and restart it. If the old environment still shows up in the kernel list, reload the editor
window.

</details>
