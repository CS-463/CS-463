# Chapter 5 demos

## Files you need

Get these files from this folder and keep them **in the same folder** on your computer:

- `ch_5_01_knn_bayes_boundary.ipynb`
- `ch_5_demos_2_bootstrap.ipynb`
- `ch_5_demos_helpers.py` (the notebook imports its demos from this file)

If you cloned the course repo, run `git pull` to get them.

No new packages are needed: if the Chapter 4 demos work for you, these will too.

<details>
<summary><b>Google Colab</b></summary>

1. Open [Colab](https://colab.research.google.com/) and upload the notebook (**File > Upload
   notebook**).
2. Click the folder icon in the left sidebar and upload `ch_5_demos_helpers.py`. Without it, the
   demo cells fail with `ModuleNotFoundError: No module named 'ch_5_demos_helpers'`.

Uploaded files disappear when the Colab session ends, so re-upload `ch_5_demos_helpers.py` each time
you come back, and download your notebook (**File > Download > Download .ipynb**) before you close
the tab.

</details>

## If the demos don't show up

| What you see | What to do |
|---|---|
| `ModuleNotFoundError: No module named 'ch_5_demos_helpers'` | `ch_5_demos_helpers.py` isn't in the same folder as the notebook. Move it there (or upload it, on Colab). |
| `ModuleNotFoundError: No module named 'ipywidgets'` | The kernel is using a different environment. Select the `CS_463` kernel. See the [Chapter 4 README](../ch_4/README.md) for adding `ipywidgets`. |
| Text like `VBox(children=(HBox(...` instead of sliders, or `Error displaying widget` | Restart the kernel, then reload the window (VS Code/Cursor) or restart JupyterLab. |
