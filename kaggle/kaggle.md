# Kaggle Setup (Recommended)

For a no-setup option with GPU acceleration, use the Kaggle notebook.

## Quick Start

1. Go to [kaggle.com](https://www.kaggle.com) → **Code** → **New Notebook**
2. In the notebook editor sidebar, click **Add Data Source** → search `eye-diseases-classification` by `gunavenkatdoddi` → add it
3. Go to **Settings** → set **Accelerator** to **GPU T4x2**
4. Open `kaggle_notebook.ipynb` from this repo, select all cells, copy, and paste into your Kaggle notebook
5. Run all cells top-to-bottom
6. Dataset path is auto-detected — no manual path editing needed

Model checkpoints are saved under `/kaggle/working/model/`.

---

## Local Setup

Download the dataset from Kaggle, extract it so class folders (`cataract`, `diabetic_retinopathy`, `glaucoma`, `normal`) are at the root, then run the notebook cells in order.
