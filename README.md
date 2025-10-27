# CHST — Tuberculosis Chest X-ray Classifier

This repository contains a Jupyter notebook, trained models, and a simple Flask web app for tuberculosis (TB) detection from chest X-rays.

Layout
------
- `96.ipynb` — Main experiment notebook (data prep, model training, evaluation, Grad-CAM explainability).
- `models/` — Saved model files and generated artifacts (model weights, overlays).
- `tb_xray_split/` — Local train/val/test splits (not tracked — see .gitignore).
- `web_app/` — Minimal Flask frontend (upload + predict) with templates and Dockerfile.
- `scripts/` — Utility scripts (e.g., Grad-CAM runner).

Getting started
---------------
1. Create a Python environment (recommended: Python 3.10+). On Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

2. Prepare data and (optionally) download dataset with Kaggle CLI.

- Place your `kaggle.json` in the project root (or set up the kaggle CLI). The notebook contains a cell to copy `kaggle.json` to `~/.kaggle`.

3. Open the notebook `96.ipynb` in Jupyter or VS Code and run cells in order.

Running the web app (local)
--------------------------
The small Flask app is in `web_app/`.

From the project root (PowerShell):

```powershell
cd web_app
.\.\Scripts\Activate.ps1  # if using venv; otherwise ensure env is active
pip install -r requirements.txt
python app.py
# app will be available at http://127.0.0.1:5000
```

Grad-CAM (explainability)
--------------------------
- The notebook contains a final cell that runs Grad-CAM for a selected saved model and saves an overlay to `models/gradcam_<modelname>_sample.png`.
- There is also a small script under `scripts/run_gradcam.py` (if present) for running Grad-CAM from the command line.

Notes and troubleshooting
-------------------------
- Models and large dataset split directories are excluded from version control by `.gitignore`. If you want to track model files, remove the relevant lines from `.gitignore`.
- If loading a saved model fails due to custom layers, supply `custom_objects` when calling `tf.keras.models.load_model` (the notebook includes guidance for this).
- If you see input-structure or dtype mismatches when running Grad-CAM, re-run the notebook cell that prints the model inputs and paste the printed output into an issue so we can adapt the loader.

License & attribution
---------------------
This project re-uses publicly available architectures (ResNet, DenseNet, EfficientNet) and an open TB chest X-ray dataset. Check dataset licensing before redistributing.

If you'd like, I can add step-by-step quickstarts for: (a) running the notebook end-to-end, (b) running batch Grad-CAM for multiple images, or (c) adding an API endpoint to return Grad-CAM overlays from the web app.
