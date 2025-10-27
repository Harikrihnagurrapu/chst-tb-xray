# TB X-ray Predictor (minimal Flask frontend)

This small Flask app provides a simple web UI to upload a chest X-ray image and get a TB/Normal prediction from one of the trained models in `../models/`.

Files created:

- `app.py` — Flask application.
- `templates/index.html` — upload form.
- `templates/result.html` — simple result page.
- `requirements.txt` — minimal dependencies.
- `Dockerfile` and `.dockerignore` for containerized runs.

New features added:

- Image preview in the upload UI.
- Decision threshold slider to tune classification cutoff.
- JSON API endpoint at `/api/predict` accepting multipart/form-data or JSON (base64 image).
- Simple prediction logging to `web_app/predictions.log` and endpoints to view/download logs (`/logs`, `/download_log`).
- Health endpoint `/health` for quick readiness checks.

How to run locally (Windows PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r web_app/requirements.txt
python web_app/app.py
```

Then open http://localhost:5000 in your browser.

Run in Docker:

```powershell
docker build -t tb-predictor:latest -f web_app/Dockerfile .
# mount your models folder into the container
docker run -p 5000:5000 -v ${PWD}\\models:/app/models tb-predictor:latest
```

API usage (JSON example):

```python
import base64
import requests

with open('some_xray.png','rb') as f:
    b64 = base64.b64encode(f.read()).decode('utf-8')

resp = requests.post('http://localhost:5000/api/predict', json={'image': b64, 'model': 'EfficientNetB4'})
print(resp.json())
```

Notes and tips:

- The app expects models in the `models/` folder (relative to the project root). If your models are saved elsewhere, either provide the absolute path in the form or update `MODEL_PATHS` in `web_app/app.py`.
- EfficientNet uses 380x380 input size; ResNet/DenseNet use 224x224. The app automatically applies the correct preprocessing.
- This app is intended for local testing/demo only. For production consider a proper WSGI server, input validation, authentication, and security hardening.
