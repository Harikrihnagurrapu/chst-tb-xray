from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file, Response
import os
import io
import base64
import time
import logging
import numpy as np
from PIL import Image
import tensorflow as tf

# Optional CORS support for API usage
try:
    from flask_cors import CORS
    _have_cors = True
except Exception:
    _have_cors = False

app = Flask(__name__)
app.secret_key = os.urandom(24)
if _have_cors:
    CORS(app)

# Map friendly names to default model paths (adjust if you have different filenames)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATHS = {
    'ResNet50': os.path.join(MODEL_DIR, 'my_resnet50_tuberculosis_model.keras'),
    'DenseNet121': os.path.join(MODEL_DIR, 'my_densenet121_tuberculosis_model.keras'),
    'EfficientNetB4': os.path.join(MODEL_DIR, 'my_efficientnetb4_tuberculosis_model_best.keras'),
}

# Lazy-loaded models cache
_loaded_models = {}

# Preprocessing configuration per model
MODEL_CONFIG = {
    'ResNet50': {'size': (224, 224), 'preprocess': 'resnet'},
    'DenseNet121': {'size': (224, 224), 'preprocess': 'densenet'},
    'EfficientNetB4': {'size': (380, 380), 'preprocess': 'efficientnet'},
}

# Setup simple logging to file
LOG_PATH = os.path.join(os.path.dirname(__file__), 'predictions.log')
logger = logging.getLogger('tb_predictor')
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(LOG_PATH)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)


# Helper: load model once
def load_model_for(name, path=None):
    if name in _loaded_models:
        return _loaded_models[name]
    model_path = path or MODEL_PATHS.get(name)
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file for {name} not found at {model_path}")
    model = tf.keras.models.load_model(model_path)
    _loaded_models[name] = model
    return model

# Helper: preprocess image bytes -> tensor
def preprocess_image(image_bytes, target_size, preprocess_type):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(target_size, Image.BILINEAR)
    arr = np.array(img).astype(np.float32)
    # expand dims
    x = np.expand_dims(arr, axis=0)
    # apply model-specific preprocess
    if preprocess_type == 'resnet':
        from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
        x = resnet_preprocess(x)
    elif preprocess_type == 'densenet':
        from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
        x = densenet_preprocess(x)
    elif preprocess_type == 'efficientnet':
        from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
        x = effnet_preprocess(x)
    else:
        # simple scaling
        x = x / 255.0
    return x

# Helper: get positive probability from model output
def extract_positive_prob(pred):
    pred = np.asarray(pred)
    if pred.ndim == 2 and pred.shape[1] == 2:
        # softmax two outputs
        return float(pred[0,1])
    if pred.ndim == 2 and pred.shape[1] == 1:
        val = float(pred[0,0])
        # if likely logits (large magnitude) apply sigmoid
        if val < 0 or val > 1:
            return float(tf.sigmoid(val).numpy())
        return val
    if pred.ndim == 1:
        val = float(pred[0])
        if val < 0 or val > 1:
            return float(tf.sigmoid(val).numpy())
        return val
    # fallback
    return float(np.squeeze(pred))

# Helper to log predictions
def log_prediction(client_ip, model_name, model_path, prob, label, duration):
    logger.info(f"ip={client_ip} model={model_name} path={model_path} prob={prob:.4f} label={label} duration={duration:.3f}s")

@app.route('/', methods=['GET'])
def index():
    # show upload form
    available_models = {k: os.path.exists(v) for k,v in MODEL_PATHS.items()}
    default_threshold = 0.5
    return render_template('index.html', models=available_models, default_threshold=default_threshold)

@app.route('/predict', methods=['POST'])
def predict():
    # This endpoint supports both form uploads (from the web UI) and can be called from the API route.
    if 'file' not in request.files:
        flash('No file part in the request')
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))

    model_name = request.form.get('model') or 'EfficientNetB4'
    custom_path = request.form.get('custom_model_path')
    # threshold (float between 0 and 1)
    try:
        threshold = float(request.form.get('threshold', 0.5))
    except Exception:
        threshold = 0.5

    # read file bytes
    image_bytes = file.read()

    start = time.time()
    try:
        # load model
        model = load_model_for(model_name, custom_path if custom_path else None)
    except Exception as e:
        flash(f'Could not load model: {e}')
        return redirect(url_for('index'))

    cfg = MODEL_CONFIG.get(model_name, {'size': (224,224), 'preprocess': None})
    try:
        x = preprocess_image(image_bytes, cfg['size'], cfg['preprocess'])
    except Exception as e:
        flash(f'Image preprocessing failed: {e}')
        return redirect(url_for('index'))

    try:
        pred = model.predict(x)
        prob_pos = extract_positive_prob(pred)
        label = 'Tuberculosis' if prob_pos >= threshold else 'Normal'
    except Exception as e:
        flash(f'Model prediction failed: {e}')
        return redirect(url_for('index'))

    duration = time.time() - start
    # log
    client_ip = request.remote_addr or 'local'
    model_path = custom_path if custom_path else (MODEL_PATHS.get(model_name) or 'unknown')
    log_prediction(client_ip, model_name, model_path, prob_pos, label, duration)

    # also produce a small base64 preview of the uploaded image to show on the result page
    preview_b64 = base64.b64encode(image_bytes).decode('utf-8')

    return render_template('result.html', label=label, probability=prob_pos, model_name=model_name, preview_b64=preview_b64, threshold=threshold)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Simple JSON API that accepts either multipart/form-data with a file (form field 'file')
    or a JSON body with a base64-encoded image under key 'image'. Optional keys: 'model', 'threshold'.
    Returns JSON: {label, probability, model}
    """
    start = time.time()
    model_name = request.form.get('model') if 'model' in request.form else (request.json.get('model') if request.is_json else 'EfficientNetB4')
    custom_path = request.form.get('custom_model_path') if 'custom_model_path' in request.form else (request.json.get('custom_model_path') if request.is_json else None)

    # threshold
    try:
        threshold = float(request.form.get('threshold')) if 'threshold' in request.form else (float(request.json.get('threshold')) if request.is_json and request.json.get('threshold') is not None else 0.5)
    except Exception:
        threshold = 0.5

    image_bytes = None
    # multipart file
    if 'file' in request.files:
        f = request.files['file']
        image_bytes = f.read()
    else:
        # JSON body with base64 image
        if not request.is_json:
            return jsonify({'error': 'No file provided and body is not JSON with a base64 image.'}), 400
        body = request.get_json()
        b64 = body.get('image')
        if not b64:
            return jsonify({'error': 'JSON must contain `image` (base64 string).'}), 400
        try:
            # allow data URLs
            if b64.startswith('data:'):
                b64 = b64.split(',', 1)[1]
            image_bytes = base64.b64decode(b64)
        except Exception as e:
            return jsonify({'error': f'Failed to decode base64 image: {e}'}), 400

    if image_bytes is None:
        return jsonify({'error': 'No image data found.'}), 400

    try:
        model = load_model_for(model_name, custom_path if custom_path else None)
    except Exception as e:
        return jsonify({'error': f'Could not load model: {e}'}), 400

    cfg = MODEL_CONFIG.get(model_name, {'size': (224,224), 'preprocess': None})
    try:
        x = preprocess_image(image_bytes, cfg['size'], cfg['preprocess'])
    except Exception as e:
        return jsonify({'error': f'Image preprocessing failed: {e}'}), 400

    try:
        pred = model.predict(x)
        prob_pos = extract_positive_prob(pred)
        label = 'Tuberculosis' if prob_pos >= threshold else 'Normal'
    except Exception as e:
        return jsonify({'error': f'Model prediction failed: {e}'}), 500

    duration = time.time() - start
    client_ip = request.remote_addr or 'api'
    model_path = custom_path if custom_path else (MODEL_PATHS.get(model_name) or 'unknown')
    log_prediction(client_ip, model_name, model_path, prob_pos, label, duration)

    return jsonify({'label': label, 'probability': float(prob_pos), 'model': model_name, 'threshold': threshold, 'duration_s': duration})


@app.route('/logs', methods=['GET'])
def view_logs():
    """Return the last N lines of the prediction log in plain text for quick inspection."""
    try:
        n = int(request.args.get('n', 200))
    except Exception:
        n = 200
    if not os.path.exists(LOG_PATH):
        return Response('No log file yet.', mimetype='text/plain')
    # Read tail efficiently
    def tail(file_path, lines=200):
        with open(file_path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            filesize = f.tell()
            blocksize = 1024
            data = b''
            blocks = -1
            while len(data.splitlines()) <= lines and -blocks * blocksize < filesize:
                try:
                    f.seek(blocks * blocksize, os.SEEK_END)
                except Exception:
                    f.seek(0)
                    data = f.read()
                    break
                data = f.read() + data
                blocks -= 1
            return b'\n'.join(data.splitlines()[-lines:]).decode('utf-8', errors='replace')
    text = tail(LOG_PATH, n)
    return Response(text, mimetype='text/plain')

@app.route('/download_log', methods=['GET'])
def download_log():
    if not os.path.exists(LOG_PATH):
        return Response('No log file to download.', mimetype='text/plain')
    return send_file(LOG_PATH, as_attachment=True, download_name='predictions.log')

@app.route('/health', methods=['GET'])
def health():
    # basic health: check TensorFlow can be imported and optionally models exist
    try:
        _ = tf.__version__
    except Exception:
        return jsonify({'status': 'error', 'reason': 'tensorflow_unavailable'}), 500
    # quick model availability
    available = {k: os.path.exists(v) for k,v in MODEL_PATHS.items()}
    return jsonify({'status': 'ok', 'tensorflow': tf.__version__, 'models_available': available})


if __name__ == '__main__':
    # For local development only. In production use a WSGI server.
    app.run(host='0.0.0.0', port=5000, debug=True)
