FROM python:3.10-slim

# Install system deps for pillow and tensorflow minimal runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY web_app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app code
COPY web_app /app/web_app

# Expose port and set a models volume so you can mount heavy model files at runtime
VOLUME ["/app/models"]
EXPOSE 5000

ENV PYTHONUNBUFFERED=1

# Run the Flask dev server (for demo). For production, replace with a WSGI server.
CMD ["python", "web_app/app.py"]
