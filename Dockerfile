# ---------------------------------------------------------------------------
# meter-reader – AI-on-the-edge compatible meter reader for x86 / arm64
# ---------------------------------------------------------------------------
FROM python:3.11-slim

# Runtime dependencies for OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Use the lightweight TFLite runtime by default (swap the comment in
# requirements.txt if you prefer the full tensorflow-cpu package)
RUN pip install --no-cache-dir -r requirements.txt

COPY meter_reader.py .

# ---- Volumes ---------------------------------------------------------------
# /sdcard  – mirrors the original ESP32 sdcard layout:
#              /sdcard/config/config.ini
#              /sdcard/config/*.tflite
#              /sdcard/config/ref0.jpg  ref1.jpg
# /images  – drop input images here; pass the path with --image
# ----------------------------------------------------------------------------
VOLUME ["/sdcard", "/images"]

ENTRYPOINT ["python", "/app/meter_reader.py"]
CMD ["--help"]
