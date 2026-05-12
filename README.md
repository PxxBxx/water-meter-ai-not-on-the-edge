# Meter Reader – Standalone AI-on-the-edge Port

A lightweight, portable Python port of the [AI-on-the-edge-device](https://github.com/jomjol/AI-on-the-edge-device) ESP32 firmware. Read water/gas/electricity meters using the same TensorFlow Lite models and configuration as the original – but running on x86-64, ARM64, Proxmox LXC, or Docker, without any ESP32 or camera hardware.

**Two tools included:**
- **Setup Wizard** – Browser-based GUI to define your meter zones visually (replacing the original ESP32 web UI)
- **Meter Reader** – Standalone CLI/API to process images and extract readings

---

## Features

### Setup Wizard (`setup_wizard.py`)
- 📸 **Reference image upload** – upload a meter photo, rotate/flip to align digits horizontally
- 🎯 **Alignment markers** – drag-select two high-contrast template regions for sub-pixel registration
- 📦 **Digit ROI drawing** – interactive canvas to define each digit zone; auto-spacing helpers
- 🔄 **Analog ROI drawing** – drag dial zones; CCW (counter-clockwise) toggle per dial
- ⚙️ **Post-processing config** – decimal shift, rate limits, consistency checks, model thresholds
- 💾 **One-click save** – writes `config.ini`, `reference.jpg`, `ref0.jpg`, `ref1.jpg` to `sdcard/config/`
- 🌐 **Browser-based UI** – works on any desktop, phone, or remote machine

### Meter Reader (`meter_reader.py`)
- ✅ **Same inference logic** as the original ESP32 firmware
- 🔢 **All 5 CNN types supported** – Digit, Analogue, DoubleHybrid10, Digit100, Analogue100
- 📷 **Image alignment** – optional fine-alignment using OpenCV template matching (reference markers)
- 🎛️ **Smart digit readout** – neighbour-weighted classification, zero-crossing detection
- 📊 **Decimal handling** – decimal shift, extended resolution, leading-NaN stripping
- 📤 **JSON output** – machine-readable results, perfect for integration with Home Assistant, InfluxDB, etc.
- 🖼️ **Debug mode** – save aligned image and every cropped ROI for troubleshooting

---

## Quick Start

### Option 1: Docker Compose (Recommended)

```bash
cd meter-reader

# Copy your existing sdcard directory (or start with an empty one)
cp -r ../sd-card ./sdcard

# 1. Run the setup wizard to configure your meter
docker compose up setup-wizard
# Open http://localhost:5000 in your browser
# Complete all 6 steps, then click "Save to sdcard"

# 2. Read a meter image
docker compose run --rm meter-reader \
  --image /path/to/meter.jpg \
  --sdcard ./sdcard \
  --pretty
```

### Option 2: Bare Python (venv)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Setup wizard
python setup_wizard.py --sdcard ./sdcard --port 5000
# then open http://localhost:5000

# Meter reading
python meter_reader.py \
  --image /path/to/meter.jpg \
  --sdcard ./sdcard \
  --pretty
```

---

## Installation

### Requirements

- **Python 3.9+** (3.11 recommended)
- **For alignment** (optional but recommended): OpenCV (`opencv-python-headless`)
- **For TFLite inference**: one of:
  - `tflite-runtime` (lightweight, ~2 MB; recommended for Docker)
  - `ai-edge-litert` (newer standalone runtime)
  - `tensorflow-cpu` (heavier, ~400 MB; works everywhere)

### Docker Build

```bash
# Meter reader image
docker build -f Dockerfile -t meter-reader:latest .

# Setup wizard image
docker build -f Dockerfile.setup -t meter-reader-setup:latest .

# Or both with Compose
docker compose build
```

### Python venv

```bash
python3 -m venv venv
source venv/bin/activate   # or: venv\Scripts\activate on Windows
pip install -r requirements.txt
```

**Note:** If you have issues with TFLite, edit `requirements.txt` to comment out the default choice and uncomment an alternative (see file comments).

### Obtaining Pre-trained Models

This repository **does not include** the pre-trained TFLite model files (`.tflite`). They are maintained by the [jomjol/AI-on-the-edge-device](https://github.com/jomjol/AI-on-the-edge-device) project.

**To get the models:**
1. Clone or visit the [AI-on-the-edge-device repository](https://github.com/jomjol/AI-on-the-edge-device)
2. Download the models from [`sd-card/config/`](https://github.com/jomjol/AI-on-the-edge-device/tree/master/sd-card/config)
3. Copy them to your local `sdcard/config/` directory (create it if it doesn't exist)

Example:
```bash
# Download a few common models
wget -P sdcard/config/ \
  https://raw.githubusercontent.com/jomjol/AI-on-the-edge-device/master/sd-card/config/dig-cont_0900_s3_q.tflite \
  https://raw.githubusercontent.com/jomjol/AI-on-the-edge-device/master/sd-card/config/ana-cont_1500_s2_q.tflite
```

See the [original project documentation](https://github.com/jomjol/AI-on-the-edge-device) for a complete list of available models.

---

## Workflow

### 1. Setup Wizard – Define Your Meter

The wizard guides you through 6 steps to create a `config.ini` and reference images:

#### **Step ① – Reference Image**
- Upload a clear photo of your meter
- Rotate to align digits horizontally (degrees, ±1° fine-tune)
- Optionally mirror or flip
- **Output**: rotated reference stored in memory

#### **Step ② – Alignment Markers**
- Drag two small rectangles over static, high-contrast features (e.g. a screw, printed logo corner)
- Each marker should be 40×40 to 120×120 pixels and unique enough for template matching
- Optional contrast enhancement (CLAHE if OpenCV available)
- **Output**: `ref0.jpg` and `ref1.jpg` (enhanced thumbnails)

#### **Step ③ – Digit ROIs**
- Create named sequences (e.g. `main`, `water`, `gas`)
- For each sequence, drag rectangles over digit zones
- Order matters: **most-significant digit first** (left to right, as displayed)
- Tools: add/delete/rename ROIs, reorder with ↑/↓, sync sizes, equidistant spacing
- **Output**: ROI coordinates and names stored

#### **Step ④ – Analog ROIs**
- Repeat step ③ for analog (gauge) dials
- Check **CCW** if the dial rotates counter-clockwise (rare)
- Keep Δx = Δy (square crops)
- **Output**: analog ROI coordinates

#### **Step ⑤ – Post-processing**
- Set model paths (e.g. `/config/dig-cont_0900_s3_q.tflite`)
- CNN thresholds for each model
- Per-group settings: decimal shift, extended resolution, max rate, consistency checks, etc.
- **Output**: stored in memory

#### **Step ⑥ – Save & Export**
- Review the generated `config.ini`
- Click **Save to sdcard**
- Files written:
  - `sdcard/config/config.ini` (new)
  - `sdcard/config/reference.jpg` (the rotated input)
  - `sdcard/config/ref0.jpg`, `ref1.jpg` (marker crops)
  - `sdcard/config/config.ini.bak` (backup of old config, if any)

---

### 2. Meter Reader – Process Images

Once configured, `meter_reader.py` reads images and outputs JSON:

```bash
python meter_reader.py \
  --image /path/to/meter.jpg \
  --sdcard ./sdcard \
  [--config ./sdcard/config/config.ini] \
  [--debug-dir /tmp/debug] \
  [--pretty]
```

**Pipeline:**
1. Load config.ini and TFLite models from sdcard
2. Load and optionally align the input image
   - Coarse rotation (from `InitialRotate`)
   - Optional fine-alignment using reference marker template matching
3. Extract ROI sub-images and resize to model input dimensions
4. Run inference for each ROI (digit classification or angle estimation)
5. Assemble multi-digit numbers with smart neighbour-weighting logic
6. Apply decimal shift, NaN handling, etc.
7. Output JSON with raw values, processed readings, and error messages

**Output example:**
```json
{
  "main": {
    "raw": "056.4321",
    "value": "56.4321",
    "error": "no error"
  },
  "gas": {
    "raw": "N12.34",
    "value": null,
    "error": "Unresolved digit (N)"
  }
}
```

**Debug mode:**
```bash
python meter_reader.py \
  --image meter.jpg \
  --sdcard ./sdcard \
  --debug-dir /tmp/debug \
  --pretty
```
Saves:
- `aligned.jpg` – the input after rotation + fine-alignment
- `digit_main.dig1.jpg`, `digit_main.dig2.jpg`, … – cropped digit ROIs
- `analog_main.ana1.jpg`, … – cropped analog ROIs

---

## Configuration (`config.ini`)

The config file mirrors the original AI-on-the-edge-device format. Structure:

```ini
[Alignment]
InitialRotate = 2.5          ; degrees, positive = CCW
FlipImageSize = false        ; mirror + flip
SearchFieldX = 40            ; template search region size
SearchFieldY = 40
AlignmentAlgo = default      ; default | highaccuracy | fast | off
/config/ref0.jpg 103 271     ; marker file and target position
/config/ref1.jpg 442 142

[Digits]
Model = /config/dig-cont_0900_s3_q.tflite
CNNGoodThreshold = 0.5
main.dig1 294 126 30 54 false   ; ROI name x y dx dy ccw
main.dig2 343 126 30 54 false
main.dig3 391 126 30 54 false

[Analog]
Model = /config/ana-cont_1500_s2_q.tflite
CNNGoodThreshold = 0.5
main.ana1 432 230 92 92 false
main.ana2 379 332 92 92 false

[PostProcessing]
main.DecimalShift = 0
main.ExtendedResolution = false
main.AllowNegativeRates = false
main.MaxRateValue = 0.05
main.AnalogDigitTransitionStart = 9.2
PreValueUse = true
ErrorMessage = true
```

**Key parameters:**

| Param | Purpose |
|-------|---------|
| `InitialRotate` | Coarse rotation angle (°); applied before fine-alignment |
| `SearchFieldX/Y` | Size of region around target marker for template matching |
| `AlignmentAlgo` | `off` disables fine-alignment; use `fast` for speed on embedded systems |
| `Model` | Path to `.tflite` model file |
| `CNNGoodThreshold` | Confidence threshold for DoubleHybrid10 models |
| `DecimalShift` | Shift decimal point in the final reading (e.g. `1` = divide by 10) |
| `ExtendedResolution` | Include fractional digit from the last analog ROI |
| `AllowNegativeRates` | Reject readings that decrease over time |
| `MaxRateValue` | Max change between consecutive readings |
| `AnalogDigitTransitionStart` | Analog value (0-10) above which digit transition logic triggers |

### Pre-Value Validation and Digit Correction

The `PreValueUse` flag enables reading validation against a previous meter reading. This helps correct common OCR errors where a single digit is misread:

**How it works:**

1. **Enable in config.ini:**
   ```ini
   [PostProcessing]
   PreValueUse = true
   ```

2. **Create `sdcard/config/prevalue.ini`** with the previous reading:
   ```
   2026-05-12_10-30-45
   1957.17
   ```
   - Line 1: Timestamp (ISO format with underscores)
   - Line 2: The previous meter value

3. **Automatic correction:**
   - If current reading is unreasonable (e.g., 1457.17 when previous was 1957.17), the reader tries substituting each digit
   - Returns the alternative that makes sense based on typical usage patterns (-10 to +500 m³ daily)
   - Includes a `"note"` field in JSON output explaining the correction

**Example output with correction:**
```json
{
  "main": {
    "raw": "1457.17",
    "value": "1957.17",
    "error": "no error",
    "note": "Corrected from 1457.17 based on previous reading 1957.17",
    "confidence": [0.95, 0.87, 0.92]
  }
}
```

**Usage:**
```bash
python meter_reader.py \
  --image meter.jpg \
  --sdcard ./sdcard \
  --prevalue ./sdcard/config/prevalue.ini \
  --pretty
```

---

## Models & Inference

> **⚠️ Note on TFLite Models:** Pre-trained `.tflite` model files are **not included** in this repository. They are the property of the [jomjol/AI-on-the-edge-device](https://github.com/jomjol/AI-on-the-edge-device) project. You can obtain them from the [official repository's sd-card/config/ directory](https://github.com/jomjol/AI-on-the-edge-device/tree/master/sd-card/config). Copy the desired `.tflite` models to your local `sdcard/config/` directory before running meter_reader.py.

### Supported Model Types

Auto-detected from model output shape:

| Type | Outputs | Logic | Usage |
|------|---------|-------|-------|
| **Digit** | 11 | Softmax; argmax → class 0-10 (10 = N/error) | Single-digit classification |
| **Analogue** | 2 | atan2 of outputs → angle → 0-10 range | Analog gauge reading |
| **DoubleHybrid10** | 10 | Softmax + neighbour weighting → 0-10 float | Hybrid digit/analog (10 classes + sub-integer) |
| **Digit100** | 100 | argmax / 10 → float | Fine-grained digit (0.0-10.0) |
| **Analogue100** | 100 | Same formula as Digit100 | Analog gauges |

### Confidence Values

Each digit/ROI includes a confidence score from the model:

- **Digit models**: Output value at the predicted class (0.0-1.0)
- **Analogue models**: Magnitude of the signal vector (strength of needle position)
- **DoubleHybrid10**: Sum of best and adjacent class outputs (fit value)
- **Digit100/Analogue100**: Output value at argmax

**Usage in Home Assistant:**

The `confidence` array in the JSON output tracks confidence for each digit:
```json
{
  "main": {
    "raw": "056.4321",
    "value": "56.4321",
    "error": "no error",
    "confidence": [0.95, 0.87, 0.92, 0.91, 0.88]
  }
}
```

Home Assistant automatically includes these in the sensor attributes. Create templates to alert on low confidence:
```yaml
automation:
  - alias: "Alert on low meter confidence"
    trigger:
      platform: template
      value_template: "{{ state_attr('sensor.water_meter', 'confidence') | list | min < 0.7 }}"
    action:
      service: notify.mobile_app_phone
      data:
        message: "⚠️ Water meter reading has low confidence ({{ state_attr('sensor.water_meter', 'confidence') }})"
```

### Inference Details

**TFLite input:** Raw RGB uint8 image, no normalization (0-255 fed as float).

**Per-ROI logic:**
1. Crop ROI from aligned image
2. Resize to model input dimensions (e.g. 32×32, 48×48)
3. Feed as float32 to interpreter
4. Extract outputs and apply type-specific formula
5. Store `result_float` (for analogue/hybrid) or `result_klasse` (for digit)

**Digit assembly (readout):**
- Most-significant ROI first
- Neighbour-weighted adjustment for zero-crossing detection
- Handles transitions between digits when one "ticks over"
- Extended resolution: include sub-integer from last analog ROI

See [meter_reader.py](meter_reader.py) for full ported logic.

---

## API Reference

### Setup Wizard (`setup_wizard.py`)

**Endpoints:**

| Method | Path | Body | Returns |
|--------|------|------|---------|
| `POST` | `/api/image/upload` | multipart form (file) | `{image: data-URI, width, height}` |
| `POST` | `/api/image/rotate` | `{rotation, flip_h, flip_v}` | `{image: data-URI}` |
| `GET` | `/api/image/current` | — | `{image: data-URI}` |
| `POST` | `/api/alignment/marker` | `{index, x, y, dx, dy, enhance}` | `{preview_original, preview_enhanced}` |
| `GET` | `/api/alignment/marker/<idx>` | — | `{preview, marker}` |
| `GET` | `/api/rois/digit` | — | `{main: […], …}` (all ROI groups) |
| `POST` | `/api/rois/digit` | `{action, …}` | `{ok}` or preview |
| `GET` | `/api/rois/analog` | — | `{…}` (same) |
| `POST` | `/api/rois/analog` | `{action, …}` | `{…}` |
| `GET` | `/api/postprocessing` | — | `{postprocessing, digit_model, …}` |
| `POST` | `/api/postprocessing` | `{postprocessing, …}` | `{ok}` |
| `GET` | `/api/config_preview` | — | `{config_ini: string}` |
| `POST` | `/api/save` | `{}` | `{ok, config_ini, saved_to, warnings}` |

### Meter Reader (`meter_reader.py`)

**CLI:**
```bash
python meter_reader.py --image IMAGE --sdcard DIR [options]
```

**Options:**
- `--image PATH` *(required)* – input meter image
- `--sdcard DIR` – root of sdcard tree (default: `./sdcard`)
- `--config PATH` – config.ini path (default: `<sdcard>/config/config.ini`)
- `--debug-dir DIR` – save intermediate images for troubleshooting
- `--pretty` – pretty-print JSON output
- `--prevalue PATH` – path to prevalue.ini for validation against previous readings (default: `<sdcard>/config/prevalue.ini`)

**Output:** JSON to stdout
```json
{"sequence_name": {"raw": "value", "value": "clean_value", "error": "message", "confidence": [0.95, 0.87, ...]}, …}
```

### Home Assistant Integration (`cron_ha.py`)

**Purpose:** Automated Home Assistant MQTT integration – publishes meter readings directly to HA with MQTT Discovery.

**CLI:**
```bash
python cron_ha.py [options]
```

**Options:**
- `--image-url URL` – URL of the meter image (default: `http://web.lan/watermeter_images/latest.jpg`)
- `--mqtt-broker HOST` – MQTT broker address (default: `192.168.1.10`)
- `--mqtt-port PORT` – MQTT broker port (default: `1883`)
- `--sdcard DIR` – root of sdcard tree (default: `./sdcard`)
- `--meter-reader PATH` – path to meter_reader.py script
- `--debug` – enable debug logging

**Features:**
- 🏠 **MQTT Discovery** – automatically creates water meter sensor in Home Assistant
- 📷 **Image Download** – fetches the latest meter image from a web URL
- 🔍 **Smart Reading** – runs meter_reader.py and extracts the value
- 📊 **Full Attributes** – publishes raw reading, confidence values, and error messages
- 🔄 **Cron-Friendly** – designed for scheduled execution (e.g., every 5 minutes)
- 📝 **Logging** – comprehensive debug logging for troubleshooting

**Configuration (environment variables):**
```bash
export MQTT_BROKER=192.168.1.10
export MQTT_PORT=1883
export MQTT_USERNAME=homeassistant
export MQTT_PASSWORD=secret
export IMAGE_URL=http://web.lan/watermeter_images/latest.jpg
export SDCARD_DIR=./sdcard
export HA_DISCOVERY_PREFIX=homeassistant
python cron_ha.py
```

**MQTT Topics:**
- **Config** (discovery): `homeassistant/sensor/water_meter_1/water_meter_total/config`
- **State** (reading): `homeassistant/sensor/water_meter_1/water_meter_total/state`
- **Availability**: `homeassistant/sensor/water_meter_1/water_meter_total/availability`

**Home Assistant Output Example:**
```json
{
  "name": "Water Meter",
  "unique_id": "water_meter_total",
  "state_topic": "homeassistant/sensor/water_meter_1/water_meter_total/state",
  "unit_of_measurement": "m³",
  "device_class": "water",
  "state_class": "total_increasing",
  "value_template": "{{ value_json.value }}",
  "json_attributes_topic": "homeassistant/sensor/water_meter_1/water_meter_total/state",
  "device": {
    "identifiers": ["water_meter_1"],
    "name": "Water Meter",
    "manufacturer": "Custom",
    "model": "DIY Analog Water Meter Reading"
  }
}
```

**State Payload:**
```json
{
  "group": "main",
  "raw": "056.4321",
  "value": "56.4321",
  "error": "no error",
  "confidence": [0.95, 0.87, 0.92]
}
```

**Requirements:**
```bash
pip install paho-mqtt requests
```

---

## Examples

### Example 1: Docker Setup & Reading

```bash
# Clone and navigate
cd meter-reader

# Step 1: Run setup wizard in Docker
docker compose up setup-wizard
# Open http://localhost:5000
# - Upload meter photo
# - Define alignment markers
# - Draw digit + analog ROIs
# - Set post-processing params
# - Save

# Step 2: Read a meter image
docker compose run --rm meter-reader \
  --image /path/to/new-meter.jpg \
  --sdcard ./sdcard \
  --pretty

# Output:
# {
#   "main": {
#     "raw": "056.4321",
#     "value": "56.4321",
#     "error": "no error"
#   }
# }
```

### Example 5: Python Bare-Metal

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Wizard
python setup_wizard.py --sdcard ./sdcard --port 5000
# http://localhost:5000 → configure meter → save

# Reading
python meter_reader.py \
  --image meter.jpg \
  --sdcard ./sdcard \
  --debug-dir /tmp/meter-debug \
  --pretty

# View debug images
open /tmp/meter-debug/aligned.jpg
ls /tmp/meter-debug/digit_*.jpg
```

### Example 3: Home Assistant Integration (cron_ha.py)

**Setup:**

1. **Install dependencies:**
   ```bash
   pip install paho-mqtt requests
   ```

2. **Configure your Home Assistant MQTT broker** (if not already done):
   - In Home Assistant: Settings → Devices & Services → Create Automation or MQTT
   - Ensure MQTT broker is running and accessible (default: port 1883)

3. **Configure meter_reader.py** (if not already done):
   ```bash
   python setup_wizard.py --sdcard ./sdcard --port 5000
   # Open http://localhost:5000 and complete the wizard
   ```

4. **Run once to test:**
   ```bash
   python cron_ha.py \
     --image-url http://web.lan/watermeter_images/latest.jpg \
     --mqtt-broker 192.168.1.10 \
     --sdcard ./sdcard \
     --debug
   ```

5. **Check Home Assistant:**
   - Settings → Devices & Services → MQTT
   - You should see a new "Water Meter" entity
   - Check Entities list: `sensor.water_meter`

**Automated scheduling (Cron):**

Every 5 minutes:
```bash
*/5 * * * * cd /home/scalp/water-meter-ai-not-on-the-edge && \
  python cron_ha.py >> /var/log/water_meter.log 2>&1
```

Or via systemd timer:

Create `/etc/systemd/system/water-meter.service`:
```ini
[Unit]
Description=Water Meter Reader
After=network.target

[Service]
Type=oneshot
User=meter
WorkingDirectory=/home/scalp/water-meter-ai-not-on-the-edge
Environment="MQTT_BROKER=192.168.1.10"
Environment="IMAGE_URL=http://web.lan/watermeter_images/latest.jpg"
Environment="SDCARD_DIR=/home/scalp/water-meter-ai-not-on-the-edge/sdcard"
ExecStart=/usr/bin/python3 /home/scalp/water-meter-ai-not-on-the-edge/cron_ha.py
StandardOutput=journal
StandardError=journal
```

Create `/etc/systemd/system/water-meter.timer`:
```ini
[Unit]
Description=Water Meter Reader Timer
Requires=water-meter.service

[Timer]
OnBootSec=1min
OnUnitActiveSec=5min
Persistent=true

[Install]
WantedBy=timers.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable water-meter.timer
sudo systemctl start water-meter.timer
sudo systemctl status water-meter.timer

# View logs
sudo journalctl -u water-meter.service -f
```

**Home Assistant Automation Example:**

Create a template sensor for daily delta:
```yaml
# configuration.yaml
template:
  - sensor:
      - name: "Water Meter Daily Delta"
        unique_id: water_meter_daily_delta
        unit_of_measurement: "m³"
        state: >
          {% set today = now().date() %}
          {% set sensor_history = state_attr('sensor.water_meter', 'history') %}
          {# Simple delta: current - midnight reading #}
          {{ (states('sensor.water_meter') | float - 0.0) | round(3) }}
```

Or trigger automations on readings:
```yaml
automation:
  - alias: "Alert if water meter jumps"
    trigger:
      platform: state
      entity_id: sensor.water_meter
    condition:
      template: "{{ (trigger.to_state.state | float - trigger.from_state.state | float) > 10 }}"
    action:
      service: notify.mobile_app_phone
      data:
        message: "⚠️ Water meter jumped by {{ (trigger.to_state.state | float - trigger.from_state.state | float) | round(2) }} m³"
```

**Troubleshooting cron_ha.py:**

1. **"Failed to connect to MQTT broker"**
   - Verify broker is running: `nc -zv 192.168.1.10 1883`
   - Check firewall allows port 1883
   - Verify `MQTT_BROKER` env var is correct

2. **"Failed to download image"**
   - Test URL: `curl http://web.lan/watermeter_images/latest.jpg > /tmp/test.jpg`
   - Check URL is accessible from the machine running cron_ha.py
   - Verify image is valid JPEG/PNG

3. **"meter_reader.py failed"**
   - Run meter_reader.py directly to debug
   - Check `--debug-dir` output from cron_ha.py logs
   - Verify sdcard/config/ has valid config.ini and models

4. **Entity not appearing in Home Assistant**
   - Check MQTT integration is enabled in HA
   - Verify `discovery_prefix` matches (default: `homeassistant`)
   - Check HA logs for MQTT discovery errors
   - Manually check MQTT topic: `mosquitto_sub -h 192.168.1.10 -t "homeassistant/#"`

### Example 4: Integration with Home Assistant / InfluxDB

```bash
# Cron job to read meter every hour
0 * * * * python /app/meter_reader.py --image /mnt/camera/latest.jpg --sdcard /mnt/sdcard > /tmp/reading.json 2>&1 && \
  curl -X POST http://influxdb:8086/write?db=meters \
       --data-binary @/tmp/reading.json
```

Or use the JSON output directly:
```python
import json
import subprocess

result = subprocess.run([
    "python", "meter_reader.py",
    "--image", "meter.jpg",
    "--sdcard", "./sdcard"
], capture_output=True, text=True)

data = json.loads(result.stdout)
for sequence_name, reading in data.items():
    if reading["error"] == "no error":
        print(f"{sequence_name}: {reading['value']} m³")
    else:
        print(f"{sequence_name}: ERROR – {reading['error']}")
```

### Example 6: Proxmox LXC Container

```bash
# Inside Proxmox LXC (Debian 12):
apt-get update && apt-get install -y python3 python3-pip python3-venv

git clone https://github.com/jomjol/AI-on-the-edge-device.git
cd AI-on-the-edge-device/meter-reader

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run setup wizard (port 5000 exposed to host network)
python setup_wizard.py --host 0.0.0.0 --port 5000 --sdcard /var/lib/meters/sdcard

# Read images on demand
python meter_reader.py --image /tmp/meter.jpg --sdcard /var/lib/meters/sdcard
```

---

## Troubleshooting

### "No TFLite runtime found"

Install one of:
```bash
pip install tflite-runtime
# OR
pip install ai-edge-litert
# OR
pip install tensorflow-cpu
```

### "Model file doesn't exist"

Ensure:
1. Models are in `sdcard/config/` with correct filenames
2. config.ini paths match (e.g. `/config/dig-cont_0900_s3_q.tflite`)
3. Run from a directory where sdcard is accessible (or use `--sdcard` path)

### "Alignment markers not found" / poor alignment

1. Use `--debug-dir` to inspect `aligned.jpg`
2. Ensure both markers are unique and high-contrast
3. Try `AlignmentAlgo = fast` or `off` in config.ini
4. Increase `SearchFieldX/Y` in config

### "Digits reading as 'N'" (unresolved)

1. Check cropped ROI images in `--debug-dir`
2. Verify ROI coordinates include the full digit
3. Increase `CNNGoodThreshold` to accept lower-confidence results (trade accuracy for recall)
4. Try extending ROI Δx/Δy slightly

### Performance / Slow reading

1. Use lightweight `tflite-runtime` instead of `tensorflow-cpu`
2. Set `AlignmentAlgo = fast` or `off`
3. Run on arm64 if on a Raspberry Pi (faster than x86 emulation)

### OpenCV not installed

The tool works without OpenCV but skips fine-alignment:
```bash
WARNING: opencv-python not installed – skipping fine alignment. Install with: pip install opencv-python-headless
```

Either install it or disable alignment in config (`AlignmentAlgo = off`).

---

## Architecture

### File Structure

```
meter-reader/
├── meter_reader.py          # Standalone inference engine
├── setup_wizard.py          # Flask web UI for configuration
├── Dockerfile               # Container for meter_reader
├── Dockerfile.setup         # Container for setup_wizard
├── docker-compose.yml       # Multi-service orchestration
├── requirements.txt         # Python dependencies
└── Readme.md               # This file
```

### Data Flow

```
┌──────────────────────────────────────────────────────────────┐
│  SETUP WIZARD (setup_wizard.py)                              │
│  ↓ Browser UI (6 steps) → API → Server-side state           │
│  ↓ Save config.ini + ref images to sdcard/config/           │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│  sdcard/config/ (persistent)                                 │
│  • config.ini            (meter configuration)               │
│  • reference.jpg         (rotated input image)               │
│  • ref0.jpg, ref1.jpg    (alignment marker crops)            │
│  • *.tflite              (TensorFlow Lite models)            │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│  METER READER (meter_reader.py)                              │
│  ↓ Load config + models                                      │
│  ↓ Input image → rotate → fine-align → crop ROIs            │
│  ↓ TFLite inference per ROI                                  │
│  ↓ Assemble digits/analogs → decimal shift → validate       │
│  ↓ Output JSON                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## Environment Variables

### Setup Wizard
- `FLASK_ENV=development` – enable debug mode
- `FLASK_DEBUG=1` – auto-reload on code changes

### Meter Reader
- None; use CLI arguments

---

## Compatibility

**Tested on:**
- macOS 13+
- Ubuntu 20.04+ / Debian 11+
- Proxmox LXC (Debian 12)
- Docker (x86-64, arm64)
- Python 3.9–3.12

**Models:** Any `.tflite` model exported from the original AI-on-the-edge-device training pipeline works directly.

---

## Licensing

This code is a port of [AI-on-the-edge-device](https://github.com/jomjol/AI-on-the-edge-device) and retains compatibility with its configuration and models. Refer to the upstream project for licensing of core logic and trained models.

---

## Contributing

Issues, PRs, and feedback welcome! Key areas for contribution:
- GPU/TPU acceleration
- Real-time monitoring API
- Additional post-processing rules
- Home Assistant / Node-RED integrations
- Model training docs for custom meters

---

## See Also

- **Original Project**: [jomjol/AI-on-the-edge-device](https://github.com/jomjol/AI-on-the-edge-device)
- **Model Zoo**: Pre-trained TFLite models in `sd-card/config/`
- **Docs**: [Official documentation](https://jomjol.github.io/AI-on-the-edge-device-docs/)

---

**Questions?** Feel free to open an issue or discussion thread!
