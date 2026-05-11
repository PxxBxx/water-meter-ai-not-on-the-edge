#!/usr/bin/env python3
"""
Meter Reader Setup Wizard

A browser-based setup tool that replicates the original AI-on-the-edge-device
web wizard – without any ESP32 or camera involvement.

Steps (mirroring the original firmware):
  1. Reference image   – upload your meter photo, set rotation/flip
  2. Alignment markers – drag-select two high-contrast reference regions
  3. Digit ROIs        – draw rectangles over each digit zone
  4. Analog ROIs       – draw rectangles over each analog dial zone
  5. Post-processing   – decimal shift, rate limits, …
  6. Save              – writes config.ini + reference images to sdcard/config/

Usage:
    python setup_wizard.py [--sdcard ./sdcard] [--port 5000] [--host 0.0.0.0]

Then open http://localhost:5000 in your browser.
"""

import argparse
import base64
import io
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request, send_file, send_from_directory

# ---------------------------------------------------------------------------
# Optional OpenCV for contrast enhancement
# ---------------------------------------------------------------------------
try:
    import cv2 as _cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

from PIL import Image, ImageEnhance, ImageOps

# ===========================================================================
# Flask app
# ===========================================================================

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB upload limit

# ---------------------------------------------------------------------------
# Server-side session state (single-user setup tool – no auth needed)
# ---------------------------------------------------------------------------
_state: Dict[str, Any] = {
    "reference_image": None,       # PIL Image – the uploaded reference picture
    "rotation": 0.0,               # degrees (positive = CCW for PIL)
    "flip_h": False,
    "flip_v": False,
    "markers": [None, None],       # [{"x","y","dx","dy"}, …] in reference coords
    "digit_groups": {},            # {"main": [{"name","x","y","dx","dy","ccw"}, …]}
    "analog_groups": {},
    "postprocessing": {},          # {"main": {field: value, …}}
    "digit_model": "/config/dig-cont_0900_s3_q.tflite",
    "analog_model": "/config/ana-cont_1500_s2_q.tflite",
    "digit_threshold": 0.5,
    "analog_threshold": 0.5,
    "alignment_algo": "default",
    "search_x": 40,
    "search_y": 40,
}


# ===========================================================================
# Helpers
# ===========================================================================

def _pil_to_b64(img: Image.Image, fmt: str = "JPEG", quality: int = 88) -> str:
    """Encode a PIL image to a base64 data-URI."""
    buf = io.BytesIO()
    rgb = img.convert("RGB")
    rgb.save(buf, format=fmt, quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def _rotated_image() -> Optional[Image.Image]:
    """Return the reference image with the current rotation and flip applied."""
    img = _state["reference_image"]
    if img is None:
        return None
    if _state["flip_h"]:
        img = ImageOps.mirror(img)
    if _state["flip_v"]:
        img = ImageOps.flip(img)
    if _state["rotation"] != 0.0:
        img = img.rotate(-_state["rotation"], expand=False, resample=Image.BICUBIC)
    return img


def _crop_region(img: Image.Image, x: int, y: int, dx: int, dy: int) -> Image.Image:
    iw, ih = img.size
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(iw, x + dx)
    y1 = min(ih, y + dy)
    return img.crop((x0, y0, x1, y1))


def _enhance_contrast(img: Image.Image) -> Image.Image:
    """Simple CLAHE-like contrast enhancement (use OpenCV if available)."""
    if _HAS_CV2:
        import numpy as np
        arr = np.array(img.convert("RGB"))
        lab = _cv2.cvtColor(arr, _cv2.COLOR_RGB2LAB)
        clahe = _cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        rgb = _cv2.cvtColor(lab, _cv2.COLOR_LAB2RGB)
        return Image.fromarray(rgb)
    # Fallback: PIL autocontrast
    return ImageOps.autocontrast(img, cutoff=1)


# ===========================================================================
# Config.ini generator
# ===========================================================================

def _build_config_ini() -> str:
    lines: List[str] = []

    lines.append("[Alignment]")
    if _state["rotation"] != 0.0:
        lines.append(f"InitialRotate = {_state['rotation']:.1f}")
    if _state["flip_h"] or _state["flip_v"]:
        lines.append(f"FlipImageSize = true")
    lines.append(f"SearchFieldX = {_state['search_x']}")
    lines.append(f"SearchFieldY = {_state['search_y']}")
    lines.append(f"AlignmentAlgo = {_state['alignment_algo']}")
    for i, m in enumerate(_state["markers"]):
        if m:
            lines.append(f"/config/ref{i}.jpg {m['target_x']} {m['target_y']}")
    lines.append("")

    # Digits
    if _state["digit_groups"]:
        lines.append("[Digits]")
        lines.append(f"Model = {_state['digit_model']}")
        lines.append(f"CNNGoodThreshold = {_state['digit_threshold']}")
        for grp_name, rois in _state["digit_groups"].items():
            for r in rois:
                ccw = "true" if r.get("ccw") else "false"
                lines.append(
                    f"{r['name']} {r['x']} {r['y']} {r['dx']} {r['dy']} {ccw}"
                )
        lines.append("")

    # Analog
    if _state["analog_groups"]:
        lines.append("[Analog]")
        lines.append(f"Model = {_state['analog_model']}")
        lines.append(f"CNNGoodThreshold = {_state['analog_threshold']}")
        for grp_name, rois in _state["analog_groups"].items():
            for r in rois:
                ccw = "true" if r.get("ccw") else "false"
                lines.append(
                    f"{r['name']} {r['x']} {r['y']} {r['dx']} {r['dy']} {ccw}"
                )
        lines.append("")

    # PostProcessing
    lines.append("[PostProcessing]")
    # Gather all group names across digit + analog
    all_groups = sorted(
        set(list(_state["digit_groups"].keys()) + list(_state["analog_groups"].keys()))
    )
    for grp in all_groups:
        pp = _state["postprocessing"].get(grp, {})
        ds = pp.get("decimal_shift", 0)
        lines.append(f"{grp}.DecimalShift = {ds}")
        ts = pp.get("analog_to_digit_transition_start", 9.2)
        lines.append(f"{grp}.AnalogDigitTransitionStart = {ts}")
        cr = pp.get("change_rate_threshold", 2)
        lines.append(f"{grp}.ChangeRateThreshold = {cr}")
        lines.append(f"PreValueUse = {'true' if pp.get('pre_value_use', True) else 'false'}")
        age = pp.get("pre_value_age_startup", 720)
        lines.append(f"PreValueAgeStartup = {age}")
        anr = "true" if pp.get("allow_negative_rates", False) else "false"
        lines.append(f"{grp}.AllowNegativeRates = {anr}")
        mrv = pp.get("max_rate_value", 0.05)
        lines.append(f"{grp}.MaxRateValue = {mrv}")
        er = "true" if pp.get("extended_resolution", False) else "false"
        lines.append(f"{grp}.ExtendedResolution = {er}")
        iln = "true" if pp.get("ignore_leading_nan", False) else "false"
        lines.append(f"{grp}.IgnoreLeadingNaN = {iln}")
        em = "true" if pp.get("error_message", True) else "false"
        lines.append(f"ErrorMessage = {em}")
        cdic = "true" if pp.get("check_digit_increase_consistency", False) else "false"
        lines.append(f"{grp}.CheckDigitIncreaseConsistency = {cdic}")
    lines.append("")

    return "\n".join(lines)


# ===========================================================================
# API routes
# ===========================================================================

# ---------- Step 1: Upload reference image ----------------------------------

@app.route("/api/image/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["file"]
    img = Image.open(f.stream).convert("RGB")
    _state["reference_image"] = img
    _state["rotation"] = 0.0
    _state["flip_h"] = False
    _state["flip_v"] = False
    rotated = _rotated_image()
    return jsonify({"image": _pil_to_b64(rotated), "width": img.width, "height": img.height})


@app.route("/api/image/rotate", methods=["POST"])
def api_rotate():
    data = request.get_json()
    _state["rotation"] = float(data.get("rotation", 0))
    _state["flip_h"] = bool(data.get("flip_h", False))
    _state["flip_v"] = bool(data.get("flip_v", False))
    rotated = _rotated_image()
    if rotated is None:
        return jsonify({"error": "No image loaded"}), 400
    return jsonify({"image": _pil_to_b64(rotated)})


@app.route("/api/image/current")
def api_current_image():
    rotated = _rotated_image()
    if rotated is None:
        return jsonify({"error": "No image loaded"}), 404
    return jsonify({"image": _pil_to_b64(rotated)})


# ---------- Step 2: Alignment markers ----------------------------------------

@app.route("/api/alignment/marker", methods=["POST"])
def api_set_marker():
    data = request.get_json()
    idx = int(data.get("index", 0))
    if idx not in (0, 1):
        return jsonify({"error": "index must be 0 or 1"}), 400

    x = int(data["x"])
    y = int(data["y"])
    dx = int(data["dx"])
    dy = int(data["dy"])

    rotated = _rotated_image()
    if rotated is None:
        return jsonify({"error": "No image loaded"}), 400

    cropped = _crop_region(rotated, x, y, dx, dy)
    enhanced = _enhance_contrast(cropped) if data.get("enhance") else cropped

    _state["markers"][idx] = {
        "x": x, "y": y, "dx": dx, "dy": dy,
        # target = top-left corner of the marker in the aligned image
        "target_x": x, "target_y": y,
    }

    return jsonify({
        "preview_original": _pil_to_b64(cropped),
        "preview_enhanced": _pil_to_b64(enhanced),
    })


@app.route("/api/alignment/marker/<int:idx>")
def api_get_marker(idx: int):
    m = _state["markers"][idx] if idx in (0, 1) else None
    if m is None:
        return jsonify({"error": "not set"}), 404
    rotated = _rotated_image()
    cropped = _crop_region(rotated, m["x"], m["y"], m["dx"], m["dy"])
    return jsonify({"preview": _pil_to_b64(cropped), "marker": m})


# ---------- Step 3 & 4: ROI management ---------------------------------------

def _roi_endpoint(roi_type: str):
    """Shared handler for /api/rois/digit and /api/rois/analog."""
    groups_key = "digit_groups" if roi_type == "digit" else "analog_groups"

    if request.method == "GET":
        return jsonify(_state[groups_key])

    data = request.get_json()
    action = data.get("action")

    if action == "set_groups":
        _state[groups_key] = data["groups"]
        return jsonify({"ok": True})

    if action == "preview_roi":
        rotated = _rotated_image()
        if rotated is None:
            return jsonify({"error": "No image loaded"}), 400
        roi = data["roi"]
        cropped = _crop_region(rotated, roi["x"], roi["y"], roi["dx"], roi["dy"])
        return jsonify({"preview": _pil_to_b64(cropped, quality=80)})

    return jsonify({"error": f"Unknown action: {action}"}), 400


@app.route("/api/rois/digit", methods=["GET", "POST"])
def api_rois_digit():
    return _roi_endpoint("digit")


@app.route("/api/rois/analog", methods=["GET", "POST"])
def api_rois_analog():
    return _roi_endpoint("analog")


# ---------- Step 5: Post-processing ------------------------------------------

@app.route("/api/postprocessing", methods=["GET", "POST"])
def api_postprocessing():
    if request.method == "GET":
        return jsonify({
            "postprocessing": _state["postprocessing"],
            "digit_model": _state["digit_model"],
            "analog_model": _state["analog_model"],
            "digit_threshold": _state["digit_threshold"],
            "analog_threshold": _state["analog_threshold"],
            "alignment_algo": _state["alignment_algo"],
            "search_x": _state["search_x"],
            "search_y": _state["search_y"],
        })

    data = request.get_json()
    _state["postprocessing"] = data.get("postprocessing", {})
    _state["digit_model"] = data.get("digit_model", _state["digit_model"])
    _state["analog_model"] = data.get("analog_model", _state["analog_model"])
    _state["digit_threshold"] = float(data.get("digit_threshold", _state["digit_threshold"]))
    _state["analog_threshold"] = float(data.get("analog_threshold", _state["analog_threshold"]))
    _state["alignment_algo"] = data.get("alignment_algo", _state["alignment_algo"])
    _state["search_x"] = int(data.get("search_x", _state["search_x"]))
    _state["search_y"] = int(data.get("search_y", _state["search_y"]))
    return jsonify({"ok": True})


# ---------- Step 6: Save to sdcard -------------------------------------------

@app.route("/api/save", methods=["POST"])
def api_save():
    sdcard = app.config["SDCARD_DIR"]
    config_dir = os.path.join(sdcard, "config")
    os.makedirs(config_dir, exist_ok=True)

    errors = []

    # 1. Write reference.jpg (rotated image)
    rotated = _rotated_image()
    if rotated is None:
        return jsonify({"error": "No reference image loaded"}), 400

    ref_path = os.path.join(config_dir, "reference.jpg")
    rotated.save(ref_path, "JPEG", quality=90)

    # 2. Write ref0.jpg and ref1.jpg (marker thumbnails)
    for i, m in enumerate(_state["markers"]):
        if m:
            cropped = _crop_region(rotated, m["x"], m["y"], m["dx"], m["dy"])
            enhanced = _enhance_contrast(cropped)
            enhanced.save(os.path.join(config_dir, f"ref{i}.jpg"), "JPEG", quality=90)
        else:
            errors.append(f"Alignment marker {i+1} not defined")

    # 3. Write config.ini
    ini_content = _build_config_ini()
    ini_path = os.path.join(config_dir, "config.ini")

    # Back up existing config
    if os.path.exists(ini_path):
        shutil.copy2(ini_path, ini_path + ".bak")

    with open(ini_path, "w", encoding="utf-8") as fh:
        fh.write(ini_content)

    return jsonify({
        "ok": True,
        "warnings": errors,
        "config_ini": ini_content,
        "saved_to": ini_path,
    })


@app.route("/api/config_preview")
def api_config_preview():
    return jsonify({"config_ini": _build_config_ini()})


# ---------- Serve the single-page UI -----------------------------------------

@app.route("/")
def index():
    return _SETUP_HTML, 200, {"Content-Type": "text/html; charset=utf-8"}


# ===========================================================================
# Embedded single-page application
# ===========================================================================

_SETUP_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Meter Reader – Setup Wizard</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:Arial,sans-serif;background:#f0f2f5;color:#222}
  header{background:#1e3a5f;color:#fff;padding:12px 20px;display:flex;align-items:center;gap:16px}
  header h1{font-size:1.2rem;font-weight:600}
  #steps{display:flex;background:#fff;border-bottom:2px solid #d0d0d0;padding:8px 20px;gap:4px;flex-wrap:wrap}
  .step-btn{padding:6px 14px;border:2px solid #ccc;border-radius:20px;background:#fff;cursor:pointer;font-size:0.85rem;color:#555;white-space:nowrap}
  .step-btn.active{background:#1e3a5f;color:#fff;border-color:#1e3a5f}
  .step-btn.done{border-color:#2ecc71;color:#2ecc71}
  .step-btn:disabled{opacity:0.4;cursor:default}
  #main{padding:16px 20px;max-width:1200px}
  .panel{display:none}.panel.active{display:block}
  h2{font-size:1.1rem;font-weight:600;margin-bottom:8px;color:#1e3a5f}
  .hint{background:#e8f4fd;border-left:4px solid #3498db;padding:8px 12px;font-size:0.85rem;margin-bottom:12px;border-radius:0 4px 4px 0}
  .row{display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap}
  .col{flex:1;min-width:220px}
  label{display:block;font-size:0.85rem;color:#555;margin-bottom:3px;margin-top:8px}
  input[type=number],input[type=text],select{width:100%;padding:5px 8px;border:1px solid #ccc;border-radius:4px;font-size:0.9rem}
  input[type=range]{width:100%}
  .btn{display:inline-block;padding:7px 16px;border-radius:4px;border:none;cursor:pointer;font-size:0.9rem;font-weight:600}
  .btn-primary{background:#1e3a5f;color:#fff}.btn-primary:hover{background:#16304f}
  .btn-success{background:#2ecc71;color:#fff}.btn-success:hover{background:#27ae60}
  .btn-danger{background:#e74c3c;color:#fff}.btn-danger:hover{background:#c0392b}
  .btn-secondary{background:#ecf0f1;color:#333;border:1px solid #ccc}.btn-secondary:hover{background:#dde1e4}
  .btn-sm{padding:3px 10px;font-size:0.8rem}
  .actions{margin-top:12px;display:flex;gap:8px;flex-wrap:wrap}
  #canvas-wrap{position:relative;display:inline-block;max-width:100%;overflow:auto;border:2px solid #ccc;background:#888}
  canvas{display:block;cursor:crosshair}
  .preview-thumb{max-height:80px;border:1px solid #ccc;border-radius:3px;margin-top:4px}
  .roi-list{list-style:none;margin-top:6px}
  .roi-list li{display:flex;align-items:center;gap:6px;padding:3px 0;border-bottom:1px solid #eee;font-size:0.85rem}
  .roi-list li.active-roi{background:#e8f4fd;border-radius:3px;padding:3px 4px}
  .badge{display:inline-block;padding:1px 7px;border-radius:10px;font-size:0.75rem;font-weight:700}
  .badge-digit{background:#3498db;color:#fff}
  .badge-analog{background:#e67e22;color:#fff}
  .group-tabs{display:flex;gap:4px;flex-wrap:wrap;margin-bottom:8px}
  .group-tab{padding:4px 12px;border-radius:12px;border:1px solid #ccc;cursor:pointer;font-size:0.82rem}
  .group-tab.active{background:#1e3a5f;color:#fff;border-color:#1e3a5f}
  .form-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
  .form-full{grid-column:1/-1}
  #toast{position:fixed;bottom:20px;right:20px;background:#333;color:#fff;padding:10px 18px;border-radius:6px;font-size:0.9rem;display:none;z-index:9999;max-width:340px}
  .section{margin-bottom:16px}
  #config-preview{white-space:pre;font-family:monospace;font-size:0.8rem;background:#1e1e1e;color:#d4d4d4;padding:12px;border-radius:6px;max-height:400px;overflow:auto}
  .checkbox-row{display:flex;align-items:center;gap:6px;margin-top:6px}
  .checkbox-row input{width:auto}
  .marker-panel{border:1px solid #ccc;border-radius:6px;padding:10px;margin-bottom:10px;background:#fafafa}
  .marker-panel h3{font-size:0.9rem;font-weight:600;margin-bottom:6px}
  .marker-imgs{display:flex;gap:8px;margin-top:6px;flex-wrap:wrap;align-items:flex-start}
  .nav-btns{display:flex;gap:8px;margin-top:16px}
</style>
</head>
<body>

<header>
  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="12" x2="15" y2="15"/></svg>
  <h1>Meter Reader – Setup Wizard</h1>
</header>

<div id="steps">
  <button class="step-btn active" id="tab0" onclick="goStep(0)">① Reference Image</button>
  <button class="step-btn" id="tab1" onclick="goStep(1)">② Alignment Markers</button>
  <button class="step-btn" id="tab2" onclick="goStep(2)">③ Digit ROIs</button>
  <button class="step-btn" id="tab3" onclick="goStep(3)">④ Analog ROIs</button>
  <button class="step-btn" id="tab4" onclick="goStep(4)">⑤ Post-processing</button>
  <button class="step-btn" id="tab5" onclick="goStep(5)">⑥ Save &amp; Export</button>
</div>

<div id="main">

<!-- ═══════════════════════ STEP 0 – REFERENCE IMAGE ══════════════════════ -->
<div class="panel active" id="panel0">
  <h2>Step 1 – Reference Image</h2>
  <div class="hint">Upload a clear photo of your meter. Adjust rotation so that the digits are <strong>horizontal</strong>. This image will be saved as <code>reference.jpg</code> and is also used to define all zones in the following steps.</div>

  <div class="row">
    <div class="col">
      <div class="section">
        <label>Upload meter photo</label>
        <input type="file" id="imgFile" accept="image/*" onchange="uploadImage()">
      </div>
      <div class="section">
        <label>Rotation (degrees, positive = clockwise)</label>
        <input type="number" id="rotAngle" value="0" step="1" min="-180" max="180" oninput="applyTransform()">
        <label>Fine-tune (±1°)</label>
        <input type="range" id="rotFine" min="-1" max="1" step="0.1" value="0" oninput="document.getElementById('rotFineVal').textContent=this.value; applyTransform()">
        <span id="rotFineVal" style="font-size:0.8rem;color:#555">0</span>°
        <div class="checkbox-row">
          <input type="checkbox" id="flipH" onchange="applyTransform()"><label for="flipH" style="margin:0">Mirror horizontally</label>
        </div>
        <div class="checkbox-row">
          <input type="checkbox" id="flipV" onchange="applyTransform()"><label for="flipV" style="margin:0">Flip vertically</label>
        </div>
      </div>
    </div>
    <div class="col" style="flex:3">
      <label>Preview (rotated/flipped)</label>
      <div id="canvas-wrap">
        <canvas id="cvs0"></canvas>
      </div>
    </div>
  </div>

  <div class="nav-btns">
    <button class="btn btn-primary" onclick="confirmStep0()">Confirm &amp; Continue →</button>
  </div>
</div>

<!-- ═══════════════════════ STEP 1 – ALIGNMENT MARKERS ═══════════════════ -->
<div class="panel" id="panel1">
  <h2>Step 2 – Alignment Markers</h2>
  <div class="hint">
    Drag two small, <strong>high-contrast, static</strong> reference regions on the image (e.g. a screw head or a printed logo corner). The system will use these to correct small camera misalignments between shots. Each marker must be unique enough for template matching.
  </div>

  <div class="row">
    <div class="col" style="min-width:180px;max-width:260px">
      <div class="marker-panel">
        <h3>Marker 1</h3>
        <button class="btn btn-secondary btn-sm" onclick="selectMarker(0)" id="msel0" style="border-color:#3498db;color:#3498db">● Select</button>
        <div class="marker-imgs" id="mpreview0"><span style="color:#aaa;font-size:0.8rem">Not set</span></div>
      </div>
      <div class="marker-panel">
        <h3>Marker 2</h3>
        <button class="btn btn-secondary btn-sm" onclick="selectMarker(1)" id="msel1">● Select</button>
        <div class="marker-imgs" id="mpreview1"><span style="color:#aaa;font-size:0.8rem">Not set</span></div>
      </div>
      <div class="checkbox-row" style="margin-top:8px">
        <input type="checkbox" id="enhanceMarker"><label for="enhanceMarker" style="margin:0">Enhance contrast for marker</label>
      </div>
      <label>Alignment algorithm</label>
      <select id="alignAlgo">
        <option value="default">default</option>
        <option value="highaccuracy">highaccuracy</option>
        <option value="fast">fast</option>
        <option value="off">off (disable)</option>
      </select>
      <label>Search field X</label>
      <input type="number" id="searchX" value="40" min="5" max="200">
      <label>Search field Y</label>
      <input type="number" id="searchY" value="40" min="5" max="200">
    </div>
    <div class="col" style="flex:4">
      <div id="canvas-wrap">
        <canvas id="cvs1"></canvas>
      </div>
      <div style="margin-top:6px;font-size:0.82rem;color:#555">Current selection: <span id="markerCoords">none</span></div>
    </div>
  </div>

  <div class="nav-btns">
    <button class="btn btn-secondary" onclick="goStep(0)">← Back</button>
    <button class="btn btn-primary" onclick="confirmStep1()">Confirm &amp; Continue →</button>
  </div>
</div>

<!-- ═══════════════════════ STEP 2 – DIGIT ROIs ═══════════════════════════ -->
<div class="panel" id="panel2">
  <h2>Step 3 – Digit ROIs</h2>
  <div class="hint">Draw a rectangle over each <strong>digit</strong> zone. Digits run <em>most-significant first</em> (left to right, as they appear on the meter). Use the <strong>equidistant spacing</strong> helpers to speed up placement.</div>
  <div id="roiUI" data-type="digit"></div>
  <div class="nav-btns">
    <button class="btn btn-secondary" onclick="goStep(1)">← Back</button>
    <button class="btn btn-primary" onclick="goStep(3)">Continue →</button>
  </div>
</div>

<!-- ═══════════════════════ STEP 3 – ANALOG ROIs ══════════════════════════ -->
<div class="panel" id="panel3">
  <h2>Step 4 – Analog ROIs</h2>
  <div class="hint">Draw a rectangle over each <strong>analog dial</strong> zone. Dials are ordered <em>most-significant first</em> too. Check <strong>CCW</strong> for counter-clockwise dials. Analog ROIs are square – keep Δx = Δy.</div>
  <div id="roiUI2" data-type="analog"></div>
  <div class="nav-btns">
    <button class="btn btn-secondary" onclick="goStep(2)">← Back</button>
    <button class="btn btn-primary" onclick="goStep(4)">Continue →</button>
  </div>
</div>

<!-- ═══════════════════════ STEP 4 – POST-PROCESSING ══════════════════════ -->
<div class="panel" id="panel4">
  <h2>Step 5 – Post-processing</h2>
  <div class="hint">These settings match the <code>[PostProcessing]</code> section of config.ini and the model paths in <code>[Digits]</code> / <code>[Analog]</code>.</div>

  <div class="section">
    <h3 style="font-size:0.95rem;margin-bottom:6px">Models</h3>
    <label>Digit model path (on sdcard)</label>
    <input type="text" id="digitModel" value="/config/dig-cont_0900_s3_q.tflite">
    <label>Digit CNN good threshold</label>
    <input type="number" id="digitThreshold" value="0.5" step="0.05" min="0" max="1">
    <label>Analog model path (on sdcard)</label>
    <input type="text" id="analogModel" value="/config/ana-cont_1500_s2_q.tflite">
    <label>Analog CNN good threshold</label>
    <input type="number" id="analogThreshold" value="0.5" step="0.05" min="0" max="1">
  </div>

  <div class="section">
    <h3 style="font-size:0.95rem;margin-bottom:6px">Alignment</h3>
    <label>Search field X</label><input type="number" id="pp_searchX" value="40">
    <label>Search field Y</label><input type="number" id="pp_searchY" value="40">
  </div>

  <div id="ppGroupsUI"></div>

  <div class="nav-btns">
    <button class="btn btn-secondary" onclick="goStep(3)">← Back</button>
    <button class="btn btn-primary" onclick="savePostProcessing(); goStep(5)">Continue →</button>
  </div>
</div>

<!-- ═══════════════════════ STEP 5 – SAVE ═════════════════════════════════ -->
<div class="panel" id="panel5">
  <h2>Step 6 – Save &amp; Export</h2>
  <div class="hint">Review the generated <code>config.ini</code> below, then click <strong>Save to sdcard</strong> to write all files.</div>

  <div class="actions">
    <button class="btn btn-secondary" onclick="previewConfig()">Refresh preview</button>
    <button class="btn btn-success" onclick="saveAll()">💾 Save to sdcard</button>
  </div>

  <div style="margin-top:12px">
    <label>Generated config.ini</label>
    <pre id="config-preview">Click "Refresh preview"…</pre>
  </div>

  <div id="saveResult" style="margin-top:10px;font-size:0.9rem"></div>

  <div class="nav-btns">
    <button class="btn btn-secondary" onclick="goStep(4)">← Back</button>
  </div>
</div>

</div><!-- #main -->

<div id="toast"></div>

<script>
/* ══════════════════════════════════════════════════════════════════════════
   Global state
   ══════════════════════════════════════════════════════════════════════════ */
let currentStep = 0;
let refImage = null;        // HTMLImageElement of the current rotated reference
let refWidth = 0, refHeight = 0;

/* shared drag-rect state */
let drag = false, rect = {startX:0,startY:0,w:0,h:0};
let activeCanvas = null, activeCtx = null;
let dragCallback = null;   /* fn(rect) called on mouseup */

// Alignment
let activeMarkerIdx = 0;
const markerData = [null, null];

// ROIs  {digit: {groupName:[{name,x,y,dx,dy,ccw}]}, analog: …}
const roiState = { digit: {}, analog: {} };
let activeGroup = { digit: "main", analog: "main" };
let activeROI   = { digit: 0, analog: 0 };

/* ══════════════════════════════════════════════════════════════════════════
   Utilities
   ══════════════════════════════════════════════════════════════════════════ */
function toast(msg, color="#333") {
  const t = document.getElementById("toast");
  t.style.background = color;
  t.textContent = msg;
  t.style.display = "block";
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => t.style.display="none", 3500);
}

async function api(path, opts={}) {
  const r = await fetch(path, { headers: {"Content-Type":"application/json"}, ...opts });
  return r.json();
}

function goStep(n) {
  document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
  document.querySelectorAll(".step-btn").forEach(b => b.classList.remove("active"));
  document.getElementById("panel"+n).classList.add("active");
  document.getElementById("tab"+n).classList.add("active");
  currentStep = n;
  if (n === 1) initAlignCanvas();
  if (n === 2) initROICanvas("digit");
  if (n === 3) initROICanvas("analog");
  if (n === 4) initPPGroups();
}

/* ══════════════════════════════════════════════════════════════════════════
   Canvas drag-rect helper  (shared by all steps)
   ══════════════════════════════════════════════════════════════════════════ */
function setupDrag(canvas, onDone) {
  if (canvas._dragSetup) return;
  canvas._dragSetup = true;

  function pos(e) {
    const b = canvas.getBoundingClientRect();
    const scaleX = canvas.width / b.width;
    const scaleY = canvas.height / b.height;
    return {
      x: Math.round((e.clientX - b.left) * scaleX),
      y: Math.round((e.clientY - b.top)  * scaleY)
    };
  }

  canvas.addEventListener("mousedown", e => {
    const p = pos(e);
    drag = true;
    rect = {startX:p.x, startY:p.y, w:0, h:0};
    activeCanvas = canvas;
    activeCtx    = canvas.getContext("2d");
    dragCallback = onDone;
  });
  canvas.addEventListener("mousemove", e => {
    if (!drag || activeCanvas !== canvas) return;
    const p = pos(e);
    rect.w = p.x - rect.startX;
    rect.h = p.y - rect.startY;
    drawCurrentRect(canvas);
  });
  canvas.addEventListener("mouseup", e => {
    if (!drag || activeCanvas !== canvas) return;
    drag = false;
    // Normalize
    if (rect.w < 0) { rect.startX += rect.w; rect.w = -rect.w; }
    if (rect.h < 0) { rect.startY += rect.h; rect.h = -rect.h; }
    if (rect.w > 4 && rect.h > 4 && dragCallback) {
      dragCallback({x:rect.startX, y:rect.startY, dx:rect.w, dy:rect.h});
    }
  });
}

function drawCurrentRect(canvas) {
  const ctx = canvas.getContext("2d");
  // Redraw base
  if (canvas._baseImg) {
    ctx.drawImage(canvas._baseImg, 0, 0, canvas.width, canvas.height);
  }
  canvas._drawExtras && canvas._drawExtras(ctx);
  ctx.strokeStyle = "#f00";
  ctx.lineWidth = 2;
  ctx.setLineDash([4,3]);
  ctx.strokeRect(rect.startX, rect.startY, rect.w, rect.h);
  ctx.setLineDash([]);
}

function loadImgToCanvas(canvas, dataURL) {
  return new Promise(resolve => {
    const img = new Image();
    img.onload = () => {
      canvas.width  = img.naturalWidth;
      canvas.height = img.naturalHeight;
      canvas.getContext("2d").drawImage(img, 0, 0);
      canvas._baseImg = img;
      resolve(img);
    };
    img.src = dataURL;
  });
}

/* ══════════════════════════════════════════════════════════════════════════
   STEP 0 – Reference image
   ══════════════════════════════════════════════════════════════════════════ */
async function uploadImage() {
  const file = document.getElementById("imgFile").files[0];
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  const r = await fetch("/api/image/upload", { method:"POST", body:form });
  const data = await r.json();
  if (data.error) { toast(data.error,"#e74c3c"); return; }
  refWidth  = data.width;
  refHeight = data.height;
  const canvas = document.getElementById("cvs0");
  await loadImgToCanvas(canvas, data.image);
  toast("Image loaded ✓","#2ecc71");
}

async function applyTransform() {
  const rotation = parseFloat(document.getElementById("rotAngle").value||0)
                 + parseFloat(document.getElementById("rotFine").value||0);
  const flip_h = document.getElementById("flipH").checked;
  const flip_v = document.getElementById("flipV").checked;
  const data = await api("/api/image/rotate", {
    method:"POST", body:JSON.stringify({rotation,flip_h,flip_v})
  });
  if (data.error) { toast(data.error,"#e74c3c"); return; }
  const canvas = document.getElementById("cvs0");
  await loadImgToCanvas(canvas, data.image);
}

async function confirmStep0() {
  const canvas = document.getElementById("cvs0");
  if (!canvas._baseImg) { toast("Please upload an image first","#e74c3c"); return; }
  document.getElementById("tab0").classList.add("done");
  goStep(1);
}

/* ══════════════════════════════════════════════════════════════════════════
   STEP 1 – Alignment markers
   ══════════════════════════════════════════════════════════════════════════ */
async function initAlignCanvas() {
  const canvas = document.getElementById("cvs1");
  const data = await api("/api/image/current");
  if (data.error) { toast("Load step 1 first","#e74c3c"); return; }
  await loadImgToCanvas(canvas, data.image);
  canvas._drawExtras = ctx => drawMarkerOverlays(ctx, canvas);
  setupDrag(canvas, onMarkerDrag);
}

function drawMarkerOverlays(ctx, canvas) {
  const colors = ["#3498db","#e67e22"];
  markerData.forEach((m, i) => {
    if (!m) return;
    ctx.strokeStyle = colors[i];
    ctx.lineWidth = 2;
    ctx.setLineDash([]);
    ctx.strokeRect(m.x, m.y, m.dx, m.dy);
    ctx.fillStyle = colors[i];
    ctx.font = "bold 14px Arial";
    ctx.fillText("M"+(i+1), m.x+3, m.y+15);
  });
}

function selectMarker(idx) {
  activeMarkerIdx = idx;
  document.getElementById("msel0").style.borderColor = idx===0?"#3498db":"#ccc";
  document.getElementById("msel1").style.borderColor = idx===1?"#e67e22":"#ccc";
  toast(`Draw marker ${idx+1} on the image`);
}

async function onMarkerDrag(r) {
  document.getElementById("markerCoords").textContent =
    `x=${r.x} y=${r.y} dx=${r.dx} dy=${r.dy}`;
  const enhance = document.getElementById("enhanceMarker").checked;
  const data = await api("/api/alignment/marker", {
    method:"POST",
    body: JSON.stringify({index:activeMarkerIdx, ...r, enhance})
  });
  if (data.error) { toast(data.error,"#e74c3c"); return; }
  markerData[activeMarkerIdx] = r;

  const el = document.getElementById("mpreview"+activeMarkerIdx);
  el.innerHTML =
    `<img class="preview-thumb" src="${data.preview_original}" title="original">` +
    `<img class="preview-thumb" src="${data.preview_enhanced}" title="enhanced (saved)">`;

  // Redraw overlays
  const canvas = document.getElementById("cvs1");
  const ctx = canvas.getContext("2d");
  ctx.drawImage(canvas._baseImg,0,0,canvas.width,canvas.height);
  drawMarkerOverlays(ctx, canvas);
}

async function confirmStep1() {
  if (!markerData[0] || !markerData[1]) {
    toast("Define both alignment markers first","#e74c3c"); return;
  }
  document.getElementById("tab1").classList.add("done");
  goStep(2);
}

/* ══════════════════════════════════════════════════════════════════════════
   STEP 2 & 3 – ROI management (shared)
   ══════════════════════════════════════════════════════════════════════════ */
function _roiContainer(type) {
  return document.getElementById(type==="digit" ? "roiUI" : "roiUI2");
}

function _roiCanvasId(type) { return type==="digit" ? "cvs2" : "cvs3"; }

async function initROICanvas(type) {
  let container = _roiContainer(type);
  // Build UI if needed
  if (!document.getElementById(_roiCanvasId(type))) {
    container.innerHTML = buildROIHtml(type);
  }
  const canvas = document.getElementById(_roiCanvasId(type));
  const data = await api("/api/image/current");
  if (data.error) { toast("Load step 1 first","#e74c3c"); return; }
  await loadImgToCanvas(canvas, data.image);
  canvas._drawExtras = ctx => drawROIOverlays(ctx, type);
  setupDrag(canvas, r => onROIDrag(r, type));
  if (!roiState[type]["main"]) roiState[type]["main"] = [];
  renderGroupTabs(type);
  renderROIList(type);
}

function buildROIHtml(type) {
  const color = type==="digit"?"#3498db":"#e67e22";
  return `
  <div class="row" style="align-items:flex-start">
    <div class="col" style="min-width:230px;max-width:270px">
      <div class="section">
        <div style="display:flex;align-items:center;justify-content:space-between">
          <label style="margin:0">Sequences</label>
          <button class="btn btn-secondary btn-sm" onclick="addGroup('${type}')">+ New</button>
        </div>
        <div class="group-tabs" id="gtabs_${type}" style="margin-top:6px"></div>
      </div>
      <div class="section">
        <div style="display:flex;align-items:center;justify-content:space-between">
          <label style="margin:0">ROIs <span class="badge badge-${type}">${type}</span></label>
          <div style="display:flex;gap:4px">
            <button class="btn btn-secondary btn-sm" onclick="addROI('${type}')">+ Add</button>
            <button class="btn btn-danger btn-sm" onclick="deleteROI('${type}')">✕</button>
          </div>
        </div>
        <ul class="roi-list" id="roilist_${type}"></ul>
      </div>
      <div class="section" id="roiform_${type}">
        <div class="form-grid">
          <div><label>x</label><input type="number" id="ri_x_${type}" oninput="syncROIFromForm('${type}')"></div>
          <div><label>Δx</label><input type="number" id="ri_dx_${type}" oninput="syncROIFromForm('${type}')"></div>
          <div><label>y</label><input type="number" id="ri_y_${type}" oninput="syncROIFromForm('${type}')"></div>
          <div><label>Δy</label><input type="number" id="ri_dy_${type}" oninput="syncROIFromForm('${type}')"></div>
          ${type==="analog"?`<div class="form-full"><div class="checkbox-row"><input type="checkbox" id="ri_ccw_${type}" onchange="syncROIFromForm('${type}')"><label for="ri_ccw_${type}" style="margin:0">Counter-clockwise (CCW)</label></div></div>`:""}
        </div>
        <div style="margin-top:6px;font-size:0.8rem;color:#555">Preview: <img id="roipreview_${type}" class="preview-thumb" src="" alt=""></div>
      </div>
      <div class="section">
        <div class="checkbox-row">
          <input type="checkbox" id="locksize_${type}" checked><label for="locksize_${type}" style="margin:0">Sync Δx/Δy between ROIs</label>
        </div>
        <div class="checkbox-row">
          <input type="checkbox" id="showall_${type}" checked onchange="drawROIOverlays(document.getElementById('${_roiCanvasId(type)}').getContext('2d'),'${type}')">
          <label for="showall_${type}" style="margin:0">Show all ROIs</label>
        </div>
      </div>
    </div>
    <div class="col" style="flex:4">
      <div id="canvas-wrap"><canvas id="${_roiCanvasId(type)}"></canvas></div>
    </div>
  </div>`;
}

function renderGroupTabs(type) {
  const el = document.getElementById("gtabs_"+type);
  if (!el) return;
  el.innerHTML = "";
  Object.keys(roiState[type]).forEach(gname => {
    const btn = document.createElement("button");
    btn.className = "group-tab" + (gname===activeGroup[type]?" active":"");
    btn.textContent = gname;
    btn.onclick = () => { activeGroup[type]=gname; activeROI[type]=0; renderGroupTabs(type); renderROIList(type); };
    el.appendChild(btn);
  });
}

function renderROIList(type) {
  const el = document.getElementById("roilist_"+type);
  if (!el) return;
  const rois = roiState[type][activeGroup[type]] || [];
  el.innerHTML = "";
  rois.forEach((r, i) => {
    const li = document.createElement("li");
    li.className = i===activeROI[type]?"active-roi":"";
    li.innerHTML = `<span style="color:#888;min-width:18px">${i+1}.</span> <span style="flex:1">${r.name}</span>
      <button class="btn btn-secondary btn-sm" onclick="moveROI('${type}',${i},-1)" ${i===0?"disabled":""}>↑</button>
      <button class="btn btn-secondary btn-sm" onclick="moveROI('${type}',${i}, 1)" ${i===rois.length-1?"disabled":""}>↓</button>`;
    li.addEventListener("click", () => { activeROI[type]=i; renderROIList(type); fillROIForm(type); });
    el.appendChild(li);
  });
  fillROIForm(type);
  redrawROICanvas(type);
}

function fillROIForm(type) {
  const rois = roiState[type][activeGroup[type]] || [];
  const r = rois[activeROI[type]];
  if (!r) return;
  const set = id => { const el=document.getElementById(id); if(el) el.value=r[id.split("_")[1]]; };
  ["ri_x","ri_y","ri_dx","ri_dy"].forEach(p => { const el=document.getElementById(p+"_"+type); if(el) el.value=r[p.replace("ri_","")]; });
  if (type==="analog") { const el=document.getElementById("ri_ccw_analog"); if(el) el.checked=!!r.ccw; }
  refreshROIPreview(type);
}

async function refreshROIPreview(type) {
  const rois = roiState[type][activeGroup[type]] || [];
  const r = rois[activeROI[type]];
  if (!r) return;
  const data = await api("/api/rois/"+type, {method:"POST", body:JSON.stringify({action:"preview_roi",roi:r})});
  const img = document.getElementById("roipreview_"+type);
  if (img && data.preview) img.src = data.preview;
}

function syncROIFromForm(type) {
  const rois = roiState[type][activeGroup[type]] || [];
  if (!rois[activeROI[type]]) return;
  const r = rois[activeROI[type]];
  const gv = id => { const el=document.getElementById(id+"_"+type); return el ? (parseFloat(el.value)||0) : 0; };
  const locksize = document.getElementById("locksize_"+type)?.checked;
  r.x  = gv("ri_x");
  r.y  = gv("ri_y");
  r.dx = gv("ri_dx");
  r.dy = gv("ri_dy");
  if (type==="analog") r.ccw = !!document.getElementById("ri_ccw_analog")?.checked;

  // Sync sizes if lock enabled
  if (locksize) {
    rois.forEach((other, i) => { if(i!==activeROI[type]){other.dx=r.dx; other.dy=r.dy;} });
  }
  redrawROICanvas(type);
  refreshROIPreview(type);
}

function onROIDrag(r, type) {
  const rois = roiState[type][activeGroup[type]];
  if (!rois || !rois.length) {
    // auto-create first ROI
    const gname = activeGroup[type];
    const name = `${gname}.${type.slice(0,3)}1`;
    roiState[type][gname].push({name, x:r.x, y:r.y, dx:r.dx, dy:r.dy, ccw:false});
    activeROI[type] = 0;
  } else {
    const roi = rois[activeROI[type]];
    if (roi) { roi.x=r.x; roi.y=r.y; roi.dx=r.dx; roi.dy=r.dy; }
  }
  renderROIList(type);
}

function addROI(type) {
  const gname = activeGroup[type];
  if (!roiState[type][gname]) roiState[type][gname]=[];
  const rois = roiState[type][gname];
  const n = rois.length+1;
  const name = prompt(`ROI name (e.g. ${gname}.${type.slice(0,3)}${n}):`, `${gname}.${type.slice(0,3)}${n}`);
  if (!name) return;
  const prev = rois[rois.length-1];
  const nx = prev ? prev.x+prev.dx+3 : 20;
  const ny = prev ? prev.y : 30;
  const dx = prev ? prev.dx : (type==="analog"?90:30);
  const dy = prev ? prev.dy : (type==="analog"?90:54);
  rois.push({name, x:nx, y:ny, dx, dy, ccw:false});
  activeROI[type]=rois.length-1;
  renderROIList(type);
}

function deleteROI(type) {
  const rois = roiState[type][activeGroup[type]];
  if (!rois || !rois.length) return;
  if (!confirm("Delete selected ROI?")) return;
  rois.splice(activeROI[type],1);
  if(activeROI[type]>=rois.length) activeROI[type]=Math.max(0,rois.length-1);
  renderROIList(type);
}

function moveROI(type, idx, dir) {
  const rois = roiState[type][activeGroup[type]];
  const ni = idx+dir;
  if (ni<0||ni>=rois.length) return;
  [rois[idx], rois[ni]] = [rois[ni], rois[idx]];
  activeROI[type]=ni;
  renderROIList(type);
}

function addGroup(type) {
  const name = prompt("Sequence name (e.g. main, water, gas):", "main");
  if (!name) return;
  if (!roiState[type][name]) roiState[type][name]=[];
  activeGroup[type]=name;
  activeROI[type]=0;
  renderGroupTabs(type);
  renderROIList(type);
}

function drawROIOverlays(ctx, type) {
  const canvas = document.getElementById(_roiCanvasId(type));
  if (!canvas || !canvas._baseImg) return;
  ctx.drawImage(canvas._baseImg,0,0,canvas.width,canvas.height);
  const showall = document.getElementById("showall_"+type)?.checked;
  const colors = ["#3498db","#2ecc71","#e67e22","#9b59b6","#1abc9c","#e74c3c","#f39c12","#34495e"];
  Object.entries(roiState[type]).forEach(([gname, rois]) => {
    rois.forEach((r, i) => {
      const isActive = gname===activeGroup[type] && i===activeROI[type];
      if (!showall && !isActive) return;
      const col = colors[i % colors.length];
      ctx.strokeStyle = isActive ? "#f00" : col;
      ctx.lineWidth = isActive ? 2 : 1;
      ctx.setLineDash(isActive ? [] : [3,2]);
      ctx.strokeRect(r.x, r.y, r.dx, r.dy);
      ctx.setLineDash([]);
      ctx.fillStyle = isActive ? "#f00" : col;
      ctx.font = "bold 11px Arial";
      ctx.fillText(r.name.split(".").pop(), r.x+2, r.y+12);
    });
  });
}

function redrawROICanvas(type) {
  const canvas = document.getElementById(_roiCanvasId(type));
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  drawROIOverlays(ctx, type);
}

/* ══════════════════════════════════════════════════════════════════════════
   STEP 4 – Post-processing per group
   ══════════════════════════════════════════════════════════════════════════ */
function allGroupNames() {
  const s = new Set([...Object.keys(roiState.digit), ...Object.keys(roiState.analog)]);
  return [...s];
}

function initPPGroups() {
  const el = document.getElementById("ppGroupsUI");
  const groups = allGroupNames();
  if (!groups.length) { el.innerHTML="<p style='color:#888'>No ROI groups defined yet.</p>"; return; }

  el.innerHTML = groups.map(g => {
    const pp = window._ppData && window._ppData[g] ? window._ppData[g] : {};
    return `
    <div class="section" style="border:1px solid #ddd;border-radius:6px;padding:10px;margin-top:10px">
      <h3 style="font-size:0.9rem;font-weight:600;margin-bottom:6px">Group: <strong>${g}</strong></h3>
      <div class="form-grid">
        <div><label>Decimal shift</label><input type="number" id="pp_ds_${g}" value="${pp.decimal_shift||0}"></div>
        <div><label>Max rate value</label><input type="number" id="pp_mrv_${g}" value="${pp.max_rate_value||0.05}" step="0.001"></div>
        <div><label>Analog→digit transition start</label><input type="number" id="pp_adts_${g}" value="${pp.analog_to_digit_transition_start||9.2}" step="0.1"></div>
        <div><label>Change rate threshold</label><input type="number" id="pp_crt_${g}" value="${pp.change_rate_threshold||2}"></div>
      </div>
      <div style="display:flex;flex-wrap:wrap;gap:12px;margin-top:6px">
        <div class="checkbox-row"><input type="checkbox" id="pp_er_${g}" ${pp.extended_resolution?"checked":""}><label for="pp_er_${g}" style="margin:0">Extended resolution</label></div>
        <div class="checkbox-row"><input type="checkbox" id="pp_anr_${g}" ${pp.allow_negative_rates?"checked":""}><label for="pp_anr_${g}" style="margin:0">Allow negative rates</label></div>
        <div class="checkbox-row"><input type="checkbox" id="pp_iln_${g}" ${pp.ignore_leading_nan?"checked":""}><label for="pp_iln_${g}" style="margin:0">Ignore leading NaN</label></div>
        <div class="checkbox-row"><input type="checkbox" id="pp_pvu_${g}" ${pp.pre_value_use!==false?"checked":""}><label for="pp_pvu_${g}" style="margin:0">Use previous value</label></div>
        <div class="checkbox-row"><input type="checkbox" id="pp_cdic_${g}" ${pp.check_digit_increase_consistency?"checked":""}><label for="pp_cdic_${g}" style="margin:0">Check digit increase consistency</label></div>
        <div class="checkbox-row"><input type="checkbox" id="pp_em_${g}" ${pp.error_message!==false?"checked":""}><label for="pp_em_${g}" style="margin:0">Error message</label></div>
      </div>
    </div>`;
  }).join("");

  // Restore model fields
  document.getElementById("digitModel").value = window._ppMeta?.digit_model || "/config/dig-cont_0900_s3_q.tflite";
  document.getElementById("analogModel").value = window._ppMeta?.analog_model || "/config/ana-cont_1500_s2_q.tflite";
  document.getElementById("digitThreshold").value = window._ppMeta?.digit_threshold ?? 0.5;
  document.getElementById("analogThreshold").value = window._ppMeta?.analog_threshold ?? 0.5;
  document.getElementById("pp_searchX").value = window._ppMeta?.search_x ?? 40;
  document.getElementById("pp_searchY").value = window._ppMeta?.search_y ?? 40;
}

async function savePostProcessing() {
  const groups = allGroupNames();
  const postprocessing = {};
  groups.forEach(g => {
    postprocessing[g] = {
      decimal_shift: parseFloat(document.getElementById("pp_ds_"+g)?.value||0),
      max_rate_value: parseFloat(document.getElementById("pp_mrv_"+g)?.value||0.05),
      analog_to_digit_transition_start: parseFloat(document.getElementById("pp_adts_"+g)?.value||9.2),
      change_rate_threshold: parseFloat(document.getElementById("pp_crt_"+g)?.value||2),
      extended_resolution: !!document.getElementById("pp_er_"+g)?.checked,
      allow_negative_rates: !!document.getElementById("pp_anr_"+g)?.checked,
      ignore_leading_nan: !!document.getElementById("pp_iln_"+g)?.checked,
      pre_value_use: !!document.getElementById("pp_pvu_"+g)?.checked,
      check_digit_increase_consistency: !!document.getElementById("pp_cdic_"+g)?.checked,
      error_message: !!document.getElementById("pp_em_"+g)?.checked,
    };
  });
  window._ppData = postprocessing;
  window._ppMeta = {
    digit_model: document.getElementById("digitModel").value,
    analog_model: document.getElementById("analogModel").value,
    digit_threshold: parseFloat(document.getElementById("digitThreshold").value),
    analog_threshold: parseFloat(document.getElementById("analogThreshold").value),
    search_x: parseInt(document.getElementById("pp_searchX").value),
    search_y: parseInt(document.getElementById("pp_searchY").value),
  };

  await api("/api/postprocessing", {
    method:"POST",
    body: JSON.stringify({postprocessing, ...window._ppMeta})
  });

  // Push ROIs to server
  await api("/api/rois/digit", {method:"POST", body:JSON.stringify({action:"set_groups", groups:roiState.digit})});
  await api("/api/rois/analog", {method:"POST", body:JSON.stringify({action:"set_groups", groups:roiState.analog})});

  document.getElementById("tab4").classList.add("done");
  toast("Post-processing saved ✓","#2ecc71");
}

/* ══════════════════════════════════════════════════════════════════════════
   STEP 5 – Save
   ══════════════════════════════════════════════════════════════════════════ */
async function previewConfig() {
  // Push latest state first
  await api("/api/rois/digit",  {method:"POST", body:JSON.stringify({action:"set_groups", groups:roiState.digit})});
  await api("/api/rois/analog", {method:"POST", body:JSON.stringify({action:"set_groups", groups:roiState.analog})});
  const data = await api("/api/config_preview");
  document.getElementById("config-preview").textContent = data.config_ini || "(empty)";
}

async function saveAll() {
  // Push latest state
  await api("/api/rois/digit",  {method:"POST", body:JSON.stringify({action:"set_groups", groups:roiState.digit})});
  await api("/api/rois/analog", {method:"POST", body:JSON.stringify({action:"set_groups", groups:roiState.analog})});

  const data = await api("/api/save", {method:"POST", body:"{}"});
  const el = document.getElementById("saveResult");
  if (data.error) {
    el.innerHTML = `<span style="color:#e74c3c">✗ ${data.error}</span>`;
    toast(data.error,"#e74c3c");
    return;
  }
  document.getElementById("config-preview").textContent = data.config_ini;
  let msg = `<span style="color:#2ecc71">✓ Saved to ${data.saved_to}</span>`;
  if (data.warnings && data.warnings.length) {
    msg += `<br><span style="color:#e67e22">Warnings: ${data.warnings.join(", ")}</span>`;
  }
  el.innerHTML = msg;
  document.getElementById("tab5").classList.add("done");
  toast("Configuration saved ✓","#2ecc71");
}

/* init */
(async () => {
  const data = await api("/api/postprocessing");
  if (data && !data.error) { window._ppData = data.postprocessing; window._ppMeta = data; }
})();
</script>
</body>
</html>
"""


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Meter Reader Setup Wizard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sdcard", default="./sdcard", metavar="DIR",
                        help="sdcard root directory (default: ./sdcard)")
    parser.add_argument("--port", type=int, default=5000, metavar="PORT",
                        help="HTTP port (default: 5000)")
    parser.add_argument("--host", default="0.0.0.0", metavar="HOST",
                        help="Listen address (default: 0.0.0.0)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable Flask debug mode")
    args = parser.parse_args()

    app.config["SDCARD_DIR"] = os.path.abspath(args.sdcard)
    os.makedirs(os.path.join(app.config["SDCARD_DIR"], "config"), exist_ok=True)

    print(f"Meter Reader Setup Wizard")
    print(f"  sdcard : {app.config['SDCARD_DIR']}")
    print(f"  Open   : http://localhost:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
