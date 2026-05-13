#!/usr/bin/env python3
"""
Meter Reader - Standalone port of AI-on-the-edge-device core logic.

Takes a meter image as input, runs the same TFLite digit/analog models,
and outputs the meter reading(s) as JSON.

Compatible with the same config.ini and .tflite model files used by the
original ESP32 firmware.

Usage:
    python meter_reader.py --image /path/to/image.jpg \\
                           --sdcard /path/to/sdcard-dir \\
                           [--config /path/to/config.ini] \\
                           [--debug-dir /tmp/debug] \\
                           [--pretty]

Outputs JSON to stdout, e.g.:
    {"main": {"raw": "056.4321", "value": "56.4321", "error": "no error"}}
"""

import argparse
import datetime
import json
import math
import os
import re
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# TFLite runtime – prefer lightweight tflite-runtime, fall back to tensorflow
# ---------------------------------------------------------------------------
try:
    import tflite_runtime.interpreter as _tfl

    def _make_interpreter(model_path: str):
        return _tfl.Interpreter(model_path=model_path)

except ImportError:
    try:
        import ai_edge_litert.interpreter as _tfl  # newer standalone package

        def _make_interpreter(model_path: str):
            return _tfl.Interpreter(model_path=model_path)

    except ImportError:
        try:
            import tensorflow as _tf

            def _make_interpreter(model_path: str):
                return _tf.lite.Interpreter(model_path=model_path)

        except ImportError:
            print(
                "ERROR: No TFLite runtime found. Install one of:\n"
                "  pip install tflite-runtime\n"
                "  pip install ai-edge-litert\n"
                "  pip install tensorflow-cpu",
                file=sys.stderr,
            )
            sys.exit(1)

# ---------------------------------------------------------------------------
# OpenCV – optional but strongly recommended for alignment
# ---------------------------------------------------------------------------
try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

def _is_url(source: str) -> bool:
    """Check if source is a URL (http://, https://, etc.)."""
    return source.startswith(("http://", "https://", "ftp://"))


def _load_image(source: str) -> Image.Image:
    """
    Load an image from either a URL or a file path.
    
    Args:
        source: Either a URL (http://, https://) or a local file path.
    
    Returns:
        PIL Image in RGB mode.
    
    Raises:
        FileNotFoundError: if the local file doesn't exist.
        urllib.error.URLError: if the URL fails.
        PIL.UnidentifiedImageError: if the image format is not recognized.
    """
    if _is_url(source):
        # Download from URL
        with urllib.request.urlopen(source) as response:
            image_data = response.read()
        image = Image.open(BytesIO(image_data))
    else:
        # Load from file path
        if not os.path.exists(source):
            raise FileNotFoundError(f"Image not found: {source}")
        image = Image.open(source)
    
    return image.convert("RGB")


# ===========================================================================
# Data structures
# ===========================================================================

class CNNType(Enum):
    AutoDetect = "autodetect"
    Digit = "digit"           # 11 outputs: argmax → class 0-10 (10 = N/NaN)
    Analogue = "analogue"     # 2 outputs: atan2 → angle → 0-10
    DoubleHybrid10 = "doublehybrid10"  # 10 outputs: softmax + neighbor weighting
    Digit100 = "digit100"     # 100 outputs: value = argmax / 10
    Analogue100 = "analogue100"  # 100 outputs, 32×32 input: same formula


@dataclass
class ROI:
    name: str
    x: int          # left edge in aligned image (pixels)
    y: int          # top edge in aligned image (pixels)
    dx: int         # width
    dy: int         # height
    ccw: bool = False       # counter-clockwise dial direction (analog only)
    result_float: float = -1.0
    result_klasse: int = -1
    is_reject: bool = False
    confidence: float = -1.0  # model confidence/probability for the result
    image: Optional[np.ndarray] = None  # resized RGB uint8 (H, W, 3)


@dataclass
class GeneralGroup:
    """Named group of ROIs – corresponds to one meter sequence / number."""
    name: str
    rois: List[ROI] = field(default_factory=list)


@dataclass
class RefInfo:
    image_file: str   # absolute path to the template thumbnail
    target_x: int     # expected top-left x in the aligned image
    target_y: int     # expected top-left y in the aligned image
    search_x: int = 40
    search_y: int = 40


@dataclass
class AlignmentConfig:
    initial_rotate: float = 0.0
    initial_flip: bool = False
    references: List[RefInfo] = field(default_factory=list)
    algo: str = "default"   # default | highaccuracy | fast | off


@dataclass
class LoggingConfig:
    """Configuration for ROI sample logging (model training data collection)."""
    enabled: bool = False
    save_location: str = ""  # /sdcard/log or similar
    retention_days: int = 3  # Keep samples for N days
    selective: Optional[List[str]] = None  # If set, only log these ROI names
    save_all_files: bool = False  # If True, save all ROIs


@dataclass
class CNNConfig:
    model_file: str = ""
    cnn_type: CNNType = CNNType.AutoDetect
    good_threshold: float = 0.5
    groups: List[GeneralGroup] = field(default_factory=list)


@dataclass
class PostProcessingNumber:
    name: str
    decimal_shift: int = 0
    extended_resolution: bool = False
    analog_to_digit_transition_start: float = 9.2
    allow_negative_rates: bool = False
    ignore_leading_nan: bool = False
    max_rate_value: float = 0.1
    use_max_rate_value: bool = False
    change_rate_threshold: int = 2
    check_digit_increase_consistency: bool = False


@dataclass
class PreValue:
    """Previous meter reading with timestamp."""
    timestamp: str = ""
    value: float = -1.0


@dataclass
class Config:
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    digits: Optional[CNNConfig] = None
    analog: Optional[CNNConfig] = None
    postprocessing: Dict[str, PostProcessingNumber] = field(default_factory=dict)
    pre_value_use: bool = False
    error_message: bool = True
    prevalue: Optional[PreValue] = None
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# ===========================================================================
# Config parser
# ===========================================================================

def _to_bool(s: str) -> bool:
    return s.strip().lower() in ("true", "yes", "1", "on")


def _parse_kv(line: str) -> Tuple[str, List[str]]:
    """
    Parse a config line that may use 'Key = Value' or 'token1 token2 ...' syntax.
    Returns (key, [value_tokens]).  Inline ; comments are stripped first.
    """
    line = line.split(";")[0].strip()
    if "=" in line:
        key, _, rest = line.partition("=")
        return key.strip(), rest.strip().split()
    parts = line.split()
    return (parts[0] if parts else ""), (parts[1:] if len(parts) > 1 else [])


def load_prevalue(prevalue_path: str) -> Optional[PreValue]:
    """Load the previous meter reading from prevalue.ini.
    Format:
        Line 1: timestamp (YYYY-MM-DD_HH-MM-SS)
        Line 2: previous reading value
    """
    if not os.path.exists(prevalue_path):
        return None

    try:
        with open(prevalue_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            if len(lines) >= 2:
                return PreValue(
                    timestamp=lines[0],
                    value=float(lines[1])
                )
    except (ValueError, IOError):
        pass

    return None


def save_prevalue(prevalue_path: str, value: float) -> bool:
    """Save the current meter reading to prevalue.ini.
    Creates the file if it doesn't exist. Updates with new timestamp and value.
    """
    try:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with open(prevalue_path, "w", encoding="utf-8") as f:
            f.write(f"{timestamp}\n{value}\n")
        return True
    except IOError:
        return False


def parse_config(config_path: str, sdcard_dir: str, prevalue_path: Optional[str] = None) -> Config:
    """Parse an AI-on-the-edge-device config.ini."""

    cfg = Config()
    section = None
    digit_cfg = CNNConfig()
    analog_cfg = CNNConfig()
    pp: Dict[str, PostProcessingNumber] = {}

    # SearchField defaults – may be overridden before reference lines
    search_x = 40
    search_y = 40

    def sdcard_path(p: str) -> str:
        """Resolve /config/... or /sdcard/... to real filesystem paths."""
        p = p.strip().strip('"').strip("'")
        if p.startswith("/sdcard/"):
            return os.path.join(sdcard_dir, p[len("/sdcard/"):])
        if p.startswith("/"):
            return os.path.join(sdcard_dir, p.lstrip("/"))
        return p

    def get_pp(name: str) -> PostProcessingNumber:
        if name not in pp:
            pp[name] = PostProcessingNumber(name=name)
        return pp[name]

    with open(config_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()

            # Skip blank lines and full-line comments
            if not line or line.startswith(";") or line.startswith("#"):
                continue

            # Section header
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1].lower()
                continue

            key, vals = _parse_kv(line)
            ku = key.upper()
            v0 = vals[0] if vals else ""

            # ---- [Alignment] ------------------------------------------------
            if section == "alignment":
                if ku == "INITIALROTATE":
                    cfg.alignment.initial_rotate = float(v0) if v0 else 0.0
                elif ku == "FLIPIMAGESIZE":
                    cfg.alignment.initial_flip = _to_bool(v0)
                elif ku == "SEARCHFIELDX":
                    search_x = int(v0) if v0 else search_x
                elif ku == "SEARCHFIELDY":
                    search_y = int(v0) if v0 else search_y
                elif ku == "ALIGNMENTALGO":
                    cfg.alignment.algo = v0.lower()
                elif ku == "ANTIALIASING":
                    pass  # not needed
                elif not ku and len(line.split()) == 3:
                    # Reference line: /config/ref0.jpg 103 271
                    parts = line.split()
                    try:
                        cfg.alignment.references.append(RefInfo(
                            image_file=sdcard_path(parts[0]),
                            target_x=int(parts[1]),
                            target_y=int(parts[2]),
                            search_x=search_x,
                            search_y=search_y,
                        ))
                    except (ValueError, IndexError):
                        pass
                elif ku.startswith("/") or key.startswith("/"):
                    # Also handle reference line when key starts with /
                    parts = line.split()
                    if len(parts) == 3:
                        try:
                            cfg.alignment.references.append(RefInfo(
                                image_file=sdcard_path(parts[0]),
                                target_x=int(parts[1]),
                                target_y=int(parts[2]),
                                search_x=search_x,
                                search_y=search_y,
                            ))
                        except (ValueError, IndexError):
                            pass

            # ---- [Digits] or [Analog] ---------------------------------------
            elif section in ("digits", "analog"):
                target_cfg = digit_cfg if section == "digits" else analog_cfg

                if ku == "MODEL":
                    target_cfg.model_file = sdcard_path(v0)
                elif ku == "CNNGOODTHRESHOLD":
                    target_cfg.good_threshold = float(v0) if v0 else 0.5
                elif ku == "ROIIMAGESLOCATION":
                    cfg.logging.save_location = sdcard_path(v0)
                    cfg.logging.enabled = True
                elif ku == "ROIIMAGESRETENTION":
                    cfg.logging.retention_days = int(v0) if v0 else 3
                elif ku == "LOGIMAGESELECT":
                    cfg.logging.selective = [s.strip() for s in v0.split(",")] if v0 else None
                elif ku == "SAVEALLFILES":
                    cfg.logging.save_all_files = _to_bool(v0)
                else:
                    # ROI line: name x y dx dy [ccw]
                    # The key is the ROI name (e.g. "main.dig1")
                    roi_parts = line.split()
                    if len(roi_parts) >= 5:
                        try:
                            roi_name = roi_parts[0]
                            x = int(roi_parts[1])
                            y = int(roi_parts[2])
                            dx = int(roi_parts[3])
                            dy = int(roi_parts[4])
                            ccw = _to_bool(roi_parts[5]) if len(roi_parts) > 5 else False

                            grp_name = roi_name.split(".")[0] if "." in roi_name else "default"
                            grp = next((g for g in target_cfg.groups if g.name == grp_name), None)
                            if grp is None:
                                grp = GeneralGroup(name=grp_name)
                                target_cfg.groups.append(grp)
                            grp.rois.append(ROI(name=roi_name, x=x, y=y, dx=dx, dy=dy, ccw=ccw))
                        except (ValueError, IndexError):
                            pass

            # ---- [PostProcessing] -------------------------------------------
            elif section == "postprocessing":
                # key may be "main.DecimalShift" or "PreValueUse"
                dot = key.find(".")
                if dot >= 0:
                    grp_name = key[:dot].strip()
                    param = key[dot + 1:].strip().upper()
                else:
                    grp_name = "default"
                    param = ku

                if param == "DECIMALSHIFT":
                    get_pp(grp_name).decimal_shift = int(v0) if v0 else 0
                elif param == "EXTENDEDRESOLUTION":
                    get_pp(grp_name).extended_resolution = _to_bool(v0)
                elif param in ("ANALOGDIGITTRANSITIONSTART", "ANALOGTODIGITTRANSITIONSTART"):
                    get_pp(grp_name).analog_to_digit_transition_start = float(v0) if v0 else 9.2
                elif param == "ALLOWNEGATIVERATES":
                    get_pp(grp_name).allow_negative_rates = _to_bool(v0)
                elif param == "IGNORELEADINGNAN":
                    get_pp(grp_name).ignore_leading_nan = _to_bool(v0)
                elif param == "MAXRATEVALUE":
                    n = get_pp(grp_name)
                    n.max_rate_value = float(v0) if v0 else 0.1
                    n.use_max_rate_value = True
                elif param == "CHANGERATETHRESHOLD":
                    get_pp(grp_name).change_rate_threshold = int(float(v0)) if v0 else 2
                elif param == "CHECKDIGITINCREASECONSISTENCY":
                    get_pp(grp_name).check_digit_increase_consistency = _to_bool(v0)
                elif param == "PREVALUEUSE":
                    cfg.pre_value_use = _to_bool(v0)
                elif param == "ERRORMESSAGE":
                    cfg.error_message = _to_bool(v0)
                # Other post-processing params (MaxRateType, PreValueAgeStartup, etc.)
                # are only relevant for time-series validation; skip for single-image mode.

    # Apply updated search fields to any references parsed before the SearchField lines
    for ref in cfg.alignment.references:
        ref.search_x = search_x
        ref.search_y = search_y

    if digit_cfg.groups:
        cfg.digits = digit_cfg
    if analog_cfg.groups:
        cfg.analog = analog_cfg
    cfg.postprocessing = pp

    # Load prevalue if available and enabled
    if cfg.pre_value_use and prevalue_path:
        cfg.prevalue = load_prevalue(prevalue_path)

    return cfg


# ===========================================================================
# Image alignment
# ===========================================================================

def align_image(image: Image.Image, alignment: AlignmentConfig) -> Image.Image:
    """
    Align the input image using the two reference template images.

    Steps:
      1. Optional initial rotation / flip (coarse)
      2. Template matching (OpenCV) to find each reference mark in the image
      3. Estimate similarity transform (rotation + scale + translation)
         that maps the found positions to the target positions
      4. Warp the image

    If OpenCV is not available or alignment is disabled, returns the image
    after the initial rotation only.
    """
    # 1. Initial rotation / flip
    if alignment.initial_flip:
        image = image.rotate(90, expand=True)

    if alignment.initial_rotate != 0.0:
        image = image.rotate(
            -alignment.initial_rotate, expand=False, resample=Image.BICUBIC
        )

    if alignment.algo == "off" or not alignment.references:
        return image

    if not _HAS_CV2:
        print(
            "WARNING: opencv-python not installed – skipping fine alignment. "
            "Install with: pip install opencv-python-headless",
            file=sys.stderr,
        )
        return image

    img_np = np.array(image.convert("RGB"))
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    src_pts: List[List[float]] = []
    dst_pts: List[List[float]] = []

    for ref in alignment.references:
        if not os.path.exists(ref.image_file):
            print(f"WARNING: Reference image not found: {ref.image_file}", file=sys.stderr)
            continue

        tmpl_img = Image.open(ref.image_file).convert("L")
        tmpl = np.array(tmpl_img)
        th, tw = tmpl.shape[:2]
        ih, iw = img_gray.shape[:2]

        # Constrain search region around the expected target position
        sx0 = max(0, ref.target_x - ref.search_x)
        sy0 = max(0, ref.target_y - ref.search_y)
        sx1 = min(iw, ref.target_x + ref.search_x + tw)
        sy1 = min(ih, ref.target_y + ref.search_y + th)

        roi_gray = img_gray[sy0:sy1, sx0:sx1]

        if roi_gray.shape[0] < th or roi_gray.shape[1] < tw:
            print(
                f"WARNING: Search region too small for ref {ref.image_file}",
                file=sys.stderr,
            )
            continue

        # Template matching (sum of squared differences, normalised)
        res = cv2.matchTemplate(roi_gray, tmpl, cv2.TM_SQDIFF_NORMED)
        _, _, min_loc, _ = cv2.minMaxLoc(res)

        # Convert back to full-image coordinates (use top-left corner of match)
        found_x = float(sx0 + min_loc[0])
        found_y = float(sy0 + min_loc[1])

        src_pts.append([found_x, found_y])
        dst_pts.append([float(ref.target_x), float(ref.target_y)])

    if len(src_pts) < 1:
        return image

    if len(src_pts) == 1:
        # Pure translation
        dx = dst_pts[0][0] - src_pts[0][0]
        dy = dst_pts[0][1] - src_pts[0][1]
        M = np.float32([[1, 0, dx], [0, 1, dy]])
    else:
        # Similarity transform (rotation + uniform scale + translation)
        src = np.array(src_pts, dtype=np.float32)
        dst = np.array(dst_pts, dtype=np.float32)
        M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC)
        if M is None:
            print("WARNING: Could not estimate alignment transform.", file=sys.stderr)
            return image

    h, w = img_np.shape[:2]
    aligned_np = cv2.warpAffine(img_np, M, (w, h), flags=cv2.INTER_LINEAR)
    return Image.fromarray(aligned_np)


# ===========================================================================
# Autotune (contrast enhancement)
# ===========================================================================

def autotune_image(image: Image.Image) -> Image.Image:
    """
    Enhance contrast on the aligned image before ROI extraction.

    Steps:
      1. Convert to grayscale to remove color variation
      2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation) via
         OpenCV for localised contrast boost – falls back to PIL autocontrast
         if OpenCV is unavailable
      3. Convert back to RGB so the rest of the pipeline is unchanged
    """
    if _HAS_CV2:
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(rgb)
    else:
        from PIL import ImageOps
        gray = image.convert("L")
        equalized = ImageOps.autocontrast(gray)
        return equalized.convert("RGB")


# ===========================================================================
# ROI extraction
# ===========================================================================

def extract_roi_images(
    image: Image.Image, group: GeneralGroup, model_w: int, model_h: int
) -> None:
    """
    Crop each ROI from the aligned image and resize it to the model's
    expected input dimensions (model_w × model_h).  Result stored in roi.image.
    """
    img_w, img_h = image.size

    for roi in group.rois:
        x0 = max(0, roi.x)
        y0 = max(0, roi.y)
        x1 = min(img_w, roi.x + roi.dx)
        y1 = min(img_h, roi.y + roi.dy)

        if x1 <= x0 or y1 <= y0:
            roi.image = np.zeros((model_h, model_w, 3), dtype=np.uint8)
            continue

        cropped = image.crop((x0, y0, x1, y1))
        resized = cropped.resize((model_w, model_h), Image.LANCZOS)
        roi.image = np.array(resized.convert("RGB"), dtype=np.uint8)


# ===========================================================================
# TFLite inference
# ===========================================================================

def _detect_cnn_type(interp) -> Tuple[CNNType, int, int, int]:
    """
    Auto-detect CNN type from model input/output tensor dimensions.
    Mirrors ClassFlowCNNGeneral::getNetworkParameter().
    """
    in_shape = interp.get_input_details()[0]["shape"]   # [1, H, W, C]
    out_shape = interp.get_output_details()[0]["shape"]  # [1, N]

    h = int(in_shape[1])
    w = int(in_shape[2])
    c = int(in_shape[3])
    n = int(out_shape[1])

    if n == 2:
        return CNNType.Analogue, w, h, c
    if n == 10:
        return CNNType.DoubleHybrid10, w, h, c
    if n == 11:
        return CNNType.Digit, w, h, c
    if n == 100:
        if w == 32 and h == 32:
            return CNNType.Analogue100, w, h, c
        return CNNType.Digit100, w, h, c

    raise ValueError(f"Unsupported model output dimension: {n}")


def _run_inference(interp, roi_image: np.ndarray) -> np.ndarray:
    """
    Feed an ROI image (H, W, 3) uint8 into the TFLite interpreter.
    Returns the raw output array (N,).

    NOTE: The original firmware feeds raw 0-255 float values without
    normalisation.  We replicate that here.
    """
    in_details = interp.get_input_details()
    data = roi_image.astype(np.float32)
    data = np.expand_dims(data, axis=0)  # (1, H, W, 3)
    interp.set_tensor(in_details[0]["index"], data)
    interp.invoke()
    out_details = interp.get_output_details()
    return interp.get_tensor(out_details[0]["index"])[0]  # (N,)


def _infer_digit(output: np.ndarray) -> Tuple[int, float, float]:
    """Digit model (11 outputs). Class 10 means the digit could not be read (N).
    Returns (class, float_result, confidence).
    """
    class_idx = int(np.argmax(output))
    confidence = float(output[class_idx])
    return class_idx, -1.0, confidence


def _infer_analogue(output: np.ndarray, ccw: bool) -> Tuple[float, float]:
    """Analogue model (2 outputs). atan2 → 0-1 → scale to 0-10.
    Returns (result_float, confidence) where confidence is the magnitude of the vector.
    """
    f1, f2 = float(output[0]), float(output[1])
    magnitude = math.sqrt(f1**2 + f2**2)  # confidence is signal strength
    angle = math.fmod(math.atan2(f1, f2) / (math.pi * 2) + 2, 1)
    if ccw:
        return 10.0 - angle * 10.0, magnitude
    return angle * 10.0, magnitude


def _infer_doublehybrid10(
    output: np.ndarray, ccw: bool, good_threshold: float
) -> Tuple[float, bool, float]:
    """
    DoubleHybrid10 model (10 outputs).
    Finds the best class then weights it by adjacent class probabilities
    to get a sub-integer float result.
    Returns (result, is_reject, confidence) where confidence is the fit value.
    """
    num = int(np.argmax(output))
    np1 = (num + 1) % 10
    nm1 = (num - 1 + 10) % 10

    v = float(output[num])
    vp = float(output[np1])
    vm = float(output[nm1])

    result = float(num)
    if vp > vm:
        result = result + vp / (vp + v)
        fit = v + vp
    else:
        result = result - vm / (v + vm)
        fit = v + vm

    if result >= 10:
        result -= 10
    if result < 0:
        result += 10

    is_reject = fit < good_threshold
    return (-1.0 if is_reject else result), is_reject, fit


def _infer_cls100(output: np.ndarray, ccw: bool) -> Tuple[float, float]:
    """Digit100 / Analogue100 model (100 outputs). value = argmax / 10.
    Returns (result_float, confidence) where confidence is the output value at argmax.
    """
    num = int(np.argmax(output))
    confidence = float(output[num])
    if ccw:
        return 10.0 - num / 10.0, confidence
    return num / 10.0, confidence


def run_cnn_for_group(
    cfg: CNNConfig,
    group: GeneralGroup,
    cnn_type: CNNType,
    interp,
) -> None:
    """Run inference for every ROI in a group and store results in the ROI."""
    for roi in group.rois:
        if roi.image is None:
            continue

        output = _run_inference(interp, roi.image)

        if cnn_type == CNNType.Digit:
            roi.result_klasse, roi.result_float, roi.confidence = _infer_digit(output)

        elif cnn_type == CNNType.Analogue:
            roi.result_float, roi.confidence = _infer_analogue(output, roi.ccw)
            roi.result_klasse = -1

        elif cnn_type == CNNType.DoubleHybrid10:
            roi.result_float, roi.is_reject, roi.confidence = _infer_doublehybrid10(
                output, roi.ccw, cfg.good_threshold
            )
            roi.result_klasse = -1

        elif cnn_type in (CNNType.Digit100, CNNType.Analogue100):
            roi.result_float, roi.confidence = _infer_cls100(output, roi.ccw)
            roi.result_klasse = -1


# ===========================================================================
# Readout assembly
# Ported from ClassFlowCNNGeneral::getReadout() and its helper functions.
# ===========================================================================

# Tuning constants (defaults from ClassFlowCNNGeneral.cpp / .h)
_DIGIT_BAND = 1                      # "band" around integer where rounding applies
_DIGIT_TRANSITION_PREDECESSOR = 3.0  # "safe" zone away from transition
_DIGIT_TRANSITION_FORWARD = 9.2      # threshold where the next digit leads


def _pointer_eval_analog_new(number: float, prev: int) -> int:
    """
    Determine the displayed integer for one analog dial, taking the
    less-significant (previous) dial value into account.
    Port of PointerEvalAnalogNew().
    """
    result_after = int(math.floor(number * 10) + 10) % 10  # fractional digit 0-9
    result_before = int(math.floor(number) + 10) % 10       # integer part 0-9

    if prev < 0:
        return result_before

    # Less-significant dial has just crossed zero → current may have ticked over
    if prev <= 1:
        if result_after > 5:
            return (result_before + 1) % 10
        return result_before

    return result_before


def _pointer_eval_hybrid_new(
    number: float,
    predecessor_value: float,
    eval_predecessor: int,
    analog_predecessor: bool = False,
    transition_start: float = 9.2,
) -> int:
    """
    Core readout logic for DoubleHybrid10 / Digit100.
    Port of PointerEvalHybridNew().

    number               – raw CNN float for the current digit (0-10)
    predecessor_value    – raw CNN float for the less-significant digit / analog
    eval_predecessor     – already-evaluated integer for the less-significant
    analog_predecessor   – True if predecessor is an analog (not a digit) ROI
    transition_start     – analog value above which transition logic triggers
    """
    result_after = int(math.floor(number * 10)) % 10
    result_before = int(math.floor(number) + 10) % 10

    if eval_predecessor < 0:
        # First (most-significant) element – just truncate with rounding guard
        return int(int(math.trunc(round((number + 10 % 10) * 100))) / 100)

    if analog_predecessor:
        # Delegate to the analog-to-digit transition helper
        return _pointer_eval_analog_to_digit_new(
            number, predecessor_value, eval_predecessor, transition_start
        )

    # Predecessor is safely away from its zero crossing → round or truncate
    if _DIGIT_TRANSITION_PREDECESSOR <= predecessor_value <= (
        10.0 - _DIGIT_TRANSITION_PREDECESSOR
    ):
        if result_after <= _DIGIT_BAND or result_after >= (10 - _DIGIT_BAND):
            return (int(round(number)) + 10) % 10
        return (int(math.trunc(number)) + 10) % 10

    # Predecessor has already crossed zero
    if eval_predecessor <= 1:
        if result_after > 5:
            return (result_before + 1) % 10
        return result_before % 10

    # Predecessor is approaching zero (9.x) but hasn't crossed yet
    if _DIGIT_TRANSITION_FORWARD >= predecessor_value or result_after >= 4:
        return result_before % 10
    return (result_before - 1 + 10) % 10


def _pointer_eval_analog_to_digit_new(
    number: float,
    analog_value: float,
    eval_predecessor: int,
    transition_start: float = 9.2,
) -> int:
    """
    Digit reading when the less-significant predecessor is an analog dial.
    Port of PointerEvalAnalogToDigitNew().
    """
    result_after = int(math.floor(number * 10)) % 10
    result_before = int(math.floor(number) + 10) % 10

    if eval_predecessor < 0:
        return int(int(math.trunc(round((number + 10 % 10) * 100))) / 100)

    if analog_value >= transition_start:
        # Analog dial approaching zero crossing
        if result_after > 5:
            return (result_before + 1) % 10
        return result_before % 10

    # Analog dial safely away from transition
    if result_after <= _DIGIT_BAND or result_after >= (10 - _DIGIT_BAND):
        return (int(round(number)) + 10) % 10
    return (int(math.trunc(number)) + 10) % 10


# ---------------------------------------------------------------------------
# Per-type readout builders
# ---------------------------------------------------------------------------

def _readout_analogue(group: GeneralGroup, extended: bool, prev: int = -1) -> str:
    """Build raw string for Analogue / Analogue100 groups."""
    rois = group.rois
    if not rois:
        return ""

    # Start from the least-significant ROI (last in list)
    prev = _pointer_eval_analog_new(rois[-1].result_float, prev)
    result = str(prev)

    if extended:
        decimal_digit = int(math.floor(rois[-1].result_float * 10) + 10) % 10
        result += str(decimal_digit)

    # Iterate from second-to-last toward most significant, prepending each
    for i in range(len(rois) - 2, -1, -1):
        prev = _pointer_eval_analog_new(rois[i].result_float, prev)
        result = str(prev) + result

    return result


def _readout_digit(group: GeneralGroup) -> str:
    """Build raw string for Digit groups."""
    result = ""
    for roi in group.rois:
        if 0 <= roi.result_klasse < 10:
            result += str(roi.result_klasse)
        else:
            result += "N"
    return result


def _readout_hybrid(
    group: GeneralGroup,
    extended: bool,
    prev: int = -1,
    before_analog: float = -1.0,
    transition_start: float = 9.2,
) -> str:
    """Build raw string for DoubleHybrid10 / Digit100 groups."""
    rois = group.rois
    if not rois:
        return ""

    result = ""
    number = rois[-1].result_float

    if 0.0 <= number < 10.0:
        if extended:
            ra = int(math.floor(number * 10)) % 10
            rb = int(math.floor(number)) % 10
            result = str(rb) + str(ra)
            prev = rb
        else:
            if before_analog >= 0:
                prev = _pointer_eval_hybrid_new(
                    rois[-1].result_float,
                    before_analog,
                    prev,
                    True,
                    transition_start,
                )
            else:
                prev = _pointer_eval_hybrid_new(
                    rois[-1].result_float, float(prev), prev
                )
            result = str(prev) if 0 <= prev < 10 else "N"
    else:
        result = "NN" if extended else "N"

    for i in range(len(rois) - 2, -1, -1):
        r = rois[i].result_float
        if 0.0 <= r < 10.0:
            prev = _pointer_eval_hybrid_new(r, rois[i + 1].result_float, prev)
            result = str(prev) + result
        else:
            prev = -1
            result = "N" + result

    return result


def _get_readout(
    cnn_cfg: CNNConfig,
    cnn_type: CNNType,
    group_idx: int,
    extended: bool = False,
    prev: int = -1,
    before_analog: float = -1.0,
    transition_start: float = 9.2,
) -> str:
    """
    Master readout for a group.
    Corresponds to ClassFlowCNNGeneral::getReadout().
    """
    if group_idx >= len(cnn_cfg.groups):
        return ""

    group = cnn_cfg.groups[group_idx]

    if cnn_type in (CNNType.Analogue, CNNType.Analogue100):
        return _readout_analogue(group, extended, prev)
    if cnn_type == CNNType.Digit:
        return _readout_digit(group)
    if cnn_type in (CNNType.DoubleHybrid10, CNNType.Digit100):
        return _readout_hybrid(group, extended, prev, before_analog, transition_start)
    return ""


# ===========================================================================
# Post-processing: decimal shift, value assembly
# Ported from ClassFlowPostProcessing::doFlow() and ShiftDecimal().
# ===========================================================================

def _shift_decimal(s: str, shift: int) -> str:
    """
    Move the decimal point in string s by `shift` positions to the right.
    Positive shift = multiply by 10^shift.
    Port of ClassFlowPostProcessing::ShiftDecimal().
    """
    if shift == 0:
        return s

    dot = s.find(".")
    if dot == -1:
        dot = len(s)
    else:
        s = s[:dot] + s[dot + 1:]

    new_dot = dot + shift

    if new_dot <= 0:
        s = "0" * (-new_dot) + s
        return "0." + s

    if new_dot >= len(s):
        s = s + "0" * (new_dot - len(s))
        return s

    return s[:new_dot] + "." + s[new_dot:]


# ---------------------------------------------------------------------------
# Digit ambiguity correction
# ---------------------------------------------------------------------------

# Digits that are commonly misread as one another by the CNN models.
COMMON_CONFUSIONS: Dict[str, List[str]] = {
    "4": ["9"],
    "9": ["4"],
    "8": ["6", "3"],
    "6": ["8"],
    "3": ["8"],
    "1": ["7"],
    "7": ["1"],
}

# Acceptable meter progression per reading cycle.
_PLAUSIBLE_MIN = -1.0    # small negative margin for rounding jitter
_MAX_FLOW_RATE = 2.5      # m³ per hour max realistic flow


def _calculate_max_plausible(prevalue_timestamp: str) -> float:
    """Calculate max plausible increase based on time elapsed and flow rate."""
    try:
        prev_time = datetime.datetime.strptime(prevalue_timestamp, "%Y-%m-%d_%H-%M-%S")
        elapsed = (datetime.datetime.now() - prev_time).total_seconds() / 3600.0
        return max(elapsed, 1.0) * _MAX_FLOW_RATE
    except (ValueError, TypeError):
        return 500.0


def _is_plausible_diff(diff: float, max_plausible: Optional[float] = None) -> bool:
    """Check if a meter reading difference is plausible."""
    if max_plausible is None:
        max_plausible = 10.0
    return _PLAUSIBLE_MIN <= diff <= max_plausible


def _pick_best_candidate(candidates: List[str], previous_value: float, max_plausible: Optional[float] = None) -> Optional[str]:
    """
    From a list of candidate strings return the best one:
    - the first candidate whose value is within the plausible range, or
    - the candidate numerically closest to the previous reading otherwise.
    """
    best: Optional[str] = None
    best_dist = float("inf")
    for c in candidates:
        try:
            val = float(c)
        except ValueError:
            continue
        diff = val - previous_value
        if _is_plausible_diff(diff, max_plausible):
            return c
        dist = abs(diff)
        if dist < best_dist:
            best = c
            best_dist = dist
    return best


def _confusion_candidates(raw_value: str) -> List[str]:
    """
    Generate candidates by substituting visually-confused digits (one at a
    time) according to COMMON_CONFUSIONS.  Only numeric positions are mutated.
    """
    candidates: List[str] = []
    for i, ch in enumerate(raw_value):
        if ch in COMMON_CONFUSIONS:
            for alt in COMMON_CONFUSIONS[ch]:
                candidates.append(raw_value[:i] + alt + raw_value[i + 1:])
    return candidates


def _all_digit_candidates(raw_value: str) -> List[str]:
    """
    Generate candidates by substituting any single digit position with 0-9.
    Brute-force fallback used when targeted confusion substitution fails.
    """
    candidates: List[str] = []
    for i, ch in enumerate(raw_value):
        if ch in "0123456789":
            for d in "0123456789":
                if d != ch:
                    candidates.append(raw_value[:i] + d + raw_value[i + 1:])
    return candidates


def _resolve_n_digits(raw_value: str, previous_value: float, max_plausible: Optional[float] = None) -> Optional[str]:
    """
    Resolve 'N' (unrecognised) digits and/or fix visually-confused digits
    using the previous meter reading as a reference.

    Strategy
    --------
    Build a per-position alternative list:
      * N        → all digits 0-9
      * confused → original digit + COMMON_CONFUSIONS alternatives
    Take the cartesian product of all mutable positions, score every resulting
    candidate and return the most plausible one.

    If no N or confused positions exist but the raw value is implausible,
    fall back to exhaustive single-digit brute-force.
    """
    import itertools

    mutable_indices: List[int] = []
    position_choices: List[List[str]] = []

    for i, ch in enumerate(raw_value):
        if ch == "N":
            mutable_indices.append(i)
            position_choices.append(list("0123456789"))
        elif ch in COMMON_CONFUSIONS:
            mutable_indices.append(i)
            position_choices.append([ch] + COMMON_CONFUSIONS[ch])

    if not mutable_indices:
        # Nothing to mutate via N or confusion map – check current plausibility.
        try:
            if _is_plausible_diff(float(raw_value) - previous_value, max_plausible):
                return raw_value
        except ValueError:
            pass
        # Last resort: single-digit brute-force.
        return _pick_best_candidate(_all_digit_candidates(raw_value), previous_value, max_plausible)

    base = list(raw_value)
    candidates: List[str] = []
    for combo in itertools.product(*position_choices):
        candidate = base[:]
        for idx, digit in zip(mutable_indices, combo):
            candidate[idx] = digit
        candidate_str = "".join(candidate)
        if "N" not in candidate_str:
            candidates.append(candidate_str)

    return _pick_best_candidate(candidates, previous_value, max_plausible)


def _apply_prevalue_correction(raw_value: str, cfg: Config, group_name: str) -> str:
    """
    Attempt to fix a plausible-but-wrong reading by comparing it against the
    previous meter value.

    Correction passes (in order):
      1. COMMON_CONFUSIONS substitutions  – targeted, single-position mutations
         of digits that the model commonly confuses.
      2. Brute-force single-digit substitution – try every digit at every
         position as a fallback.

    Accepts the first candidate that falls within the plausible progression
    range; otherwise keeps the original.
    """
    if not cfg.prevalue or cfg.prevalue.value < 0 or "N" in raw_value or "." not in raw_value:
        return raw_value

    try:
        previous = cfg.prevalue.value
        max_plausible = _calculate_max_plausible(cfg.prevalue.timestamp)
        diff = float(raw_value) - previous
        if _is_plausible_diff(diff, max_plausible):
            return raw_value  # already fine

        # Pass 1 – targeted confusion substitutions
        best = _pick_best_candidate(_confusion_candidates(raw_value), previous, max_plausible)
        if best is not None:
            try:
                if _is_plausible_diff(float(best) - previous, max_plausible):
                    return best
            except ValueError:
                pass

        # Pass 2 – brute-force single-digit substitution
        best2 = _pick_best_candidate(_all_digit_candidates(raw_value), previous, max_plausible)
        if best2 is not None:
            return best2

        return raw_value

    except (ValueError, TypeError):
        return raw_value


def _assemble_reading(cfg: Config, group_name: str) -> Dict:
    """
    Assemble the final reading for one named group (sequence).
    Mirrors the per-number loop in ClassFlowPostProcessing::doFlow().
    Also includes per-ROI confidence values.
    """
    pp = (
        cfg.postprocessing.get(group_name)
        or cfg.postprocessing.get("default")
        or PostProcessingNumber(name=group_name)
    )

    # Find this group in digit / analog configs
    digit_idx = -1
    analog_idx = -1

    if cfg.digits:
        for i, g in enumerate(cfg.digits.groups):
            if g.name == group_name:
                digit_idx = i
                break

    if cfg.analog:
        for i, g in enumerate(cfg.analog.groups):
            if g.name == group_name:
                analog_idx = i
                break

    raw_value = ""
    previous_value = -1  # integer value of the first (most-significant) analog digit
    confidences: List[float] = []  # collect confidence values for each ROI

    # Collect ROIs in order for confidence tracking
    all_rois: List[ROI] = []

    # Step 1 – Analog readout
    if analog_idx >= 0 and cfg.analog is not None:
        analog_raw = _get_readout(
            cfg.analog,
            cfg.analog.cnn_type,
            analog_idx,
            pp.extended_resolution,
        )
        raw_value = analog_raw
        if raw_value and raw_value[0].isdigit():
            previous_value = int(raw_value[0])
        if cfg.analog.groups[analog_idx].rois:
            all_rois.extend(cfg.analog.groups[analog_idx].rois)

    # Step 2 – Digit readout (prepended before analog)
    if digit_idx >= 0 and cfg.digits is not None:
        # Float value of the most-significant analog ROI (ROI[0]) for transition logic
        before_analog = -1.0
        if analog_idx >= 0 and cfg.analog:
            grp = cfg.analog.groups[analog_idx]
            if grp.rois:
                before_analog = grp.rois[0].result_float

        if analog_idx >= 0:
            # Combined digit + analog: insert decimal between them
            digit_raw = _get_readout(
                cfg.digits,
                cfg.digits.cnn_type,
                digit_idx,
                False,
                previous_value,
                before_analog,
                pp.analog_to_digit_transition_start,
            )
            raw_value = digit_raw + "." + raw_value
        else:
            # Digit only
            digit_raw = _get_readout(
                cfg.digits,
                cfg.digits.cnn_type,
                digit_idx,
                pp.extended_resolution,
                previous_value,
            )
            raw_value = digit_raw
        if cfg.digits.groups[digit_idx].rois:
            all_rois.extend(cfg.digits.groups[digit_idx].rois)

    # Collect confidences from all ROIs
    for roi in all_rois:
        if roi.confidence >= 0:
            confidences.append(roi.confidence)

    if not raw_value:
        result = {"raw": "", "value": None, "error": "No ROIs configured"}
        if confidences:
            result["confidence"] = confidences
        return result

    # Step 3 – Decimal shift
    raw_value = _shift_decimal(raw_value, pp.decimal_shift)

    # Step 4 – Strip leading NaN if configured
    if pp.ignore_leading_nan:
        while len(raw_value) > 1 and raw_value[0] == "N":
            raw_value = raw_value[1:]

    # Step 5 – Handle unresolved N
    if "N" in raw_value:
        # Attempt to fill N positions using the previous meter reading.
        if cfg.pre_value_use and cfg.prevalue and cfg.prevalue.value >= 0:
            max_plausible = _calculate_max_plausible(cfg.prevalue.timestamp)
            fixed = _resolve_n_digits(raw_value, cfg.prevalue.value, max_plausible)
            if fixed is not None and "N" not in fixed:
                raw_value = fixed  # resolved – continue with normal post-processing
            else:
                result = {"raw": raw_value, "value": None, "error": "Unresolved digit (N)"}
                if confidences:
                    result["confidence"] = confidences
                return result
        else:
            result = {"raw": raw_value, "value": None, "error": "Unresolved digit (N)"}
            if confidences:
                result["confidence"] = confidences
            return result

    # Step 6 – Strip leading zeros
    value_str = raw_value
    while len(value_str) > 1 and value_str[0] == "0" and value_str[1:2] != ".":
        value_str = value_str[1:]

    try:
        float(value_str)  # validate
    except ValueError:
        result = {"raw": raw_value, "value": None, "error": f"Cannot parse: {value_str}"}
        if confidences:
            result["confidence"] = confidences
        return result

    # Apply pre-value correction if enabled and available
    if cfg.pre_value_use and cfg.prevalue:
        corrected_value = _apply_prevalue_correction(value_str, cfg, group_name)
        if corrected_value != value_str:
            result = {
                "raw": raw_value,
                "value": corrected_value,
                "error": "no error",
                "note": f"Corrected from {value_str} based on previous reading {cfg.prevalue.value}"
            }
            if confidences:
                result["confidence"] = confidences
            return result

    result = {"raw": raw_value, "value": value_str, "error": "no error"}
    if confidences:
        result["confidence"] = confidences
    return result


# ===========================================================================
# ROI sample logging (training data collection)
# ===========================================================================

def _cleanup_old_samples(log_dir: str, retention_days: int) -> None:
    """Delete ROI sample files older than retention_days."""
    if not os.path.exists(log_dir):
        return
    
    cutoff_time = time.time() - (retention_days * 86400)
    try:
        for filename in os.listdir(log_dir):
            filepath = os.path.join(log_dir, filename)
            if os.path.isfile(filepath) and filepath.endswith('.jpg'):
                if os.path.getmtime(filepath) < cutoff_time:
                    os.remove(filepath)
    except (OSError, IOError) as e:
        print(f"WARNING: Could not clean up old samples: {e}", file=sys.stderr)


def _should_log_roi(roi_name: str, selective: Optional[List[str]]) -> bool:
    """Check if ROI should be logged based on selective filter."""
    if selective is None:
        return True
    # Extract the short name (e.g. "dig1" from "main.dig1")
    short_name = roi_name.split(".")[-1] if "." in roi_name else roi_name
    return short_name in selective or roi_name in selective


def _save_roi_sample(
    roi: ROI,
    group_name: str,
    cnn_type: CNNType,
    log_dir: str,
    selective: Optional[List[str]] = None,
) -> None:
    """Save a cropped ROI image with its inference result in the filename."""
    if roi.image is None or not _should_log_roi(roi.name, selective):
        return
    
    os.makedirs(log_dir, exist_ok=True)
    
    # Format: {group}_{roi_name}_{result}_{timestamp}.jpg
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if cnn_type == CNNType.Digit:
        result_str = str(roi.result_klasse) if roi.result_klasse >= 0 else "N"
    else:
        # Analogue / Hybrid / 100-class models
        if roi.result_float >= 0:
            result_str = f"{roi.result_float:.2f}".replace(".", "_")
        else:
            result_str = "N"
    
    filename = f"{group_name}_{roi.name.split('.')[-1]}_{result_str}_{timestamp}.jpg"
    filepath = os.path.join(log_dir, filename)
    
    try:
        Image.fromarray(roi.image).save(filepath)
    except (OSError, IOError) as e:
        print(f"WARNING: Could not save ROI sample: {e}", file=sys.stderr)


# ===========================================================================
# Main pipeline
# ===========================================================================

def _all_group_names(cfg: Config) -> List[str]:
    seen: List[str] = []
    for src in (cfg.digits, cfg.analog):
        if src:
            for g in src.groups:
                if g.name not in seen:
                    seen.append(g.name)
    return seen


def run(
    image_path: str,
    config_path: str,
    sdcard_dir: str,
    debug_dir: Optional[str] = None,
    prevalue_path: Optional[str] = None,
    log_samples_dir: Optional[str] = None,
    dry_run: bool = False,
    autotune: bool = False,
) -> Dict:
    """
    Full pipeline:
      1. Parse config.ini and load prevalue.ini if available
      2. Load and align the image
      3. Load TFLite models, detect CNN types
      4. Extract ROI sub-images and run inference
      5. Optionally save ROI samples for training
      6. Assemble and return readings

    Args:
        dry_run: If True, do not update prevalue.ini
        autotune: If True, apply grayscale + CLAHE contrast enhancement after alignment
    """

    # 1. Config (also loads prevalue if pre_value_use is enabled)
    if prevalue_path is None:
        prevalue_path = os.path.join(sdcard_dir, "config", "prevalue.ini")
    cfg = parse_config(config_path, sdcard_dir, prevalue_path)
    
    # Override logging config if --log-samples was provided
    if log_samples_dir:
        cfg.logging.enabled = True
        cfg.logging.save_location = log_samples_dir

    # 2. Image loading + alignment
    try:
        image = _load_image(image_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Could not load image from {image_path}: {e}", file=sys.stderr)
        sys.exit(1)
    aligned = align_image(image, cfg.alignment)

    if autotune:
        aligned = autotune_image(aligned)

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        aligned.save(os.path.join(debug_dir, "aligned.jpg"))

    # 3. Models
    digit_interp = None
    digit_type: Optional[CNNType] = None
    digit_w = digit_h = 0

    analog_interp = None
    analog_type: Optional[CNNType] = None
    analog_w = analog_h = 0

    if cfg.digits:
        if not os.path.exists(cfg.digits.model_file):
            print(f"ERROR: Digit model not found: {cfg.digits.model_file}", file=sys.stderr)
        else:
            digit_interp = _make_interpreter(cfg.digits.model_file)
            digit_interp.allocate_tensors()
            digit_type, digit_w, digit_h, _ = _detect_cnn_type(digit_interp)
            cfg.digits.cnn_type = digit_type

    if cfg.analog:
        if not os.path.exists(cfg.analog.model_file):
            print(f"ERROR: Analog model not found: {cfg.analog.model_file}", file=sys.stderr)
        else:
            analog_interp = _make_interpreter(cfg.analog.model_file)
            analog_interp.allocate_tensors()
            analog_type, analog_w, analog_h, _ = _detect_cnn_type(analog_interp)
            cfg.analog.cnn_type = analog_type

    # 4. Extract ROIs and run inference
    if cfg.digits and digit_interp and digit_type is not None:
        for grp in cfg.digits.groups:
            extract_roi_images(aligned, grp, digit_w, digit_h)
            if debug_dir:
                for roi in grp.rois:
                    if roi.image is not None:
                        Image.fromarray(roi.image).save(
                            os.path.join(debug_dir, f"digit_{roi.name}.jpg")
                        )
            run_cnn_for_group(cfg.digits, grp, digit_type, digit_interp)
            
            # Log ROI samples if enabled
            if cfg.logging.enabled and cfg.logging.save_location:
                for roi in grp.rois:
                    _save_roi_sample(roi, grp.name, digit_type, cfg.logging.save_location, cfg.logging.selective)

    if cfg.analog and analog_interp and analog_type is not None:
        for grp in cfg.analog.groups:
            extract_roi_images(aligned, grp, analog_w, analog_h)
            if debug_dir:
                for roi in grp.rois:
                    if roi.image is not None:
                        Image.fromarray(roi.image).save(
                            os.path.join(debug_dir, f"analog_{roi.name}.jpg")
                        )
            run_cnn_for_group(cfg.analog, grp, analog_type, analog_interp)
            
            # Log ROI samples if enabled
            if cfg.logging.enabled and cfg.logging.save_location:
                for roi in grp.rois:
                    _save_roi_sample(roi, grp.name, analog_type, cfg.logging.save_location, cfg.logging.selective)

    # Cleanup old samples if logging is enabled
    if cfg.logging.enabled and cfg.logging.save_location:
        _cleanup_old_samples(cfg.logging.save_location, cfg.logging.retention_days)

    # 5. Assemble readings
    results: Dict = {}
    for name in _all_group_names(cfg):
        results[name] = _assemble_reading(cfg, name)

    # 6. Update prevalue.ini with successful reading(s) for next time (unless dry_run)
    if not dry_run and prevalue_path and cfg.pre_value_use:
        for name, reading in results.items():
            if isinstance(reading, dict) and reading.get("error") == "no error":
                value = reading.get("value")
                if value is not None:
                    try:
                        float_value = float(value)
                        if save_prevalue(prevalue_path, float_value):
                            print(f"Updated prevalue to {float_value}", file=sys.stderr)
                        break  # Save the first successful reading group
                    except ValueError:
                        pass

    return results


# ===========================================================================
# CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "AI-on-the-edge meter reader – x86 / Docker port.\n"
            "Reads a meter image using the same TFLite models and config.ini "
            "as the original ESP32 firmware."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--image", required=True, metavar="PATH_OR_URL",
        help="Input meter image: local file path (JPEG, PNG, …) or URL (http://, https://)"
    )
    parser.add_argument(
        "--sdcard", default="./sdcard", metavar="DIR",
        help=(
            "Root of the 'sdcard' directory tree that contains config/, "
            "the .tflite models, and the reference images. "
            "Default: ./sdcard"
        ),
    )
    parser.add_argument(
        "--config", default=None, metavar="PATH",
        help=(
            "Path to config.ini. "
            "Default: <sdcard>/config/config.ini"
        ),
    )
    parser.add_argument(
        "--debug-dir", default=None, metavar="DIR",
        help="Save intermediate images (aligned frame, cropped ROIs) here."
    )
    parser.add_argument(
        "--prevalue", default=None, metavar="PATH",
        help=(
            "Path to prevalue.ini (previous meter reading). "
            "Default: <sdcard>/config/prevalue.ini"
        ),
    )
    parser.add_argument(
        "--pretty", action="store_true",
        help="Pretty-print JSON output."
    )
    parser.add_argument(
        "--log-samples", default=None, metavar="DIR",
        help="Enable ROI sample logging for model retraining. Saves cropped ROI images here."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run inference but do not update prevalue.ini."
    )
    parser.add_argument(
        "--autotune", action="store_true",
        help="Apply grayscale conversion and CLAHE contrast enhancement after alignment."
    )
    args = parser.parse_args()

    config_path = args.config or os.path.join(args.sdcard, "config", "config.ini")

    # Validate image source (URL or file path)
    if not _is_url(args.image) and not os.path.exists(args.image):
        print(f"ERROR: Image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(config_path):
        print(f"ERROR: config.ini not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    results = run(
        image_path=args.image,
        config_path=config_path,
        sdcard_dir=args.sdcard,
        debug_dir=args.debug_dir,
        prevalue_path=args.prevalue,
        log_samples_dir=args.log_samples,
        dry_run=args.dry_run,
        autotune=args.autotune,
    )

    print(json.dumps(results, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()
