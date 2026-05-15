#!/usr/bin/env python3
"""
Test script for meter_reader.py validation against sample images.

Scans debug/samples/ folder for JPG files named VALUE.jpg or VALUE_X.jpg
(e.g., 1234.56.jpg or 1234.56_2.jpg for alternative pictures),
runs meter_reader.py with --autotune and --dry-run, and compares the output
value with the expected value from the filename.

Usage:
    python test_samples.py [--tolerance 0.01]

Options:
    --tolerance FLOAT  : Allowed difference between expected and actual value (default: 0.01)
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run_meter_reader(image_path: str) -> dict:
    """
    Run meter_reader.py and return the parsed JSON output.
    
    Returns:
        dict: The parsed JSON output with meter readings
    """
    cmd = [
        sys.executable, "meter_reader.py",
        "--autotune",
        "--image", image_path,
        "--sdcard", "./sdcard",
        "--dry-run"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"meter_reader.py failed: {result.stderr}")
    
    return json.loads(result.stdout)


def extract_value_from_output(output: dict) -> float:
    """
    Extract the meter reading value from meter_reader.py output.
    
    Looks for the first successful reading (error == "no error").
    
    Returns:
        float: The meter reading value
    """
    for group_name, reading in output.items():
        if isinstance(reading, dict) and reading.get("error") == "no error":
            value = reading.get("value")
            if value is not None:
                return float(value)
    
    raise ValueError("No successful reading found in output")


def test_sample(image_path: Path, tolerance: float) -> tuple:
    """
    Test a single sample image.
    
    Args:
        image_path: Path to the JPG file
        tolerance: Allowed difference between expected and actual value
    
    Returns:
        (passed: bool, expected: float, actual: float, error_msg: str)
    """
    # Extract expected value from filename
    # Supports patterns like "1234.56.jpg" or "1234.56_2.jpg" (alternative pictures)
    # In both cases, expected value is 1234.56
    filename = image_path.stem  # Remove .jpg extension
    
    # Strip optional _X suffix (e.g., "_2", "_3", etc.)
    if "_" in filename:
        filename = filename.split("_")[0]
    
    try:
        expected = float(filename)
    except ValueError:
        return False, None, None, f"Invalid filename format (not a float): {filename}"
    
    try:
        output = run_meter_reader(str(image_path))
        actual = extract_value_from_output(output)
    except Exception as e:
        return False, expected, None, str(e)
    
    diff = abs(actual - expected)
    passed = diff <= tolerance
    
    return passed, expected, actual, f"Diff: {diff:.6f}"


def main():
    parser = argparse.ArgumentParser(
        description="Test meter_reader.py against sample images"
    )
    parser.add_argument(
        "--tolerance", type=float, default=0.01,
        help="Allowed difference between expected and actual value (default: 0.01)"
    )
    args = parser.parse_args()
    
    samples_dir = Path("debug/samples")
    
    if not samples_dir.exists():
        print(f"ERROR: {samples_dir} directory not found", file=sys.stderr)
        sys.exit(1)
    
    # Find all JPG files
    jpg_files = sorted(samples_dir.glob("*.jpg"))
    
    if not jpg_files:
        print(f"No JPG files found in {samples_dir}")
        sys.exit(0)
    
    passed_count = 0
    failed_count = 0
    errors_count = 0
    
    print(f"\nTesting {len(jpg_files)} sample(s) with tolerance={args.tolerance}...\n")
    print(f"{'File':<20} {'Expected':<12} {'Actual':<12} {'Status':<8} {'Details'}")
    print("-" * 80)
    
    for image_path in jpg_files:
        passed, expected, actual, msg = test_sample(image_path, args.tolerance)
        
        filename = image_path.name
        expected_str = f"{expected:.2f}" if expected is not None else "N/A"
        actual_str = f"{actual:.2f}" if actual is not None else "N/A"
        
        if passed:
            status = "✓ PASS"
            passed_count += 1
        elif expected is None:
            status = "✗ ERROR"
            errors_count += 1
        else:
            status = "✗ FAIL"
            failed_count += 1
        
        print(f"{filename:<20} {expected_str:<12} {actual_str:<12} {status:<8} {msg}")
    
    print("-" * 80)
    print(f"\nResults: {passed_count} passed, {failed_count} failed, {errors_count} errors")
    
    if failed_count > 0 or errors_count > 0:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
