#!/bin/bash
# Water Meter Cron Job Script
# Activates venv and runs cron_ha.py, logging all output

cd "$(dirname "$0")" || exit 1
source .venv/bin/activate
python cron_ha.py >> cron.log 2>&1
