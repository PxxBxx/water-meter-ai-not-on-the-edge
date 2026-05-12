#!/usr/bin/env python3
"""
Home Assistant MQTT Publisher for Water Meter Reading
======================================================

Publishes water meter readings from meter_reader.py to Home Assistant via MQTT.

1. Configures a new water meter entity in Home Assistant (MQTT Discovery)
2. Downloads the latest meter image from a web server
3. Runs meter_reader.py to extract the meter reading
4. Publishes the reading to Home Assistant

Requirements:
    pip install paho-mqtt requests

Configuration:
    Edit the MQTT_BROKER, HA_IP, IMAGE_URL, SDCARD_DIR, and METER_READER_SCRIPT
    variables below, or pass them as environment variables.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from typing import Optional

import paho.mqtt.client as mqtt

# ===========================================================================
# Configuration
# ===========================================================================

# MQTT Broker
MQTT_BROKER = os.getenv("MQTT_BROKER", "192.168.1.10")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_USERNAME = os.getenv("MQTT_USERNAME", "")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "")

# Home Assistant MQTT Discovery
HA_DISCOVERY_PREFIX = os.getenv("HA_DISCOVERY_PREFIX", "homeassistant")
DEVICE_ID = "water_meter"
ENTITY_UNIQUE_ID = "water_meter_total"

# Topics
STATE_TOPIC = f"{HA_DISCOVERY_PREFIX}/sensor/{DEVICE_ID}/{ENTITY_UNIQUE_ID}/state"
CONFIG_TOPIC = f"{HA_DISCOVERY_PREFIX}/sensor/{DEVICE_ID}/{ENTITY_UNIQUE_ID}/config"
AVAILABILITY_TOPIC = f"{HA_DISCOVERY_PREFIX}/sensor/{DEVICE_ID}/{ENTITY_UNIQUE_ID}/availability"

# Image source
IMAGE_URL = os.getenv(
    "IMAGE_URL",
    "http://web.lan/watermeter_images/latest.jpg"
)

# Paths
SDCARD_DIR = os.getenv("SDCARD_DIR", "./sdcard")
METER_READER_SCRIPT = os.getenv(
    "METER_READER_SCRIPT",
    os.path.join(os.path.dirname(__file__), "meter_reader.py")
)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ===========================================================================
# Home Assistant Configuration Payload
# ===========================================================================

def get_config_payload() -> dict:
    """Build the Home Assistant MQTT Discovery config payload."""
    return {
        "name": "Water Meter",
        "unique_id": ENTITY_UNIQUE_ID,
        "state_topic": STATE_TOPIC,
        "availability_topic": AVAILABILITY_TOPIC,
        "unit_of_measurement": "m³",
        "device_class": "water",
        "state_class": "total_increasing",
        "value_template": "{{ value_json.value }}",
        "json_attributes_topic": STATE_TOPIC,
        "json_attributes_template": "{{ value_json | tojson }}",
        "device": {
            "identifiers": [DEVICE_ID],
            "name": "Water Meter",
            "manufacturer": "Custom",
            "model": "DIY Analog Water Meter Reading"
        }
    }


# ===========================================================================
# MQTT Functions
# ===========================================================================

def on_connect(client, connect_flags, auth_data, rc, properties=None):
    """MQTT connection callback."""
    if rc == 0:
        logger.info("Connected to MQTT broker")
    else:
        logger.error(f"Failed to connect to MQTT broker: rc={rc}")


def on_disconnect(client, disconnect_flags, auth_data, rc, properties=None):
    """MQTT disconnection callback."""
    if rc != 0:
        logger.warning(f"Unexpected disconnection: rc={rc}")


def publish_mqtt(client: mqtt.Client, topic: str, payload: dict, retain: bool = True) -> bool:
    """Publish a JSON payload to MQTT."""
    try:
        msg_info = client.publish(
            topic,
            payload=json.dumps(payload),
            qos=1,
            retain=retain
        )
        msg_info.wait_for_publish()
        logger.info(f"Published to {topic}")
        return True
    except Exception as e:
        logger.error(f"Failed to publish to {topic}: {e}")
        return False


def setup_mqtt_client() -> mqtt.Client:
    """Create and connect MQTT client."""
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect

    if MQTT_USERNAME and MQTT_PASSWORD:
        client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        client.loop_start()
        return client
    except Exception as e:
        logger.error(f"Failed to connect to MQTT broker: {e}")
        raise


# ===========================================================================
# Meter Reader
# ===========================================================================

def run_meter_reader(image_url: str) -> Optional[dict]:
    """
    Run meter_reader.py and return the JSON output.
    
    Accepts both local file paths and URLs (http://, https://, etc.).
    
    Returns a dict like:
        {
            "main": {
                "raw": "056.4321",
                "value": "56.4321",
                "error": "no error",
                "confidence": [0.95, 0.87, 0.92]
            }
        }
    """
    cmd = [
        sys.executable,
        METER_READER_SCRIPT,
        "--image", image_url,
        "--sdcard", SDCARD_DIR,
    ]

    try:
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            logger.error(f"meter_reader.py failed: {result.stderr}")
            return None

        output = json.loads(result.stdout)
        logger.info(f"Meter reading: {output}")
        return output

    except json.JSONDecodeError:
        logger.error("Failed to parse meter_reader.py output as JSON")
        return None
    except subprocess.TimeoutExpired:
        logger.error("meter_reader.py timed out")
        return None
    except Exception as e:
        logger.error(f"Failed to run meter_reader.py: {e}")
        return None


# ===========================================================================
# Main Pipeline
# ===========================================================================

def main() -> int:
    """Main pipeline: read meter from image URL and publish to MQTT."""
    parser = argparse.ArgumentParser(
        description="Publish water meter readings to Home Assistant via MQTT"
    )
    parser.add_argument(
        "--image-url", default=IMAGE_URL,
        help="URL or path of the meter image (default: from env or config)"
    )
    parser.add_argument(
        "--mqtt-broker", default=MQTT_BROKER,
        help="MQTT broker address (default: from env or 192.168.1.10)"
    )
    parser.add_argument(
        "--mqtt-port", type=int, default=MQTT_PORT,
        help="MQTT broker port (default: 1883)"
    )
    parser.add_argument(
        "--sdcard", default=SDCARD_DIR,
        help="Path to sdcard directory (default: ./sdcard)"
    )
    parser.add_argument(
        "--meter-reader", default=METER_READER_SCRIPT,
        help="Path to meter_reader.py script"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Validate prerequisites
    if not os.path.exists(args.meter_reader):
        logger.error(f"meter_reader.py not found at {args.meter_reader}")
        return 1

    # 1. Run meter reader on the image URL (downloads if needed)
    reading = run_meter_reader(args.image_url)
    if reading is None:
        return 1

    # 2. Connect to MQTT and publish
    try:
        client = setup_mqtt_client()

        # Publish Home Assistant discovery config
        config_payload = get_config_payload()
        publish_mqtt(client, CONFIG_TOPIC, config_payload, retain=True)

        # Publish the meter reading(s)
        for group_name, group_data in reading.items():
            if isinstance(group_data, dict):
                # Publish per-group (main, secondary, etc.)
                state_payload = {
                    "group": group_name,
                    **group_data
                }
                publish_mqtt(client, STATE_TOPIC, state_payload, retain=True)

                # Log success
                if group_data.get("error") == "no error":
                    value = group_data.get("value")
                    confidence = group_data.get("confidence")
                    logger.info(
                        f"Published {group_name}: {value} m³ "
                        f"(confidence: {confidence})"
                    )
                else:
                    logger.warning(
                        f"Error reading {group_name}: {group_data.get('error')}"
                    )

        # Publish availability
        publish_mqtt(client, AVAILABILITY_TOPIC, {"state": "online"})

        client.loop_stop()
        client.disconnect()
        return 0

    except Exception as e:
        logger.error(f"MQTT error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
