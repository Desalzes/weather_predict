#!/usr/bin/env bash
# vps_setup.sh — Run on the VPS to set up the Weather scanner.
# Usage: ssh root@95.216.159.10 'bash -s' < deploy/vps_setup.sh
set -euo pipefail

INSTALL_DIR="/opt/weather"
VENV_DIR="$INSTALL_DIR/.venv"
SERVICE_NAME="weather-scanner"
LOG_DIR="$INSTALL_DIR/logs"

echo "=== Weather Scanner VPS Setup ==="

# 1. Create directory structure
echo "[1/6] Creating directories..."
mkdir -p "$INSTALL_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$INSTALL_DIR/data/calibration_models"
mkdir -p "$INSTALL_DIR/data/forecast_archive"
mkdir -p "$INSTALL_DIR/data/station_actuals"
mkdir -p "$INSTALL_DIR/data/paper_trades"
mkdir -p "$INSTALL_DIR/data/hrrr_cache"

# 2. Install Python 3.12 if not available
echo "[2/6] Checking Python..."
if command -v python3.12 &>/dev/null; then
    PYTHON=python3.12
elif command -v python3 &>/dev/null; then
    PYTHON=python3
    echo "  Warning: using $(python3 --version). Python 3.12+ recommended."
else
    echo "  Installing Python 3.12..."
    apt-get update -qq && apt-get install -y -qq python3.12 python3.12-venv python3.12-dev
    PYTHON=python3.12
fi

# 3. Create venv
echo "[3/6] Creating virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
    $PYTHON -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/pip" install --upgrade pip -q

# 4. Install dependencies
echo "[4/6] Installing dependencies..."
"$VENV_DIR/bin/pip" install -r "$INSTALL_DIR/requirements.txt" -q

# 5. Create config from example if not exists
echo "[5/6] Checking config..."
if [ ! -f "$INSTALL_DIR/config.json" ]; then
    cp "$INSTALL_DIR/config.example.json" "$INSTALL_DIR/config.json"
    echo "  Created config.json from example. Edit it to add API keys:"
    echo "    nano $INSTALL_DIR/config.json"
    echo "  Required: kalshi_api_key_id, kalshi_private_key_path, ncei_api_token"
fi

# 6. Install systemd units
echo "[6/6] Installing systemd units..."
cp "$INSTALL_DIR/deploy/weather-scanner.service" /etc/systemd/system/
cp "$INSTALL_DIR/deploy/weather-scanner.timer" /etc/systemd/system/
cp "$INSTALL_DIR/deploy/weather-settle.service" /etc/systemd/system/
cp "$INSTALL_DIR/deploy/weather-settle.timer" /etc/systemd/system/
cp "$INSTALL_DIR/deploy/weather-autopilot.service" /etc/systemd/system/
cp "$INSTALL_DIR/deploy/weather-autopilot.timer" /etc/systemd/system/
systemctl daemon-reload
systemctl enable weather-scanner.timer
systemctl enable weather-settle.timer
systemctl enable weather-autopilot.timer

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit config:    nano $INSTALL_DIR/config.json"
echo "  2. Add API key:    nano $INSTALL_DIR/api-credentials.txt"
echo "  3. Start timers:   systemctl start weather-scanner.timer weather-settle.timer weather-autopilot.timer"
echo "  4. Check status:   systemctl list-timers weather-*"
echo "  5. Manual scan:    cd $INSTALL_DIR && .venv/bin/python main.py --once"
echo "  6. Manual retrain: cd $INSTALL_DIR && .venv/bin/python scripts/autopilot_weekly.py"
echo "  7. View logs:      journalctl -u weather-scanner -f"
echo ""
