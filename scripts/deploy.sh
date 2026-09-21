#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$HOME/hamburg_ai_assistant"
SERVICE_NAME="hamburg-ai-bot.service"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"

cd "$PROJECT_DIR"

echo "Installing dependencies"
"$VENV_PYTHON" -m pip install -r requirements.txt

echo "Restarting $SERVICE_NAME"
sudo systemctl restart "$SERVICE_NAME"

sleep 3

echo "Checking service status"
systemctl status "$SERVICE_NAME" --no-pager --full

echo "Deployment completed"
