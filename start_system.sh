#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BITNET_MODEL_DIR="bitnet_cpp/model"
BITNET_MODEL_PATH="$BITNET_MODEL_DIR/ggml-model-i2_s.gguf"
BITNET_MODEL_URL="https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf"

if [ ! -f "$BITNET_MODEL_PATH" ]; then
    echo "BitNet model not found, downloading..."
    mkdir -p "$BITNET_MODEL_DIR"
    wget -O "$BITNET_MODEL_PATH" "$BITNET_MODEL_URL"
else
    echo "BitNet model already present at $BITNET_MODEL_PATH"
fi

echo "YOLO model weights will be automatically downloaded by the yolo_service container on first use."

if command -v docker-compose >/dev/null 2>&1; then
    DC_CMD="docker-compose"
else
    DC_CMD="docker compose"
fi

$DC_CMD up --build -d