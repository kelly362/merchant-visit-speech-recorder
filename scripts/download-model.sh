#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="$ROOT/models"
MODEL_NAME="sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
ARCHIVE="$MODEL_NAME.tar.bz2"
URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$ARCHIVE"

mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

if [ -d "$MODEL_NAME" ]; then
  echo "Model already exists: $MODEL_DIR/$MODEL_NAME"
  exit 0
fi

echo "Downloading $MODEL_NAME ..."
curl -L -o "$ARCHIVE" "$URL"
tar xjf "$ARCHIVE"
echo "Ready: $MODEL_DIR/$MODEL_NAME"
