#!/usr/bin/env bash
set -e

if ! command -v python >/dev/null 2>&1; then
  echo "Python not found. Attempting to install..."
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y python3 python3-pip
    sudo ln -sf /usr/bin/python3 /usr/bin/python
  elif command -v brew >/dev/null 2>&1; then
    brew install python
  else
    echo "No supported package manager found. Please install Python manually."
    exit 1
  fi
fi

if ! command -v pip >/dev/null 2>&1; then
  echo "pip not found. Attempting to install..."
  python -m ensurepip --upgrade
fi

python -m pip install --upgrade pip
pip install -r requirements-dev.txt

pytest