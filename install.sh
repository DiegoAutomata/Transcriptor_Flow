#!/bin/bash
set -e

echo "==> Instalando dependencias del sistema..."
sudo apt install -y xdotool

echo "==> Instalando dependencias Python..."
pip3 install faster-whisper sounddevice numpy pynput

echo ""
echo "✓ Instalación completa. Ejecutá: bash run.sh"
