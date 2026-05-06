#!/bin/bash
# Ejecución manual para desarrollo/debug
cd "$(dirname "$0")"
source .venv/bin/activate 2>/dev/null || true
exec python -m src.daemon "$@"
