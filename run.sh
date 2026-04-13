#!/bin/bash
cd "$(dirname "$0")"
export DISPLAY="${DISPLAY:-:0}"
.venv/bin/python transcriptor_flow.py
