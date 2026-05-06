"""Constantes y paths del proyecto."""

import os
from pathlib import Path

APP_NAME = "transcriptor-flow"
VERSION  = "6.0.0"

SAMPLE_RATE       = 16000
BLOCK_SIZE        = int(SAMPLE_RATE * 0.1)
REALTIME_INTERVAL = 0.6
MIN_AUDIO_BLOCKS  = 6

WHISPER_LANGUAGE         = "es"
WHISPER_BEAM_SIZE        = 3
WHISPER_VAD_THRESHOLD    = 0.5
WHISPER_NO_SPEECH_THRESH = 0.6
WHISPER_CPU_THREADS      = 4

BASE_DIR    = Path(os.environ.get("TRANSCRIPTOR_FLOW_HOME", Path.home() / ".local" / "share" / APP_NAME))
LOG_DIR     = BASE_DIR / "logs"
LOG_FILE    = LOG_DIR / "transcriptor.log"
LOG_MAX_BYTES = 1_048_576  # 1 MB
LOG_BACKUPS   = 3
