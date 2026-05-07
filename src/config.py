"""Constantes y paths del proyecto."""

import os
from pathlib import Path

APP_NAME = "transcriptor-flow"
VERSION  = "6.0.0"

SAMPLE_RATE       = 16000
BLOCK_SIZE        = int(SAMPLE_RATE * 0.1)
REALTIME_INTERVAL = 0.3
MIN_AUDIO_BLOCKS  = 3
MIN_RECORDING_S   = 0.4  # Duración mínima para considerar una grabación válida

WHISPER_LANGUAGE         = "es"
WHISPER_BEAM_SIZE        = 5
WHISPER_CPU_THREADS      = 4

# VAD: más permisivo para español, evita cortar frases
WHISPER_VAD_THRESHOLD      = 0.2
WHISPER_NO_SPEECH_THRESH   = 0.3
WHISPER_VAD_MIN_SILENCE_MS = 100

# Realtime (tiny): más rápido, más agresivo
RT_VAD_THRESHOLD      = 0.4
RT_NO_SPEECH_THRESH   = 0.5
RT_VAD_MIN_SILENCE_MS = 200

BASE_DIR    = Path(os.environ.get("TRANSCRIPTOR_FLOW_HOME", Path.home() / ".local" / "share" / APP_NAME))
LOG_DIR     = BASE_DIR / "logs"
LOG_FILE    = LOG_DIR / "transcriptor.log"
LOG_MAX_BYTES = 1_048_576  # 1 MB
LOG_BACKUPS   = 3
