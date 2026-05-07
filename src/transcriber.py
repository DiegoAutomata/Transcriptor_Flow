"""Transcripción con faster-whisper (tiny para realtime, small para final)."""

import logging

import numpy as np
from faster_whisper import WhisperModel

from .config import (
    SAMPLE_RATE,
    WHISPER_LANGUAGE,
    WHISPER_BEAM_SIZE,
    WHISPER_VAD_THRESHOLD,
    WHISPER_NO_SPEECH_THRESH,
    WHISPER_VAD_MIN_SILENCE_MS,
    WHISPER_CPU_THREADS,
    RT_VAD_THRESHOLD,
    RT_NO_SPEECH_THRESH,
    RT_VAD_MIN_SILENCE_MS,
)

logger = logging.getLogger(__name__)


class Transcriber:
    """Carga y gestiona modelos faster-whisper: tiny (preview), base (preview), small (final)."""

    def __init__(self) -> None:
        logger.info("Cargando modelo tiny (preview rápido)…")
        self._tiny = WhisperModel(
            "tiny", device="cpu", compute_type="int8",
            cpu_threads=WHISPER_CPU_THREADS,
        )
        logger.info("Cargando modelo small (final)…")
        self._small = WhisperModel(
            "small", device="cpu", compute_type="int8",
            cpu_threads=WHISPER_CPU_THREADS,
        )
        logger.info("Modelos cargados.")

    def transcribe_realtime(self, audio: np.ndarray) -> str:
        """Transcribe con modelo tiny + beam_size=1 + sin VAD (máxima velocidad)."""
        return self._transcribe(audio, self._tiny, beam_size=1, use_vad=False,
                                no_speech_threshold=0.99)

    def transcribe_final(self, audio: np.ndarray) -> str:
        """Transcribe con modelo small + beam_size=5 (preciso, resultado final)."""
        return self._transcribe(audio, self._small,
                                vad_threshold=WHISPER_VAD_THRESHOLD,
                                no_speech_threshold=WHISPER_NO_SPEECH_THRESH,
                                min_silence_ms=WHISPER_VAD_MIN_SILENCE_MS,
                                condition_on_previous=True)

    def _transcribe(self, audio: np.ndarray, model: WhisperModel,
                    beam_size: int = WHISPER_BEAM_SIZE,
                    vad_threshold: float = 0.5, no_speech_threshold: float = 0.6,
                    min_silence_ms: int = 100, condition_on_previous: bool = False,
                    use_vad: bool = True) -> str:
        try:
            # faster-whisper acepta numpy array directamente — sin WAV intermedio
            audio_f32 = audio.astype(np.float32)

            transcribe_kwargs = {
                "language": WHISPER_LANGUAGE,
                "beam_size": beam_size,
                "temperature": 0,
                "condition_on_previous_text": condition_on_previous,
                "repetition_penalty": 1.0,
                "prompt_reset_on_temperature": True,
                "no_speech_threshold": no_speech_threshold,
            }
            if use_vad:
                transcribe_kwargs["vad_filter"] = True
                transcribe_kwargs["vad_parameters"] = {
                    "threshold": vad_threshold,
                    "min_silence_duration_ms": min_silence_ms,
                }

            segments, _info = model.transcribe(audio_f32, **transcribe_kwargs)
            return " ".join(s.text.strip() for s in segments).strip()
        except Exception:
            logger.exception("Error en transcripción")
            return ""
