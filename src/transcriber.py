"""Transcripción con faster-whisper (tiny para realtime, small para final)."""

import os
import wave
import tempfile
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
    """Carga y gestiona modelos faster-whisper: base (preview), small (final)."""

    def __init__(self) -> None:
        logger.info("Cargando modelo base (preview)…")
        self._base = WhisperModel(
            "base", device="cpu", compute_type="int8",
            cpu_threads=WHISPER_CPU_THREADS,
        )
        logger.info("Cargando modelo small (final)…")
        self._small = WhisperModel(
            "small", device="cpu", compute_type="int8",
            cpu_threads=WHISPER_CPU_THREADS,
        )
        logger.info("Modelos cargados.")

    def transcribe_realtime(self, audio: np.ndarray) -> str:
        """Transcribe con modelo base + beam_size=1 (rápido, ~0.4s para preview)."""
        return self._transcribe(audio, self._base, beam_size=1, use_vad=False,
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
        tmp = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp = f.name
            pcm = (audio * 32767).astype(np.int16)
            with wave.open(tmp, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(pcm.tobytes())

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

            segments, _info = model.transcribe(tmp, **transcribe_kwargs)
            return " ".join(s.text.strip() for s in segments).strip()
        except Exception:
            logger.exception("Error en transcripción")
            return ""
        finally:
            if tmp:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
