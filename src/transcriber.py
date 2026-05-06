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
    """Carga y gestiona los modelos tiny y small de faster-whisper."""

    def __init__(self) -> None:
        logger.info("Cargando modelo tiny (preview)…")
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
        """Transcribe con modelo tiny (rápido, preview en vivo)."""
        return self._transcribe(audio, self._tiny, vad_threshold=RT_VAD_THRESHOLD,
                                no_speech_threshold=RT_NO_SPEECH_THRESH,
                                min_silence_ms=RT_VAD_MIN_SILENCE_MS)

    def transcribe_final(self, audio: np.ndarray) -> str:
        """Transcribe con modelo small (preciso, resultado final)."""
        return self._transcribe(audio, self._small, vad_threshold=WHISPER_VAD_THRESHOLD,
                                no_speech_threshold=WHISPER_NO_SPEECH_THRESH,
                                min_silence_ms=WHISPER_VAD_MIN_SILENCE_MS,
                                condition_on_previous=True)

    def _transcribe(self, audio: np.ndarray, model: WhisperModel,
                    vad_threshold: float = 0.5, no_speech_threshold: float = 0.6,
                    min_silence_ms: int = 100, condition_on_previous: bool = False) -> str:
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

            segments, _info = model.transcribe(
                tmp,
                language=WHISPER_LANGUAGE,
                beam_size=WHISPER_BEAM_SIZE,
                temperature=0,
                vad_filter=True,
                vad_parameters={
                    "threshold": vad_threshold,
                    "min_silence_duration_ms": min_silence_ms,
                },
                no_speech_threshold=no_speech_threshold,
                condition_on_previous_text=condition_on_previous,
                repetition_penalty=1.0,
                prompt_reset_on_temperature=True,
            )
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
