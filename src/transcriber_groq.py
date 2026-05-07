"""Transcripción via Groq API — whisper-large-v3-turbo (~800ms, cloud).

Inspirado en la arquitectura de sflow (daniel-carreon/sflow).
"""

import io
import wave
import logging
import os

import numpy as np
from groq import Groq

from .config import SAMPLE_RATE, WHISPER_LANGUAGE

logger = logging.getLogger(__name__)

GROQ_MODEL = "whisper-large-v3-turbo"


def _get_client() -> Groq | None:
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return None
    return Groq(api_key=api_key)


def transcribe(audio: np.ndarray, client: Groq | None = None) -> str:
    """Envía audio a Groq API y retorna texto transcrito.

    Args:
        audio: numpy array float32, mono, 16kHz.
        client: instancia de Groq (opcional, se crea si es None).

    Returns:
        Texto transcrito o "" si falla o no hay voz.
    """
    if client is None:
        client = _get_client()

    if client is None:
        logger.warning("Groq API key no configurada. Usá GROQ_API_KEY en .env")
        return ""

    if audio is None or len(audio) == 0:
        return ""

    # Filtrar silencio: no enviar audio sin voz a la API
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    if rms < 0.005:
        logger.debug("Audio descartado (silencio, RMS=%.5f)", rms)
        return ""

    try:
        # Convertir numpy → WAV bytes en memoria (sin disco)
        pcm = (audio * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())
        buf.seek(0)
        buf.name = "audio.wav"

        response = client.audio.transcriptions.create(
            model=GROQ_MODEL,
            file=buf,
            language=WHISPER_LANGUAGE,
            response_format="json",
            temperature=0,
        )
        return response.text.strip()
    except Exception:
        logger.exception("Error en transcripción Groq")
        return ""


def is_available() -> bool:
    """Verifica si Groq está configurado y accesible."""
    return bool(os.environ.get("GROQ_API_KEY", ""))
