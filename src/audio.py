"""Captura de audio desde el micrófono — PortAudio + fallback PulseAudio."""

import threading
import subprocess
import logging
import numpy as np
import sounddevice as sd

from .config import SAMPLE_RATE, BLOCK_SIZE

logger = logging.getLogger(__name__)


class AudioCapture:
    """Stream de micrófono con callback por bloque y acumulación."""

    def __init__(self):
        self._stream: sd.InputStream | None = None
        self._pulse_proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._buffer: list[np.ndarray] = []
        self._recording = False
        self._use_portaudio = True

    def start(self, prefer_pulse: bool = False) -> None:
        with self._lock:
            if self._recording:
                return
            self._recording = True
            self._buffer.clear()

        if prefer_pulse:
            logger.info("Modo WSL: usando PulseAudio (parec) directamente.")
            self._recording = False
            self._stream = None
            self._start_pulseaudio()
            return

        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype=np.float32,
                blocksize=BLOCK_SIZE,
                callback=self._callback,
            )
            self._stream.start()
            self._use_portaudio = True
            logger.info("Audio iniciado vía PortAudio.")
        except Exception:
            logger.warning("PortAudio falló, intentando PulseAudio (parec)...")
            self._recording = False
            self._stream = None
            self._start_pulseaudio()

    def _start_pulseaudio(self) -> None:
        """Usa parec para grabar desde el source por defecto."""
        try:
            self._pulse_proc = subprocess.Popen(
                [
                    "parec",
                    "--format=s16le",
                    f"--rate={SAMPLE_RATE}",
                    "--channels=1",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            self._use_portaudio = False
            self._recording = True

            # Hilo lector: convierte s16le a float32 y acumula
            self._pulse_thread = threading.Thread(
                target=self._pulse_reader, daemon=True
            )
            self._pulse_thread.start()
            logger.info("Audio iniciado vía PulseAudio (parec).")
        except FileNotFoundError:
            logger.error(
                "parec no encontrado. Instalá pulseaudio-utils:\n"
                "  sudo apt install pulseaudio-utils"
            )
            raise RuntimeError("No hay backend de audio disponible") from None
        except Exception:
            logger.exception("Error al iniciar parec")
            raise

    def _pulse_reader(self) -> None:
        """Lee stdout de parec, convierte s16le → float32, acumula en buffer."""
        bytes_per_block = BLOCK_SIZE * 2  # s16le = 2 bytes por sample
        while self._recording and self._pulse_proc:
            chunk = self._pulse_proc.stdout.read(bytes_per_block)
            if not chunk:
                break
            # Convertir s16le a float32 [-1, 1]
            samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            with self._lock:
                self._buffer.append(samples)

    def stop(self) -> np.ndarray | None:
        with self._lock:
            if not self._recording:
                return None
            self._recording = False

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        elif self._pulse_proc is not None:
            self._pulse_proc.terminate()
            try:
                self._pulse_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._pulse_proc.kill()
            self._pulse_proc = None

        with self._lock:
            buf = list(self._buffer)
            self._buffer.clear()

        if not buf:
            return None
        return np.concatenate(buf)

    @property
    def buffer_snapshot(self) -> np.ndarray | None:
        """Copia del buffer acumulado (sin vaciarlo)."""
        with self._lock:
            if not self._buffer:
                return None
            return np.concatenate(list(self._buffer))

    @property
    def block_count(self) -> int:
        with self._lock:
            return len(self._buffer)

    def _callback(self, indata: np.ndarray, _frames: int, _time: object, _status: int) -> None:
        if not self._recording:
            return
        with self._lock:
            self._buffer.append(indata[:, 0].copy())
