"""Captura de audio desde el micrófono."""

import threading
import numpy as np
import sounddevice as sd

from .config import SAMPLE_RATE, BLOCK_SIZE


class AudioCapture:
    """Stream de micrófono con callback por bloque y acumulación."""

    def __init__(self):
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()
        self._buffer: list[np.ndarray] = []
        self._recording = False

    def start(self) -> None:
        with self._lock:
            if self._recording:
                return
            self._recording = True
            self._buffer.clear()

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.float32,
            blocksize=BLOCK_SIZE,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> np.ndarray | None:
        with self._lock:
            if not self._recording:
                return None
            self._recording = False

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

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
