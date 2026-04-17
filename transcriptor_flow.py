#!/usr/bin/env python3
"""
Transcriptor Flow — Dictado en tiempo real con Ctrl+Alt (push-to-talk)
El texto aparece frase por frase mientras hablás, sin esperar a soltar las teclas.
"""

import os
import wave
import queue
import tempfile
import threading
import subprocess
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from pynput import keyboard

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16000
BLOCK_SIZE    = int(SAMPLE_RATE * 0.1)   # 100ms por callback

RMS_THRESHOLD     = 400   # energía mínima para considerar voz (escala int16)
SILENCE_BLOCKS    = 5    # 5 × 100ms = 500ms de silencio → fin de frase
MAX_BUF_BLOCKS    = 30   # 3s acumulados → flush forzado aunque no haya silencio
MIN_SPEECH_BLOCKS = 3    # 300ms mínimo para transcribir


class Transcriptor:
    def __init__(self):
        print("Cargando modelo Whisper...")
        self.model = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=4)
        print("Listo. Mantén Ctrl+Alt para dictar.\n")

        self.ctrl = self.alt = self.recording = False
        self.stream = None
        self._lock = threading.Lock()

        # Estado VAD
        self._buf: list[np.ndarray] = []  # fragmentos de audio del utterance actual
        self._sil = 0      # bloques de silencio consecutivos
        self._sph = 0      # bloques con voz acumulados
        self._active = False

        # Cola → worker serializa transcripción e inyección
        self._q: queue.Queue = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

    # ── Audio ──────────────────────────────────────────────────────────────────

    def _callback(self, indata, _frames, _time_info, _status):
        if not self.recording:
            return

        chunk = indata[:, 0]  # float32, mono
        rms = float(np.sqrt(np.mean((chunk * 32767) ** 2)))

        if rms > RMS_THRESHOLD:
            self._active = True
            self._sil = 0
            self._sph += 1
            self._buf.append(chunk.copy())
            # Flush forzado si se acumula demasiado audio sin pausas
            if len(self._buf) >= MAX_BUF_BLOCKS:
                self._flush()
        elif self._active:
            self._sil += 1
            self._buf.append(chunk.copy())
            if self._sil >= SILENCE_BLOCKS:
                self._flush()

    def _flush(self, final=False):
        """Encola el fragmento actual para transcripción y resetea el estado."""
        min_sph = 1 if final else MIN_SPEECH_BLOCKS
        if self._sph >= min_sph and self._buf:
            self._q.put(np.concatenate(self._buf))
        self._buf = []
        self._sil = self._sph = 0
        self._active = False

    # ── Start / Stop ───────────────────────────────────────────────────────────

    def start(self):
        with self._lock:
            if self.recording:
                return
            self.recording = True
            self._buf = []
            self._sil = self._sph = 0
            self._active = False

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1,
            dtype=np.float32, blocksize=BLOCK_SIZE,
            callback=self._callback,
        )
        self.stream.start()
        print("[●] Grabando...")

    def stop(self):
        with self._lock:
            if not self.recording:
                return
            self.recording = False

        self.stream.stop()
        self.stream.close()
        self.stream = None

        # Transcribir lo que quedó en el buffer (frase incompleta al soltar teclas)
        self._flush(final=True)
        print("[◼] Detenido.")

    # ── Worker: transcripción + inyección secuencial ───────────────────────────

    def _worker(self):
        while True:
            audio = self._q.get()
            self._transcribe(audio)
            self._q.task_done()

    def _transcribe(self, audio: np.ndarray):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name

        pcm = (audio * 32767).astype(np.int16)
        with wave.open(tmp, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())

        segs, _ = self.model.transcribe(
            tmp,
            language="es",
            beam_size=5,
            temperature=0,
            vad_filter=True,
            vad_parameters={"threshold": 0.5},
            no_speech_threshold=0.6,
            condition_on_previous_text=False,
        )
        text = " ".join(s.text.strip() for s in segs).strip()
        os.unlink(tmp)

        if text:
            print(f"  → {text}")
            subprocess.run(
                ["xdotool", "type", "--clearmodifiers", "--delay", "5", text + " "],
                check=False,
            )

    # ── Teclado ────────────────────────────────────────────────────────────────

    def _check(self):
        want = self.ctrl and self.alt
        if want and not self.recording:
            self.start()
        elif not want and self.recording:
            self.stop()

    def on_press(self, key):
        if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            self.ctrl = True
        elif key in (keyboard.Key.alt_l, keyboard.Key.alt_r):
            self.alt = True
        self._check()

    def on_release(self, key):
        if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            self.ctrl = False
        elif key in (keyboard.Key.alt_l, keyboard.Key.alt_r):
            self.alt = False
        self._check()

    def run(self):
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as lst:
            lst.join()


if __name__ == "__main__":
    Transcriptor().run()
