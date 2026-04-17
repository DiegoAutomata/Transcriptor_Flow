#!/usr/bin/env python3
"""
Transcriptor Flow v5 — Dictado en tiempo real
Ctrl+Alt: iniciar. Soltar: detener.
- Preview incremental cada 0.6s (modelo tiny) mientras hablás
- Corrección final (modelo small) al soltar
"""

import os
import wave
import threading
import tempfile
import subprocess
import time
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from pynput import keyboard

SAMPLE_RATE       = 16000
BLOCK_SIZE        = int(SAMPLE_RATE * 0.1)  # 100ms por callback
REALTIME_INTERVAL = 0.6                      # segundos entre actualizaciones
MIN_AUDIO_BLOCKS  = 6                        # 600ms mínimo antes del primer intento


class Transcriptor:
    def __init__(self):
        print("Cargando modelos Whisper (tiny + small)...")
        self.tiny  = WhisperModel("tiny",  device="cpu", compute_type="int8", cpu_threads=4)
        self.small = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=4)
        print("Listo. Mantén Ctrl+Alt para dictar.\n")

        self.ctrl = self.alt = self.recording = False
        self.stream = None
        self._lock        = threading.Lock()
        self._buf_lock    = threading.Lock()
        self._inject_lock = threading.Lock()
        self._model_lock  = threading.Lock()  # un modelo a la vez en CPU

        self._full_buf: list[np.ndarray] = []
        self._injected  = ""
        self._rt_event  = threading.Event()
        self._rt_thread: threading.Thread | None = None

    # ── Audio ──────────────────────────────────────────────────────────────────

    def _callback(self, indata, _f, _t, _s):
        if not self.recording:
            return
        with self._buf_lock:
            self._full_buf.append(indata[:, 0].copy())

    # ── Start / Stop ───────────────────────────────────────────────────────────

    def start(self):
        with self._lock:
            if self.recording:
                return
            self.recording = True
            self._injected = ""
            self._full_buf = []

        self._rt_event.clear()
        self._rt_thread = threading.Thread(target=self._realtime_loop, daemon=True)

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1,
            dtype=np.float32, blocksize=BLOCK_SIZE,
            callback=self._callback,
        )
        self.stream.start()
        self._rt_thread.start()
        print(f"[● {time.strftime('%H:%M:%S')}] Grabando...")

    def stop(self):
        with self._lock:
            if not self.recording:
                return
            self.recording = False

        # Señalamos stop: el loop realtime no inyectará más
        # NO hacemos join() — eso bloqueaba 1-5s innecesariamente
        self._rt_event.set()

        self.stream.stop()
        self.stream.close()
        self.stream = None

        with self._buf_lock:
            buf = list(self._full_buf)

        if not buf:
            print("[◼] Detenido (sin audio).")
            return

        print(f"[◼ {time.strftime('%H:%M:%S')}] Procesando final...")
        audio = np.concatenate(buf)

        # _model_lock espera si tiny está corriendo, sin bloquear el hilo principal
        final = self._transcribe(audio, self.small)
        print(f"  → {final}  ({time.strftime('%H:%M:%S')})")

        with self._inject_lock:
            # Las teclas ya están sueltas: reemplazar todo con la versión precisa
            self._replace_injected(final)
            if final:
                self._xdotool_type(" ")
        self._injected = ""

    # ── Loop realtime (modelo tiny) ────────────────────────────────────────────

    def _realtime_loop(self):
        while not self._rt_event.wait(REALTIME_INTERVAL):
            with self._buf_lock:
                if len(self._full_buf) < MIN_AUDIO_BLOCKS:
                    continue
                audio = np.concatenate(self._full_buf)

            t0   = time.time()
            text = self._transcribe(audio, self.tiny)
            dt   = time.time() - t0

            if text:
                print(f"  [rt {dt:.1f}s] {text}")
                with self._inject_lock:
                    if not self._rt_event.is_set():
                        self._append_injected(text)

    # ── Transcripción ─────────────────────────────────────────────────────────

    def _transcribe(self, audio: np.ndarray, model: WhisperModel) -> str:
        with self._model_lock:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp = f.name
            try:
                pcm = (audio * 32767).astype(np.int16)
                with wave.open(tmp, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(pcm.tobytes())

                segs, _ = model.transcribe(
                    tmp, language="es", beam_size=5, temperature=0,
                    vad_filter=True, vad_parameters={"threshold": 0.5},
                    no_speech_threshold=0.6, condition_on_previous_text=False,
                )
                return " ".join(s.text.strip() for s in segs).strip()
            finally:
                os.unlink(tmp)

    # ── Inyección ─────────────────────────────────────────────────────────────

    def _append_injected(self, new_text: str):
        """
        Durante grabación (Ctrl+Alt presionados): solo añade texto nuevo al final.
        NUNCA hace backspace — Ctrl+BackSpace borraría palabras enteras.
        """
        if new_text == self._injected or not new_text.startswith(self._injected):
            return
        delta = new_text[len(self._injected):]
        if delta:
            self._xdotool_type(delta)
            self._injected = new_text

    def _replace_injected(self, new_text: str):
        """
        Al soltar teclas: reemplaza todo el texto previo con la versión final.
        Aquí sí es seguro usar backspace (Ctrl/Alt ya sueltos).
        """
        if self._injected:
            self._xdotool_backspace(len(self._injected))
        if new_text:
            self._xdotool_type(new_text)
        self._injected = new_text

    def _xdotool_type(self, text: str):
        # SIN --clearmodifiers: ese flag manda eventos sintéticos de "soltar Ctrl/Alt"
        # que pynput intercepta y llama stop() cortando la grabación prematuramente.
        subprocess.run(
            ["xdotool", "type", "--delay", "5", text],
            check=False,
        )

    def _xdotool_backspace(self, n: int):
        if n > 0:
            # --clearmodifiers aquí es seguro: las teclas ya están sueltas
            subprocess.run(
                ["xdotool", "key", "--clearmodifiers", "--repeat", str(n), "BackSpace"],
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
