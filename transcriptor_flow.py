#!/usr/bin/env python3
"""
Transcriptor Flow — Dictado por voz con Ctrl+Alt (push-to-talk)
Transcripción local con faster-whisper (gratis, sin API)
"""

import os
import time
import wave
import tempfile
import threading
import subprocess
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from pynput import keyboard

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
CHANNELS = 1
MODEL_SIZE = "base"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
CPU_THREADS = 4

# ── Estado global ─────────────────────────────────────────────────────────────
ctrl_pressed = False
alt_pressed = False
recording = False
audio_frames = []
stream = None
model = None
lock = threading.Lock()


def audio_callback(indata, _frames, _time_info, _status):
    if recording:
        audio_frames.append(indata.copy())


def start_recording():
    global recording, audio_frames, stream
    with lock:
        if recording:
            return
        audio_frames = []
        recording = True

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        callback=audio_callback,
        blocksize=1024,
    )
    stream.start()
    print("[●] Grabando...")


def stop_and_transcribe():
    global recording, stream

    with lock:
        if not recording:
            return
        recording = False

    if stream:
        stream.stop()
        stream.close()
        stream = None

    if not audio_frames:
        return

    audio_data = np.concatenate(audio_frames, axis=0)
    duration = len(audio_data) / SAMPLE_RATE

    if duration < 0.3:
        return

    print(f"[◼] {duration:.1f}s — transcribiendo...")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        audio_int16 = (audio_data * 32767).astype(np.int16)
        with wave.open(tmp_path, "w") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())

        segments, info = model.transcribe(
            tmp_path,
            beam_size=1,
            best_of=1,
            language="es",
            vad_filter=True,
            condition_on_previous_text=False,
        )

        text = " ".join(seg.text.strip() for seg in segments).strip()

        if not text:
            return

        print(f"[✓] {text}")

        # Pausa para que el SO registre que se soltaron Ctrl+Alt
        time.sleep(0.2)
        subprocess.run(
            ["xdotool", "type", "--clearmodifiers", "--delay", "20", text + " "],
            check=True,
        )

    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def on_press(key):
    global ctrl_pressed, alt_pressed
    if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
        ctrl_pressed = True
    elif key in (keyboard.Key.alt_l, keyboard.Key.alt_r):
        alt_pressed = True

    # Iniciar grabación en cuanto ambas modificadoras estén presionadas
    if ctrl_pressed and alt_pressed and not recording:
        threading.Thread(target=start_recording, daemon=True).start()


def on_release(key):
    global ctrl_pressed, alt_pressed
    if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
        ctrl_pressed = False
    elif key in (keyboard.Key.alt_l, keyboard.Key.alt_r):
        alt_pressed = False
    else:
        return

    # Detener en cuanto se suelte cualquiera de las dos
    if recording:
        threading.Thread(target=stop_and_transcribe, daemon=True).start()


def main():
    global model

    print("Cargando modelo Whisper...")
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE, cpu_threads=CPU_THREADS)
    print("Listo. Mantén Ctrl+Alt para grabar.")

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        try:
            listener.join()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
