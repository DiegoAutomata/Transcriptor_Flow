#!/usr/bin/env python3
"""
Transcriptor Flow — Dictado por voz con Ctrl+Alt+Space (push-to-talk)
Transcripción local con faster-whisper (gratis, sin API)
"""

import os
import sys
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
SAMPLE_RATE = 16000   # Hz requerido por Whisper
CHANNELS = 1
MODEL_SIZE = "tiny"   # tiny ~150MB RAM, base ~250MB. Cambiar a "base" para más precisión.
DEVICE = "cpu"
COMPUTE_TYPE = "int8"  # más rápido en CPU sin GPU

# ── Estado global ─────────────────────────────────────────────────────────────
ctrl_pressed = False
alt_pressed = False
recording = False
audio_frames = []
stream = None
model = None
lock = threading.Lock()


def notify(title, message, urgency="normal"):
    """Envía notificación del escritorio (no bloqueante)."""
    try:
        subprocess.Popen(
            ["notify-send", "-u", urgency, "-t", "3000", title, message],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        print(f"[{title}] {message}")


def audio_callback(indata, frames, time_info, status):
    """Callback de sounddevice — acumula frames mientras graba."""
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
    notify("🎙 Transcriptor Flow", "Grabando...", urgency="low")
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
        print("[!] Sin audio capturado.")
        return

    # Concatenar frames y guardar WAV temporal
    audio_data = np.concatenate(audio_frames, axis=0)
    duration = len(audio_data) / SAMPLE_RATE

    if duration < 0.3:
        print("[!] Audio demasiado corto, ignorado.")
        return

    print(f"[◼] Grabación terminada ({duration:.1f}s). Transcribiendo...")
    notify("⚙ Transcriptor Flow", "Transcribiendo...", urgency="low")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Guardar WAV
        audio_int16 = (audio_data * 32767).astype(np.int16)
        with wave.open(tmp_path, "w") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())

        # Transcribir
        segments, info = model.transcribe(
            tmp_path,
            beam_size=1,
            best_of=1,
            language=None,  # auto-detect
            vad_filter=True,  # filtra silencio
        )

        text = " ".join(seg.text.strip() for seg in segments).strip()

        if not text:
            print("[!] No se detectó habla.")
            return

        print(f"[✓] Detectado ({info.language}, {info.language_probability:.0%}): {text}")

        # Inyectar texto en el cursor activo
        # Pequeña pausa para que el usuario suelte las teclas antes de escribir
        time.sleep(0.15)
        subprocess.run(
            ["xdotool", "type", "--clearmodifiers", "--delay", "20", text + " "],
            check=True,
        )

    except subprocess.CalledProcessError as e:
        notify("✗ Transcriptor Flow", f"Error al escribir: {e}", urgency="critical")
        print(f"[ERROR xdotool] {e}")
    except Exception as e:
        notify("✗ Transcriptor Flow", f"Error: {e}", urgency="critical")
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
    elif key == keyboard.Key.space:
        if ctrl_pressed and alt_pressed:
            # Lanzar en hilo para no bloquear el listener
            threading.Thread(target=start_recording, daemon=True).start()


def on_release(key):
    global ctrl_pressed, alt_pressed
    if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
        ctrl_pressed = False
        if recording:
            threading.Thread(target=stop_and_transcribe, daemon=True).start()
    elif key in (keyboard.Key.alt_l, keyboard.Key.alt_r):
        alt_pressed = False
        if recording:
            threading.Thread(target=stop_and_transcribe, daemon=True).start()
    elif key == keyboard.Key.space:
        if recording:
            threading.Thread(target=stop_and_transcribe, daemon=True).start()
    # Ctrl+C para salir limpiamente
    elif key == keyboard.Key.esc:
        notify("Transcriptor Flow", "Cerrando...", urgency="low")
        return False


def main():
    global model

    print("Transcriptor Flow iniciando...")
    print(f"Cargando modelo Whisper '{MODEL_SIZE}'...")

    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

    print("Modelo cargado.")
    print("Mantené Ctrl+Alt+Space para grabar. ESC para salir.")
    notify("Transcriptor Flow", "Listo ✓  Mantén Ctrl+Alt+Space para grabar")

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        try:
            listener.join()
        except KeyboardInterrupt:
            pass

    print("\nCerrando Transcriptor Flow.")


if __name__ == "__main__":
    main()
