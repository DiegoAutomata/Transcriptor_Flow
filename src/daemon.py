"""Orquestador principal — demonio con tray icon, logging, señales y realtime loop."""

import os
import sys
import time
import signal
import subprocess
import logging
import logging.handlers
import threading
from pathlib import Path

import numpy as np

from . import config
from .audio import AudioCapture
from .transcriber import Transcriber
from .injector import TextInjector
from .keyboard_handler import KeyboardHandler
from .notifier import notify
from .tray_icon import TrayIcon


logger = logging.getLogger("transcriptor-flow")


def _is_wsl() -> bool:
    """Detecta si se está ejecutando dentro de WSL."""
    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def setup_logging(debug: bool = False) -> None:
    """Configura logging rotativo a archivo."""
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    handler = logging.handlers.RotatingFileHandler(
        config.LOG_FILE,
        maxBytes=config.LOG_MAX_BYTES,
        backupCount=config.LOG_BACKUPS,
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    root = logging.getLogger()
    root.setLevel(logging.DEBUG if debug else logging.INFO)
    root.addHandler(handler)

    # Silenciar bibliotecas ruidosas
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)
    logging.getLogger("sounddevice").setLevel(logging.WARNING)


class Daemon:
    """Orquestador del servicio Transcriptor Flow."""

    def __init__(self, debug: bool = False) -> None:
        self._debug = debug
        self._shutdown = threading.Event()
        self._recording = False
        self._lock = threading.Lock()

        self._audio = AudioCapture()
        self._transcriber: Transcriber | None = None
        self._injector = TextInjector()
        self._tray: TrayIcon | None = None
        self._keyboard: KeyboardHandler | None = None
        self._bridge_proc: subprocess.Popen | None = None

        # Estado de texto inyectado durante la sesión de grabación
        self._injected_text = ""
        self._raw_preview_text = ""
        self._last_transcription = ""

        # Hilo de transcripción en tiempo real
        self._rt_stop = threading.Event()
        self._rt_thread: threading.Thread | None = None

    # ── Ciclo de vida ──────────────────────────────────────────────────────

    def run(self) -> None:
        """Inicia el demonio."""
        setup_logging(self._debug)
        logger.info("Transcriptor Flow %s iniciando…", config.VERSION)

        self._tray = TrayIcon(on_exit=self.shutdown)
        if not _is_wsl():
            self._tray.start()
            self._tray.set_idle()

        self._transcriber = Transcriber()

        mode = "socket" if _is_wsl() else "pynput"
        logger.info("Modo de teclado: %s", mode)
        self._keyboard = KeyboardHandler(
            on_activate=self._start_recording,
            on_deactivate=self._stop_recording,
            on_preview=self._get_preview_text,
            mode=mode,
        )
        self._keyboard.start()

        # En WSL, iniciar el bridge de Windows si Python está disponible
        if _is_wsl():
            self._start_win32_bridge()

        notify("Transcriptor Flow", "Listo — mantén Ctrl+Alt para dictar.", urgency="low")
        logger.info("Demonio listo. Ctrl+Alt para dictar.")

        # Manejar señales de sistema
        signal.signal(signal.SIGTERM, lambda s, f: self.shutdown())
        signal.signal(signal.SIGINT,  lambda s, f: self.shutdown())

        try:
            self._shutdown.wait()
        except KeyboardInterrupt:
            pass

        logger.info("Demonio finalizado.")

    def shutdown(self) -> None:
        """Apagado limpio de todos los componentes."""
        if self._shutdown.is_set():
            return
        logger.info("Apagando…")

        self._stop_recording()

        if self._keyboard:
            self._keyboard.stop()

        if self._tray:
            self._tray.stop()

        # Matar el bridge de Windows si existe
        if hasattr(self, "_bridge_proc") and self._bridge_proc:
            try:
                self._bridge_proc.terminate()
            except Exception:
                pass

        self._shutdown.set()
        logger.info("Transcriptor Flow detenido.")

    # ── Grabación ──────────────────────────────────────────────────────────

    def _start_recording(self) -> None:
        with self._lock:
            if self._recording:
                return
            self._recording = True
            self._injected_text = ""
            self._raw_preview_text = ""

        try:
            self._audio.start(prefer_pulse=_is_wsl())
        except Exception:
            logger.exception("No se pudo iniciar la captura de audio")
            with self._lock:
                self._recording = False
            if self._tray:
                self._tray.set_error("Micrófono no disponible")
            return
        self._rt_stop.clear()
        self._rt_thread = threading.Thread(target=self._realtime_loop, daemon=True)
        self._rt_thread.start()

        if self._tray and not _is_wsl():
            self._tray.set_recording()

        logger.info("Grabación iniciada.")

    def _stop_recording(self) -> str:
        with self._lock:
            if not self._recording:
                return ""
            self._recording = False

        self._rt_stop.set()
        audio = self._audio.stop()

        if self._tray:
            self._tray.set_idle()

        if audio is None or len(audio) == 0:
            logger.info("Sin audio para transcribir.")
            return ""

        logger.info("Transcribiendo final con modelo small…")
        try:
            final_text = self._transcriber.transcribe_final(audio)
        except Exception:
            logger.exception("Error en transcripción final")
            if self._tray:
                self._tray.set_error("Error al transcribir")
            return ""

        if final_text:
            logger.info("Texto final: %s", final_text)

        self._injector.replace_all(final_text, self._injected_text)

        if final_text:
            self._injector.type_text(" ")
        self._injected_text = ""
        self._last_transcription = final_text or ""
        logger.info("Grabación finalizada.")
        return self._last_transcription

    def _get_preview_text(self) -> str:
        """Devuelve el texto en tiempo real crudo (para el bridge)."""
        return self._raw_preview_text

    # ── Loop realtime ──────────────────────────────────────────────────────

    def _realtime_loop(self) -> None:
        """Transcripción incremental cada 0.6s con modelo tiny."""
        while not self._rt_stop.wait(config.REALTIME_INTERVAL):
            if not self._recording:
                continue

            if self._audio.block_count < config.MIN_AUDIO_BLOCKS:
                continue

            snapshot = self._audio.buffer_snapshot
            if snapshot is None:
                continue

            t0 = time.time()
            try:
                text = self._transcriber.transcribe_realtime(snapshot)
            except Exception:
                logger.exception("Error en transcripción realtime")
                continue

            dt = time.time() - t0

            if text:
                logger.info("[rt %.1fs] %s", dt, text)
                self._raw_preview_text = text
                if not self._rt_stop.is_set():
                    self._injected_text = self._injector.append_delta(text, self._injected_text)

    # ── Bridge Win32 (WSL) ────────────────────────────────────────────────

    def _start_win32_bridge(self) -> None:
        """Muestra instrucciones para iniciar el bridge en Windows."""
        logger.info(
            "Modo WSL detectado. El bridge Win32 debe ejecutarse en Windows.\n"
            "  Opción 1: Ejecutá manualmente:\n"
            "    python \\\\wsl$\\Ubuntu\\home\\diego\\Transcriptor-Flow\\src\\bridge_win32.py\n"
            "  Opción 2: Doble clic en start_bridge.bat (en el proyecto)\n"
            "  Opción 3: Agregá start_bridge.bat al inicio de Windows"
        )
        self._bridge_proc = None


def main() -> None:
    """Punto de entrada."""
    debug = "--debug" in sys.argv
    daemon = Daemon(debug=debug)
    daemon.run()


if __name__ == "__main__":
    main()
