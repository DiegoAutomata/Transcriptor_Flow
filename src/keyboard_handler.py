"""Escucha global de teclado — soporta pynput (X11) y socket TCP (WSL/remoto)."""

import socket
import threading
import logging
from typing import Callable

from .injector import TextInjector

logger = logging.getLogger(__name__)

HOTKEY_PORT = 19876
HOTKEY_HOST = "127.0.0.1"


class KeyboardHandler:
    """Detecta Ctrl+Alt sostenido para activar/desactivar dictado.

    Modos:
    - "pynput" (default): usa pynput para X11 nativo
    - "socket": levanta un servidor TCP y espera comandos "start"/"stop"
    """

    def __init__(
        self,
        on_activate: Callable[[], None],
        on_deactivate: Callable[[], str],
        mode: str = "pynput",
    ) -> None:
        self._on_activate = on_activate
        self._on_deactivate = on_deactivate
        self._mode = mode
        self._listener: threading.Thread | None = None
        self._server: socket.socket | None = None
        self._active = False

    def start(self) -> None:
        if self._mode == "socket":
            self._start_socket()
        else:
            self._start_pynput()

    def stop(self) -> None:
        if self._mode == "socket":
            if self._server:
                self._server.close()
                self._server = None
            logger.info("Servidor TCP detenido.")
        else:
            self._stop_pynput()

    # ── Modo pynput ──────────────────────────────────────────────────────

    def _start_pynput(self) -> None:
        logger.info("Usando pynput (X11) para escucha de teclado.")
        from pynput import keyboard

        self._ctrl = False
        self._alt = False
        self._last_state = False
        self._active = False

        def on_press(key):
            if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
                self._ctrl = True
            elif key in (keyboard.Key.alt_l, keyboard.Key.alt_r):
                self._alt = True
            self._check()

        def on_release(key):
            if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
                self._ctrl = False
            elif key in (keyboard.Key.alt_l, keyboard.Key.alt_r):
                self._alt = False
            self._check()

        self._pynput_listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release,
        )
        self._pynput_listener.start()
        logger.info("Escucha de teclado iniciada (pynput).")

    def _stop_pynput(self) -> None:
        if hasattr(self, "_pynput_listener") and self._pynput_listener:
            self._pynput_listener.stop()
        logger.info("Escucha de teclado detenida (pynput).")

    def _check(self) -> None:
        want = self._ctrl and self._alt
        if want == self._last_state:
            return
        self._last_state = want
        if want:
            self._on_activate()
        else:
            self._on_deactivate()

    # ── Modo socket ──────────────────────────────────────────────────────

    def _start_socket(self) -> None:
        logger.info("Usando servidor TCP en %s:%d para triggers externos.", HOTKEY_HOST, HOTKEY_PORT)
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((HOTKEY_HOST, HOTKEY_PORT))
        self._server.listen(1)

        self._running = True
        self._listener = threading.Thread(target=self._socket_loop, daemon=True)
        self._listener.start()
        logger.info("Servidor TCP iniciado en %s:%d.", HOTKEY_HOST, HOTKEY_PORT)

    def _socket_loop(self) -> None:
        while getattr(self, "_running", False):
            try:
                self._server.settimeout(1.0)
                conn, addr = self._server.accept()
            except (socket.timeout, OSError):
                continue
            except Exception:
                logger.exception("Error en accept()")
                break

            try:
                data = conn.recv(1024).decode("utf-8").strip().lower()
                logger.info("Comando recibido de %s: %s", addr, data)
                if data == "start":
                    if self._active:
                        conn.sendall(b"ok: already recording\n")
                    else:
                        self._active = True
                        self._on_activate()
                        conn.sendall(b"ok: recording\n")
                elif data == "stop" and self._active:
                    self._active = False
                    text = self._on_deactivate()
                    conn.sendall(f"text: {text}\n".encode("utf-8"))
                elif data == "ping":
                    conn.sendall(b"pong\n")
                else:
                    conn.sendall(b"unknown command\n")
            except Exception:
                logger.exception("Error procesando comando")
            finally:
                try:
                    conn.close()
                except OSError:
                    pass
