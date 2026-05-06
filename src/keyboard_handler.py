"""Escucha global de teclado: Ctrl+Alt = dictar."""

import logging
from typing import Callable

from pynput import keyboard

from .injector import TextInjector

logger = logging.getLogger(__name__)


class KeyboardHandler:
    """Detecta Ctrl+Alt sostenido para activar/desactivar dictado."""

    def __init__(
        self,
        on_activate: Callable[[], None],
        on_deactivate: Callable[[], None],
    ) -> None:
        self._on_activate = on_activate
        self._on_deactivate = on_deactivate
        self._ctrl = False
        self._alt = False
        self._last_state = False
        self._listener: keyboard.Listener | None = None

    def start(self) -> None:
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()
        logger.info("Escucha de teclado iniciada.")

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
            logger.info("Escucha de teclado detenida.")

    @property
    def is_active(self) -> bool:
        return self._ctrl and self._alt

    @property
    def recording(self) -> bool:
        return self._listener is not None

    def _check(self) -> None:
        want = self._ctrl and self._alt
        if want == self._last_state:
            return
        self._last_state = want
        if want:
            self._on_activate()
        else:
            self._on_deactivate()

    def _should_ignore(self, key: keyboard.Key | keyboard.KeyCode | None) -> bool:
        """Ignora eventos sintéticos generados por xdotool --clearmodifiers."""
        if not TextInjector.is_injecting:
            return False
        return key in (
            keyboard.Key.ctrl_l, keyboard.Key.ctrl_r,
            keyboard.Key.alt_l,  keyboard.Key.alt_r,
        )

    def _on_press(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        if self._should_ignore(key):
            return
        if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            self._ctrl = True
        elif key in (keyboard.Key.alt_l, keyboard.Key.alt_r):
            self._alt = True
        self._check()

    def _on_release(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        if self._should_ignore(key):
            return
        if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            self._ctrl = False
        elif key in (keyboard.Key.alt_l, keyboard.Key.alt_r):
            self._alt = False
        self._check()
