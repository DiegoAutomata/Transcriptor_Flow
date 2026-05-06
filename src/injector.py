"""Inyección de texto en la aplicación activa vía xdotool."""

import subprocess
import logging

logger = logging.getLogger(__name__)


class TextInjector:
    """Escribe y borra texto en la ventana activa usando xdotool."""

    _injecting = False

    @classmethod
    @property
    def is_injecting(cls) -> bool:
        return cls._injecting

    def type_text(self, text: str) -> None:
        if not text:
            return
        TextInjector._injecting = True
        try:
            subprocess.run(
                ["xdotool", "type", "--clearmodifiers", "--delay", "5", text],
                check=False, timeout=2,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("xdotool type falló")
        finally:
            TextInjector._injecting = False

    def backspace(self, count: int) -> None:
        if count <= 0:
            return
        TextInjector._injecting = True
        try:
            subprocess.run(
                ["xdotool", "key", "--clearmodifiers", "--repeat", str(count), "BackSpace"],
                check=False, timeout=2,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("xdotool backspace falló")
        finally:
            TextInjector._injecting = False

    def append_delta(self, new_text: str, current_injected: str) -> str:
        """Añade solo el delta de texto nuevo al final (sin borrar)."""
        if not new_text.startswith(current_injected):
            return current_injected
        delta = new_text[len(current_injected):]
        if delta:
            self.type_text(delta)
            return new_text
        return current_injected

    def replace_all(self, new_text: str, old_text: str) -> str:
        """Borra todo el texto inyectado y escribe la versión final."""
        if old_text:
            self.backspace(len(old_text))
        if new_text:
            self.type_text(new_text)
        return new_text
