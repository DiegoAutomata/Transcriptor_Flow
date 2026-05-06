"""Icono en la bandeja del sistema con indicador de estado."""

import threading
import logging
from typing import Callable

from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


try:
    import pystray
    PYSTRAY_AVAILABLE = True
except ImportError:
    PYSTRAY_AVAILABLE = False
    logger.warning("pystray no disponible — el icono de bandeja no funcionará")


class TrayIcon:
    """Gestiona el icono de bandeja con estados: idle, recording, error."""

    def __init__(self, on_exit: Callable[[], None]) -> None:
        self._on_exit = on_exit
        self._icon: "pystray.Icon | None" = None
        self._thread: threading.Thread | None = None
        self._status_item: "pystray.MenuItem | None" = None
        self._idle_image = _make_circle((72, 196, 113, 255))   # verde
        self._recording_image = _make_circle((237, 66, 69, 255))  # rojo
        self._error_image = _make_circle((240, 178, 50, 255))  # ámbar

    def start(self) -> None:
        if not PYSTRAY_AVAILABLE:
            logger.warning("Saltando icono de bandeja: pystray no disponible.")
            return

        self._status_item = pystray.MenuItem(
            "● Listo", lambda: None, enabled=False,
        )

        menu = pystray.Menu(
            self._status_item,
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Salir", self._on_exit),
        )

        self._icon = pystray.Icon(
            "transcriptor-flow",
            icon=self._idle_image,
            title="Transcriptor Flow",
            menu=menu,
        )

        self._thread = threading.Thread(target=self._icon.run, daemon=True)
        self._thread.start()
        logger.info("Icono de bandeja iniciado.")

    def stop(self) -> None:
        if self._icon is not None:
            self._icon.stop()
            logger.info("Icono de bandeja detenido.")

    def set_idle(self) -> None:
        """Estado: esperando Ctrl+Alt."""
        self._update("● Listo", self._idle_image)

    def set_recording(self) -> None:
        """Estado: grabando y transcribiendo."""
        self._update("⏺ Grabando", self._recording_image)

    def set_error(self, message: str = "") -> None:
        """Estado: error detectado."""
        label = f"⚠ {message}" if message else "⚠ Error"
        self._update(label, self._error_image)

    def _update(self, label: str, image: Image.Image) -> None:
        if self._icon is None:
            return
        try:
            self._icon.icon = image
            if self._status_item is not None:
                self._status_item.text = label
            self._icon.update_menu()
        except Exception:
            logger.debug("Error al actualizar icono de bandeja", exc_info=True)


def _make_circle(fill: tuple[int, int, int, int], size: int = 48) -> Image.Image:
    """Crea un círculo de color sólido con transparencia en las esquinas."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    margin = 3
    draw.ellipse([margin, margin, size - margin, size - margin], fill=fill)
    return img
