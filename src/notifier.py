"""Notificaciones de escritorio vía notify-send."""

import subprocess
import logging

logger = logging.getLogger(__name__)

APP_NAME = "Transcriptor Flow"


def notify(title: str, message: str = "", urgency: str = "normal") -> None:
    """Envía una notificación de escritorio. No lanza error si falla."""
    try:
        subprocess.run(
            [
                "notify-send",
                "--app-name", APP_NAME,
                "--urgency", urgency,
                title,
                message,
            ],
            check=False,
            timeout=3,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.debug("notify-send no disponible: %s", e)
