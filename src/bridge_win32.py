"""Bridge Win32: escucha Ctrl+Alt en Windows y notifica al daemon via TCP.

Diseñado para ejecutarse en Windows (no en WSL) y comunicarse con el
daemon de Transcriptor Flow corriendo dentro de WSL via localhost.
"""

import socket
import time
import ctypes
import os
import sys

HOST = "127.0.0.1"
PORT = 19876
POLL_INTERVAL = 0.05  # 50ms = 20 polls/segundo

VK_CONTROL = 0x11
VK_MENU = 0x12    # Alt

user32 = ctypes.windll.user32


def is_pressed(vk_code: int) -> bool:
    """GetAsyncKeyState: bit más significativo = tecla presionada."""
    return (user32.GetAsyncKeyState(vk_code) & 0x8000) != 0


def send_command(cmd: str) -> bool:
    """Envía un comando al daemon via TCP. Retorna True si ok."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        sock.connect((HOST, PORT))
        sock.sendall(cmd.encode("utf-8"))
        resp = sock.recv(1024)
        sock.close()
        return resp.startswith(b"ok")
    except (ConnectionRefusedError, socket.timeout, OSError):
        return False


def main():
    print(f"Transcriptor Flow Bridge — esperando Ctrl+Alt en {HOST}:{PORT}")
    print("(Ctrl+C para salir)")

    was_active = False
    connected = False

    # Verificar conectividad con el daemon al inicio
    print("Verificando conexión con el daemon...")
    for attempt in range(10):
        if send_command("ping"):
            connected = True
            print("  Conectado al daemon.")
            break
        time.sleep(1)
    if not connected:
        print("  ⚠ No se pudo conectar al daemon. ¿Está corriendo?")
        print("  Iniciá: systemctl --user start transcriptor-flow")
        print("  Reintentando en cada trigger...")
        print()

    try:
        while True:
            ctrl = is_pressed(VK_CONTROL)
            alt = is_pressed(VK_MENU)
            active = ctrl and alt

            if active and not was_active:
                print(f"[{time.strftime('%H:%M:%S')}] Ctrl+Alt → start")
                if send_command("start"):
                    print("  ✓ grabando")
                else:
                    print("  ✗ no se pudo conectar")

            elif not active and was_active:
                print(f"[{time.strftime('%H:%M:%S')}] Ctrl+Alt soltado → stop")
                send_command("stop")

            was_active = active
            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\nBridge detenido.")


if __name__ == "__main__":
    main()
