# Transcriptor Flow

Dictado por voz push-to-talk para Linux y WSL2. Mantenés **Ctrl+Alt**, hablás, y tu voz se convierte en texto inyectado en la aplicación que tengas enfocada. Sin fricción, sin UI innecesaria.

## Cómo funciona

```
Presionás Ctrl+Alt  →  grabación comienza
Hablás              →  audio se acumula en buffer
Soltás Ctrl+Alt     →  transcripción vía Groq API (~1s) o modelo local
                    →  texto se inyecta en la app activa
```

**Modo WSL2/Windows**: Un bridge en PowerShell detecta Ctrl+Alt (Win32 API) y se comunica con el daemon en WSL vía TCP. El bridge inyecta el texto transcrito en cualquier app de Windows — navegador, chat, Bloc de notas, terminal, etc.

**Modo Linux nativo**: El daemon usa `pynput` para detectar Ctrl+Alt y `xdotool` para inyectar texto en apps X11.

## Instalación

### Requisitos

- Python 3.10+
- Dependencias del sistema: `portaudio`, `pulseaudio-utils`, `xdotool` (solo Linux nativo)

### Instalación automática

```bash
cd ~/Transcriptor-Flow
bash install.sh
```

Esto:
1. Instala dependencias del sistema
2. Crea un virtualenv y instala paquetes Python
3. Configura el servicio systemd para auto-inicio
4. Copia el bridge de Windows a la carpeta de inicio

### Configuración de la API de Groq (recomendado)

Creá un archivo `.env` en la raíz del proyecto:

```env
GROQ_API_KEY=gsk_tu_api_key
```

Conseguí tu API key gratuita en [console.groq.com](https://console.groq.com).

**Sin Groq**: el sistema usa `faster-whisper` local (modelo `small`, ~6s por transcripción). Más lento pero sin dependencia de internet.

## Uso

### Windows + WSL2

1. Asegurate de que WSL esté corriendo y el daemon activo:
   ```bash
   systemctl --user status transcriptor-flow
   ```

2. En PowerShell (Windows), ejecutá el bridge:
   ```powershell
   \\wsl$\Ubuntu\home\diego\Transcriptor-Flow\StartBridgeHidden.vbs
   ```
   El bridge queda en segundo plano. Ctrl+Alt funciona en cualquier app.

3. Para auto-inicio: el VBS ya está copiado a la carpeta de inicio de Windows (`shell:startup`). Arranca solo al iniciar sesión.

### Linux nativo

El daemon arranca automáticamente vía systemd. Mantené Ctrl+Alt y hablá.

### Comandos verbales

Mientras dictás, podés decir estos comandos y se convierten en puntuación real:

| Decís | Se convierte en |
|-------|-----------------|
| `punto` | `.` |
| `coma` | `,` |
| `dos puntos` | `:` |
| `punto y coma` | `;` |
| `nueva línea` | salto de línea |
| `signo de interrogación` | `?` |
| `signo de exclamación` | `!` |
| `guion` | `-` |
| `arroba` | `@` |

## Arquitectura

```
┌──────────────────── WINDOWS ────────────────────┐
│  start_bridge.ps1 / StartBridgeHidden.vbs       │
│  · Detecta Ctrl+Alt (GetAsyncKeyState)          │
│  · Envía comandos TCP al daemon                 │
│  · Inyecta texto transcrito vía clipboard+Ctrl+V│
└──────────────────────┬──────────────────────────┘
                       │ TCP :19876
┌──────────────────── WSL2 / LINUX ───────────────┐
│  Daemon (systemd user service)                  │
│  · AudioCapture: sounddevice o PulseAudio       │
│  · Transcriber: Groq API o faster-whisper local │
│  · Cleanup: comandos verbales, puntuación, VAD  │
│  · TextInjector: xdotool (Linux nativo)         │
└─────────────────────────────────────────────────┘
```

## Archivos clave

| Archivo | Propósito |
|---------|-----------|
| `src/daemon.py` | Orquestador principal |
| `src/audio.py` | Captura de audio (PortAudio + PulseAudio fallback) |
| `src/transcriber.py` | Transcripción local con faster-whisper |
| `src/transcriber_groq.py` | Transcripción cloud con Groq API |
| `src/cleanup.py` | Post-procesamiento: comandos verbales, puntuación, anti-alucinaciones |
| `src/keyboard_handler.py` | Detección de hotkey (pynput o TCP socket) |
| `src/injector.py` | Inyección de texto vía xdotool (Linux nativo) |
| `start_bridge.ps1` | Bridge PowerShell para Windows |
| `StartBridgeHidden.vbs` | Lanzador silencioso para auto-inicio |
| `transcriptor-flow.service` | Servicio systemd user |

## Comandos útiles

```bash
# Ver estado del daemon
systemctl --user status transcriptor-flow

# Reiniciar daemon
systemctl --user restart transcriptor-flow

# Ver logs
tail -f ~/.local/share/transcriptor-flow/logs/transcriptor.log

# Ejecutar en modo debug
cd ~/Transcriptor-Flow && .venv/bin/python -m src.daemon --debug
```

## Solución de problemas

| Problema | Causa probable | Solución |
|----------|---------------|----------|
| Ctrl+Alt no hace nada | Bridge no está corriendo | Ejecutá `StartBridgeHidden.vbs` |
| No transcribe | Daemon no está corriendo | `systemctl --user restart transcriptor-flow` |
| Texto no aparece en la app | Bridge no inyecta | Revisá que el bridge esté corriendo y la app tenga foco |
| "Groq no configurado" en logs | Falta `.env` con API key | Creá `.env` con `GROQ_API_KEY=...` |
| Latencia alta | Usando modelo local | Configurá Groq API para ~1s |
| Audio no se captura | PulseAudio no configurado en WSL | Verificá `PULSE_SERVER` en el service |

## Licencia

MIT
