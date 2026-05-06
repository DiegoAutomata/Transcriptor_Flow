#!/bin/bash
# ============================================================
# Transcriptor Flow — Instalador completo
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   Transcriptor Flow — Instalación v6     ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════╝${NC}"
echo ""

# ── 1. Dependencias del sistema ──────────────────────────────────────
echo -e "${YELLOW}[1/5]${NC} Instalando dependencias del sistema…"
MISSING=()

for cmd in xdotool python3 pip3; do
    if ! command -v "$cmd" &>/dev/null; then
        MISSING+=("$cmd")
    fi
done

# libportaudio2 (necesario para sounddevice)
if ! dpkg -s libportaudio2 &>/dev/null 2>&1; then
    MISSING+=("libportaudio2")
fi

if [ ${#MISSING[@]} -gt 0 ]; then
    echo "  Paquetes a instalar: ${MISSING[*]}"
    echo "  Ejecutando: sudo apt install -y ${MISSING[*]}"
    sudo apt update -qq && sudo apt install -y "${MISSING[@]}"
fi

# ── 2. Entorno virtual Python ──────────────────────────────────────────
echo -e "${YELLOW}[2/5]${NC} Creando entorno virtual Python…"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
else
    echo "  .venv ya existe, se usará el existente."
fi

# Activar venv y actualizar pip
source .venv/bin/activate
pip install --quiet --upgrade pip setuptools wheel

# ── 3. Dependencias Python ─────────────────────────────────────────────
echo -e "${YELLOW}[3/5]${NC} Instalando dependencias Python…"
pip install --quiet -r requirements.txt
echo ""

# ── 4. Instalar servicio systemd de usuario ────────────────────────────
echo -e "${YELLOW}[4/5]${NC} Configurando servicio systemd (usuario)…"

SYSTEMD_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$SYSTEMD_DIR"

# Reemplazar %h por el HOME real en la unidad de servicio
sed "s|%h|$HOME|g" transcriptor-flow.service > "$SYSTEMD_DIR/transcriptor-flow.service"

echo "  Unidad instalada en: $SYSTEMD_DIR/transcriptor-flow.service"

# Recargar systemd del usuario
systemctl --user daemon-reload 2>/dev/null || true

# Habilitar el servicio
systemctl --user enable transcriptor-flow.service 2>/dev/null || true
echo "  Servicio habilitado (auto-inicio al login)."

# ── 5. Verificación final ──────────────────────────────────────────────
echo -e "${YELLOW}[5/5]${NC} Verificando instalación…"
if systemctl --user is-enabled transcriptor-flow.service &>/dev/null; then
    echo -e "  Estado: ${GREEN}habilitado${NC}"
else
    echo -e "  Estado: ${YELLOW}no habilitado (¡puede requerir login gráfico!)${NC}"
fi

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║      ✓ Instalación completa              ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Iniciar ahora:    ${CYAN}systemctl --user start transcriptor-flow${NC}"
echo -e "  Ver estado:       ${CYAN}systemctl --user status transcriptor-flow${NC}"
echo -e "  Ver logs:         ${CYAN}journalctl --user -u transcriptor-flow -f${NC}"
echo -e "  Detener:           ${CYAN}systemctl --user stop transcriptor-flow${NC}"
echo -e "  Deshabilitar:      ${CYAN}systemctl --user disable transcriptor-flow${NC}"
echo ""
echo -e "  Logs de la app:    ${CYAN}~/.local/share/transcriptor-flow/logs/${NC}"
echo ""
