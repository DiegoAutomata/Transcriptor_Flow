"""Post-procesamiento de texto transcrito — comandos verbales, puntuación, 
filtro de alucinaciones (inspirado en sflow de daniel-carreon)."""

import re
import logging

logger = logging.getLogger(__name__)

# Filtro de alucinaciones comunes de Whisper en español.
# Si la transcripción completa matchea uno de estos patrones, se descarta.
_HALLUCINATION_PATTERNS: list[str] = [
    "gracias por ver",
    "gracias por ver el video",
    "suscríbete",
    "suscríbete al canal",
    "suscríbete a mi canal",
    "no olvides suscribirte",
    "dale like",
    "amara.org",
    "subtítulos por",
    "subtítulos por la comunidad",
    "hasta la próxima",
    "hasta luego",
    "adiós",
    "nos vemos",
    "gracias por escuchar",
    "bienvenidos a mi canal",
    "bienvenidos al canal",
    "hola a todos",
    "música",
    "[música]",
    "(música)",
    "[música suave]",
    "[aplausos]",
    "[risas]",
]

# Comandos verbales → reemplazo textual.
# Se aplican sobre toda la transcripción, preservando mayúsculas.
_VERBAL_COMMANDS: list[tuple[str, str]] = [
    # Puntuación
    (r"\bpunto\b", "."),
    (r"\bcoma\b", ","),
    (r"\bpunto y coma\b", ";"),
    (r"\bdos puntos\b", ":"),
    (r"\bpuntos suspensivos\b", "..."),
    (r"\bsigno de interrogación\b", "?"),
    (r"\bsigno de exclamación\b", "!"),
    (r"\bguion\b", "-"),
    (r"\bguión\b", "-"),
    # Formato
    (r"\bnueva línea\b", "\n"),
    (r"\bsalto de línea\b", "\n"),
    (r"\bpunto y aparte\b", ".\n"),
    (r"\bmayúscula\b", ""),  # Se maneja con capitalize después
    # Símbolos
    (r"\barroba\b", "@"),
    (r"\bparéntesis\b", "()"),
    (r"\babre paréntesis\b", "("),
    (r"\bcierra paréntesis\b", ")"),
    (r"\bcomillas\b", "\""),
    # Acciones
    (r"\benter\b", ""),
    (r"\bdale enter\b", ""),
    (r"\bpresiona enter\b", ""),
]

# Palabras filler a eliminar (solo si están rodeadas de espacios).
_FILLER_WORDS: list[str] = [
    r"\beh\b", r"\bem\b", r"\buh\b", r"\bhm\b", r"\bumm\b",
    r"\beste\b", r"\beste\s+este\b",
]

# Compilar patrones una sola vez
_HALLUCINATION_REGEX = re.compile(
    r"^\s*(" + "|".join(re.escape(p) for p in _HALLUCINATION_PATTERNS) + r")\s*$",
    re.IGNORECASE,
)


def is_hallucination(text: str) -> bool:
    """Detecta alucinaciones conocidas de Whisper. Retorna True si debe descartarse."""
    stripped = text.strip().lower()
    if not stripped:
        return True
    return bool(_HALLUCINATION_REGEX.match(stripped))


def apply_verbal_commands(text: str) -> str:
    """Aplica comandos verbales (regex) sobre el texto."""
    result = text
    for pattern, replacement in _VERBAL_COMMANDS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


def clean(text: str) -> str:
    """Pipeline completo de limpieza: comandos verbales + puntuación + filler."""
    if not text or not text.strip():
        return ""

    # 1. Comandos verbales
    text = apply_verbal_commands(text)

    # 2. Eliminar filler words rodeadas de espacios
    for filler in _FILLER_WORDS:
        text = re.sub(filler, " ", text, flags=re.IGNORECASE)

    # 3. Normalizar espacios múltiples
    text = re.sub(r"\s{2,}", " ", text)

    # 4. Capitalizar primera letra de la oración
    if text:
        text = text[0].upper() + text[1:]

    # 5. Añadir punto final si no hay puntuación al final
    text = text.rstrip()
    if text and text[-1] not in {'.', '!', '?', ':', ';', ',', '\n'}:
        text += "."

    # 6. Limpiar espacios antes de puntuación
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)

    # 7. Asegurar espacio después de puntuación (excepto al final)
    text = re.sub(r"([.,!?;:])([^\s\d])", r"\1 \2", text)

    return text


def process(text: str) -> str:
    """Pipeline completo: filtra alucinaciones, aplica limpieza."""
    if is_hallucination(text):
        logger.info("Transcripción descartada (alucinación detectada): %s", text)
        return ""
    return clean(text)
