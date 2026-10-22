from __future__ import annotations

import re
import unicodedata

# Common typographical ligatures to expand
_LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
}

# Control characters to strip (preserving \t = 0x09 and \n = 0x0a)
_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Zero-width characters to remove
_ZERO_WIDTH_CHARS = ["\u00ad", "\u200b", "\u200c", "\u200d", "\ufeff"]

# Unicode space variants to normalize to standard ASCII space
_SPACE_VARIANTS = [
    "\u00a0",  # non-breaking space
    "\u202f",  # narrow no-break space
    "\u3000",  # ideographic space
] + [chr(code) for code in range(0x2000, 0x200B)]  # en-quad through hair space


def normalize_text(text: str) -> str:
    """Conservatively normalizes text without altering policy IDs, numbers, punctuation, or dates.

    - Normalizes Unicode using NFC standard composition.
    - Expands standard typographic ligatures (fi, fl, ff, etc.).
    - Removes zero-width characters and soft hyphens.
    - Converts Unicode space variations to standard ASCII spaces.
    - Strips non-printable control characters while preserving newlines and tabs.
    - Normalizes line endings to UNIX newline (\\n).
    - Collapses consecutive horizontal whitespace within lines.
    - Preserves paragraph structure by collapsing 3+ newlines to 2.
    """
    if not text:
        return ""

    # Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)

    # Ligature replacement
    for lig, rep in _LIGATURES.items():
        if lig in text:
            text = text.replace(lig, rep)

    # Remove zero-width characters and soft hyphens
    for zw in _ZERO_WIDTH_CHARS:
        if zw in text:
            text = text.replace(zw, "")

    # Normalize unicode whitespace variants to standard space
    for sp in _SPACE_VARIANTS:
        if sp in text:
            text = text.replace(sp, " ")

    # Strip control characters (preserving \t and \n)
    text = _CONTROL_CHAR_PATTERN.sub("", text)

    # Normalize line breaks to \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Clean up whitespace per line
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Collapse excessive blank lines (3+ newlines -> 2 newlines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
