"""Renders one emoji to a small transparent PNG with the system colour-emoji font (macOS Apple Color Emoji).

TouchDesigner's Text TOP cannot draw colour emoji, so the server sends the picture with the LLM result.
"""

import io
import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

APPLE_EMOJI_FONT = "/System/Library/Fonts/Apple Color Emoji.ttc"
FONT_STRIKE = 160  # Apple Color Emoji is a bitmap (sbix) font; 160 px is one of its native sizes
CANVAS = 180
EMOJI_SIZE = 128


def render_emoji(emoji: str, font_path: str | Path = APPLE_EMOJI_FONT) -> bytes | None:
    """PNG bytes (EMOJI_SIZE x EMOJI_SIZE, RGBA) or None when there is nothing to draw or no font."""
    if not emoji or not Path(font_path).exists():
        return None
    try:
        font = ImageFont.truetype(str(font_path), FONT_STRIKE)
        image = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
        ImageDraw.Draw(image).text(
            (CANVAS // 2, CANVAS // 2), emoji, font=font, embedded_color=True, anchor="mm"
        )
        image = image.resize((EMOJI_SIZE, EMOJI_SIZE), Image.Resampling.LANCZOS)
    except (OSError, ValueError) as exc:  # unsupported glyph or font problem: the visuals skip the burst
        logger.warning("emoji render failed for %r: %s", emoji, exc)
        return None
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
