import io
from pathlib import Path

import pytest
from PIL import Image

from tdml.nodes.emoji_render import APPLE_EMOJI_FONT, EMOJI_SIZE, render_emoji

needs_font = pytest.mark.skipif(
    not Path(APPLE_EMOJI_FONT).exists(), reason="Apple Color Emoji font not available"
)


@needs_font
def test_renders_a_colour_emoji_png():
    png = render_emoji("🌧️")

    image = Image.open(io.BytesIO(png))
    assert image.size == (EMOJI_SIZE, EMOJI_SIZE)
    assert image.mode == "RGBA"
    drawn = (
        [p for p in image.get_flattened_data() if p[3] > 128]
        if hasattr(image, "get_flattened_data")
        else [p for p in image.getdata() if p[3] > 128]
    )
    assert len(drawn) > 1000
    assert len({(r // 32, g // 32, b // 32) for r, g, b, _ in drawn}) > 10  # colourful, not a mono glyph


def test_missing_font_returns_none(tmp_path):
    assert render_emoji("🌧️", font_path=tmp_path / "nope.ttc") is None


def test_empty_text_returns_none():
    assert render_emoji("") is None


def test_render_errors_return_none(monkeypatch, tmp_path):
    font = tmp_path / "font.ttc"
    font.write_bytes(b"")

    def broken_font(*args, **kwargs):
        raise ValueError("unsupported glyph")

    monkeypatch.setattr("tdml.nodes.emoji_render.ImageFont.truetype", broken_font)

    assert render_emoji("🌧️", font_path=font) is None
