"""Render a text file as a terminal-style PNG screenshot."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def render_screenshot(text_path: str, image_path: str, title: str) -> None:
    text = Path(text_path).read_text(encoding="utf-8")
    lines = text.replace("\t", "    ").splitlines() or [""]
    try:
        font = ImageFont.truetype("consola.ttf", 16)
        title_font = ImageFont.truetype("segoeui.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
        title_font = font

    line_h = 22
    pad_x, pad_y = 24, 20
    header_h = 48
    width = min(1200, max(720, max(len(line) for line in lines) * 9 + pad_x * 2))
    height = header_h + pad_y * 2 + line_h * (len(lines) + 1)

    img = Image.new("RGB", (width, height), "#0d1117")
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, width, header_h), fill="#161b22")
    draw.ellipse((16, 16, 32, 32), fill="#ff5f56")
    draw.ellipse((40, 16, 56, 32), fill="#ffbd2e")
    draw.ellipse((64, 16, 80, 32), fill="#27c93f")
    draw.text((100, 14), title, fill="#c9d1d9", font=title_font)

    y = header_h + pad_y
    for line in lines:
        draw.text((pad_x, y), line[:160], fill="#d4d4d4", font=font)
        y += line_h

    Path(image_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(image_path)
    print(f"Screenshot saved: {image_path}")
