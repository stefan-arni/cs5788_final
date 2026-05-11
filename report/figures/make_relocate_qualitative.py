"""Stitch the relocation qualitative-results figure for the report.

Reads the synthetic-move PNGs that quick_test.py writes to
relocate/data/results/ and saves a single-row figure
(input | baseline | ours) as a PDF that main.tex picks up via
\\includegraphics{figures/relocate_qualitative.pdf}.

Run from this directory:
    cd report/figures && python make_relocate_qualitative.py
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# (path, column_caption)
# Source paths assume the standard layout: report/figures/ is two levels
# below the project root, and quick_test.py writes into
# relocate/data/results/.
COLS = [
    ("../../relocate/data/results/quick_test_input.png",    "input"),
    ("../../relocate/data/results/quick_test_baseline.png", "SDEdit baseline"),
    ("../../relocate/data/results/quick_test_ours.png",     "DDPM + noise shift (ours)"),
]

CELL = 360          # per-cell pixel size
LABEL_H = 26        # caption strip height
GAP = 8             # gap between cells
OUT_PDF = Path("relocate_qualitative.pdf")


def load_or_blank(path: str) -> Image.Image:
    p = Path(path)
    if p.exists():
        return Image.open(p).convert("RGB").resize((CELL, CELL))
    print(f"  missing: {path}", file=sys.stderr)
    img = Image.new("RGB", (CELL, CELL), (240, 240, 240))
    d = ImageDraw.Draw(img)
    d.text((10, 10), f"missing\n{p.name}", fill=(120, 120, 120))
    return img


def main():
    n = len(COLS)
    W = CELL * n + GAP * (n - 1)
    H = CELL + LABEL_H
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except IOError:
        font = ImageFont.load_default()

    for c, (path, caption) in enumerate(COLS):
        x = c * (CELL + GAP)
        canvas.paste(load_or_blank(path), (x, 0))
        draw.text((x + 4, CELL + 4), caption, fill="black", font=font)

    canvas.save(OUT_PDF, "PDF", resolution=144.0)
    print(f"wrote {OUT_PDF.resolve()}")


if __name__ == "__main__":
    main()
