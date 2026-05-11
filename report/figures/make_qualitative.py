"""Stitch the qualitative-results figure for the report.

Takes two (source, edit) image pairs and saves a 2-row grid as a PDF
suitable for inclusion in main.tex via \\includegraphics.

Both rows are edits on the same demo photo (data/demo/headline.jpg)
with two different target prompts. To populate, on Colab or locally:

    # Row 1: golden retriever -> red fox
    python scripts/edit_single.py data/demo/headline.jpg \\
        --src "a photograph of a golden retriever sitting on grass" \\
        --tgt "a photograph of a red fox sitting on grass" \\
        --schedule linear --mask-mode attention
    mv outputs/edit_single/headline__linear__attention__src.png \\
        outputs/edit_single/headline_fox__src.png
    mv outputs/edit_single/headline__linear__attention__edit.png \\
        outputs/edit_single/headline_fox__edit.png

    # Row 2: golden retriever -> black cat
    python scripts/edit_single.py data/demo/headline.jpg \\
        --src "a photograph of a golden retriever sitting on grass" \\
        --tgt "a photograph of a black cat sitting on grass" \\
        --schedule linear --mask-mode attention
    mv outputs/edit_single/headline__linear__attention__src.png \\
        outputs/edit_single/headline_cat__src.png
    mv outputs/edit_single/headline__linear__attention__edit.png \\
        outputs/edit_single/headline_cat__edit.png

Then:
    cd report/figures && python make_qualitative.py
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# (left_image_path, right_image_path, row_caption)
# Paths are relative to this file's directory (report/figures/).
PAIRS = [
    ("../../outputs/edit_single/headline_fox__src.png",
     "../../outputs/edit_single/headline_fox__edit.png",
     "golden retriever -> red fox"),
]

CELL = 320          # per-cell pixel size
LABEL_H = 26        # caption strip height per row
GAP = 8             # gap between cells
OUT_PDF = Path("qualitative.pdf")


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
    rows = len(PAIRS)
    W = CELL * 2 + GAP
    H = (CELL + LABEL_H + GAP) * rows
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except IOError:
        font = ImageFont.load_default()

    for r, (src_path, edit_path, caption) in enumerate(PAIRS):
        y = r * (CELL + LABEL_H + GAP)
        canvas.paste(load_or_blank(src_path), (0, y))
        canvas.paste(load_or_blank(edit_path), (CELL + GAP, y))
        draw.text((4, y + CELL + 4), caption, fill="black", font=font)

    canvas.save(OUT_PDF, "PDF", resolution=144.0)
    print(f"wrote {OUT_PDF.resolve()}")


if __name__ == "__main__":
    main()
