#!/usr/bin/env python3
"""Generate the muser-book social card (1200x630 PNG).

Deterministic output: no randomness, no external assets. The committed
assets/muser-book-social-card.png is the render used as the GitHub social
preview. Re-run after changing any text:

    python3 scripts/generate_social_card.py --check   # verify committed PNG
    python3 scripts/generate_social_card.py           # regenerate
"""

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1200, 630
BG = (26, 24, 20)          # warm dark paper
FG = (240, 246, 252)       # near-white
ACCENT = (202, 144, 62)    # mdbook rust orange
DIM = (139, 148, 158)      # muted grey

FONT_PATHS = [
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/SUPPLEMENTARY/Arial Bold.ttf",
]


def font(size: int) -> ImageFont.FreeTypeFont:
    for p in FONT_PATHS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def render() -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # spine on the left edge, like a bound book
    d.rectangle([0, 0, 22, H], fill=ACCENT)
    d.line([34, 0, 34, H], fill=(60, 55, 48), width=3)

    x = 96
    d.text((x, 78), "How to Write", font=font(104), fill=FG)
    d.text((x, 196), "an Inference Engine", font=font(104), fill=FG)
    d.text((x, 348), "The Muser book — Muse Glimmer on Apple Metal,",
           font=font(44), fill=ACCENT)
    d.text((x, 404), "kvpack, and disaggregated prefill.",
           font=font(44), fill=ACCENT)
    d.text((x, 494), "40 chapters · 424-term glossary · every number receipted",
           font=font(34), fill=DIM)
    d.text((x, 560), "github.com/High-Performance-AI-Lab/muser-book",
           font=font(30), fill=DIM)
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify the committed PNG matches a fresh render")
    args = ap.parse_args()

    out = Path(__file__).resolve().parent.parent / "assets" / "muser-book-social-card.png"
    fresh = render()

    if args.check:
        if not out.exists():
            print("FAIL: committed card missing:", out)
            return 1
        committed = Image.open(out).convert("RGB")
        if committed.tobytes() != fresh.tobytes():
            print("FAIL: committed card differs from fresh render")
            return 1
        print("PASS: social card matches render")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    fresh.save(out, format="PNG", optimize=True)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
