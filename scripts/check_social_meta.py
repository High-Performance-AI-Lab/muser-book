#!/usr/bin/env python3
"""Fail if a rendered book page cannot produce the intended social card."""

from __future__ import annotations

import argparse
import struct
import sys
from html.parser import HTMLParser
from pathlib import Path


CARD_URL = (
    "https://highperformanceailab.com/muser-book/"
    "muser-book-social-card.png"
)
BOOK_URL = "https://highperformanceailab.com/muser-book/"
REQUIRED = {
    "og:site_name",
    "og:title",
    "og:description",
    "og:type",
    "og:url",
    "og:image",
    "og:image:secure_url",
    "og:image:type",
    "og:image:width",
    "og:image:height",
    "og:image:alt",
    "twitter:card",
    "twitter:title",
    "twitter:description",
    "twitter:image",
    "twitter:image:alt",
}


class HeadMeta(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.values: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "meta":
            return
        values = dict(attrs)
        key = values.get("property") or values.get("name")
        content = values.get("content")
        if key and content is not None:
            self.values[key] = content


def png_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()[:24]
    if len(data) != 24 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise ValueError(f"{path} is not a PNG with an IHDR header")
    return struct.unpack(">II", data[16:24])


def check_page(path: Path) -> list[str]:
    parser = HeadMeta()
    parser.feed(path.read_text(encoding="utf-8"))
    errors = [f"{path}: missing {key}" for key in sorted(REQUIRED - parser.values.keys())]
    expected = {
        "og:url": BOOK_URL,
        "og:image": CARD_URL,
        "og:image:secure_url": CARD_URL,
        "og:image:type": "image/png",
        "og:image:width": "1200",
        "og:image:height": "630",
        "twitter:card": "summary_large_image",
        "twitter:image": CARD_URL,
    }
    for key, value in expected.items():
        if parser.values.get(key) != value:
            errors.append(f"{path}: {key} is not {value!r}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("build_dir", nargs="?", default="book")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    build_dir = (root / args.build_dir).resolve()
    source_card = root / "assets" / "muser-book-social-card.png"
    served_card = root / "src" / "muser-book-social-card.png"
    built_card = build_dir / "muser-book-social-card.png"
    errors: list[str] = []

    for card in (source_card, served_card, built_card):
        if not card.is_file():
            errors.append(f"missing social card: {card}")
        elif png_dimensions(card) != (1200, 630):
            errors.append(f"{card}: expected 1200x630 PNG")
    if source_card.is_file() and served_card.is_file():
        if source_card.read_bytes() != served_card.read_bytes():
            errors.append("repository and served social cards differ")
    if source_card.is_file() and built_card.is_file():
        if source_card.read_bytes() != built_card.read_bytes():
            errors.append("repository and built social cards differ")

    pages = sorted(build_dir.rglob("*.html"))
    if not pages:
        errors.append(f"no rendered HTML pages below {build_dir}")
    for page in pages:
        errors.extend(check_page(page))

    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(f"PASS: {len(pages)} pages carry complete Open Graph and X card metadata")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
