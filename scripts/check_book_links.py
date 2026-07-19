#!/usr/bin/env python3
"""Check local links and fragment targets in the generated mdBook output."""

from __future__ import annotations

import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlsplit


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.urls: list[str] = []
        self.anchors: set[str] = set()

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, Optional[str]]]
    ) -> None:
        values = dict(attrs)
        if value := values.get("id"):
            self.anchors.add(value)
        if tag == "a" and (value := values.get("name")):
            self.anchors.add(value)
        attribute = {
            "a": "href",
            "img": "src",
            "link": "href",
            "script": "src",
            "source": "src",
        }.get(tag)
        if attribute and (value := values.get(attribute)):
            self.urls.append(value)


def parse_page(path: Path) -> PageParser:
    parser = PageParser()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    return parser


def main() -> None:
    output = Path(sys.argv[1] if len(sys.argv) > 1 else "book").resolve()
    if not output.is_dir():
        raise SystemExit(f"book output not found: {output}")

    pages = sorted(output.rglob("*.html"))
    parsed = {page: parse_page(page) for page in pages}
    errors: set[str] = set()
    checked = 0

    for page, data in parsed.items():
        for raw_url in data.urls:
            url = urlsplit(raw_url)
            if url.scheme or url.netloc or raw_url.startswith("//"):
                continue
            if not url.path and not url.fragment:
                continue
            relative_path = unquote(url.path)
            if relative_path.startswith("/"):
                target = output / relative_path.lstrip("/")
            else:
                target = page.parent / relative_path if relative_path else page
            target = target.resolve()
            try:
                target.relative_to(output)
            except ValueError:
                errors.add(
                    f"{page.relative_to(output)}: link escapes output: {raw_url}"
                )
                continue
            if target.is_dir():
                target /= "index.html"
            checked += 1
            if not target.is_file():
                errors.add(
                    f"{page.relative_to(output)}: missing target: {raw_url}"
                )
                continue
            if url.fragment and target.suffix.lower() == ".html":
                target_data = parsed.get(target)
                if target_data is None:
                    target_data = parse_page(target)
                    parsed[target] = target_data
                fragment = unquote(url.fragment)
                if fragment not in target_data.anchors:
                    errors.add(
                        f"{page.relative_to(output)}: missing fragment: {raw_url}"
                    )

    if errors:
        raise SystemExit("\n".join(sorted(errors)))
    print(f"book link check passed: {checked} local references across {len(pages)} pages")


if __name__ == "__main__":
    main()
