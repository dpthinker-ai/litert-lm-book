#!/usr/bin/env python3
"""Check that SUMMARY, manuscript H1 titles, and PDF units stay aligned."""

from __future__ import annotations

import ast
import re
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
SUMMARY_ENTRY = re.compile(r"^\s*-?\s*\[([^]]+)]\(([^)]+\.md)\)\s*$")


def normalize(title: str) -> str:
    return " ".join(title.replace("·", " ").split())


def first_h1(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    raise ValueError(f"missing H1: {path.relative_to(REPO)}")


def pdf_unit_titles() -> list[str]:
    tree = ast.parse((REPO / "scripts/finalize_pdf.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "UNITS"
            for target in node.targets
        ):
            return [item[0] for item in ast.literal_eval(node.value)]
    raise ValueError("UNITS not found in scripts/finalize_pdf.py")


def main() -> None:
    entries: list[tuple[str, str]] = []
    for line in (REPO / "SUMMARY.md").read_text(encoding="utf-8").splitlines():
        match = SUMMARY_ENTRY.match(line)
        if match:
            entries.append((match.group(1), match.group(2)))

    errors: list[str] = []
    for label, rel_path in entries:
        if rel_path == "cover.md":
            continue
        h1 = first_h1(REPO / rel_path)
        if normalize(label) != normalize(h1):
            errors.append(f"SUMMARY/H1 mismatch: {label!r} != {h1!r} ({rel_path})")

    summary_units = [label for label, path in entries if path != "cover.md"]
    pdf_units = pdf_unit_titles()
    if [normalize(item) for item in summary_units] != [normalize(item) for item in pdf_units]:
        errors.append("SUMMARY order/titles do not match finalize_pdf.py UNITS")

    if errors:
        raise SystemExit("\n".join(errors))
    print(f"book consistency check passed: {len(summary_units)} units")


if __name__ == "__main__":
    main()
