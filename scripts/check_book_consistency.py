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


NUMBERED = re.compile(r"^(\d+(?:\.\d+)+)\s*(.*)$")
SUB_ENTRY = re.compile(r"^\s+-\s*\[([^]]+)]\(\)\s*$")
HEADING = re.compile(r"^(#{2,3})\s+(\d+(?:\.\d+)+)\s*(.*)$")


def normalize_section(title: str) -> str:
    return " ".join(title.replace("`", "").replace("\u3000", " ").split())


def numbered_headings(path: Path) -> list[tuple[str, str]]:
    """H2/H3 headings that start with a dotted section number (x.y or x.y.z)."""
    found: list[tuple[str, str]] = []
    fenced = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        match = HEADING.match(line)
        if match:
            found.append((match.group(2), normalize_section(match.group(3))))
    return found


def check_section_titles() -> list[str]:
    """SUMMARY 的节/小节条目必须与所属文件的 H2/H3 标题逐条一致（数字与文字）。
    finalize_pdf.py 的打印目录与书签直接使用 SUMMARY 文字，故此处是唯一防线。"""
    errors: list[str] = []
    current: Path | None = None
    expected: dict[Path, list[tuple[str, str]]] = {}
    for line in (REPO / "SUMMARY.md").read_text(encoding="utf-8").splitlines():
        top = SUMMARY_ENTRY.match(line)
        if top:
            current = REPO / top.group(2)
            expected.setdefault(current, [])
            continue
        sub = SUB_ENTRY.match(line)
        if sub and current is not None:
            numbered = NUMBERED.match(sub.group(1).strip())
            if numbered:
                expected[current].append(
                    (numbered.group(1), normalize_section(numbered.group(2)))
                )
    for path, listed in expected.items():
        if path.name == "cover.md" or not path.exists():
            continue
        actual = numbered_headings(path)
        if listed == actual:
            continue
        rel = path.relative_to(REPO)
        listed_map, actual_map = dict(listed), dict(actual)
        for num, title in listed:
            if num not in actual_map:
                errors.append(f"SUMMARY section not in manuscript: {num} {title!r} ({rel})")
            elif actual_map[num] != title:
                errors.append(
                    f"SUMMARY/heading mismatch: {num} {title!r} != {actual_map[num]!r} ({rel})"
                )
        for num, title in actual:
            if num not in listed_map:
                errors.append(f"manuscript section missing from SUMMARY: {num} {title!r} ({rel})")
        if set(listed_map) == set(actual_map) and [n for n, _ in listed] != [n for n, _ in actual]:
            errors.append(f"SUMMARY section order differs from manuscript ({rel})")
    return errors


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

    errors.extend(check_section_titles())

    summary_units = [label for label, path in entries if path != "cover.md"]
    pdf_units = pdf_unit_titles()
    if [normalize(item) for item in summary_units] != [normalize(item) for item in pdf_units]:
        errors.append("SUMMARY order/titles do not match finalize_pdf.py UNITS")

    if errors:
        raise SystemExit("\n".join(errors))
    print(f"book consistency check passed: {len(summary_units)} units")


if __name__ == "__main__":
    main()
