#!/usr/bin/env python3
"""Check LiteRT-LM source anchors used by the manuscript.

Check anchor bounds and quoted source lines against the frozen release.
Quoted lines must occur verbatim, in order, at or after their first anchor.
This does not decide whether a quotation supports the surrounding prose;
that remains part of the chapter fact-check pass.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = REPO.parent / "LiteRT-LM-v0.17.0"
DEFAULT_LITERT_SOURCE = REPO.parent / "LiteRT-moe-9fe5be4"
EXPECTED_LITERT_COMMIT = "9fe5be45564c868408e6514c8aabb83e211a0911"
EXPECTED_SOURCE_COMMIT = "e9fd8c53ff968071774206163027dd84bedfe925"
CODE_SPAN = re.compile(r"`([^`\n]+)`")
SOURCE_REF = re.compile(
    r"(?P<path>[A-Za-z0-9_.+-]+(?:/[A-Za-z0-9_.+-]+)*\."
    r"(?:bzl|c|cc|cmake|cpp|fbs|g4|h|hpp|inc|java|js|kt|m|md|mm|proto|py|rs|swift|ts))"
    r":(?P<start>\d+)(?:-(?P<end>\d+))?"
)
SHORTHAND_REF = re.compile(
    r"(?::|\.(?:bzl|c|cc|cmake|cpp|fbs|g4|h|hpp|inc|java|js|kt|m|md|mm|proto|py|rs|swift|ts):)"
    r"\d+(?:-\d+)?"
)
EXTERNAL_PREFIXES = ("llama.cpp/", "MLC-LLM/", "ExecuTorch/")
FENCED_BLOCK = re.compile(r"^```[^\n]*\n(.*?)^```", re.MULTILINE | re.DOTALL)
ANNOTATION = re.compile(r"\s+(?://|#)\s*\(\d+\)\s*$")
OMISSION = re.compile(r"\s*(?://|#)\s*\.\.\.")


def source_target(path: str, source: Path, litert_source: Path | None = None) -> Path:
    if path.startswith("LiteRT/"):
        if litert_source is None:
            raise ValueError("LiteRT source required for dependency quotation")
        return litert_source / path.removeprefix("LiteRT/")
    return source / path


def check_source_excerpts(
    text: str, source: Path, litert_source: Path | None = None
) -> tuple[int, list[str]]:
    """Check local source quotations, preserving indentation and source order."""
    errors: list[str] = []
    checked = 0
    for block in FENCED_BLOCK.finditer(text):
        body = block.group(1)
        refs = [
            ref for ref in SOURCE_REF.finditer(body)
            if not ref.group("path").startswith(EXTERNAL_PREFIXES)
        ]
        if not refs:
            continue
        checked += 1
        first_line = text[:block.start()].count("\n") + 2
        paths = {ref.group("path") for ref in refs}
        if len(paths) != 1:
            errors.append(f"{first_line}: split quotations from different files")
            continue
        target = source_target(refs[0].group("path"), source, litert_source)
        if not target.is_file():
            continue  # The anchor check reports the missing file.
        source_lines = target.read_text(encoding="utf-8").splitlines()
        quoted_lines = body.splitlines()
        if len(quoted_lines) > 30:
            errors.append(f"{first_line}: source quotation exceeds 30 lines")
        cursor = max(0, int(refs[0].group("start")) - 1)
        for offset, line in enumerate(quoted_lines):
            if not line.strip() or OMISSION.match(line):
                continue
            if line.lstrip().startswith(("//", "#")) and SOURCE_REF.search(line):
                continue
            original = ANNOTATION.sub("", line)
            try:
                cursor = source_lines.index(original, cursor) + 1
            except ValueError:
                errors.append(
                    f"{first_line + offset}: quoted line absent or out of order "
                    f"in {refs[0].group('path')}: {original!r}"
                )
    return checked, errors


def manuscript_files() -> list[Path]:
    files = [REPO / "preface.md"]
    files.extend(sorted((REPO / "chapters").glob("*/chapter.md")))
    files.extend(sorted((REPO / "appendix").glob("*.md")))
    return [path for path in files if path.is_file()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(os.environ.get("LITERT_LM_SOURCE", DEFAULT_SOURCE)),
        help="LiteRT-LM v0.17.0 source worktree",
    )
    parser.add_argument(
        "--litert-source", type=Path,
        default=Path(os.environ.get("LITERT_SOURCE", DEFAULT_LITERT_SOURCE)),
        help="LiteRT checkout pinned by LiteRT-LM v0.17.0",
    )
    args = parser.parse_args()
    litert_source = args.litert_source.resolve()
    try:
        litert_commit = subprocess.run(
            ["git", "-C", str(litert_source), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"cannot read frozen LiteRT checkout: {litert_source}") from error
    if litert_commit != EXPECTED_LITERT_COMMIT:
        raise SystemExit(f"LiteRT commit mismatch: expected {EXPECTED_LITERT_COMMIT}, got {litert_commit}")

    source = args.source.resolve()
    if not source.is_dir():
        raise SystemExit(f"source worktree not found: {source}")

    source_commit: str | None = None
    if (source / ".git").exists():
        try:
            source_commit = subprocess.run(
                ["git", "-C", str(source), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise SystemExit(f"cannot resolve source commit: {error}") from error
        if source_commit != EXPECTED_SOURCE_COMMIT:
            raise SystemExit(
                "source worktree is not LiteRT-LM v0.17.0: "
                f"expected {EXPECTED_SOURCE_COMMIT}, got {source_commit}"
            )

    errors: list[str] = []
    checked: set[tuple[str, int, int]] = set()
    line_counts: dict[Path, int] = {}
    excerpt_count = 0

    for manuscript in manuscript_files():
        relative = manuscript.relative_to(REPO)
        count, excerpt_errors = check_source_excerpts(
            manuscript.read_text(encoding="utf-8"), source, litert_source
        )
        excerpt_count += count
        errors.extend(f"{relative}:{error}" for error in excerpt_errors)
        in_fence = False
        for number, line in enumerate(
            manuscript.read_text(encoding="utf-8").splitlines(), 1
        ):
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                stripped = line.strip()
                if stripped.startswith(("// ", "# ")):
                    for ref in SOURCE_REF.finditer(stripped):
                        ref_path = ref.group("path")
                        if ref_path.startswith(EXTERNAL_PREFIXES):
                            continue
                        start = int(ref.group("start"))
                        end = int(ref.group("end") or start)
                        key = (ref_path, start, end)
                        checked.add(key)
                        target = source_target(ref_path, source, litert_source)
                        if not target.is_file():
                            errors.append(
                                f"{relative}:{number}: missing source file {ref_path}"
                            )
                            continue
                        if target not in line_counts:
                            line_counts[target] = sum(
                                1 for _ in target.open(encoding="utf-8", errors="ignore")
                            )
                        if end > line_counts[target] or start < 1:
                            errors.append(
                                f"{relative}:{number}: line range out of bounds "
                                f"{ref_path}:{start}-{end}"
                            )
                continue
            for span_match in CODE_SPAN.finditer(line):
                span = span_match.group(1)
                shorthand = SHORTHAND_REF.fullmatch(span.strip())
                if shorthand:
                    errors.append(
                        f"{relative}:{number}: incomplete source anchor `{span}`"
                    )
                    continue
                for ref in SOURCE_REF.finditer(span):
                    ref_path = ref.group("path")
                    if ref_path.startswith(EXTERNAL_PREFIXES):
                        continue
                    start = int(ref.group("start"))
                    end = int(ref.group("end") or start)
                    key = (ref_path, start, end)
                    checked.add(key)
                    target = source_target(ref_path, source, litert_source)
                    if not target.is_file():
                        candidates = sorted(source.rglob(ref_path))
                        suggestion = ""
                        if len(candidates) == 1:
                            suggestion = (
                                "; use " + str(candidates[0].relative_to(source))
                            )
                        elif candidates:
                            suggestion = f"; {len(candidates)} suffix matches"
                        errors.append(
                            f"{relative}:{number}: source file not found: "
                            f"{ref_path}{suggestion}"
                        )
                        continue
                    if end < start:
                        errors.append(
                            f"{relative}:{number}: reversed source range "
                            f"{ref_path}:{start}-{end}"
                        )
                        continue
                    if target not in line_counts:
                        with target.open(encoding="utf-8", errors="replace") as handle:
                            line_counts[target] = sum(1 for _ in handle)
                    if end > line_counts[target]:
                        errors.append(
                            f"{relative}:{number}: source line out of range: "
                            f"{ref_path}:{start}-{end} "
                            f"(file has {line_counts[target]} lines)"
                        )

    dependency_paths = sorted({
        path.removeprefix("LiteRT/")
        for path, _, _ in checked if path.startswith("LiteRT/")
    })
    for path in dependency_paths:
        original = subprocess.run(
            ["git", "-C", str(litert_source), "show", f"{EXPECTED_LITERT_COMMIT}:{path}"],
            capture_output=True,
        )
        target = litert_source / path
        if original.returncode != 0 or not target.is_file():
            errors.append(f"LiteRT source file missing from checkout or commit: {path}")
            continue
        if target.read_bytes() != original.stdout:
            errors.append(f"LiteRT quoted source differs from frozen commit: {path}")
    if errors:
        raise SystemExit("\n".join(errors))
    commit_note = f" @ {source_commit[:12]}" if source_commit else ""
    print(
        f"source anchor check passed: {len(checked)} unique anchors, "
        f"{excerpt_count} verbatim quotations "
        f"against {source}{commit_note}; LiteRT @ {litert_commit[:12]}"
    )


if __name__ == "__main__":
    main()
