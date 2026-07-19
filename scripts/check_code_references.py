#!/usr/bin/env python3
"""Check LiteRT-LM source anchors used by the manuscript.

The check is intentionally mechanical: every ``path:line`` anchor must name an
existing file in the frozen source worktree, and every referenced line must be
inside that file.  It does not decide whether the cited code supports the
surrounding prose; that remains part of the chapter fact-check pass.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = REPO.parent / "LiteRT-LM-v0.13.1"
EXPECTED_SOURCE_COMMIT = "a0afb5a56acd106b23a2b2385b8469834dc268c0"
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
        help="LiteRT-LM v0.13.1 source worktree",
    )
    args = parser.parse_args()
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
                "source worktree is not LiteRT-LM v0.13.1: "
                f"expected {EXPECTED_SOURCE_COMMIT}, got {source_commit}"
            )

    errors: list[str] = []
    checked: set[tuple[str, int, int]] = set()
    line_counts: dict[Path, int] = {}

    for manuscript in manuscript_files():
        relative = manuscript.relative_to(REPO)
        for number, line in enumerate(
            manuscript.read_text(encoding="utf-8").splitlines(), 1
        ):
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
                    target = source / ref_path
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

    if errors:
        raise SystemExit("\n".join(errors))
    commit_note = f" @ {source_commit[:12]}" if source_commit else ""
    print(
        f"source anchor check passed: {len(checked)} unique anchors "
        f"against {source}{commit_note}"
    )


if __name__ == "__main__":
    main()
