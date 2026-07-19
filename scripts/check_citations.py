#!/usr/bin/env python3
"""Validate semantic citations and source de-duplication across all appendices."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit


REPO = Path(__file__).resolve().parent.parent
SUMMARY_ENTRY = re.compile(r"^\s*-?\s*\[[^]]+]\(([^)]+\.md)\)\s*$")
FOOTNOTE_DEF = re.compile(r"^\[\^([^]]+)]\:\s*(.*)$")
FOOTNOTE_REF = re.compile(r"\[\^([^]]+)]")
URL = re.compile(r"https?://[^\s<>()]+(?:\([^\s<>()]*\)[^\s<>()]*)*")
ACCESS_DATE = re.compile(r"访问(?:日期)?[：:\s]*20\d{2}-\d{2}-\d{2}")
DATE = re.compile(r"20\d{2}-\d{2}-\d{2}")
ISSUE_URL = re.compile(
    r"github\.com/([^/\s]+)/([^/\s]+)/issues/(\d+)", re.IGNORECASE
)
ISSUE_MENTION = re.compile(r"LiteRT-LM#(\d+)")
NONCANONICAL_ISSUE = re.compile(r"(?:\bissue\s*#|`#)(\d+)`?", re.IGNORECASE)
DOI = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
TITLE_MARKUP = re.compile(r"(?:\[[^]]{3,}]\(https?://|\*[^*\n]{3,}\*)")
SOURCE_PREFIX = re.compile(r"^[^，,\n]{2,}[，,]")
FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
INDIRECT_SOURCE_APPENDIX = re.compile(
    r"(?:出处|来源|参考文献|外部文献|文献)[^。！？\n]{0,16}"
    r"(?:见|详见|参见|列于|收录于|汇总于|汇总在)[^。！？\n]{0,8}附录\s*[A-Z]"
    r"|(?:见|详见|参见)附录\s*[A-Z][^。！？\n]{0,12}"
    r"(?:出处|来源|参考文献|外部文献|文献)"
)
APPENDIX_DIR = Path("appendix")
LITERT_LM_REPO = ("google-ai-edge", "litert-lm")
TRACKING_QUERY_KEYS = {"fbclid", "gclid"}


def manuscript_paths() -> list[Path]:
    paths: list[Path] = []
    for line in (REPO / "SUMMARY.md").read_text(encoding="utf-8").splitlines():
        match = SUMMARY_ENTRY.match(line)
        if match:
            paths.append(Path(match.group(1)))
    return paths


def is_appendix(path: Path) -> bool:
    return bool(path.parts) and path.parts[0] == APPENDIX_DIR.name


def normalize_url(raw: str) -> str:
    raw = raw.rstrip(".,;，。；：]")
    while raw.endswith(")") and raw.count(")") > raw.count("("):
        raw = raw[:-1]
    parts = urlsplit(raw)
    host = parts.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if host in {"doi.org", "dx.doi.org"}:
        host = "doi.org"
        path = unquote(parts.path)
    else:
        path = parts.path
    path = path.rstrip("/") or "/"
    query_items = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_")
        and key.lower() not in TRACKING_QUERY_KEYS
    ]
    scheme = "https" if parts.scheme.lower() in {"http", "https"} else parts.scheme
    return urlunsplit((scheme, host, path, urlencode(sorted(query_items)), ""))


def urls(text: str) -> set[str]:
    return {normalize_url(item) for item in URL.findall(text)}


def issue_keys(text: str) -> set[tuple[str, str, str]]:
    return {
        (owner.lower(), repo.lower(), issue_number)
        for owner, repo, issue_number in ISSUE_URL.findall(text)
    }


def format_issue(issue_key: tuple[str, str, str]) -> str:
    owner, repo, issue_number = issue_key
    return f"{owner}/{repo}#{issue_number}"


def body_lines(text: str):
    """Yield non-code, non-definition lines with source line numbers."""
    fence_char = ""
    fence_length = 0
    in_definition = False
    for line_number, line in enumerate(text.splitlines(), 1):
        fence = FENCE.match(line)
        if fence:
            marker = fence.group(1)
            if not fence_char:
                fence_char = marker[0]
                fence_length = len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_length:
                fence_char = ""
                fence_length = 0
            in_definition = False
            continue
        if fence_char or line.startswith("    "):
            continue
        if FOOTNOTE_DEF.match(line):
            in_definition = True
            continue
        if in_definition and line.startswith("    "):
            continue
        in_definition = False
        yield line_number, line


def footnotes(text: str) -> tuple[dict[str, str], list[str], list[str]]:
    definitions: dict[str, str] = {}
    duplicate_definitions: list[str] = []
    references: list[str] = []
    lines = text.splitlines()
    fence_char = ""
    fence_length = 0

    for index, line in enumerate(lines):
        fence = FENCE.match(line)
        if fence:
            marker = fence.group(1)
            if not fence_char:
                fence_char = marker[0]
                fence_length = len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_length:
                fence_char = ""
                fence_length = 0
            continue
        if fence_char or line.startswith("    "):
            continue
        match = FOOTNOTE_DEF.match(line)
        if match:
            footnote_id, body = match.groups()
            continuation: list[str] = []
            cursor = index + 1
            while cursor < len(lines) and lines[cursor].startswith("    "):
                continuation.append(lines[cursor].strip())
                cursor += 1
            if footnote_id in definitions:
                duplicate_definitions.append(footnote_id)
            definitions[footnote_id] = " ".join([body, *continuation]).strip()
            continue
        references.extend(FOOTNOTE_REF.findall(line))

    return definitions, references, duplicate_definitions


def main() -> None:
    errors: list[str] = []
    global_definitions: dict[str, Path] = {}
    global_reference_count = 0
    main_source_urls: set[str] = set()
    main_issue_keys: set[tuple[str, str, str]] = set()
    main_dois: set[str] = set()
    appendix_sources: dict[
        Path, tuple[set[str], set[tuple[str, str, str]], set[str]]
    ] = {}

    for relative_path in manuscript_paths():
        text = (REPO / relative_path).read_text(encoding="utf-8")
        definitions, references, duplicate_definitions = footnotes(text)
        path_source_urls: set[str] = set()
        path_issue_keys: set[tuple[str, str, str]] = set()
        path_dois: set[str] = set()
        global_reference_count += len(references)

        for footnote_id in duplicate_definitions:
            errors.append(f"{relative_path}: duplicate footnote definition [^{footnote_id}]")

        for footnote_id in definitions:
            previous = global_definitions.get(footnote_id)
            if previous is not None:
                errors.append(
                    f"duplicate footnote id [^{footnote_id}] in {previous} and {relative_path}"
                )
            global_definitions[footnote_id] = relative_path

        defined = set(definitions)
        referenced = set(references)
        for footnote_id in sorted(referenced - defined):
            errors.append(f"{relative_path}: undefined footnote [^{footnote_id}]")
        for footnote_id in sorted(defined - referenced):
            errors.append(f"{relative_path}: unreferenced footnote [^{footnote_id}]")

        for footnote_id, body in definitions.items():
            source_urls = urls(body)
            if not source_urls:
                errors.append(
                    f"{relative_path}: citation footnote [^{footnote_id}] lacks a URL"
                )
            if not SOURCE_PREFIX.search(body) or not TITLE_MARKUP.search(body):
                errors.append(
                    f"{relative_path}: citation footnote [^{footnote_id}] lacks an institution/author or marked title"
                )
            if not ACCESS_DATE.search(body):
                errors.append(
                    f"{relative_path}: external footnote [^{footnote_id}] lacks an access date"
                )
            for _, _, issue_number in ISSUE_URL.findall(body):
                if not re.search(rf"issue\s*#\s*{issue_number}\b", body, re.IGNORECASE):
                    errors.append(
                        f"{relative_path}: issue footnote [^{footnote_id}] lacks issue #{issue_number}"
                    )
                without_access_date = ACCESS_DATE.sub("", body)
                if not DATE.search(without_access_date) or not re.search(
                    r"\*[^*\n]{5,}\*", body
                ):
                    errors.append(
                        f"{relative_path}: issue footnote [^{footnote_id}] lacks its title or publication date"
                    )
            path_source_urls.update(source_urls)
            path_issue_keys.update(issue_keys(body))
            path_dois.update(item.lower().rstrip(".,;") for item in DOI.findall(body))

        if is_appendix(relative_path):
            appendix_sources[relative_path] = (
                path_source_urls,
                path_issue_keys,
                path_dois,
            )
        else:
            main_source_urls.update(path_source_urls)
            main_issue_keys.update(path_issue_keys)
            main_dois.update(path_dois)

        for line_number, line in body_lines(text):
            if INDIRECT_SOURCE_APPENDIX.search(line):
                errors.append(
                    f"{relative_path}:{line_number}: indirect source reference to an appendix remains"
                )
            direct_urls = urls(line)
            if direct_urls:
                errors.append(
                    f"{relative_path}:{line_number}: external URL must be in a citation footnote"
                )
            if DOI.search(line):
                errors.append(
                    f"{relative_path}:{line_number}: external DOI must be in a citation footnote"
                )
            for issue_number in NONCANONICAL_ISSUE.findall(line):
                errors.append(
                    f"{relative_path}:{line_number}: write issue as LiteRT-LM#{issue_number}"
                )
            for issue_number in ISSUE_MENTION.findall(line):
                expected_issue = (*LITERT_LM_REPO, issue_number)
                matching_footnote = any(
                    expected_issue in issue_keys(definitions.get(ref, ""))
                    for ref in FOOTNOTE_REF.findall(line)
                )
                if not matching_footnote:
                    errors.append(
                        f"{relative_path}:{line_number}: LiteRT-LM#{issue_number} lacks a same-line source footnote"
                    )

    appendix_items = sorted(appendix_sources.items(), key=lambda item: str(item[0]))
    for index, (appendix_path, source_sets) in enumerate(appendix_items):
        source_urls, source_issue_keys, dois = source_sets
        for duplicate_url in sorted(main_source_urls & source_urls):
            errors.append(
                f"{appendix_path} duplicates a main-matter source: {duplicate_url}"
            )
        for issue_key in sorted(main_issue_keys & source_issue_keys):
            errors.append(
                f"{appendix_path} duplicates main-matter issue {format_issue(issue_key)}"
            )
        for doi in sorted(main_dois & dois):
            errors.append(f"{appendix_path} duplicates a main-matter DOI: {doi}")

        for other_path, other_source_sets in appendix_items[index + 1 :]:
            other_urls, other_issue_keys, other_dois = other_source_sets
            for duplicate_url in sorted(source_urls & other_urls):
                errors.append(
                    f"source duplicated across {appendix_path} and {other_path}: {duplicate_url}"
                )
            for issue_key in sorted(source_issue_keys & other_issue_keys):
                errors.append(
                    f"issue {format_issue(issue_key)} duplicated across "
                    f"{appendix_path} and {other_path}"
                )
            for doi in sorted(dois & other_dois):
                errors.append(
                    f"DOI {doi} duplicated across {appendix_path} and {other_path}"
                )

    if errors:
        raise SystemExit("\n".join(errors))
    print(
        "citation check passed: "
        f"{len(global_definitions)} footnote definitions, "
        f"{global_reference_count} references, {len(main_source_urls)} main-matter sources, "
        f"{sum(len(items[0]) for items in appendix_sources.values())} appendix source placements"
    )


if __name__ == "__main__":
    main()
