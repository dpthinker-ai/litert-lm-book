#!/usr/bin/env python3
"""Paginate a continuous WKWebView PDF into A4 pages.

The companion JSON contains DOM coordinates for unit starts and blocks that
should not be cut. The source remains vector; each clipped segment is embedded
as a Form XObject, so text extraction, MathJax SVG and book SVGs stay sharp.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
from pathlib import Path

import fitz

from finalize_pdf import UNITS as FINAL_PDF_UNITS


EPSILON = 0.75

# content_bottom 边界检查的容差（px）：WebKit 按整像素落行，小数行距
# （如 1.7 × 13.3px = 22.6px）会在连续 PDF 总高度上累计亚像素误差，
# 允许一行级的高度差；真正的截断事故（字体/图片未加载）远超此量级。
CONTENT_BOTTOM_TOLERANCE = 48.0

# keep 块拆分校验的细条容差（px）：切点距块边界 3px 以内视为不可见的
# padding/margin 细条（如封面底与下一单元起点的亚像素重叠），不判为拆分；
# 真实拆分（切在块中部）远超此量级。
SPLIT_SLIVER = 3.0

# 允许在行边界跨页拆分的 keep 块类型（其余整块不拆）
SPLITTABLE_TYPES = {"p", "pre"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="continuous WebKit PDF")
    parser.add_argument("metadata", type=Path, help="DOM layout JSON")
    parser.add_argument("output", type=Path, help="A4 raw PDF")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def ordered_note_keys(
    references: list[dict], start: float, end: float
) -> list[str]:
    keys = []
    seen = set()
    for reference in references:
        y = float(reference["y"])
        if y + EPSILON < start or y >= end - EPSILON:
            continue
        key = reference["key"]
        if key not in seen:
            keys.append(key)
            seen.add(key)
    return keys


def adjust_boundary(
    start: float,
    end: float,
    unit_starts: list[float],
    ranges: list[tuple[float, float, str]],
) -> tuple[float, str | None]:
    reason = None
    unit_index = bisect.bisect_right(unit_starts, start + EPSILON)
    if unit_index < len(unit_starts):
        unit_y = unit_starts[unit_index]
        if unit_y < end - EPSILON:
            end = unit_y
            reason = "unit"

    # A range crosses the proposed cut if it starts on this page and ends
    # after the cut. Choose the earliest such start, including overlapping
    # heading-plus-following-block and paragraph ranges.
    # 段落（p）与代码块（pre）允许在行边界跨页拆开：仅当当前页剩余空间
    # 放不下其高度的 30% 时才整块推入下一页，避免页底大片空白；
    # 图、表格、列表、标题、侧栏等保持整块不拆（或按行/条目拆）。
    while True:
        conflicts = [
            (top, bottom, typ) for top, bottom, typ in ranges
            if top > start + EPSILON and top < end - EPSILON and
            bottom > end + EPSILON
        ]
        if not conflicts:
            break
        keep_candidates = [
            top for top, bottom, typ in conflicts
            if typ not in SPLITTABLE_TYPES or
            end - top < (bottom - top) * 0.3
        ]
        if not keep_candidates:
            break
        keep_end = min(keep_candidates)
        if keep_end >= end - EPSILON:
            break
        end = keep_end
        reason = "keep"
    return end, reason


def snap_candidates(
    line_bottoms: list[float], ranges: list[tuple[float, float, str]]
) -> list[float]:
    """页尾对齐候选：文本行底线 ∪ 不可拆 keep 块的顶/底。

    纯文本行吸附在「无文本区域」（大图、SVG、空白）前会失效——图块内部没有
    行底线，切点会被留在图前造成残页。块的顶/底是同样合法的页边界。"""
    cands = set(line_bottoms)
    for top, bottom, typ in ranges:
        if typ in SPLITTABLE_TYPES:
            continue
        cands.add(top)
        cands.add(bottom)
    return sorted(cands)


def choose_pages(
    metadata: dict, total_height: float, line_bottoms: list[float]
) -> tuple[list[dict], dict]:
    usable = float(metadata["contentHeight"])
    unit_starts = sorted(float(item["y"]) for item in metadata["unitStarts"])
    ranges = [
        (float(item["top"]), float(item["bottom"]), item["type"])
        for item in metadata["keepRanges"]
        if float(item["height"]) <= usable + EPSILON
    ]
    references = sorted(
        metadata.get("footnoteRefs", []), key=lambda item: float(item["y"])
    )
    notes = {}
    for item in metadata.get("footnotes", []):
        key = item["key"]
        if key in notes:
            raise RuntimeError(f"duplicate footnote bank item: {key}")
        notes[key] = item
    for reference in references:
        if reference["key"] not in notes:
            raise RuntimeError(
                f"unknown footnote definition: {reference['key']}"
            )
    gap = float(metadata.get("footnoteGap", 0.0))
    max_footnotes = float(metadata.get("maxFootnoteHeight", usable))

    def footnote_height(keys: list[str]) -> float:
        if not keys:
            return 0.0
        return gap + sum(float(notes[key]["height"]) for key in keys)

    pages = []
    stats = {"unit_cuts": 0, "keep_cuts": 0, "footnote_cuts": 0}
    start = 0.0
    cands = snap_candidates(line_bottoms, ranges)
    while start < total_height - EPSILON:
        # A unit's trailing margin can be slightly taller than the previous
        # page's remaining space. Skip only DOM-confirmed empty space instead
        # of emitting a page for it; figures and other non-text blocks remain.
        for unit in metadata['unitStarts']:
            previous_bottom = unit.get('previousBottom')
            if (previous_bottom is not None and
                    float(previous_bottom) - EPSILON <= start < float(unit['y'])):
                start = float(unit['y'])
        nominal = min(total_height, start + usable)
        # 页尾对齐到行边界或不可拆块的顶/底，避免切在行中间或图前残页
        # Snapping may retreat but must not advance: two tolerance-based
        # snaps can otherwise accumulate beyond one page's height tolerance.
        idx = bisect.bisect_right(cands, nominal) - 1
        if idx >= 0 and cands[idx] > start + EPSILON:
            nominal = cands[idx]
        end, reason = adjust_boundary(start, nominal, unit_starts, ranges)

        # If the provisional page contains too many notes, move the cut before
        # the first reference that would cross the limit. adjust_boundary then
        # rolls the cut back to the start of its paragraph, row, or list item.
        # A block that cannot be moved intact will fail validation below.
        while True:
            provisional_keys = ordered_note_keys(references, start, end)
            reserved = footnote_height(provisional_keys)
            if reserved <= max_footnotes + EPSILON:
                break
            seen = set()
            running = gap
            overflow_reference = None
            for reference in references:
                y = float(reference["y"])
                if y + EPSILON < start or y >= end - EPSILON:
                    continue
                key = reference["key"]
                if key in seen:
                    continue
                seen.add(key)
                running += float(notes[key]["height"])
                if running > max_footnotes + EPSILON:
                    overflow_reference = reference
                    break
            if overflow_reference is None:
                raise RuntimeError("cannot locate overflowing footnote reference")
            adjusted, adjusted_reason = adjust_boundary(
                start,
                min(end, float(overflow_reference["y"])),
                unit_starts,
                ranges,
            )
            if adjusted <= start + EPSILON:
                joined = ", ".join(provisional_keys)
                raise RuntimeError(
                    f"footnotes exceed page limit at source y={start:.3f}: {joined}"
                )
            end = adjusted
            reason = adjusted_reason or "footnote"

        if reserved > 0:
            note_limit = min(end, start + usable - reserved)
            adjusted, adjusted_reason = adjust_boundary(
                start, note_limit, unit_starts, ranges
            )
            if adjusted < end - EPSILON:
                end = adjusted
                reason = adjusted_reason or "footnote"

        # 脚注溢出等调整路径产生的页尾同样对齐到行边界。
        # 对齐后复用 adjust_boundary 的迭代回退（会循环处理嵌套的 keep 冲突，
        # 如「标题块 ⊃ 列表顶」的逐级回退）；仅当对齐位置落入「跨越页首的
        # 不可拆块」（含亚像素重叠）时改推进到块底，且不得越过单元起点。
        idx = bisect.bisect_right(cands, end) - 1
        if idx >= 0 and cands[idx] > start + EPSILON:
            snapped = cands[idx]
            span_top = None
            for top, bottom, typ in ranges:
                if typ in SPLITTABLE_TYPES:
                    continue
                if top <= start + EPSILON and top < snapped < bottom:
                    span_top = (top, bottom)
                    break
            if span_top is not None:
                top, bottom = span_top
                nxt = min(bottom, start + usable)
                for u in unit_starts:
                    if u > snapped - EPSILON and u < nxt - EPSILON:
                        nxt = u
                        break
                end = nxt
            else:
                end, _ = adjust_boundary(start, snapped, unit_starts, ranges)

        if end <= start + EPSILON:
            near = [(round(t, 1), round(b, 1), ty) for t, b, ty in ranges
                    if t <= start + 50 and b >= start - 50]
            raise RuntimeError(
                f"pagination made no progress at source y={start:.3f}: "
                f"snapped={snapped:.3f} nominal={nominal:.3f} "
                f"near_ranges={near}"
            )

        page_keys = ordered_note_keys(references, start, end)
        page_footnote_height = footnote_height(page_keys)
        if page_footnote_height > max_footnotes + EPSILON:
            joined = ", ".join(page_keys)
            raise RuntimeError(
                f"footnotes exceed page limit at source y={start:.3f}: {joined}"
            )
        pages.append({
            "start": start,
            "end": end,
            "footnotes": page_keys,
            "footnoteHeight": page_footnote_height,
        })
        if reason == "unit":
            stats["unit_cuts"] += 1
        elif reason == "keep":
            stats["keep_cuts"] += 1
        elif reason == "footnote":
            stats["footnote_cuts"] += 1
        start = end

    reference_count = 0
    placement_count = 0
    distinct_placements = set()
    for page in pages:
        page_references = [
            reference for reference in references
            if float(reference["y"]) + EPSILON >= float(page["start"]) and
            float(reference["y"]) < float(page["end"]) - EPSILON
        ]
        reference_count += len(page_references)
        placement_count += len(page["footnotes"])
        distinct_placements.update(page["footnotes"])
    stats["footnote_references"] = reference_count
    stats["footnote_placements"] = placement_count
    stats["same_page_dedup"] = reference_count - placement_count
    stats["cross_page_repeats"] = placement_count - len(distinct_placements)

    return pages, stats


def validate_pages(pages: list[dict], metadata: dict) -> None:
    usable = float(metadata["contentHeight"])
    if metadata.get("oversized"):
        raise RuntimeError(f"oversized unbreakable blocks: {metadata['oversized']}")
    expected_units = len(FINAL_PDF_UNITS)
    actual_units = len(metadata.get("unitStarts", []))
    if actual_units != expected_units:
        raise RuntimeError(
            f"expected {expected_units} unit starts after the cover, "
            f"found {actual_units}"
        )
    if int(metadata.get("mathContainers", 0)) <= 0:
        raise RuntimeError("expected rendered MathJax containers")

    boundaries = sorted({float(page[edge]) for page in pages
                         for edge in ('start', 'end')})
    for page in pages:
        body_height = float(page["end"]) - float(page["start"])
        total_height = body_height + float(page["footnoteHeight"])
        if total_height > usable + EPSILON:
            raise RuntimeError("page content exceeds the A4 printable height")

    for item in metadata["keepRanges"]:
        top = float(item["top"])
        bottom = float(item["bottom"])
        height = float(item["height"])
        if height > usable + EPSILON:
            continue
        for cut in boundaries[1:-1]:
            if top + SPLIT_SLIVER < cut < bottom - SPLIT_SLIVER:
                if item["type"] in {"p", "pre"}:
                    continue  # 段落与代码块允许在行边界跨页拆分
                page = next(item for item in pages
                            if abs(item["end"] - cut) <= EPSILON or
                            abs(item["start"] - cut) <= EPSILON)
                nearby_units = [
                    unit["text"] for unit in metadata["unitStarts"]
                    if abs(float(unit["y"]) - cut) <= EPSILON
                ]
                raise RuntimeError(
                    f"cut at {cut:.3f} splits {item['type']} "
                    f"range {top:.3f}-{bottom:.3f}; "
                    f"page={page['start']:.3f}-{page['end']:.3f}, "
                    f"footnotes={page['footnotes']}, units={nearby_units}, "
                    f"text={item.get('text', '')!r}"
                )

    cuts = boundaries[1:-1]
    for unit in metadata["unitStarts"]:
        y = float(unit["y"])
        if not any(abs(cut - y) <= EPSILON for cut in cuts):
            raise RuntimeError(f"unit does not start a page: {unit['text']}")


def source_offsets(source: fitz.Document) -> list[float]:
    offsets = [0.0]
    for page in source:
        offsets.append(offsets[-1] + page.rect.height)
    return offsets


def copy_uri_links(
    output_page: fitz.Page,
    source_page: fitz.Page,
    clip: fitz.Rect,
    destination: fitz.Rect,
) -> None:
    scale_x = destination.width / clip.width
    scale_y = destination.height / clip.height
    for link in source_page.get_links():
        if link.get("kind") != fitz.LINK_URI or not link.get("uri"):
            continue
        source_rect = fitz.Rect(link["from"])
        overlap = source_rect & clip
        if overlap.is_empty:
            continue
        mapped = fitz.Rect(
            destination.x0 + (overlap.x0 - clip.x0) * scale_x,
            destination.y0 + (overlap.y0 - clip.y0) * scale_y,
            destination.x0 + (overlap.x1 - clip.x0) * scale_x,
            destination.y0 + (overlap.y1 - clip.y0) * scale_y,
        )
        output_page.insert_link({
            "kind": fitz.LINK_URI,
            "from": mapped,
            "uri": link["uri"],
        })


def place_source_range(
    output_page: fitz.Page,
    source: fitz.Document,
    offsets: list[float],
    page_width: float,
    start: float,
    end: float,
    destination_y: float,
    scale: float,
) -> None:
    if start < -EPSILON or end > offsets[-1] + EPSILON or end <= start:
        raise RuntimeError(f"invalid source range {start:.3f}-{end:.3f}")
    cursor = start
    while cursor < end - EPSILON:
        source_index = bisect.bisect_right(offsets, cursor + EPSILON) - 1
        source_index = min(source_index, source.page_count - 1)
        segment_end = min(end, offsets[source_index + 1])
        local_start = cursor - offsets[source_index]
        local_end = segment_end - offsets[source_index]
        clip = fitz.Rect(0, local_start, page_width, local_end)
        segment_y = destination_y + (cursor - start) * scale
        destination = fitz.Rect(
            0,
            segment_y,
            page_width * scale,
            segment_y + clip.height * scale,
        )
        output_page.show_pdf_page(
            destination,
            source,
            source_index,
            clip=clip,
            keep_proportion=False,
        )
        copy_uri_links(output_page, source[source_index], clip, destination)
        cursor = segment_end


def render_pages(
    source: fitz.Document,
    metadata: dict,
    pages: list[dict],
) -> fitz.Document:
    page_width = float(metadata["pageWidth"])
    expected_height = float(metadata["height"])
    offsets = source_offsets(source)
    if abs(offsets[-1] - expected_height) > 1.0:
        raise RuntimeError(
            f"source height {offsets[-1]:.3f} != DOM height {expected_height:.3f}"
        )
    for page in source:
        if abs(page.rect.width - page_width) > 0.5:
            raise RuntimeError("continuous PDF pages have inconsistent widths")

    a4_width, a4_height = fitz.paper_size("a4")
    scale = a4_width / page_width
    top_margin = float(metadata["pageTop"]) * scale
    side_margin = float(metadata["pageSide"]) * scale
    usable = float(metadata["contentHeight"])
    gap = float(metadata.get("footnoteGap", 0.0))
    notes = {item["key"]: item for item in metadata.get("footnotes", [])}
    output = fitz.open()

    for spec in pages:
        start = float(spec["start"])
        end = float(spec["end"])
        page = output.new_page(width=a4_width, height=a4_height)
        place_source_range(
            page, source, offsets, page_width, start, end, top_margin, scale
        )

        if spec["footnotes"]:
            footnote_top = top_margin + (
                usable - float(spec["footnoteHeight"])
            ) * scale
            line_y = footnote_top + 3 * scale
            page.draw_line(
                (side_margin, line_y),
                (side_margin + 90 * scale, line_y),
                color=(0.55, 0.58, 0.62),
                width=0.55,
            )
            cursor_y = footnote_top + gap * scale
            for key in spec["footnotes"]:
                note = notes[key]
                # Every referencing page carries the full source; pagination
                # already reserves the full note height, including repeats.
                place_source_range(
                    page, source, offsets, page_width,
                    float(note["top"]), float(note["bottom"]), cursor_y, scale,
                )
                cursor_y += float(note["height"]) * scale

    return output


def main() -> None:
    args = parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    source = fitz.open(args.source)
    total_height = sum(page.rect.height for page in source)
    # 分页终点 = 最后一个正文 keep 块的底部。DOM 测得的 bodyBottom 可能
    # 包含页尾的脚注银行区（银行区只供脚注切片引用，不得当正文分页），
    # 因此以 keep 底部为硬上限。
    keep_bottom = max(
        [float(item["bottom"]) for item in metadata["keepRanges"]] +
        [float(item["y"]) for item in metadata["unitStarts"]]
    )
    content_bottom = min(
        float(metadata.get("bodyBottom") or keep_bottom),
        keep_bottom,
        total_height,
    )
    if content_bottom > total_height + CONTENT_BOTTOM_TOLERANCE:
        raise RuntimeError("DOM content extends beyond the continuous PDF")
    # 容差内的亚像素/整像素差：把 DOM 测得的底部钳到画布高度，
    # 超出的部分至多一行残线，分页按画布实际高度收尾。
    content_bottom = min(content_bottom, total_height)
    # 文本行底线（源 PDF 全局坐标），供页尾对齐使用
    line_bottoms: list[float] = []
    offsets = source_offsets(source)
    for pno, page in enumerate(source):
        dct = page.get_text("dict")
        for block in dct["blocks"]:
            for line in block.get("lines", []):
                txt = "".join(span["text"] for span in line.get("spans", [])).strip()
                if txt:
                    line_bottoms.append(line["bbox"][3] + offsets[pno])
    line_bottoms.sort()
    pages, stats = choose_pages(metadata, content_bottom, line_bottoms)
    validate_pages(pages, metadata)
    output = render_pages(source, metadata, pages)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.save(args.output, deflate=True, garbage=4)

    fills = [
        (
            float(page["end"]) - float(page["start"]) +
            float(page["footnoteHeight"])
        ) / float(metadata["contentHeight"])
        for page in pages
    ]
    print(
        f"paginated {source.page_count} continuous pages into "
        f"{output.page_count} A4 pages; "
        f"unit cuts={stats['unit_cuts']}, keep cuts={stats['keep_cuts']}, "
        f"footnote cuts={stats['footnote_cuts']}, "
        f"footnote refs={stats['footnote_references']}, "
        f"placements={stats['footnote_placements']}, "
        f"same-page dedup={stats['same_page_dedup']}, "
        f"cross-page repeats={stats['cross_page_repeats']}, "
        f"mean fill={sum(fills) / len(fills):.1%}"
    )
    if args.debug:
        print(
            f"source height={total_height:.3f}, "
            f"min fill={min(fills):.1%}, max fill={max(fills):.1%}, "
            f"external links={sum(len(page.get_links()) for page in output)}"
        )


if __name__ == "__main__":
    main()
