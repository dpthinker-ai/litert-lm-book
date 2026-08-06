#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""校验成书 PDF 的侧边栏大纲（书签/目录）是否完整，缺失即失败退出。

用法：check_pdf_outline.py <book.pdf>

这是「每次编译 PDF 都必须有侧边栏目录」的硬性门禁：
大纲非空、条目数不低于最低下限、包含节级（三级）条目、页码全部在范围内。
任一项不满足则以非零码退出，make_pdf.sh 会因此终止，不会产出缺目录的 PDF。
依赖：PyMuPDF(fitz)。"""
import sys
import fitz

MIN_ENTRIES = 24  # 封面+目录+前言+4 分部+11 章+尾声+5 附录 = 24（两级结构的最低下限）
REQUIRE_LEVEL3 = True  # 侧边栏必须含节级条目（与打印目录一致）


def main() -> int:
    if len(sys.argv) != 2:
        print("用法：check_pdf_outline.py <book.pdf>", file=sys.stderr)
        return 2
    path = sys.argv[1]
    doc = fitz.open(path)
    toc = doc.get_toc()
    errors = []
    if not toc:
        errors.append("PDF 大纲为空：侧边栏没有目录。finalize_pdf.py 未写入书签？")
    else:
        if len(toc) < MIN_ENTRIES:
            errors.append(f"PDF 大纲条目数不足：{len(toc)} < {MIN_ENTRIES}")
        if REQUIRE_LEVEL3 and not any(item[0] >= 3 for item in toc):
            errors.append("PDF 大纲缺少节级（三级）条目，侧边栏目录不完整")
        out_of_range = [item for item in toc if not (1 <= item[2] <= doc.page_count)]
        if out_of_range:
            errors.append(f"PDF 大纲存在越界页码 {len(out_of_range)} 条（首条：{out_of_range[0]}）")
    if errors:
        for e in errors:
            print(f"error: {e}", file=sys.stderr)
        print(f"error: 侧边栏目录校验失败：{path}", file=sys.stderr)
        return 1
    n3 = sum(1 for item in toc if item[0] == 3)
    n4 = sum(1 for item in toc if item[0] == 4)
    print(f"OK: 侧边栏目录 {len(toc)} 条（三级 {n3}、四级 {n4}），页码全部在范围内：{path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
