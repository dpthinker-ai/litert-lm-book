#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 Chrome 渲染的 book-raw.pdf 加工成成书 PDF：
   ① 封面后插入「目录」（分部 + 章 + 节 + 小节 + 页码，按需多页，续页不印标题）
   ② 写入可跳转的 PDF 书签/大纲（分部 -> 章 -> 节 -> 小节，四级）
   ③ 每页底部盖「章名 · 页码」页脚（封面/目录页不盖）
渲染仍由 Chrome 负责（MathJax、SVG 正确）；本脚本只做结构化后处理。
用法：finalize_pdf.py <in.pdf> <out.pdf> [--debug]
依赖：PyMuPDF(fitz)。CJK 用 fitz 内置 "china-ss"，失败则回退 macOS PingFang。"""
import sys, re, unicodedata
import fitz  # PyMuPDF

# 结构：顺序即成书顺序。detect=定位该单元起始页的正则（匹配页面最顶一行标题）；
# part=所属部分（部分导页与独立单元为 None）；short=页脚短名。
UNITS = [
    ("前言",                                    r"^\s*前[言⾔]\s*$",           None,                 "前言"),
    ("第一部分 · 约束、指标与系统概览",          r"^\s*第\s*一\s*部分",       None,                         "第一部分 系统概览"),
    ("第 1 章 端侧 LLM 与 LiteRT-LM：三类物理约束与同类运行时对照", r"^\s*第\s*1\s*章",          "第一部分 · 约束、指标与系统概览", "第 1 章 三类物理约束"),
    ("第 2 章 从运行到架构：benchmark、Roofline 与五层视图", r"^\s*第\s*2\s*章",   "第一部分 · 约束、指标与系统概览", "第 2 章 从运行到架构"),
    ("第二部分 · 推理流水线",                    r"^\s*第\s*二\s*部分",       None,                         "第二部分 推理流水线"),
    ("第 3 章 输入侧：从 Engine API 到 token 序列", r"^\s*第\s*3\s*章",       "第二部分 · 推理流水线",           "第 3 章 输入侧"),
    ("第 4 章 Prefill：并行处理提示词",          r"^\s*第\s*4\s*章",          "第二部分 · 推理流水线",           "第 4 章 Prefill"),
    ("第 5 章 Decode：单步解码循环",             r"^\s*第\s*5\s*章",          "第二部分 · 推理流水线",           "第 5 章 Decode"),
    ("第三部分 · 资源管理与推理优化",  r"^\s*第\s*三\s*部分",       None,                         "第三部分 资源管理与优化"),
    ("第 6 章 KV cache：容量、带宽与会话生命周期", r"^\s*第\s*6\s*章",          "第三部分 · 资源管理与推理优化", "第 6 章 KV cache"),
    ("第 7 章 模型文件与权重：量化、容器格式与 LoRA", r"^\s*第\s*7\s*章",     "第三部分 · 资源管理与推理优化", "第 7 章 模型文件与权重"),
    ("第 8 章 异构算力：CPU、GPU 与 NPU",        r"^\s*第\s*8\s*章",          "第三部分 · 资源管理与推理优化", "第 8 章 异构算力"),
    ("第 9 章 一次前向，多个 token：投机解码与 MTP", r"^\s*第\s*9\s*章",       "第三部分 · 资源管理与推理优化", "第 9 章 投机解码"),
    ("第 10 章 端侧 MoE：稀疏激活、专家执行与内存管理", r"^\s*第\s*10\s*章", "第三部分 · 资源管理与推理优化", "第 10 章 端侧 MoE"),
    ("第四部分 · 多模态、工具调用与跨语言集成",  r"^\s*第\s*四\s*部分",       None,                         "第四部分 多模态与集成"),
    ("第 11 章 多模态与工具调用：视觉/音频编码、约束解码与函数调用", r"^\s*第\s*11\s*章",         "第四部分 · 多模态、工具调用与跨语言集成",   "第 11 章 多模态与工具"),
    ("第 12 章 多语言绑定：C ABI、JNI 与 Embind", r"^\s*第\s*12\s*章",         "第四部分 · 多模态、工具调用与跨语言集成",   "第 12 章 多语言绑定"),
    ("尾声 · 实践入口与待验证问题",                r"^\s*尾声",                 None,                 "尾声"),
    ("附录 A 术语表",                            r"^\s*附录\s*A",             None,                 "附录 A 术语表"),
    ("附录 B 代码地图",                          r"^\s*附录\s*B",             None,                 "附录 B 代码地图"),
    ("附录 C 环境搭建与实验复现",                r"^\s*附录\s*C",             None,                 "附录 C 复现指南"),
    ("附录 D 基准数据集",                        r"^\s*附录\s*D",             None,                 "附录 D 基准数据集"),
    ("附录 E 练习提示与参考答案",                r"^\s*附录\s*E",             None,                 "附录 E 练习答案"),
]

PART_TITLES = {
    "第一部分 · 约束、指标与系统概览",
    "第二部分 · 推理流水线",
    "第三部分 · 资源管理与推理优化",
    "第四部分 · 多模态、工具调用与跨语言集成",
}

ACCENT = (0x0b/255, 0x57/255, 0xd0/255)
INK    = (0x1f/255, 0x23/255, 0x28/255)
DIM    = (0x8b/255, 0x94/255, 0x9e/255)


def get_font():
    # 优先嵌入真实字体文件，避免依赖阅读器的 Adobe-GB1/CJK language pack。
    for path in ("/System/Library/Fonts/PingFang.ttc",
                 "/System/Library/Fonts/STHeiti Light.ttc",
                 "/System/Library/Fonts/Supplemental/Songti.ttc"):
        try:
            return fitz.Font(fontfile=path), "book-cjk", path
        except Exception:
            pass
    for name in ("china-ss", "china-s"):
        try:
            return fitz.Font(name), name, None
        except Exception:
            pass
    raise RuntimeError("找不到可用的中文字体")


def page_top_line(page):
    """返回该页最顶部一行的文字与其字号（用于识别章起始页）。"""
    d = page.get_text("dict")
    lines = []
    for b in d.get("blocks", []):
        if b.get("type", 0) != 0:
            continue
        for l in b.get("lines", []):
            spans = l.get("spans", [])
            if not spans:
                continue
            y = min(s["bbox"][1] for s in spans)
            size = max(s["size"] for s in spans)
            txt = "".join(s["text"] for s in spans)
            lines.append((y, size, txt))
    if not lines:
        return "", 0.0
    lines.sort(key=lambda t: t[0])
    # WebKit/PyMuPDF 偶尔会把同一标题同时提取为局部字形行和完整文本行，
    # 两者的 y 坐标只相差几个千分点。选顶部窄带内最长的文本，避免把
    # “附录 E · ……”误识别成只有“附录”的局部行。
    top_y = lines[0][0]
    top_band = [line for line in lines if line[0] <= top_y + 0.5]
    _, size, txt = max(top_band, key=lambda line: (len(line[2]), line[1]))
    # WebKit 的 PDF 字体偶尔把常用汉字编码为康熙部首兼容字（如“一”→“⼀”）。
    # 定位前统一做兼容分解，避免同一标题因字体编码差异而匹配失败。
    return unicodedata.normalize("NFKC", txt), size


def detect_starts(doc):
    """在 raw pdf 里按顺序定位每个单元的起始页（0-based）。单调向后扫，避免误配。"""
    starts = []
    scan = 0
    for title, rx, part, short in UNITS:
        pat = re.compile(rx)
        found = None
        for p in range(scan, doc.page_count):
            top, size = page_top_line(doc[p])
            # 起始页：最顶一行匹配标题正则，且是大字号标题（阈值放宽到 18pt）
            if pat.search(top) and size >= 18:
                found = p
                break
        if found is None:  # 放宽：不看字号再扫一遍
            for p in range(scan, doc.page_count):
                top, _ = page_top_line(doc[p])
                if pat.search(top):
                    found = p
                    break
        if found is None:
            raise RuntimeError(f"定位失败：{title}（正则 {rx}）")
        starts.append(found)
        scan = found + 1
    return starts  # raw 0-based


def draw_toc_pages(doc, font, entries):
    """entries: [(kind, title, printed_no)]  kind in {part, chapter, section, subsection, plain}
    Draws TOC across as many pages as needed. Returns number of TOC pages used.
    排版参照 llm-inference-handbook：分部/章/节/小节四级，字号阶梯 13/11/9.5/8.5，
    长标题与页码之间画点线引导；续页不印标题，从同一上边距直接接排条目。"""
    W, H = doc[0].rect.width, doc[0].rect.height
    ml, mr, mt, mb = 62, 62, 70, 50  # margins
    page_idx = 0  # TOC page index (0 = first TOC page inserted after cover)

    def new_toc_page():
        nonlocal page_idx
        page = doc.new_page(pno=1 + page_idx, width=W, height=H)
        page_idx += 1
        tw = fitz.TextWriter(page.rect)
        if page_idx == 1:
            tw.append((ml, mt), "目录", font=font, fontsize=26)
            page.draw_line((ml, mt + 12), (ml + 70, mt + 12), color=ACCENT, width=2)
            return tw, mt + 46
        return tw, mt  # 续页不印“目录（续）”，直接接排条目

    tw, y = new_toc_page()

    for kind, title, no in entries:
        # Determine layout
        if kind == "part":
            indent, size, dy = ml, 13, 22
            y += 12  # extra gap before part
        elif kind == "chapter":
            indent, size, dy = ml + 16, 11, 20
        elif kind == "section":
            indent, size, dy = ml + 34, 9.5, 19
            y += 2  # slight gap
        elif kind == "subsection":
            indent, size, dy = ml + 52, 8.5, 16
        else:  # plain：前言/尾声/附录
            indent, size, dy = ml, 11, 20

        # Check page overflow
        if y + dy > H - mb:
            tw.write_text(doc[1 + page_idx - 1], color=INK)
            tw, y = new_toc_page()

        # Title
        tw.append((indent, y), title, font=font, fontsize=size)
        # Page number（右对齐）
        num = str(no)
        num_w = font.text_length(num, fontsize=size)
        tw.append((W - mr - num_w, y), num, font=font, fontsize=size)
        # 点线引导：标题右端到页码左端之间
        xs = indent + font.text_length(title, fontsize=size) + 7
        xe = W - mr - num_w - 7
        if xe - xs > 12:
            doc[1 + page_idx - 1].draw_line(
                (xs, y - size * 0.3), (xe, y - size * 0.3),
                color=DIM, width=0.9, dashes="[1] 7", lineCap=1,
            )
        y += dy

    # Write last page
    tw.write_text(doc[1 + page_idx - 1], color=INK)
    return page_idx


def load_section_titles():
    """Parse SUMMARY.md to get section headings per chapter unit index.
    Returns dict: unit_index -> [(section_number, section_title)]"""
    import os
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    summary = os.path.join(repo, "SUMMARY.md")
    sections = {}
    current_ch = None
    current_sec = None
    ch_pattern = re.compile(r"第\s*(\d+)\s*章")
    sec_pattern = re.compile(r"(\d+\.\d+(?:\.\d+)?)")
    for raw in open(summary):
        line = raw.strip()
        m = ch_pattern.search(line)
        if m and "[" in line:
            current_ch = int(m.group(1))
            sections[current_ch] = []
            current_sec = None
            continue
        if raw.startswith("- ["):  # 顶层条目（部分导页、尾声、附录）不属于任何章
            current_ch = None
            continue
        # Extract title from markdown link: "- [title]()" or "  - [title]()"
        link_match = re.match(r"\s*-?\s*\[([^\]]+)\]\(\)", line)
        if not link_match:
            continue
        title = link_match.group(1)
        m = sec_pattern.match(title)
        if m and current_ch:
            sections[current_ch].append((m.group(1), title))
    return sections


def span_size_counts(page, counts):
    """把该页各字号的字符数累加进 counts（用于估计全书正文字号）。"""
    for b in page.get_text("dict").get("blocks", []):
        if b.get("type", 0) != 0:
            continue
        for l in b.get("lines", []):
            for s in l.get("spans", []):
                if s["text"].strip():
                    key = round(s["size"], 1)
                    counts[key] = counts.get(key, 0) + len(s["text"])


def heading_blocks(page, body_size):
    """返回该页字号明显大于正文（≥ 1.15 倍）的文本块及文本行，文字已压缩。
    正文字号须在全部章节页面上统一估计：单页按众数估计会在代码块占比高的页面
    把正文误判为标题。"""
    blocks = []
    for b in page.get_text("dict").get("blocks", []):
        if b.get("type", 0) != 0:
            continue
        spans = [s for l in b.get("lines", []) for s in l.get("spans", [])
                 if s["text"].strip()]
        if spans and max(s["size"] for s in spans) >= body_size * 1.15:
            blocks.append(squash_text("".join(s["text"] for s in spans)))
            # WebKit 可把局部字形行与完整标题放进同一个块；块前缀此时可能
            # 是缺标点的“11 5”，仍须识别后面的完整“11.5 …”文本行。
            for line in b.get("lines", []):
                line_spans = [s for s in line.get("spans", []) if s["text"].strip()]
                if line_spans and max(s["size"] for s in line_spans) >= body_size * 1.15:
                    blocks.append(squash_text("".join(s["text"] for s in line_spans)))
    return blocks


def find_section_pages(doc, chapter_ranges, chapter_unit_indices, raw_starts):
    """按节号在各章页面范围内定位标题所在页。
    只认字号大于正文、且以节号开头的文本块或文本行，不逐字比对标题：WebKit 的 PDF 字体
    会把部分汉字编码成 NFKC 无法还原的部首兼容字（如“⻚”“⻬”），标题里的代码字
    还会被重复提取。打印用的标题文字取自 SUMMARY.md（去掉反引号），其与章内标题
    的一致性由 check_book_consistency.py 在构建时保证。
    Returns list of (unit_index, section_title, page, depth)；depth 为节号里的句点数
    （x.y 为 1，x.y.z 为 2），层级只由节号决定，与标题文字里的句点无关。"""
    results = []
    all_sections = load_section_titles()
    counts = {}
    for start_page, end_page in chapter_ranges.values():
        for p in range(start_page, min(end_page + 1, doc.page_count)):
            span_size_counts(doc[p], counts)
    body_size = max(counts, key=counts.get) if counts else 10.0

    for ch_idx, (start_page, end_page) in chapter_ranges.items():
        title = UNITS[ch_idx][0]
        m = re.search(r"第\s*(\d+)\s*章", title)
        if not m:
            continue
        ch_num = int(m.group(1))
        if ch_num not in all_sections:
            continue
        pages = range(start_page, min(end_page + 1, doc.page_count))
        blocks_by_page = {p: heading_blocks(doc[p], body_size) for p in pages}
        scan_from = start_page  # SUMMARY 顺序即成书顺序：只向后扫，避免图内文字等误配
        for sec_num, sec_title in all_sections[ch_num]:
            # 节号后不能再接数字（排除 1.10 匹配 1.1、6.2.1 匹配 6.2），
            # 但允许紧跟代码字里的句点（如“2.7.litertlm”）。
            pat = re.compile(r"^" + re.escape(sec_num) + r"(?!\d)(?!\.\d)")
            display_title = sec_title.replace("`", "")
            for p in range(scan_from, pages.stop):
                if any(pat.match(b) for b in blocks_by_page[p]):
                    results.append((chapter_unit_indices[ch_idx], display_title, p,
                                    sec_num.count(".")))
                    scan_from = p
                    break
            else:
                raise RuntimeError(f"目录定位失败：{sec_title}；拒绝生成缺少节级书签的 PDF")
    return results


def squash_text(s):
    """NFKC 规范化、去反引号、去全部空白，用于标题与页面文本的宽松匹配。"""
    s = unicodedata.normalize("NFKC", s).replace("`", "")
    return re.sub(r"\s+", "", s)


def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]
    debug = "--debug" in sys.argv
    doc = fitz.open(src)
    font, fname, fontfile = get_font()

    raw_starts = detect_starts(doc)  # 0-based in raw
    if debug:
        print(f"[字体] {fname}   [总页] {doc.page_count}")
        for (u, rs) in zip(UNITS, raw_starts):
            print(f"  raw p{rs:>3}  {u[0]}")

    # Build chapter page ranges for section detection
    chapter_ranges = {}     # unit_index -> (start_page, end_page)
    chapter_unit_indices = {}  # unit_index -> unit_index in UNITS
    for i in range(len(raw_starts)):
        start = raw_starts[i]
        end = raw_starts[i + 1] - 1 if i + 1 < len(raw_starts) else doc.page_count - 1
        if UNITS[i][2] is not None:  # has a part (is a chapter)
            chapter_ranges[i] = (start, end)
            chapter_unit_indices[i] = i

    # Find section pages within chapters
    section_pages = find_section_pages(doc, chapter_ranges, chapter_unit_indices, raw_starts)
    if debug:
        for ui, title, page, depth in section_pages:
            print(f"  section p{page:>3}  {title}")

    # 插入一页目录（在封面之后，index=1）。插入后所有 raw 页后移 1。
    W, H = doc[0].rect.width, doc[0].rect.height
    # 印刷页码：封面=0(不印)、目录=不印、正文从前言起 = raw 0-based 索引本身
    #   （因 cover=raw0, preface=raw1 -> 印刷 1；插目录后物理页= raw+1）
    printed = {i: rs for i, rs in enumerate(raw_starts)}  # unit i -> printed no = raw index
    phys = {i: rs + 1 for i, rs in enumerate(raw_starts)}  # 插目录后物理页(0-based)

    # Build section page mapping: unit_index -> [(section_title, printed_page)]
    section_by_unit = {}
    for ui, title, page, depth in section_pages:
        if ui not in section_by_unit:
            section_by_unit[ui] = []
        section_by_unit[ui].append((title, page, depth))

    # 目录条目
    entries, cur_part = [], None
    for i, (title, rx, part, short) in enumerate(UNITS):
        if title in PART_TITLES:
            entries.append(("part", title, printed[i]))
            cur_part = title
        elif part:
            if part != cur_part:
                raise RuntimeError(f"章节缺少对应的部分导页：{title} -> {part}")
            entries.append(("chapter", title, printed[i]))
            # 节与小节都进打印目录，与 PDF 书签大纲的三、四级一致
            if i in section_by_unit:
                for sec_title, sec_page, depth in section_by_unit[i]:
                    kind = "subsection" if depth >= 2 else "section"
                    entries.append((kind, sec_title, sec_page))
        else:
            cur_part = None
            entries.append(("plain", title, printed[i]))

    toc_pages = draw_toc_pages(doc, font, entries)

    # Adjust phys: N TOC pages inserted instead of original 1-page assumption.
    # phys was raw+1; now it should be raw + toc_pages.
    for i in phys:
        phys[i] = raw_starts[i] + toc_pages

    # 书签/大纲（fitz 用 1-based 物理页）。分部一级、章二级、节三级、小节四级；
    # 无部单元一级。与打印目录共用 section_by_unit，保证两侧一致。
    outline, cur_part = [], None
    outline.append([1, "封面", 1])
    outline.append([1, "目录", 2])
    for i, (title, rx, part, short) in enumerate(UNITS):
        tgt = phys[i] + 1  # 1-based
        if title in PART_TITLES:
            outline.append([1, title, tgt])
            cur_part = title
        elif part:
            if part != cur_part:
                raise RuntimeError(f"章节缺少对应的部分导页：{title} -> {part}")
            outline.append([2, title, tgt])
            if i in section_by_unit:
                for sec_title, sec_page, depth in section_by_unit[i]:
                    level = 4 if depth >= 2 else 3
                    outline.append([level, sec_title, sec_page + toc_pages + 1])
        else:
            cur_part = None
            outline.append([1, title, tgt])
    doc.set_toc(outline)

    # 页脚：每页底部居中「章名 · 页码」。封面(phys0)、目录(phys1)不盖。
    # 建立 物理页 -> 单元 的覆盖区间
    unit_phys = [phys[i] for i in range(len(UNITS))]
    def unit_of(p):
        u = None
        for i, sp in enumerate(unit_phys):
            if p >= sp:
                u = i
            else:
                break
        return u
    for p in range(doc.page_count):
        if p <= toc_pages:  # 封面 + 目录页
            continue
        ui = unit_of(p)
        if ui is None:
            continue
        printed_no = p - toc_pages  # 物理页(0-based) -> 印刷号
        short = UNITS[ui][3]
        text = f"{short} · {printed_no}"
        pg = doc[p]
        size = 8.2
        tlen = font.text_length(text, fontsize=size)
        x = (pg.rect.width - tlen) / 2
        y = pg.rect.height - 26
        # 用 insert_text（TextWriter 写 Chrome 既有页会坐标错乱，insert_text 正常）
        pg.insert_text(
            (x, y), text, fontname=fname, fontfile=fontfile,
            fontsize=size, color=DIM,
        )

    if fontfile is not None:
        doc.subset_fonts()
    doc.save(dst, deflate=True, garbage=4)
    print(f"[完成] {dst}  共 {doc.page_count} 页（含封面+目录），书签 {len(outline)} 条，字体 {fname}")


if __name__ == "__main__":
    main()
