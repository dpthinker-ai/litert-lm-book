#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 Chrome 渲染的 book-raw.pdf 加工成成书 PDF：
   ① 封面后插一页「目录」（分部 + 章 + 页码）
   ② 写入可跳转的 PDF 书签/大纲（分部 -> 章，两级）
   ③ 每页底部盖「章名 · 页码」页脚（封面/目录页不盖）
渲染仍由 Chrome 负责（MathJax、SVG 正确）；本脚本只做结构化后处理。
用法：finalize_pdf.py <in.pdf> <out.pdf> [--debug]
依赖：PyMuPDF(fitz)。CJK 用 fitz 内置 "china-ss"，失败则回退 macOS PingFang。"""
import sys, re
import fitz  # PyMuPDF

# 结构：顺序即成书顺序。detect=定位该单元起始页的正则（匹配页面最顶一行标题）；
# part=所属部（None 表示不归部，做顶层书签）；short=页脚短名。
UNITS = [
    ("前言",                                    r"^\s*前言\s*$",              None,                 "前言"),
    ("第 1 章 端侧 LLM：约束与总览",             r"^\s*第\s*1\s*章",          "第一部 · 起点",       "第 1 章 约束与总览"),
    ("第 2 章 跑起来与鸟瞰：从 benchmark 数字到五层架构", r"^\s*第\s*2\s*章",   "第一部 · 起点",       "第 2 章 跑起来与鸟瞰"),
    ("第 3 章 输入侧：从 Engine API 到 token 序列", r"^\s*第\s*3\s*章",       "第二部 · 推理流水线", "第 3 章 输入侧"),
    ("第 4 章 Prefill：并行处理提示词",          r"^\s*第\s*4\s*章",          "第二部 · 推理流水线", "第 4 章 Prefill"),
    ("第 5 章 Decode：单步解码循环",             r"^\s*第\s*5\s*章",          "第二部 · 推理流水线", "第 5 章 Decode"),
    ("第 6 章 KV cache 与会话状态",              r"^\s*第\s*6\s*章",          "第三部 · 性能优化",   "第 6 章 KV cache"),
    ("第 7 章 模型的形态：量化、.litertlm 格式与 LoRA", r"^\s*第\s*7\s*章",     "第三部 · 性能优化",   "第 7 章 模型的形态"),
    ("第 8 章 异构算力：CPU、GPU 与 NPU",        r"^\s*第\s*8\s*章",          "第三部 · 性能优化",   "第 8 章 异构算力"),
    ("第 9 章 一次前向，多个 token：推测解码与 MTP", r"^\s*第\s*9\s*章",       "第三部 · 性能优化",   "第 9 章 推测解码"),
    ("第 10 章 多模态输入、约束解码与工具调用",   r"^\s*第\s*10\s*章",         "第四部 · 能力与工程", "第 10 章 多模态与工具"),
    ("第 11 章 一套核心，六种语言：C ABI、绑定与工程纪律", r"^\s*第\s*11\s*章",  "第四部 · 能力与工程", "第 11 章 六种语言"),
    ("尾声 · 出发",                              r"^\s*尾声",                 None,                 "尾声"),
    ("附录 A 术语表",                            r"^\s*附录\s*A",             None,                 "附录 A 术语表"),
    ("附录 B 代码地图",                          r"^\s*附录\s*B",             None,                 "附录 B 代码地图"),
    ("附录 C 环境搭建与实验复现",                r"^\s*附录\s*C",             None,                 "附录 C 复现指南"),
    ("附录 D 基准数据集",                        r"^\s*附录\s*D",             None,                 "附录 D 基准数据集"),
    ("附录 E 练习提示与参考答案",                r"^\s*附录\s*E",             None,                 "附录 E 练习答案"),
]

ACCENT = (0x0b/255, 0x57/255, 0xd0/255)
INK    = (0x1f/255, 0x23/255, 0x28/255)
DIM    = (0x8b/255, 0x94/255, 0x9e/255)


def get_font():
    for name in ("china-ss", "china-s"):
        try:
            return fitz.Font(name), name
        except Exception:
            pass
    # 回退 macOS 系统字体
    for path in ("/System/Library/Fonts/PingFang.ttc",
                 "/System/Library/Fonts/STHeiti Light.ttc",
                 "/System/Library/Fonts/Supplemental/Songti.ttc"):
        try:
            return fitz.Font(fontfile=path), "sys"
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
    return lines[0][2], lines[0][1]


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


def draw_toc_page(page, font, entries):
    """entries: [(kind, title, printed_no)]  kind in {part, chapter, plain}"""
    W, H = page.rect.width, page.rect.height
    ml, mr, mt = 62, 62, 70
    tw = fitz.TextWriter(page.rect)
    # 标题「目录」
    tw.append((ml, mt), "目录", font=font, fontsize=26)
    y = mt + 46
    for kind, title, no in entries:
        if kind == "part":
            y += 12
            tw.append((ml, y), title, font=font, fontsize=13)
            y += 22
            continue
        indent = ml + (16 if kind == "chapter" else 0)
        size = 10.5
        # 标题
        tw.append((indent, y), title, font=font, fontsize=size)
        # 右对齐页码
        num = str(no)
        num_w = font.text_length(num, fontsize=size)
        tw.append((W - mr - num_w, y), num, font=font, fontsize=size)
        y += 19
    tw.write_text(page, color=INK)
    # 「目录」下方一条强调色细线
    page.draw_line((ml, mt + 12), (ml + 70, mt + 12), color=ACCENT, width=2)


def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]
    debug = "--debug" in sys.argv
    doc = fitz.open(src)
    font, fname = get_font()

    raw_starts = detect_starts(doc)  # 0-based in raw
    if debug:
        print(f"[字体] {fname}   [总页] {doc.page_count}")
        for (u, rs) in zip(UNITS, raw_starts):
            print(f"  raw p{rs:>3}  {u[0]}")

    # 插入一页目录（在封面之后，index=1）。插入后所有 raw 页后移 1。
    W, H = doc[0].rect.width, doc[0].rect.height
    # 印刷页码：封面=0(不印)、目录=不印、正文从前言起 = raw 0-based 索引本身
    #   （因 cover=raw0, preface=raw1 -> 印刷 1；插目录后物理页= raw+1）
    printed = {i: rs for i, rs in enumerate(raw_starts)}  # unit i -> printed no = raw index
    phys = {i: rs + 1 for i, rs in enumerate(raw_starts)}  # 插目录后物理页(0-based)

    # 目录条目
    entries, cur_part = [], None
    for i, (title, rx, part, short) in enumerate(UNITS):
        if part and part != cur_part:
            entries.append(("part", part, None))
            cur_part = part
        kind = "chapter" if part else "plain"
        entries.append((kind, title, printed[i]))

    toc_page = doc.new_page(pno=1, width=W, height=H)
    draw_toc_page(toc_page, font, entries)

    # 书签/大纲（fitz 用 1-based 物理页）。分部为一级，章为二级；无部单元一级。
    outline, cur_part = [], None
    outline.append([1, "封面", 1])
    outline.append([1, "目录", 2])
    for i, (title, rx, part, short) in enumerate(UNITS):
        tgt = phys[i] + 1  # 1-based
        if part:
            if part != cur_part:
                outline.append([1, part, tgt])  # 部指向其首章页
                cur_part = part
            outline.append([2, title, tgt])
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
        if p <= 1:  # 封面、目录
            continue
        ui = unit_of(p)
        if ui is None:
            continue
        printed_no = p - 1  # 物理页(0-based) -> 印刷号：preface phys2 -> 1
        short = UNITS[ui][3]
        text = f"{short} · {printed_no}"
        pg = doc[p]
        size = 8.2
        tlen = font.text_length(text, fontsize=size)
        x = (pg.rect.width - tlen) / 2
        y = pg.rect.height - 26
        # 用 insert_text（TextWriter 写 Chrome 既有页会坐标错乱，insert_text 正常）
        pg.insert_text((x, y), text, fontname="china-ss", fontsize=size, color=DIM)

    doc.save(dst, deflate=True, garbage=4)
    print(f"[完成] {dst}  共 {doc.page_count} 页（含封面+目录），书签 {len(outline)} 条，字体 {fname}")


if __name__ == "__main__":
    main()
