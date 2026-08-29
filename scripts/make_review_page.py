#!/usr/bin/env python3
"""把 mdBook 构建产物中的一章组装成自包含的审阅页 HTML。

用法: python3 make_review_page.py <book/xxx.html> <输出.html> <章节名>
- 抽取 <main> 内容
- 相对链接（跨章引用）拆成纯文本；#锚点与 http(s) 链接保留
- <img src="*.svg"> 就地内联为 <svg>（跟随页面主题变量）
- 含 \\( 或 $$ 的章节内联本地 MathJax（tex-svg）
"""
import datetime
import re
import subprocess
import sys
from pathlib import Path

BOOK_ROOT = Path("/Users/dpthinker/workspace/litert-lm-book/book")
MATHJAX = BOOK_ROOT / "assets/mathjax4-tex-svg.js"

src, out, chapter_name = sys.argv[1], sys.argv[2], sys.argv[3]
src = Path(src)
html = src.read_text(encoding="utf-8")
main = re.search(r"<main>(.*?)</main>", html, re.S).group(1)

# 相对链接拆成纯文本（保留页内 # 锚点与外链）
def unwrap(m):
    href, text = m.group(1), m.group(2)
    if href.startswith(("#", "http://", "https://", "mailto:")):
        return m.group(0)
    return text
main = re.sub(r'<a\s[^>]*href="([^"]+)"[^>]*>(.*?)</a>', unwrap, main, flags=re.S)

# 内联 SVG 配图
def inline_svg(m):
    rel = m.group(1)
    p = (src.parent / rel).resolve()
    if not p.exists() or p.suffix != ".svg":
        return m.group(0)
    svg = p.read_text(encoding="utf-8")
    svg = re.sub(r"^<\?xml[^>]*\?>\s*", "", svg)
    return svg
main = re.sub(r'<img\s[^>]*src="([^"]+)"[^>]*/?>', inline_svg, main)

needs_math = ("\\(" in main) or ("$$" in main)
math_tag = ""
if needs_math:
    math_tag = "<script>\n" + MATHJAX.read_text(encoding="utf-8") + "\n</script>"

CSS = r"""
:root {
  --fig-bg:#ffffff; --fig-surface:#f6f8fa; --fig-surface-2:#eef1f5;
  --fig-accent:#0b57d0; --fig-accent-2:#6639ba;
  --fig-border:#d0d7de; --fig-border-2:#e6eaef;
  --fig-text:#1f2328; --fig-text-soft:#57606a; --fig-text-dim:#8b949e;
  --fig-green:#1a7f37;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --fig-bg:#0d1117; --fig-surface:#161b22; --fig-surface-2:#1c2330;
    --fig-accent:#4f9cff; --fig-accent-2:#7c5cff;
    --fig-border:#2a3340; --fig-border-2:#21282f;
    --fig-text:#e6edf3; --fig-text-soft:#b6c2cf; --fig-text-dim:#8b98a5;
    --fig-green:#3fb950;
  }
}
:root[data-theme="dark"] {
  --fig-bg:#0d1117; --fig-surface:#161b22; --fig-surface-2:#1c2330;
  --fig-accent:#4f9cff; --fig-accent-2:#7c5cff;
  --fig-border:#2a3340; --fig-border-2:#21282f;
  --fig-text:#e6edf3; --fig-text-soft:#b6c2cf; --fig-text-dim:#8b98a5;
  --fig-green:#3fb950;
}
body {
  background: var(--fig-bg); color: var(--fig-text); margin: 0;
  font-family: "PingFang SC","Hiragino Sans GB","Source Han Sans SC",
    "Noto Sans CJK SC","Microsoft YaHei",-apple-system,BlinkMacSystemFont,
    "Segoe UI",Roboto,sans-serif;
}
a { color: var(--fig-accent); }
a:focus-visible { outline: 2px solid var(--fig-accent); outline-offset: 2px; }

.review-bar {
  border-bottom: 1px solid var(--fig-border-2);
  padding: 14px 16px; font-size: 0.82em; line-height: 1.7;
  color: var(--fig-text-dim);
}
.review-bar .inner { max-width: 720px; margin: 0 auto; }
.review-bar b { color: var(--fig-text-soft); font-weight: 600; }

.content main {
  display: block; max-width: 720px; margin: 0 auto;
  padding: 20px 16px 80px; line-height: 1.85; letter-spacing: 0.008em;
  font-size: 16px;
}
.content main p { margin: 0 0 1.1em; }
.content main h1 { font-size: 1.95em; font-weight: 800; line-height: 1.35; margin: 18px 0 24px; letter-spacing: 0.01em; text-wrap: balance; }
.content main h2 { font-size: 1.42em; font-weight: 700; line-height: 1.4; margin: 40px 0 15px; padding-bottom: 6px; border-bottom: 2px solid var(--fig-border-2); text-wrap: balance; }
.content main h3 { font-size: 1.2em; font-weight: 700; margin: 30px 0 12px; }
.content main h4 { font-size: 1.05em; font-weight: 700; color: var(--fig-text-soft); margin: 24px 0 9px; }
.content main a.header { color: inherit; text-decoration: none; }
.content main > blockquote {
  border-left: 3px solid var(--fig-accent); background: var(--fig-surface);
  padding: 12px 16px; margin: 0 0 30px; border-radius: 0 8px 8px 0;
  color: var(--fig-text-soft); font-size: 0.97em;
}
.content main > blockquote p { margin: 0; }
.content main :not(pre) > code {
  font-family: "SF Mono","JetBrains Mono","Fira Code",Menlo,Consolas,monospace;
  font-size: 0.85em; padding: 0.1em 0.36em; border-radius: 4px;
  background: var(--fig-surface-2); color: var(--fig-accent-2);
  border: 1px solid var(--fig-border-2);
}
.content main pre {
  border: 1px solid var(--fig-border); border-radius: 8px;
  padding: 12px 15px; margin: 18px 0; line-height: 1.62;
  background: var(--fig-surface); overflow-x: auto;
}
.content main pre, .content main pre code {
  font-family: "SF Mono","JetBrains Mono","Fira Code",Menlo,Consolas,monospace;
  font-size: 0.86em; background: none; border: none; color: inherit; padding: 0;
}
.content main pre { background: var(--fig-surface); }
.table-wrapper { overflow-x: auto; }
.content main table { width: 100%; margin: 20px 0; border-collapse: collapse; font-size: 0.93em; line-height: 1.6; }
.content main table thead th { background: var(--fig-surface); font-weight: 700; text-align: left; }
.content main table th, .content main table td { border: 1px solid var(--fig-border); padding: 7px 10px; }
.content main table tbody tr:nth-child(even) td { background: var(--fig-surface); }
.content main figure { margin: 26px 0; text-align: center; }
.content main figure svg, .content main svg { max-width: 100%; height: auto; }
.content main figcaption { margin-top: 9px; font-size: 0.88em; color: var(--fig-text-dim); }
.content main ul, .content main ol { margin: 0 0 16px; padding-left: 22px; }
.content main li { margin: 5px 0; }
.content main li > p { margin: 0 0 6px; }
.content main hr { border: none; border-top: 1px solid var(--fig-border-2); margin: 34px 0; }
.aside-compare, .aside-version {
  border-left: 3px solid var(--fig-accent); background: var(--fig-surface);
  padding: 0.6rem 1rem; margin: 1rem 0; border-radius: 0 6px 6px 0;
}
.aside-version { border-left-color: var(--fig-text-dim); }
.aside-compare::before { content: "对照视野"; font-weight: 700; font-size: 0.85em; display: block; margin-bottom: 0.3rem; }
.aside-version::before { content: "版本注记"; font-weight: 700; font-size: 0.85em; display: block; margin-bottom: 0.3rem; color: var(--fig-text-dim); }
.content main .footnote-reference a { text-decoration: none; }
.content main .footnote-definition { color: var(--fig-text-soft); font-size: 0.86em; line-height: 1.58; }
.content main .footnote-definition > li { margin: 0.45em 0; }
.content main .footnote-definition a { overflow-wrap: anywhere; }
"""

rev = subprocess.run(
    ["git", "-C", str(BOOK_ROOT.parent), "rev-parse", "--short", "HEAD"],
    capture_output=True, text=True).stdout.strip() or "未知"
today = datetime.date.today().isoformat()

page = f"""<title>《端侧大模型推理》{chapter_name}</title>
<style>{CSS}</style>
<div class="review-bar"><div class="inner">
<b>《端侧大模型推理：原理与 LiteRT-LM 实现》审阅版</b> · 对应提交 {rev} · {today}<br>
选中正文文字加评论并发送给 Claude：修订会直接落回书稿仓库，完成后在你的评论下回复。
</div></div>
<div class="content"><main>
{main}
</main></div>
{math_tag}
"""
Path(out).write_text(page, encoding="utf-8")
print(f"{out}: {len(page)} 字节, math={'有' if needs_math else '无'}")
