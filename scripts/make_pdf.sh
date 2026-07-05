#!/usr/bin/env bash
# 成书 PDF 生成管线：mdbook 构建 -> Chrome 渲染(A4, 保 MathJax/SVG) -> fitz 后处理(目录/书签/页脚)
# 产物：dist/book.pdf（成书）与 dist/book-raw.pdf（后处理前的中间件）
set -euo pipefail
export PATH="/opt/homebrew/bin:$PATH"
cd "$(dirname "$0")/.."
BOOK="$PWD"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

echo "==> mdbook build"
mdbook build >/dev/null

echo "==> Chrome 渲染 A4 原始 PDF"
mkdir -p "$BOOK/dist"
"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer \
  --virtual-time-budget=40000 --run-all-compositor-stages-before-draw \
  --print-to-pdf="$BOOK/dist/book-raw.pdf" "file://$BOOK/book/print.html" 2>/dev/null

echo "==> fitz 后处理（目录 + 书签 + 页脚）"
python3 "$BOOK/scripts/finalize_pdf.py" "$BOOK/dist/book-raw.pdf" "$BOOK/dist/book.pdf" "$@"

echo "==> 完成：dist/book.pdf"
pdfinfo "$BOOK/dist/book.pdf" 2>/dev/null | grep -iE '^Pages|^Page size' || true
