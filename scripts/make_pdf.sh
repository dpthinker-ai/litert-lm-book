#!/usr/bin/env bash
# 成书 PDF 管线：mdBook -> 本地渲染 -> A4 分页 -> 目录/书签/页脚。
# 产物：dist/book.pdf 与 dist/book-raw.pdf。
set -euo pipefail
export PATH="/opt/homebrew/bin:$PATH"
cd "$(dirname "$0")/.."
BOOK="$PWD"
PDF_TMP="$(mktemp -d "${TMPDIR:-/tmp}/litert-lm-book-pdf.XXXXXX")"

cleanup() {
  rm -rf -- "$PDF_TMP"
}
trap cleanup EXIT

echo "==> staged mdbook build"
bash "$BOOK/scripts/build_book.sh" >/dev/null

mkdir -p "$BOOK/dist"

if [[ "$(uname -s)" == "Darwin" ]]; then
  echo "==> WebKit 连续矢量排版"
  CLANG_MODULE_CACHE_PATH="$PDF_TMP/clang-cache" xcrun clang \
    -fobjc-arc -framework AppKit -framework WebKit \
    "$BOOK/scripts/render_pdf_webkit.m" \
    -o "$PDF_TMP/render_pdf_webkit"
  "$PDF_TMP/render_pdf_webkit" \
    "$BOOK/book/print.html" \
    "$PDF_TMP/book-long.pdf" \
    "$PDF_TMP/layout.json" \
    "$BOOK/scripts/prepare_pdf_dom.js"

  echo "==> 按 DOM 边界分页为 A4"
  python3 "$BOOK/scripts/paginate_webkit_pdf.py" \
    "$PDF_TMP/book-long.pdf" \
    "$PDF_TMP/layout.json" \
    "$PDF_TMP/book-raw.pdf"
else
  if grep -q 'class="footnote-reference"' "$BOOK/book/print.html"; then
    echo "error: semantic same-page footnotes currently require the macOS WebKit PDF pipeline" >&2
    echo "error: refusing to emit a Chromium PDF with chapter-end footnote definitions" >&2
    exit 1
  fi
  echo "==> Chromium 渲染 A4 原始 PDF"
  CHROME_BIN="${CHROME_BIN:-}"
  if [[ -z "$CHROME_BIN" ]]; then
    CHROME_BIN="$(command -v google-chrome || command -v chromium || true)"
  fi
  if [[ -z "$CHROME_BIN" ]]; then
    echo "error: set CHROME_BIN to a Chrome/Chromium executable" >&2
    exit 1
  fi
  "$CHROME_BIN" \
    --headless=new --disable-gpu --no-pdf-header-footer \
    --user-data-dir="$PDF_TMP/chrome-profile" \
    --no-first-run --no-default-browser-check --disable-extensions \
    --disable-background-networking --virtual-time-budget=40000 \
    --run-all-compositor-stages-before-draw \
    --print-to-pdf="$PDF_TMP/book-raw.pdf" \
    "file://$BOOK/book/print.html"
fi

echo "==> fitz 后处理（目录 + 书签 + 页脚）"
python3 "$BOOK/scripts/finalize_pdf.py" \
  "$PDF_TMP/book-raw.pdf" \
  "$PDF_TMP/book.pdf" \
  "$@"

mv -f "$PDF_TMP/book-raw.pdf" "$BOOK/dist/book-raw.pdf"
mv -f "$PDF_TMP/book.pdf" "$BOOK/dist/book.pdf"

echo "==> 完成：dist/book.pdf"
pdfinfo "$BOOK/dist/book.pdf" 2>/dev/null | grep -iE '^Pages|^Page size' || true
