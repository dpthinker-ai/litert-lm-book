#!/usr/bin/env bash
# Build the HTML book from the staged public-source allowlist.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$REPO_DIR/book"
BUILD_LOG="$(mktemp "${TMPDIR:-/tmp}/litert-lm-book-build.XXXXXX")"

cleanup() {
  rm -f -- "$BUILD_LOG"
}
trap cleanup EXIT

python3 "$SCRIPT_DIR/check_book_consistency.py"
python3 "$SCRIPT_DIR/check_citations.py"
bash "$SCRIPT_DIR/lint_prose.sh"
python3 "$SCRIPT_DIR/check_code_references.py"
bash "$SCRIPT_DIR/stage_book.sh"
if ! mdbook build "$REPO_DIR" >"$BUILD_LOG" 2>&1; then
  cat "$BUILD_LOG" >&2
  exit 1
fi
cat "$BUILD_LOG"
if grep -q ' WARN ' "$BUILD_LOG"; then
  echo "error: mdBook emitted warnings" >&2
  exit 1
fi

for forbidden in .git .agents .claude archive experiments scripts dist; do
  if [[ -e "$OUTPUT_DIR/$forbidden" ]]; then
    echo "error: forbidden path leaked into book output: $forbidden" >&2
    exit 1
  fi
done

if find "$OUTPUT_DIR" -type f -name '*.litertlm' -print -quit | grep -q .; then
  echo "error: model artifact leaked into book output" >&2
  exit 1
fi

if find "$OUTPUT_DIR" -type f \( -name '.DS_Store' -o -name 'review.html' \) \
  -print -quit | grep -q .; then
  echo "error: internal metadata leaked into book output" >&2
  exit 1
fi

python3 "$SCRIPT_DIR/check_book_links.py" "$OUTPUT_DIR"
