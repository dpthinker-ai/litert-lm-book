#!/usr/bin/env bash
# Materialize the public book sources from an explicit allowlist.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
STAGE_DIR="$REPO_DIR/.mdbook-src"
TEMP_STAGE="$(mktemp -d "${TMPDIR:-/tmp}/litert-lm-book-stage.XXXXXX")"

cleanup() {
  rm -rf -- "$TEMP_STAGE"
}
trap cleanup EXIT

cp "$REPO_DIR/SUMMARY.md" "$REPO_DIR/preface.md" "$REPO_DIR/cover.md" \
  "$TEMP_STAGE/"

for source_dir in assets parts chapters appendix; do
  mkdir -p "$TEMP_STAGE/$source_dir"
  rsync -a \
    --exclude='.DS_Store' \
    --exclude='review.md' \
    "$REPO_DIR/$source_dir/" "$TEMP_STAGE/$source_dir/"
done

if grep -R -n --include='*.md' '<!--' "$TEMP_STAGE"; then
  echo "error: internal HTML comment found in publishable manuscript" >&2
  exit 1
fi

mkdir -p "$STAGE_DIR"
rsync -a --delete "$TEMP_STAGE/" "$STAGE_DIR/"
