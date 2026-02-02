#!/bin/sh
set -e

REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
OUT_DIR="${1:-/tmp/minrss-public}"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

rsync -a --delete \
  --exclude ".git" \
  --exclude "scripts" \
  --exclude "README.public.md" \
  "$REPO_ROOT/" "$OUT_DIR/"

# Replace README with public version.
cp "$REPO_ROOT/README.public.md" "$OUT_DIR/README.md"

# Remove local-only data from public build (ask functionality).
python3 "$REPO_ROOT/scripts/publicize.py" "$OUT_DIR/minrss.py"

# Scrub any accidental private absolute paths in README.
sed -i '' 's#/Users/[^ ]*/#/#g' "$OUT_DIR/README.md" || true

# Fail if private paths or LLM env vars leak into public tree.
if rg -n "/Users/|MINRSS_LLM_MODEL" "$OUT_DIR" --glob '!publish.sh' >/dev/null; then
  echo "Publish failed: private paths or LLM references detected in public tree." >&2
  rg -n "/Users/|MINRSS_LLM_MODEL" "$OUT_DIR" --glob '!publish.sh' || true
  exit 1
fi

echo "Public repo prepared in: $OUT_DIR"
