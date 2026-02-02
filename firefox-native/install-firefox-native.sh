#!/bin/sh
set -e
SRC_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
HOST_DST_DIR="$HOME/Library/Application Support/Mozilla/NativeMessagingHosts"
HOST_PATH="$SRC_DIR/host/minrss_native_host.py"
mkdir -p "$HOST_DST_DIR"
cat > "$HOST_DST_DIR/minrss_native_host.json" <<JSON
{
  "name": "minrss_native_host",
  "description": "Native host for minrss",
  "path": "$HOST_PATH",
  "type": "stdio",
  "allowed_extensions": ["minrss@local"]
}
JSON
chmod 644 "$HOST_DST_DIR/minrss_native_host.json"
echo "Installed native host manifest to: $HOST_DST_DIR/minrss_native_host.json"
