#!/usr/bin/env python3
import json
import os
import struct
import subprocess
import sys
import shutil
from pathlib import Path

def read_message():
    raw_len = sys.stdin.buffer.read(4)
    if len(raw_len) == 0:
        return None
    msg_len = struct.unpack("<I", raw_len)[0]
    data = sys.stdin.buffer.read(msg_len)
    if not data:
        return None
    return json.loads(data.decode("utf-8"))


def send_message(payload):
    data = json.dumps(payload).encode("utf-8")
    sys.stdout.buffer.write(struct.pack("<I", len(data)))
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


def main():
    msg = read_message()
    if not msg:
        return
    url = msg.get("url")
    if not url:
        send_message({"ok": False, "error": "no url"})
        return
    try:
        env = os.environ.copy()
        rss_bin = shutil.which("rss")
        if not rss_bin:
            repo_root = Path(__file__).resolve().parents[2]
            rss_bin = str(repo_root / "minrss")
        result = subprocess.run([rss_bin, "add", url], capture_output=True, text=True, env=env)
        send_message({"ok": result.returncode == 0, "stdout": result.stdout, "stderr": result.stderr})
    except Exception as e:
        send_message({"ok": False, "error": str(e)})


if __name__ == "__main__":
    main()
