#!/bin/sh
# Usage: open-with-minrss.sh <url> [...]
# Adds each URL as a feed to minrss.

if [ "$#" -eq 0 ]; then
  echo "No URL provided"
  exit 1
fi

for url in "$@"; do
  rss add "$url"
  if [ "$?" -ne 0 ]; then
    echo "Failed to add: $url"
  fi
done
