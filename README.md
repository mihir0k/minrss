# minrss

minrss is a tiny, fast terminal RSS reader focused on clean reading and zero fuss. It stores feeds locally, syncs on demand, and gives you fast, focused views of new items without getting buried by any single feed.

Key ideas:
- Everything is local (SQLite).
- Add/remove/list feeds instantly.
- Fast list views with per‑feed caps.
- Clean reading experience in a pager.
- Fuzzy picking with `fzf` for speed.

## Install

Clone the repo and run it directly:

```bash
./minrss init
```

(Optional) install a short alias (example uses `~/.local/bin`):

```bash
ln -s /path/to/minrss/minrss ~/.local/bin/rss
```

## Quick start

```bash
./minrss init
./minrss add https://example.com/feed.xml
./minrss refresh
./minrss recent --clean
./minrss read 123
```

## How it works

- Feeds are stored in `~/.minrss/minrss.db`.
- `refresh` pulls new items (ETag/Last‑Modified supported).
- List views default to **max 3 items per feed** and **100 items per page**.
- `--clean` shows short previews so you can scan quickly.

## Commands

General:
- `init` create the local SQLite DB in `~/.minrss/minrss.db`
- `help [topic]` show general help or help for a specific command
- `add <url> [--tag TAG] [--no-check]` add a feed with optional tags (repeatable); validates feed by default
- `remove <id|url>` remove a feed
- `list [--ranked]` list feeds (optionally sorted by reads)
- `refresh [--feed <id|url>]` fetch and store new items

Views:
- `entries [--feed <id|url>] [--tag TAG] [--unread] [--starred] [--clean] [--days N] [--per-feed N] [--limit N] [--page N]` list items
- `recent [--feed <id|url>] [--tag TAG] [--unread] [--starred] [--clean] [--days N] [--per-feed N] [--limit N] [--page N]` combined feed for recent items (default 7 days)
- `latest [--feed <id|url>] [--tag TAG] [--unread] [--starred] [--clean] [--days N] [--limit N] [--page N]` newest item per feed
- `search <query> [--feed <id|url>] [--tag TAG] [--unread] [--starred] [--clean] [--days N] [--per-feed N] [--limit N] [--page N]` search items

Reading:
- `read <item_id>` show an item in a pager and mark as read
- `mark <item_id> [--read|--unread]` mark an item (default: read)
- `open <item_id>` open the item link in your default browser
- `star <item_id>` star an item
- `unstar <item_id>` unstar an item

Pickers (requires `fzf`):
- `pick feed` fuzzy‑pick a feed
- `pick item [--open] [--once]` fuzzy‑pick an item to read/open
- `pick entries|recent|latest|search` fuzzy‑pick from those views (supports `--feed`, `--tag`, `--unread`, `--days`, `--per-feed`)

Views (saved filters):
- `view add <name> [--feed <id|url>] [--tag TAG] [--unread] [--starred] [--days N] [--per-feed N] [--limit N] [--latest-per-feed]` save a custom view
- `view run <name>` run a saved view
- `view list` list saved views
- `view remove <name>` delete a saved view

## Pager

`minrss` uses `$MINRSS_PAGER`, then `$PAGER`, then falls back to `less -R`.

## Firefox right‑click add

### Option A: "Open With" extension

1) Install a Firefox "Open With" style extension and add a custom app.
2) Point it to `open-with-minrss.sh` in this repo.
3) Right‑click a feed link and choose the custom app to add it.

Script path (from repo root):

```
./open-with-minrss.sh
```

### Option B: Native right‑click (one‑click add)

This adds a right‑click menu item that sends the link URL to a local native host, which runs `rss add <url>`.

Setup (macOS):

1) Install the native host manifest:

```
./firefox-native/install-firefox-native.sh
```

2) Load the extension temporarily:
   - Open `about:debugging#/runtime/this-firefox`
   - Click "Load Temporary Add‑on"
   - Select `firefox-native/extension/manifest.json`

3) Right‑click a feed link and choose "Add feed to minrss".

Notes:
- Temporary add‑ons reset on Firefox restart. For a permanent install you need a signed extension (Developer Edition makes this easier).
- The add‑on shows a small system notification on success/failure.
- The install script writes the correct absolute path for your machine.
