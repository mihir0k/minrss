#!/usr/bin/env python3
import argparse
import json
import os
import sqlite3
import subprocess
import shutil
import tempfile
import re
import sys
import textwrap
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from datetime import timedelta
from html import unescape
from html.parser import HTMLParser
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Optional, Tuple

DB_DIR = os.path.expanduser("~/.minrss")
DB_PATH = os.path.join(DB_DIR, "minrss.db")
USER_AGENT = "minrss/0.1 (+https://example.local)"
DEFAULT_LIMIT = 100
DEFAULT_PER_FEED = 3
DEFAULT_ASK_LIMIT = 60
DEFAULT_ASK_TOP = 10
DEFAULT_ASK_DAYS = 7


def ensure_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS feeds (
            id INTEGER PRIMARY KEY,
            url TEXT NOT NULL UNIQUE,
            title TEXT,
            site_url TEXT,
            tags TEXT,
            etag TEXT,
            last_modified TEXT,
            added_at TEXT NOT NULL
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY,
            feed_id INTEGER NOT NULL,
            uid TEXT NOT NULL,
            title TEXT,
            link TEXT,
            author TEXT,
            published TEXT,
            summary TEXT,
            content TEXT,
            categories TEXT,
            read INTEGER NOT NULL DEFAULT 0,
            starred INTEGER NOT NULL DEFAULT 0,
            added_at TEXT NOT NULL,
            UNIQUE(feed_id, uid),
            FOREIGN KEY(feed_id) REFERENCES feeds(id) ON DELETE CASCADE
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS views (
            name TEXT PRIMARY KEY,
            args TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_items_feed ON items(feed_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_items_read ON items(read);")
    # Lightweight migration for older DBs.
    cols = {row[1] for row in conn.execute("PRAGMA table_info(items)").fetchall()}
    if "summary" not in cols:
        conn.execute("ALTER TABLE items ADD COLUMN summary TEXT;")
    if "content" not in cols:
        conn.execute("ALTER TABLE items ADD COLUMN content TEXT;")
    if "categories" not in cols:
        conn.execute("ALTER TABLE items ADD COLUMN categories TEXT;")
    if "starred" not in cols:
        conn.execute("ALTER TABLE items ADD COLUMN starred INTEGER NOT NULL DEFAULT 0;")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_items_starred ON items(starred);")
    conn.commit()


def db_connect() -> sqlite3.Connection:
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    ensure_db(conn)
    return conn


def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def cutoff_iso(days: int) -> str:
    return (datetime.utcnow() - timedelta(days=days)).replace(microsecond=0).isoformat() + "Z"


def parse_http_date(value: str) -> Optional[str]:
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=None)
        return dt.isoformat()
    except Exception:
        return None


def fetch_url(url: str, etag: Optional[str], last_modified: Optional[str]) -> Tuple[int, Optional[bytes], Optional[str], Optional[str]]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if etag:
        req.add_header("If-None-Match", etag)
    if last_modified:
        req.add_header("If-Modified-Since", last_modified)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
            new_etag = resp.headers.get("ETag")
            new_lm = resp.headers.get("Last-Modified")
            return resp.status, data, new_etag, new_lm
    except urllib.error.HTTPError as e:
        if e.code == 304:
            return 304, None, etag, last_modified
        raise


def strip_ns(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def text_or_none(el: Optional[ET.Element]) -> Optional[str]:
    if el is None:
        return None
    text = (el.text or "").strip()
    return text or None


def get_child(parent: ET.Element, names: Tuple[str, ...]) -> Optional[ET.Element]:
    for child in parent:
        if strip_ns(child.tag) in names:
            return child
    return None


def get_children(parent: ET.Element, names: Tuple[str, ...]):
    for child in parent:
        if strip_ns(child.tag) in names:
            yield child


def normalize_datetime(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    parsed = parse_http_date(value)
    if parsed:
        return parsed
    return value


def parse_feed(data: bytes):
    root = ET.fromstring(data)
    root_tag = strip_ns(root.tag).lower()

    if root_tag == "rss" or root_tag == "rdf":
        channel = get_child(root, ("channel",)) or root
        feed_title = text_or_none(get_child(channel, ("title",)))
        site_url = text_or_none(get_child(channel, ("link",)))
        items = []
        for item in get_children(channel, ("item",)):
            title = text_or_none(get_child(item, ("title",)))
            link = text_or_none(get_child(item, ("link",)))
            guid = text_or_none(get_child(item, ("guid",)))
            pub_date = text_or_none(get_child(item, ("pubDate", "date")))
            author = text_or_none(get_child(item, ("author", "creator")))
            desc = text_or_none(get_child(item, ("description", "summary")))
            content = None
            for child in item:
                if strip_ns(child.tag).lower() in ("encoded", "content"):
                    content = text_or_none(child)
            categories = [text_or_none(c) for c in get_children(item, ("category",))]
            categories = ", ".join([c for c in categories if c]) or None
            uid = guid or link or (title or "") + "|" + (pub_date or "")
            items.append(
                {
                    "uid": uid,
                    "title": title,
                    "link": link,
                    "author": author,
                    "published": normalize_datetime(pub_date),
                    "summary": desc,
                    "content": content,
                    "categories": categories,
                }
            )
        return {"title": feed_title, "site_url": site_url, "items": items}

    if root_tag == "feed":
        feed_title = text_or_none(get_child(root, ("title",)))
        site_url = None
        for link in get_children(root, ("link",)):
            rel = link.attrib.get("rel", "alternate")
            if rel == "alternate":
                site_url = link.attrib.get("href") or text_or_none(link)
                break
        items = []
        for entry in get_children(root, ("entry",)):
            title = text_or_none(get_child(entry, ("title",)))
            link = None
            for lnk in get_children(entry, ("link",)):
                rel = lnk.attrib.get("rel", "alternate")
                if rel == "alternate":
                    link = lnk.attrib.get("href") or text_or_none(lnk)
                    break
            uid = text_or_none(get_child(entry, ("id",))) or link or title or ""
            author = None
            author_el = get_child(entry, ("author",))
            if author_el is not None:
                author = text_or_none(get_child(author_el, ("name",)))
            published = text_or_none(get_child(entry, ("published", "updated")))
            summary = text_or_none(get_child(entry, ("summary",)))
            content = text_or_none(get_child(entry, ("content",)))
            categories = []
            for c in get_children(entry, ("category",)):
                term = c.attrib.get("term") or text_or_none(c)
                if term:
                    categories.append(term)
            categories = ", ".join(categories) or None
            items.append(
                {
                    "uid": uid,
                    "title": title,
                    "link": link,
                    "author": author,
                    "published": normalize_datetime(published),
                    "summary": summary,
                    "content": content,
                    "categories": categories,
                }
            )
        return {"title": feed_title, "site_url": site_url, "items": items}

    raise ValueError("Unsupported feed format")


def cmd_init(_: argparse.Namespace) -> None:
    conn = db_connect()
    conn.close()
    print(f"Initialized {DB_PATH}")


def cmd_add(args: argparse.Namespace) -> None:
    conn = db_connect()
    url = args.url
    tags = None
    if args.tags:
        cleaned = [t.strip() for t in args.tags if t.strip()]
        tags = ",".join(cleaned) if cleaned else None
    title = None
    site_url = None
    if not args.no_check:
        try:
            status, data, _, _ = fetch_url(url, None, None)
            if status == 304 or not data:
                raise ValueError("empty response")
            parsed = parse_feed(data)
            title = parsed.get("title")
            site_url = parsed.get("site_url")
        except Exception as e:
            conn.close()
            print(f"Invalid feed: {e}")
            sys.exit(1)
    conn.execute(
        "INSERT OR IGNORE INTO feeds(url, title, site_url, tags, added_at) VALUES (?, ?, ?, ?, ?)",
        (url, title, site_url, tags, now_iso()),
    )
    conn.commit()
    conn.close()
    print("Added" if tags is None else f"Added (tags: {tags})")


def cmd_remove(args: argparse.Namespace) -> None:
    conn = db_connect()
    val = args.feed
    if val.isdigit():
        cur = conn.execute("DELETE FROM feeds WHERE id = ?", (int(val),))
    else:
        cur = conn.execute("DELETE FROM feeds WHERE url = ?", (val,))
    conn.commit()
    conn.close()
    print(f"Removed {cur.rowcount} feed(s)")


def cmd_purge(args: argparse.Namespace) -> None:
    conn = db_connect()
    val = args.feed
    if val.isdigit():
        feed_id = int(val)
    else:
        row = conn.execute("SELECT id FROM feeds WHERE url = ?", (val,)).fetchone()
        if not row:
            conn.close()
            print("Feed not found")
            return
        feed_id = row["id"]
    cur = conn.execute("DELETE FROM items WHERE feed_id = ?", (feed_id,))
    if not args.keep_feed:
        conn.execute("DELETE FROM feeds WHERE id = ?", (feed_id,))
    conn.commit()
    conn.close()
    print(f"Purged {cur.rowcount} item(s)")


def cmd_list(args: argparse.Namespace) -> None:
    conn = db_connect()
    if args.ranked:
        rows = conn.execute(
            """
            SELECT feeds.id, feeds.url, feeds.title, feeds.tags,
                   SUM(CASE WHEN items.read = 1 THEN 1 ELSE 0 END) AS read_count
            FROM feeds
            LEFT JOIN items ON items.feed_id = feeds.id
            GROUP BY feeds.id
            ORDER BY read_count DESC, feeds.id ASC
            """
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, url, title, tags FROM feeds ORDER BY id"
        ).fetchall()
    conn.close()
    for r in rows:
        title = r["title"] or "(untitled)"
        tags = f" [{r['tags']}]" if r["tags"] else ""
        if args.ranked:
            count = r["read_count"] or 0
            print(f"{r['id']:>3}  {title}{tags}  ({count} reads)\n     {r['url']}")
        else:
            print(f"{r['id']:>3}  {title}{tags}\n     {r['url']}")


def refresh_one(conn: sqlite3.Connection, row: sqlite3.Row) -> None:
    status, data, new_etag, new_lm = fetch_url(row["url"], row["etag"], row["last_modified"])
    if status == 304:
        # If feed has no items, force a full fetch to repopulate.
        count = conn.execute(
            "SELECT COUNT(1) AS c FROM items WHERE feed_id = ?",
            (row["id"],),
        ).fetchone()["c"]
        if count > 0:
            return
        status, data, new_etag, new_lm = fetch_url(row["url"], None, None)
    if not data:
        return
    parsed = parse_feed(data)
    feed_title = parsed.get("title")
    site_url = parsed.get("site_url")
    conn.execute(
        "UPDATE feeds SET title = ?, site_url = ?, etag = ?, last_modified = ? WHERE id = ?",
        (
            feed_title,
            site_url,
            new_etag or row["etag"],
            new_lm or row["last_modified"],
            row["id"],
        ),
    )
    for item in parsed["items"]:
        conn.execute(
            """
            INSERT OR IGNORE INTO items
            (feed_id, uid, title, link, author, published, summary, content, categories, added_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["id"],
                item.get("uid"),
                item.get("title"),
                item.get("link"),
                item.get("author"),
                item.get("published"),
                item.get("summary"),
                item.get("content"),
                item.get("categories"),
                now_iso(),
            ),
        )


def cmd_refresh(args: argparse.Namespace) -> None:
    conn = db_connect()
    if args.feed:
        val = args.feed
        if val.isdigit():
            rows = conn.execute("SELECT * FROM feeds WHERE id = ?", (int(val),)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM feeds WHERE url = ?", (val,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM feeds ORDER BY id").fetchall()

    for row in rows:
        try:
            refresh_one(conn, row)
            conn.commit()
            print(f"Refreshed {row['id']}")
        except Exception as e:
            print(f"Failed {row['id']} {row['url']}: {e}")
    conn.close()


def build_filters(args: argparse.Namespace):
    where = []
    params = []
    if getattr(args, "unread", False):
        where.append("items.read = 0")
    if getattr(args, "starred", False):
        where.append("items.starred = 1")
    if getattr(args, "feed", None):
        if args.feed.isdigit():
            where.append("items.feed_id = ?")
            params.append(int(args.feed))
        else:
            where.append("items.feed_id = (SELECT id FROM feeds WHERE url = ?)")
            params.append(args.feed)
    tags = getattr(args, "tags", None) or []
    if tags:
        for tag in tags:
            where.append("(',' || feeds.tags || ',') LIKE ?")
            params.append(f"%,{tag},%")
    days = getattr(args, "days", None)
    if days is not None:
        where.append("items.added_at >= ?")
        params.append(cutoff_iso(int(days)))
    clause = "WHERE " + " AND ".join(where) if where else ""
    limit_val = int(getattr(args, "limit", DEFAULT_LIMIT) or DEFAULT_LIMIT)
    page_val = int(getattr(args, "page", 1) or 1)
    if page_val < 1:
        page_val = 1
    offset_val = (page_val - 1) * limit_val
    per_feed_val = getattr(args, "per_feed", DEFAULT_PER_FEED)
    if per_feed_val is not None:
        per_feed_val = int(per_feed_val)
        if per_feed_val < 1:
            per_feed_val = None
    return clause, params, limit_val, offset_val, per_feed_val


def extract_preview(value: str, width: int = 88, lines: int = 2) -> str:
    if not value:
        return ""
    text = strip_html(value)
    text = " ".join(text.split())
    wrapped = textwrap.wrap(text, width=width)
    return "\n".join(wrapped[:lines])


def item_columns(conn: sqlite3.Connection) -> set:
    return {row[1] for row in conn.execute("PRAGMA table_info(items)").fetchall()}


def has_item_columns(conn: sqlite3.Connection, cols) -> bool:
    existing = item_columns(conn)
    return all(c in existing for c in cols)


def print_entries(rows, clean: bool = False) -> None:
    for r in rows:
        status = " " if r["read"] else "*"
        title = r["title"] or "(untitled)"
        date = r["published"] or ""
        feed_title = r["feed_title"] or "(feed)"
        star = "S" if r["starred"] else " "
        print(f"{status}{star} {r['id']:>6}  {date:>20}  {title}")
        if clean:
            has_summary = "summary" in r.keys()
            has_content = "content" in r.keys()
            summary = r["summary"] if has_summary else ""
            content = r["content"] if has_content else ""
            preview = extract_preview(summary or content or "")
            if preview:
                print(f"        {preview}")
        print(f"        {feed_title}")


def fetch_entries(conn: sqlite3.Connection, args: argparse.Namespace, include_body: bool = False):
    clause, params, limit_val, offset_val, per_feed_val = build_filters(args)
    if include_body and not has_item_columns(conn, ("summary", "content")):
        include_body = False
    inner_body_cols = ", items.summary, items.content" if include_body else ""
    outer_body_cols = ", summary, content" if include_body else ""
    inner = f"""
        SELECT items.id, items.title, items.published, items.read, items.starred,
               feeds.title AS feed_title{inner_body_cols},
               ROW_NUMBER() OVER (
                   PARTITION BY items.feed_id
                   ORDER BY items.published IS NULL, items.published DESC, items.id DESC
               ) AS rn
        FROM items JOIN feeds ON feeds.id = items.feed_id
        {clause}
    """
    outer_where = ""
    outer_params = []
    if per_feed_val:
        outer_where = "WHERE rn <= ?"
        outer_params.append(per_feed_val)
    query = f"""
        SELECT id, title, published, read, starred, feed_title{outer_body_cols}
        FROM ({inner}) sub
        {outer_where}
        ORDER BY published IS NULL, published DESC, id DESC
        LIMIT ? OFFSET ?
    """
    return conn.execute(query, params + outer_params + [limit_val, offset_val]).fetchall()


def fetch_latest_per_feed(conn: sqlite3.Connection, args: argparse.Namespace, include_body: bool = False):
    clause, params, limit_val, offset_val, _ = build_filters(args)
    if include_body and not has_item_columns(conn, ("summary", "content")):
        include_body = False
    body_cols = ", items.summary, items.content" if include_body else ""
    query = f"""
        SELECT items.id, items.title, items.published, items.read, items.starred, feeds.title AS feed_title{body_cols}
        FROM items
        JOIN feeds ON feeds.id = items.feed_id
        JOIN (
            SELECT feed_id, MAX(added_at) AS max_added
            FROM items
            GROUP BY feed_id
        ) latest ON latest.feed_id = items.feed_id AND items.added_at = latest.max_added
        {clause}
        ORDER BY items.added_at DESC, items.id DESC
        LIMIT ? OFFSET ?
    """
    return conn.execute(query, params + [limit_val, offset_val]).fetchall()


def cmd_entries(args: argparse.Namespace) -> None:
    conn = db_connect()
    rows = fetch_entries(conn, args, include_body=args.clean)
    conn.close()
    print_entries(rows, clean=args.clean)


def cmd_recent(args: argparse.Namespace) -> None:
    if args.days is None:
        args.days = 7
    conn = db_connect()
    rows = fetch_entries(conn, args, include_body=args.clean)
    conn.close()
    print_entries(rows, clean=args.clean)


def cmd_latest(args: argparse.Namespace) -> None:
    conn = db_connect()
    rows = fetch_latest_per_feed(conn, args, include_body=args.clean)
    conn.close()
    print_entries(rows, clean=args.clean)


def pager(text: str) -> None:
    pager_cmd = os.environ.get("MINRSS_PAGER") or os.environ.get("PAGER") or "less -R"
    try:
        import subprocess

        p = subprocess.Popen(pager_cmd, shell=True, stdin=subprocess.PIPE)
        p.stdin.write(text.encode("utf-8", errors="replace"))
        p.stdin.close()
        p.wait()
    except Exception:
        print(text)


class _HTMLStripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"br"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"p", "div", "li", "tr"}:
            self.parts.append("\n\n")

    def get_text(self) -> str:
        return "".join(self.parts)


def strip_html(value: str) -> str:
    stripper = _HTMLStripper()
    try:
        stripper.feed(value)
        text = stripper.get_text()
    except Exception:
        text = value
    return unescape(text)


def format_body(value: str, width: int = 88) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    paras = [p.strip() for p in value.split("\n\n") if p.strip()]
    if not paras:
        return ""
    wrapped = [textwrap.fill(p, width=width) for p in paras]
    return "\n\n".join(wrapped)


def cmd_read(args: argparse.Namespace) -> None:
    conn = db_connect()
    row = conn.execute(
        """
        SELECT items.*, feeds.title AS feed_title
        FROM items JOIN feeds ON feeds.id = items.feed_id
        WHERE items.id = ?
        """,
        (int(args.item_id),),
    ).fetchone()
    if not row:
        conn.close()
        print("Not found")
        return
    conn.execute("UPDATE items SET read = 1 WHERE id = ?", (int(args.item_id),))
    conn.commit()
    conn.close()

    parts = []
    parts.append(row["title"] or "(untitled)")
    parts.append("=" * 80)
    if row["feed_title"]:
        parts.append(f"Feed: {row['feed_title']}")
    if row["author"]:
        parts.append(f"Author: {row['author']}")
    if row["published"]:
        parts.append(f"Published: {row['published']}")
    if row["categories"]:
        parts.append(f"Categories: {row['categories']}")
    if row["link"]:
        parts.append(f"Link: {row['link']}")
    parts.append("")
    body = row["content"] or row["summary"] or ""
    if body:
        body = format_body(strip_html(body))
    else:
        body = "(no content)"
    parts.append(body)
    pager("\n".join(parts))


def cmd_mark(args: argparse.Namespace) -> None:
    conn = db_connect()
    read = 0 if args.unread else 1
    cur = conn.execute("UPDATE items SET read = ? WHERE id = ?", (read, int(args.item_id)))
    conn.commit()
    conn.close()
    print(f"Updated {cur.rowcount} item(s)")


def cmd_star(args: argparse.Namespace) -> None:
    conn = db_connect()
    cur = conn.execute("UPDATE items SET starred = 1 WHERE id = ?", (int(args.item_id),))
    conn.commit()
    conn.close()
    print(f"Starred {cur.rowcount} item(s)")


def cmd_unstar(args: argparse.Namespace) -> None:
    conn = db_connect()
    cur = conn.execute("UPDATE items SET starred = 0 WHERE id = ?", (int(args.item_id),))
    conn.commit()
    conn.close()
    print(f"Unstarred {cur.rowcount} item(s)")


def cmd_search(args: argparse.Namespace) -> None:
    conn = db_connect()
    include_body = args.clean or has_item_columns(conn, ("summary", "content"))
    clause, params, limit = build_filters(args)
    q = f"%{args.query}%"
    if include_body:
        if clause:
            clause = clause + " AND (items.title LIKE ? OR items.summary LIKE ? OR items.content LIKE ? OR items.categories LIKE ?)"
        else:
            clause = "WHERE (items.title LIKE ? OR items.summary LIKE ? OR items.content LIKE ? OR items.categories LIKE ?)"
        params.extend([q, q, q, q])
        select_body = ", items.summary, items.content"
    else:
        if clause:
            clause = clause + " AND (items.title LIKE ?)"
        else:
            clause = "WHERE (items.title LIKE ?)"
        params.append(q)
        select_body = ""
    rows = conn.execute(
        f"""
        SELECT items.id, items.title, items.published, items.read, items.starred, feeds.title AS feed_title{select_body}
        FROM items JOIN feeds ON feeds.id = items.feed_id
        {clause}
        ORDER BY items.published IS NULL, items.published DESC, items.id DESC
        {limit}
        """,
        params,
    ).fetchall()
    conn.close()
    print_entries(rows, clean=args.clean)


def run_llama(prompt: str, args: argparse.Namespace) -> str:
    llm_bin = args.llm_bin or os.environ.get("MINRSS_LLM_BIN") or "llama-cli"
    if not llm_model:
    extra = args.llm_args or os.environ.get("MINRSS_LLM_ARGS")
    if not extra:
        # Reasonable defaults to avoid interactive loops and keep output clean.
        extra = "-n 256 --no-display-prompt --no-show-timings --no-conversation --single-turn --simple-io --log-disable"
    cmd = [llm_bin, "-m", llm_model, "-p", prompt]
    if extra:
        cmd.extend(extra.split())
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "llama-cli failed")
    return result.stdout.strip()


def cmd_ask(args: argparse.Namespace) -> None:
    conn = db_connect()
    rows = fetch_entries(conn, args, include_body=True)
    conn.close()
    if not rows:
        print("No items")
        return
    chunks = []
    for r in rows:
        has_summary = "summary" in r.keys()
        has_content = "content" in r.keys()
        summary = r["summary"] if has_summary else ""
        content = r["content"] if has_content else ""
        body = summary or content or ""
        preview = extract_preview(body, width=100, lines=4)
        feed = r["feed_title"] or "(feed)"
        title = r["title"] or "(untitled)"
        published = r["published"] or ""
        chunks.append(f"[{r['id']}] {feed} | {published} | {title}\n{preview}")
    context = "\n\n".join(chunks)
    if args.max_chars and len(context) > args.max_chars:
        context = context[: args.max_chars]
    prompt = f"""You are a fast relevance filter for RSS items.
Only use the content shown. Do NOT add extra commentary.

Query: {args.query}

Items:
{context}

Return exactly {args.top} lines, each in this strict format:
ID - reason

No other text."""
    try:
        output = run_llama(prompt, args)
    except Exception as e:
        print(f"LLM error: {e}")
        return
    if not args.fzf:
        print(output)
        return
    ids = parse_ask_ids(output)
    if not ids:
        print("No ranked ids parsed from LLM output.")
        return
    rows = conn.execute(
        f"""
        SELECT items.id, items.title, items.published, items.read, items.starred,
               items.summary, items.content, feeds.title AS feed_title
        FROM items JOIN feeds ON feeds.id = items.feed_id
        WHERE items.id IN ({",".join("?" for _ in ids)})
        """,
        ids,
    ).fetchall()
    conn.close()
    by_id = {r["id"]: r for r in rows}
    ordered = [by_id[i] for i in ids if i in by_id]
    if not ordered:
        print(output)
        return
    tmpdir = tempfile.mkdtemp(prefix="minrss_ask_")
    lines = []
    for r in ordered:
        body = (r["summary"] or "") or (r["content"] or "")
        preview = format_body(extract_preview(body, width=100, lines=6), width=100)
        path = os.path.join(tmpdir, f"{r['id']}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(preview or "(no preview)")
        title = r["title"] or "(untitled)"
        feed = r["feed_title"] or "(feed)"
        date = r["published"] or ""
        lines.append(f"{r['id']}\\t{date}\\t{title}\\t{feed}")
    try:
        fzf = require_fzf()
        proc = subprocess.run(
            [
                fzf,
                "--delimiter",
                "\\t",
                "--with-nth",
                "2,3,4",
                "--preview",
                f"cat {tmpdir}/{{1}}.txt",
                "--bind",
                "esc:abort",
                "--prompt",
                "ask> ",
            ],
            input="\\n".join(lines),
            text=True,
            capture_output=True,
        )
        if proc.returncode != 0:
            return
        choice = proc.stdout.strip()
        if not choice:
            return
        item_id = choice.split("\\t", 1)[0]
        if args.open:
            cmd_open(argparse.Namespace(item_id=item_id))
        else:
            cmd_read(argparse.Namespace(item_id=item_id))
    finally:
        try:
            for name in os.listdir(tmpdir):
                os.remove(os.path.join(tmpdir, name))
            os.rmdir(tmpdir)
        except Exception:
            pass


def cmd_open(args: argparse.Namespace) -> None:
    conn = db_connect()
    row = conn.execute("SELECT link FROM items WHERE id = ?", (int(args.item_id),)).fetchone()
    conn.close()
    if not row or not row["link"]:
        print("No link")
        return
    link = row["link"]
    # Prefer OS opener if available, otherwise print the URL.
    if sys.platform.startswith("darwin"):
        os.system(f"open {link!r}")
    elif os.name == "nt":
        os.system(f"start {link}")
    else:
        os.system(f"xdg-open {link!r}")


def require_fzf() -> str:
    fzf = shutil.which("fzf")
    if not fzf:
        raise RuntimeError("fzf not found in PATH")
    return fzf


def run_fzf(lines, prompt: str) -> str:
    fzf = require_fzf()
    proc = subprocess.run(
        [fzf, "--prompt", prompt],
        input="\n".join(lines),
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def cmd_pick(args: argparse.Namespace) -> None:
    conn = db_connect()
    if args.kind == "feed":
        rows = conn.execute(
            "SELECT id, title, url, tags FROM feeds ORDER BY id"
        ).fetchall()
        conn.close()
        lines = []
        for r in rows:
            title = r["title"] or "(untitled)"
            tags = f"[{r['tags']}]" if r["tags"] else ""
            lines.append(f"{r['id']}\t{title} {tags}\t{r['url']}")
        choice = run_fzf(lines, "feed> ")
        if not choice:
            return
        feed_id = choice.split("\t", 1)[0]
        if args.print_id:
            print(feed_id)
            return
        entries_args = argparse.Namespace(
            feed=feed_id,
            tags=[],
            unread=False,
            starred=False,
            clean=args.clean,
            days=None,
            per_feed=None,
            limit=DEFAULT_LIMIT,
            page=1,
        )
        cmd_entries(entries_args)
    else:
        if args.kind == "recent":
            if args.days is None:
                args.days = 7
            rows = fetch_entries(conn, args, include_body=True)
        elif args.kind == "latest":
            rows = fetch_latest_per_feed(conn, args, include_body=True)
        elif args.kind == "entries":
            rows = fetch_entries(conn, args, include_body=True)
        elif args.kind == "search":
            include_body = True
            clause, params, limit_val, offset_val, per_feed_val = build_filters(args)
            q = f"%{args.query}%"
            if clause:
                clause = clause + " AND (items.title LIKE ? OR items.summary LIKE ? OR items.content LIKE ? OR items.categories LIKE ?)"
            else:
                clause = "WHERE (items.title LIKE ? OR items.summary LIKE ? OR items.content LIKE ? OR items.categories LIKE ?)"
            params.extend([q, q, q, q])
            inner_body_cols = ", items.summary, items.content"
            inner = f"""
                SELECT items.id, items.title, items.published, items.read, items.starred,
                       feeds.title AS feed_title{inner_body_cols},
                       ROW_NUMBER() OVER (
                           PARTITION BY items.feed_id
                           ORDER BY items.published IS NULL, items.published DESC, items.id DESC
                       ) AS rn
                FROM items JOIN feeds ON feeds.id = items.feed_id
                {clause}
            """
            outer_where = ""
            outer_params = []
            if per_feed_val:
                outer_where = "WHERE rn <= ?"
                outer_params.append(per_feed_val)
            query = f"""
                SELECT id, title, published, read, starred, feed_title, summary, content
                FROM ({inner}) sub
                {outer_where}
                ORDER BY published IS NULL, published DESC, id DESC
                LIMIT ? OFFSET ?
            """
            rows = conn.execute(query, params + outer_params + [limit_val, offset_val]).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT items.id, items.title, items.published, items.read, items.starred, feeds.title AS feed_title,
                       items.summary, items.content
                FROM items JOIN feeds ON feeds.id = items.feed_id
                ORDER BY items.published IS NULL, items.published DESC, items.id DESC
                LIMIT 500
                """
            ).fetchall()
        conn.close()
        if not rows:
            return
        tmpdir = tempfile.mkdtemp(prefix="minrss_pick_")
        lines = []
        for r in rows:
            status = " " if r["read"] else "*"
            star = "S" if r["starred"] else " "
            title = r["title"] or "(untitled)"
            feed_title = r["feed_title"] or "(feed)"
            date = r["published"] or ""
            body = (r["summary"] or "") or (r["content"] or "")
            preview = extract_preview(body, width=100, lines=6)
            path = os.path.join(tmpdir, f"{r['id']}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(preview or "(no preview)")
            lines.append(f"{r['id']}\t{status}{star} {date} {title}\t{feed_title}")
        try:
            fzf = require_fzf()
            while True:
                proc = subprocess.run(
                    [
                        fzf,
                        "--delimiter",
                        "\t",
                        "--with-nth",
                        "2,3",
                        "--preview",
                        f"cat {tmpdir}/{{1}}.txt",
                        "--bind",
                        "esc:abort",
                        "--prompt",
                        f"{args.kind}> ",
                    ],
                    input="\n".join(lines),
                    text=True,
                    capture_output=True,
                )
                if proc.returncode != 0:
                    return
                choice = proc.stdout.strip()
                if not choice:
                    return
                item_id = choice.split("\t", 1)[0]
                if args.print_id:
                    print(item_id)
                    return
                if args.open:
                    cmd_open(argparse.Namespace(item_id=item_id))
                else:
                    cmd_read(argparse.Namespace(item_id=item_id))
                if args.once:
                    return
        finally:
            try:
                for name in os.listdir(tmpdir):
                    os.remove(os.path.join(tmpdir, name))
                os.rmdir(tmpdir)
            except Exception:
                pass


def cmd_view_add(args: argparse.Namespace) -> None:
    conn = db_connect()
    payload = {
        "feed": args.feed,
        "tags": args.tags,
        "unread": args.unread,
        "starred": args.starred,
        "limit": args.limit,
        "per_feed": args.per_feed,
        "days": args.days,
        "latest_per_feed": args.latest_per_feed,
    }
    conn.execute(
        "INSERT OR REPLACE INTO views(name, args, created_at) VALUES (?, ?, ?)",
        (args.name, json.dumps(payload), now_iso()),
    )
    conn.commit()
    conn.close()
    print(f"Saved view {args.name}")


def cmd_view_list(_: argparse.Namespace) -> None:
    conn = db_connect()
    rows = conn.execute("SELECT name, created_at FROM views ORDER BY name").fetchall()
    conn.close()
    for r in rows:
        print(f"{r['name']}  {r['created_at']}")


def cmd_view_remove(args: argparse.Namespace) -> None:
    conn = db_connect()
    cur = conn.execute("DELETE FROM views WHERE name = ?", (args.name,))
    conn.commit()
    conn.close()
    print(f"Removed {cur.rowcount} view(s)")


def cmd_view_run(args: argparse.Namespace) -> None:
    conn = db_connect()
    row = conn.execute("SELECT args FROM views WHERE name = ?", (args.name,)).fetchone()
    if not row:
        conn.close()
        print("Not found")
        return
    saved = json.loads(row["args"])
    merged = argparse.Namespace(
        feed=saved.get("feed"),
        tags=saved.get("tags") or [],
        unread=bool(saved.get("unread")),
        starred=bool(saved.get("starred")),
        limit=saved.get("limit") or DEFAULT_LIMIT,
        per_feed=saved.get("per_feed") or DEFAULT_PER_FEED,
        days=saved.get("days"),
    )
    if saved.get("latest_per_feed"):
        rows = fetch_latest_per_feed(conn, merged)
    else:
        rows = fetch_entries(conn, merged)
    conn.close()
    print_entries(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="minrss")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init")
    help_p = sub.add_parser("help")
    help_p.add_argument("topic", nargs="?")

    add_p = sub.add_parser("add")
    add_p.add_argument("url")
    add_p.add_argument("--tag", dest="tags", action="append", default=[])
    add_p.add_argument("--no-check", action="store_true")

    rm_p = sub.add_parser("remove")
    rm_p.add_argument("feed")

    list_p = sub.add_parser("list")
    list_p.add_argument("--ranked", action="store_true")

    purge_p = sub.add_parser("purge")
    purge_p.add_argument("--feed", required=True)
    purge_p.add_argument("--keep-feed", action="store_true")

    ref_p = sub.add_parser("refresh")
    ref_p.add_argument("--feed")

    ent_p = sub.add_parser("entries")
    ent_p.add_argument("--feed")
    ent_p.add_argument("--tag", dest="tags", action="append", default=[])
    ent_p.add_argument("--unread", action="store_true")
    ent_p.add_argument("--starred", action="store_true")
    ent_p.add_argument("--clean", action="store_true")
    ent_p.add_argument("--days", type=int)
    ent_p.add_argument("--per-feed", type=int, default=DEFAULT_PER_FEED)
    ent_p.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    ent_p.add_argument("--page", type=int, default=1)

    recent_p = sub.add_parser("recent")
    recent_p.add_argument("--feed")
    recent_p.add_argument("--tag", dest="tags", action="append", default=[])
    recent_p.add_argument("--unread", action="store_true")
    recent_p.add_argument("--starred", action="store_true")
    recent_p.add_argument("--clean", action="store_true")
    recent_p.add_argument("--days", type=int)
    recent_p.add_argument("--per-feed", type=int, default=DEFAULT_PER_FEED)
    recent_p.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    recent_p.add_argument("--page", type=int, default=1)

    latest_p = sub.add_parser("latest")
    latest_p.add_argument("--feed")
    latest_p.add_argument("--tag", dest="tags", action="append", default=[])
    latest_p.add_argument("--unread", action="store_true")
    latest_p.add_argument("--starred", action="store_true")
    latest_p.add_argument("--clean", action="store_true")
    latest_p.add_argument("--days", type=int)
    latest_p.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    latest_p.add_argument("--page", type=int, default=1)

    read_p = sub.add_parser("read")
    read_p.add_argument("item_id")

    mark_p = sub.add_parser("mark")
    mark_p.add_argument("item_id")
    mark_p.add_argument("--read", action="store_true")
    mark_p.add_argument("--unread", action="store_true")

    star_p = sub.add_parser("star")
    star_p.add_argument("item_id")

    unstar_p = sub.add_parser("unstar")
    unstar_p.add_argument("item_id")

    search_p = sub.add_parser("search")
    search_p.add_argument("query")
    search_p.add_argument("--feed")
    search_p.add_argument("--tag", dest="tags", action="append", default=[])
    search_p.add_argument("--unread", action="store_true")
    search_p.add_argument("--starred", action="store_true")
    search_p.add_argument("--clean", action="store_true")
    search_p.add_argument("--days", type=int)
    search_p.add_argument("--per-feed", type=int, default=DEFAULT_PER_FEED)
    search_p.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    search_p.add_argument("--page", type=int, default=1)

    open_p = sub.add_parser("open")
    open_p.add_argument("item_id")

    view_p = sub.add_parser("view")
    view_sub = view_p.add_subparsers(dest="view_cmd", required=True)

    view_add = view_sub.add_parser("add")
    view_add.add_argument("name")
    view_add.add_argument("--feed")
    view_add.add_argument("--tag", dest="tags", action="append", default=[])
    view_add.add_argument("--unread", action="store_true")
    view_add.add_argument("--starred", action="store_true")
    view_add.add_argument("--days", type=int)
    view_add.add_argument("--per-feed", type=int, default=DEFAULT_PER_FEED)
    view_add.add_argument("--limit", type=int)
    view_add.add_argument("--latest-per-feed", action="store_true")

    view_run = view_sub.add_parser("run")
    view_run.add_argument("name")

    view_sub.add_parser("list")

    view_rm = view_sub.add_parser("remove")
    view_rm.add_argument("name")

    return p


def print_help_topic(topic: str) -> None:
    topics = {
        "init": """init

Create the local SQLite database.

Example:
  rss init
""",
        "add": """add

Add a feed (validated by default).

Usage:
  rss add <url> [--tag TAG] [--no-check]

Examples:
  rss add https://example.com/feed.xml
  rss add https://example.com/feed.xml --tag tech --tag reading
  rss add https://example.com/feed.xml --no-check
""",
        "remove": """remove

Remove a feed by id or URL.

Usage:
  rss remove <id|url>
""",
        "list": """list

List feeds.

Usage:
  rss list [--ranked]
""",
        "refresh": """refresh

Fetch new items for all feeds (or one feed).

Usage:
  rss refresh [--feed <id|url>]
""",
        "entries": """entries

List items across feeds.

Usage:
  rss entries [filters]

Common filters:
  --feed <id|url>   --tag TAG   --unread   --starred
  --days N          --per-feed N   --limit N   --page N
  --clean
""",
        "recent": """recent

Combined view of recent items (default 7 days).

Usage:
  rss recent [filters]
""",
        "latest": """latest

Newest item per feed.

Usage:
  rss latest [filters]
""",
        "read": """read

Read an item in the pager (marks read).

Usage:
  rss read <item_id>
""",
        "mark": """mark

Mark an item read/unread.

Usage:
  rss mark <item_id> [--read|--unread]
""",
        "star": """star

Star an item.

Usage:
  rss star <item_id>
""",
        "unstar": """unstar

Unstar an item.

Usage:
  rss unstar <item_id>
""",
        "open": """open

Open an item link in the browser.

Usage:
  rss open <item_id>
""",
        "search": """search

Search items by keyword.

Usage:
  rss search <query> [filters]
""",
        "pick": """pick

Fuzzy-pick with fzf (loops by default).

Usage:
  rss pick feed
  rss pick item [--open] [--once]
  rss pick entries|recent|latest|search [filters]
""",
        "view": """view

Saved custom views.

Usage:
  rss view add <name> [filters] [--latest-per-feed]
  rss view run <name>
  rss view list
  rss view remove <name>
""",
    }
    print(topics.get(topic, f"Unknown help topic: {topic}"))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "init":
        cmd_init(args)
    elif args.cmd == "help":
        if args.topic:
            print_help_topic(args.topic)
        else:
            parser.print_help()
    elif args.cmd == "add":
        cmd_add(args)
    elif args.cmd == "remove":
        cmd_remove(args)
    elif args.cmd == "list":
        cmd_list(args)
    elif args.cmd == "purge":
        cmd_purge(args)
    elif args.cmd == "refresh":
        cmd_refresh(args)
    elif args.cmd == "entries":
        cmd_entries(args)
    elif args.cmd == "recent":
        cmd_recent(args)
    elif args.cmd == "latest":
        cmd_latest(args)
    elif args.cmd == "read":
        cmd_read(args)
    elif args.cmd == "mark":
        cmd_mark(args)
    elif args.cmd == "star":
        cmd_star(args)
    elif args.cmd == "unstar":
        cmd_unstar(args)
    elif args.cmd == "search":
        cmd_search(args)
    elif args.cmd == "open":
        cmd_open(args)
    elif args.cmd == "pick":
        try:
            cmd_pick(args)
        except Exception as e:
            print(f"Pick error: {e}")
    elif args.cmd == "view":
        if args.view_cmd == "add":
            cmd_view_add(args)
        elif args.view_cmd == "list":
            cmd_view_list(args)
        elif args.view_cmd == "remove":
            cmd_view_remove(args)
        elif args.view_cmd == "run":
            cmd_view_run(args)
        else:
            parser.print_help()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
