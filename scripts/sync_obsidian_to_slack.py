#!/usr/bin/env python3
"""Sync Obsidian daily lab log to #lab-log-francisco on Slack.

Posts one thread per day: a header message ("Lab log: YYYY-MM-DD") as the
parent, and each bold-header section as a sequence of thread replies with
images interleaved where they appear in the original text.

On subsequent runs, new sections are added and modified sections are edited
in place (matched by header text). Checks the last 5 days by default so
missed runs (e.g. laptop was off) get caught up.

Usage:
    python sync_obsidian_to_slack.py              # Sync last 5 days
    python sync_obsidian_to_slack.py 2026-02-27   # Sync a specific date only
    python sync_obsidian_to_slack.py --dry-run    # Preview without posting

Config:
    Store your Slack bot token in ~/.config/obsidian-slack-sync/token
    Or set the SLACK_BOT_TOKEN environment variable.
"""

import os
import re
import sys
import json
import hashlib
import argparse
from datetime import date, timedelta
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: 'requests' package required. Install with: pip install requests")
    sys.exit(1)

# --- Configuration ---
VAULT_DIR = Path.home() / "Documents" / "Francisco's Vault"
LOG_DIR = VAULT_DIR / "AI Safety" / "Activation Plateaus Project"
STATE_FILE = Path.home() / ".local" / "share" / "obsidian-slack-sync" / "state.json"
TOKEN_FILE = Path.home() / ".config" / "obsidian-slack-sync" / "token"
SLACK_CHANNEL = "C0A83J9HF8W"  # #lab-log-francisco
SLACK_API = "https://slack.com/api"
LOOKBACK_DAYS = 5

IMAGE_RE = re.compile(
    r"!\[\[([^\]]+\.(?:png|jpg|jpeg|gif|svg))\]\]", re.IGNORECASE
)


def get_token():
    """Get Slack bot token from env var or config file."""
    token = os.environ.get("SLACK_BOT_TOKEN")
    if token:
        return token.strip()
    if TOKEN_FILE.exists():
        return TOKEN_FILE.read_text().strip()
    return None


# --- Slack API helpers ---


def slack_post(token, channel, text, thread_ts=None):
    """Post a message to Slack. Returns message ts on success."""
    payload = {"channel": channel, "text": text}
    if thread_ts:
        payload["thread_ts"] = thread_ts
    resp = requests.post(
        f"{SLACK_API}/chat.postMessage",
        headers={"Authorization": f"Bearer {token}"},
        json=payload,
    )
    data = resp.json()
    if not data.get("ok"):
        print(f"  Slack error: {data.get('error', 'unknown')}")
        return None
    return data.get("ts")


def slack_update(token, channel, ts, text):
    """Edit an existing Slack message. Returns True on success."""
    resp = requests.post(
        f"{SLACK_API}/chat.update",
        headers={"Authorization": f"Bearer {token}"},
        json={"channel": channel, "ts": ts, "text": text},
    )
    data = resp.json()
    if not data.get("ok"):
        print(f"  Slack update error: {data.get('error', 'unknown')}")
        return False
    return True


def slack_upload_file(token, channel, filepath, thread_ts=None):
    """Upload a file to Slack using the v2 upload flow."""
    filepath = Path(filepath)
    if not filepath.exists():
        return None

    file_size = filepath.stat().st_size
    filename = filepath.name

    resp = requests.post(
        f"{SLACK_API}/files.getUploadURLExternal",
        headers={"Authorization": f"Bearer {token}"},
        data={"filename": filename, "length": file_size},
    )
    data = resp.json()
    if not data.get("ok"):
        print(f"  Upload URL error: {data.get('error')}")
        return None

    upload_url = data["upload_url"]
    file_id = data["file_id"]

    with open(filepath, "rb") as f:
        resp = requests.post(upload_url, files={"file": (filename, f)})
    if resp.status_code != 200:
        print(f"  File upload failed: HTTP {resp.status_code}")
        return None

    complete_data = {
        "files": [{"id": file_id, "title": filename}],
        "channel_id": channel,
    }
    if thread_ts:
        complete_data["thread_ts"] = thread_ts

    resp = requests.post(
        f"{SLACK_API}/files.completeUploadExternal",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=complete_data,
    )
    data = resp.json()
    if not data.get("ok"):
        print(f"  Complete upload error: {data.get('error')}")
        return None

    return file_id


# --- Content processing ---


def get_log_content(log_date):
    """Read the Obsidian log file for a given date."""
    log_file = LOG_DIR / f"{log_date} Log.md"
    if log_file.exists():
        return log_file.read_text()
    return None


def strip_task_list(content):
    """Remove checkbox task list from the top of the file."""
    lines = content.split("\n")
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("- [") or (stripped == "" and i > 0):
            i += 1
        else:
            break
    while i < len(lines) and lines[i].strip() == "":
        i += 1
    return "\n".join(lines[i:])


def extract_header(section):
    """Extract the bold header text from a section, or None if no header."""
    m = re.match(r"^\s*\*\*([^*]+)\*\*\s*$", section.split("\n")[0])
    if m:
        return m.group(1).strip()
    return None


def split_into_sections(content):
    """Split content into sections by bold headers (**...**).

    Returns list of (header_or_None, full_section_text) tuples.
    """
    sections = []
    current_lines = []

    for line in content.split("\n"):
        if re.match(r"^\s*\*\*[^*]+\*\*\s*$", line):
            text = "\n".join(current_lines).strip()
            if text:
                sections.append(text)
            current_lines = [line]
        else:
            current_lines.append(line)

    text = "\n".join(current_lines).strip()
    if text:
        sections.append(text)

    return [(extract_header(s), s) for s in sections]


def split_section_into_parts(section_text):
    """Split a section into alternating text and image parts.

    Returns a list of ("text", slack_text) or ("image", filename) tuples,
    preserving the order images appear in the original text.
    """
    parts = []
    # Split on image embeds, keeping the captures
    pieces = IMAGE_RE.split(section_text)
    # pieces alternates: text, image_filename, text, image_filename, ...
    for idx, piece in enumerate(pieces):
        if idx % 2 == 0:
            # Text chunk — convert to slack format
            slack_text = obsidian_to_slack_raw(piece)
            if slack_text:
                parts.append(("text", slack_text))
        else:
            # Image filename
            parts.append(("image", piece))
    return parts


def find_image(filename):
    """Find an image file in the Obsidian vault."""
    path = VAULT_DIR / filename
    if path.exists():
        return path
    path = LOG_DIR / filename
    if path.exists():
        return path
    return None


def obsidian_to_slack_raw(text):
    """Convert Obsidian markdown to Slack mrkdwn format.

    Like obsidian_to_slack but does NOT strip image embeds (caller handles those).
    """
    # Wikilinks
    text = re.sub(r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    # Bold: **text** -> *text*
    text = re.sub(r"\*\*([^*]+)\*\*", r"*\1*", text)
    # Markdown links: [text](url) -> <url|text>
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<\2|\1>", text)
    # LaTeX block equations
    text = re.sub(
        r"\\begin\{equation\}(.*?)\\end\{equation\}",
        lambda m: "[equation: " + m.group(1).strip() + "]",
        text,
        flags=re.DOTALL,
    )
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --- State management ---
#
# State format (v3):
# {
#   "2026-03-05": {
#     "thread_ts": "1234567890.123456",
#     "sections": {
#       "Morning work": {
#         "hash": "abc123",
#         "parts": [
#           {"type": "text", "ts": "1234567890.234567"},
#           {"type": "image", "name": "plot.png"},
#           {"type": "text", "ts": "1234567890.345678"}
#         ]
#       }
#     }
#   }
# }


def section_hash(text):
    """Short hash of section content for deduplication."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def migrate_state(state):
    """Migrate old state formats to current format.

    v1 (lists of hashes) and v2 (sections with single ts) are dropped
    since we can't recover thread structure from them.
    """
    migrated = False
    for key, val in list(state.items()):
        if isinstance(val, list):
            del state[key]
            migrated = True
        elif isinstance(val, dict) and "sections" in val:
            # Check if any section uses old v2 format (has "ts" key instead of "parts")
            for sec_val in val["sections"].values():
                if "ts" in sec_val and "parts" not in sec_val:
                    # v2 format — drop entire day to re-sync
                    del state[key]
                    migrated = True
                    break
    if migrated:
        save_state(state)
        print("Migrated state file (old entries dropped, will re-sync)")
    return state


# --- Main logic ---


def section_key(header, index):
    """State key for a section: header text, or _noheader_N for headerless ones."""
    if header:
        return header
    return f"_noheader_{index}"


def post_section_parts(parts, token, thread_ts, dry_run=False):
    """Post a sequence of text/image parts as thread replies.

    Returns list of part state dicts for storage, or None on failure.
    """
    part_states = []
    for part_type, content in parts:
        if part_type == "text":
            if dry_run:
                preview = content[:200] + ("..." if len(content) > 200 else "")
                print(f"    [text] {preview}")
                part_states.append({"type": "text", "ts": "dry-run"})
            else:
                ts = slack_post(token, SLACK_CHANNEL, content, thread_ts=thread_ts)
                if not ts:
                    return None
                part_states.append({"type": "text", "ts": ts})
        elif part_type == "image":
            img_path = find_image(content)
            if dry_run:
                status = "found" if img_path else "NOT FOUND"
                print(f"    [image] {content} ({status})")
                part_states.append({"type": "image", "name": content})
            else:
                if img_path:
                    slack_upload_file(token, SLACK_CHANNEL, img_path, thread_ts=thread_ts)
                    print(f"    Uploaded {content}")
                else:
                    print(f"    Image not found: {content}")
                part_states.append({"type": "image", "name": content})
    return part_states


def update_section_parts(new_parts, old_part_states, token, thread_ts, dry_run=False):
    """Update an existing section's parts.

    Edits text messages in place where possible, uploads new images,
    appends any extra parts that didn't exist before.
    """
    updated_states = []
    old_text_ts = [p["ts"] for p in old_part_states if p["type"] == "text"]
    old_images = {p["name"] for p in old_part_states if p["type"] == "image"}
    text_idx = 0

    for part_type, content in new_parts:
        if part_type == "text":
            if text_idx < len(old_text_ts):
                # Edit existing text message
                ts = old_text_ts[text_idx]
                if dry_run:
                    preview = content[:200] + ("..." if len(content) > 200 else "")
                    print(f"    [edit text] {preview}")
                else:
                    slack_update(token, SLACK_CHANNEL, ts, content)
                updated_states.append({"type": "text", "ts": ts})
                text_idx += 1
            else:
                # New text part — append
                if dry_run:
                    preview = content[:200] + ("..." if len(content) > 200 else "")
                    print(f"    [new text] {preview}")
                    updated_states.append({"type": "text", "ts": "dry-run"})
                else:
                    ts = slack_post(token, SLACK_CHANNEL, content, thread_ts=thread_ts)
                    if ts:
                        updated_states.append({"type": "text", "ts": ts})
        elif part_type == "image":
            if content in old_images:
                # Already uploaded — keep it
                if dry_run:
                    print(f"    [image, already uploaded] {content}")
                updated_states.append({"type": "image", "name": content})
            else:
                # New image — upload
                img_path = find_image(content)
                if dry_run:
                    status = "found" if img_path else "NOT FOUND"
                    print(f"    [new image] {content} ({status})")
                else:
                    if img_path:
                        slack_upload_file(token, SLACK_CHANNEL, img_path, thread_ts=thread_ts)
                        print(f"    Uploaded {content}")
                    else:
                        print(f"    Image not found: {content}")
                updated_states.append({"type": "image", "name": content})

    return updated_states


def sync_log(log_date, dry_run=False, token=None):
    """Sync a day's log to Slack as a thread.

    Creates a parent message ("Lab log: YYYY-MM-DD") if needed, then
    posts each section as a sequence of text/image thread replies.
    Edited sections get updated in place (matched by header text).
    """
    content = get_log_content(log_date)
    if not content:
        return

    content = strip_task_list(content)
    if not content.strip():
        return

    sections = split_into_sections(content)
    state = load_state()
    day_state = state.get(log_date, {})
    thread_ts = day_state.get("thread_ts")
    posted_sections = day_state.get("sections", {})

    new_count = 0
    updated_count = 0

    # Create thread parent if needed
    if not thread_ts:
        header_text = f"Lab log: {log_date}"
        if dry_run:
            print(f"\n=== Would create thread: {header_text} ===")
        else:
            thread_ts = slack_post(token, SLACK_CHANNEL, header_text)
            if not thread_ts:
                print(f"  Failed to create thread for {log_date}")
                return
            day_state["thread_ts"] = thread_ts
            print(f"  Created thread for {log_date}")

    for i, (header, section_text) in enumerate(sections):
        key = section_key(header, i)
        h = section_hash(section_text)
        existing = posted_sections.get(key)

        if existing and existing["hash"] == h:
            continue

        parts = split_section_into_parts(section_text)
        if not parts:
            continue

        if dry_run:
            action = "UPDATE" if existing else "NEW"
            print(f"\n  --- [{action}] {key} ---")

        if existing:
            part_states = update_section_parts(
                parts, existing.get("parts", []), token, thread_ts, dry_run
            )
            posted_sections[key] = {"hash": h, "parts": part_states}
            updated_count += 1
            if not dry_run:
                print(f"  Updated: {key}")
        else:
            part_states = post_section_parts(parts, token, thread_ts, dry_run)
            if part_states is not None:
                posted_sections[key] = {"hash": h, "parts": part_states}
                new_count += 1
                if not dry_run:
                    print(f"  Posted: {key}")

    if not dry_run:
        day_state["sections"] = posted_sections
        state[log_date] = day_state
        save_state(state)
        if new_count or updated_count:
            print(f"  {log_date}: {new_count} new, {updated_count} updated")
    else:
        total = sum(
            1 for i, (h, s) in enumerate(sections)
            if section_hash(s) != (posted_sections.get(section_key(h, i)) or {}).get("hash")
        )
        if total:
            print(f"\nDry run: {total} section(s) would be posted/updated for {log_date}")


def sync_recent(days, dry_run=False):
    """Sync the last N days of logs."""
    token = None
    if not dry_run:
        token = get_token()
        if not token:
            print(f"Error: No Slack bot token found.")
            print(f"Set SLACK_BOT_TOKEN env var or put token in {TOKEN_FILE}")
            return

    state = load_state()
    migrate_state(state)

    today = date.today()
    for offset in range(days - 1, -1, -1):
        log_date = (today - timedelta(days=offset)).isoformat()
        sync_log(log_date, dry_run=dry_run, token=token)


def main():
    parser = argparse.ArgumentParser(
        description="Sync Obsidian lab log to #lab-log-francisco on Slack"
    )
    parser.add_argument(
        "date",
        nargs="?",
        default=None,
        help="Date to sync (YYYY-MM-DD). If omitted, syncs last 5 days.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be posted without actually posting",
    )
    args = parser.parse_args()

    if args.date:
        token = None
        if not args.dry_run:
            token = get_token()
            if not token:
                print(f"Error: No Slack bot token found.")
                print(f"Set SLACK_BOT_TOKEN env var or put token in {TOKEN_FILE}")
                return
        sync_log(args.date, dry_run=args.dry_run, token=token)
    else:
        sync_recent(LOOKBACK_DAYS, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
