#!/usr/bin/env python3
"""Sync Obsidian daily lab log to #lab-log-francisco on Slack.

Reads today's log file, splits into sections by bold headers,
converts Obsidian markdown to Slack format, and posts new sections.
Images referenced with ![[file.png]] are uploaded as thread replies.

Usage:
    python sync_obsidian_to_slack.py              # Sync today's log
    python sync_obsidian_to_slack.py 2026-02-27   # Sync a specific date
    python sync_obsidian_to_slack.py --dry-run    # Preview without posting
    python sync_obsidian_to_slack.py --install-cron  # Print cron setup

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
from datetime import date
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


def get_token():
    """Get Slack bot token from env var or config file."""
    token = os.environ.get("SLACK_BOT_TOKEN")
    if token:
        return token.strip()
    if TOKEN_FILE.exists():
        return TOKEN_FILE.read_text().strip()
    return None


def slack_post(token, channel, text):
    """Post a message to Slack. Returns message ts on success."""
    resp = requests.post(
        f"{SLACK_API}/chat.postMessage",
        headers={"Authorization": f"Bearer {token}"},
        json={"channel": channel, "text": text},
    )
    data = resp.json()
    if not data.get("ok"):
        print(f"  Slack error: {data.get('error', 'unknown')}")
        return None
    return data.get("ts")


def slack_upload_file(token, channel, filepath, thread_ts=None):
    """Upload a file to Slack using the v2 upload flow."""
    filepath = Path(filepath)
    if not filepath.exists():
        return None

    file_size = filepath.stat().st_size
    filename = filepath.name

    # Step 1: Get upload URL
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

    # Step 2: Upload file content
    with open(filepath, "rb") as f:
        resp = requests.post(upload_url, files={"file": (filename, f)})
    if resp.status_code != 200:
        print(f"  File upload failed: HTTP {resp.status_code}")
        return None

    # Step 3: Complete upload and share in channel/thread
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
    # Skip lines that are task items or blank lines between them
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("- [") or (stripped == "" and i > 0):
            i += 1
        else:
            break
    # Also skip any leading blank lines after the task block
    while i < len(lines) and lines[i].strip() == "":
        i += 1
    return "\n".join(lines[i:])


def split_into_sections(content):
    """Split content into sections by bold headers (**...**).

    Each section starts with a bold header line (if present) and includes
    all content up to the next bold header. Content before the first bold
    header (if any) forms its own section.
    """
    sections = []
    current_lines = []

    for line in content.split("\n"):
        if re.match(r"^\s*\*\*[^*]+\*\*\s*$", line):
            # Bold header — start a new section
            text = "\n".join(current_lines).strip()
            if text:
                sections.append(text)
            current_lines = [line]
        else:
            current_lines.append(line)

    # Last section
    text = "\n".join(current_lines).strip()
    if text:
        sections.append(text)

    return sections


def find_image(filename):
    """Find an image file in the Obsidian vault.

    Obsidian stores attachments at the vault root by default.
    Also checks the project directory as a fallback.
    """
    # Vault root (where Obsidian stores attachments by default)
    path = VAULT_DIR / filename
    if path.exists():
        return path
    # Project directory
    path = LOG_DIR / filename
    if path.exists():
        return path
    return None


def extract_images(text):
    """Extract image filenames referenced via ![[file.png]] in text."""
    return re.findall(
        r"!\[\[([^\]]+\.(?:png|jpg|jpeg|gif|svg))\]\]", text, re.IGNORECASE
    )


def obsidian_to_slack(text):
    """Convert Obsidian markdown to Slack mrkdwn format.

    Handles: image embeds, wikilinks, bold markers, markdown links,
    LaTeX blocks. Leaves inline LaTeX ($...$) as-is since it's
    readable enough in plain text.
    """
    # Remove image embeds (handled separately via file upload)
    text = re.sub(
        r"!\[\[([^\]]+\.(?:png|jpg|jpeg|gif|svg))\]\]\n?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Wikilinks: [[target|display]] -> display, [[target]] -> target
    text = re.sub(r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)

    # Bold: **text** -> *text*  (Slack uses single * for bold)
    text = re.sub(r"\*\*([^*]+)\*\*", r"*\1*", text)

    # Markdown links: [text](url) -> <url|text>
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<\2|\1>", text)

    # LaTeX block equations -> bracketed notation
    text = re.sub(
        r"\\begin\{equation\}(.*?)\\end\{equation\}",
        lambda m: "[equation: " + m.group(1).strip() + "]",
        text,
        flags=re.DOTALL,
    )

    # Collapse 3+ consecutive blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# --- State management ---


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


# --- Main logic ---


def sync_log(log_date, dry_run=False):
    """Sync a day's log to Slack. Only posts sections not yet posted."""
    if not dry_run:
        token = get_token()
        if not token:
            print(f"Error: No Slack bot token found.")
            print(f"Set SLACK_BOT_TOKEN env var or put token in {TOKEN_FILE}")
            return
    else:
        token = None

    content = get_log_content(log_date)
    if not content:
        print(f"No log found for {log_date}")
        return

    content = strip_task_list(content)
    if not content.strip():
        print(f"Log for {log_date} is empty after stripping task list")
        return

    sections = split_into_sections(content)
    state = load_state()
    posted = set(state.get(log_date, []))
    new_count = 0

    for i, section in enumerate(sections):
        h = section_hash(section)
        if h in posted:
            continue

        slack_text = obsidian_to_slack(section)
        if not slack_text:
            continue

        images = extract_images(section)

        if dry_run:
            preview = slack_text[:300] + ("..." if len(slack_text) > 300 else "")
            print(f"\n--- Section {i+1} ({len(slack_text)} chars) ---")
            print(preview)
            if images:
                found = []
                missing = []
                for img in images:
                    if find_image(img):
                        found.append(img)
                    else:
                        missing.append(img)
                if found:
                    print(f"  Images to upload: {', '.join(found)}")
                if missing:
                    print(f"  Images not found: {', '.join(missing)}")
        else:
            ts = slack_post(token, SLACK_CHANNEL, slack_text)
            if ts:
                # Upload images as thread replies
                for img_name in images:
                    img_path = find_image(img_name)
                    if img_path:
                        slack_upload_file(token, SLACK_CHANNEL, img_path, thread_ts=ts)
                        print(f"  Uploaded {img_name}")
                    else:
                        print(f"  Image not found: {img_name}")

                posted.add(h)
                new_count += 1
                print(f"  Posted section {i+1} ({len(slack_text)} chars)")

    if not dry_run:
        state[log_date] = list(posted)
        save_state(state)
        print(f"\nDone: {new_count} new section(s) posted for {log_date}")
    else:
        new = sum(1 for s in sections if section_hash(s) not in posted)
        print(f"\nDry run: {new} section(s) would be posted for {log_date}")


def install_cron():
    """Print cron entries for twice-daily sync."""
    script_path = Path(__file__).resolve()
    python = sys.executable
    cron_lines = [
        f"0 8 * * 1-5 {python} {script_path}",
        f"0 18 * * 1-5 {python} {script_path}",
    ]
    print("Add these lines to your crontab (crontab -e):\n")
    for line in cron_lines:
        print(f"  {line}")
    print(f"\nMake sure your bot token is in {TOKEN_FILE}")
    print("(Or set SLACK_BOT_TOKEN in the cron environment)")


def main():
    parser = argparse.ArgumentParser(
        description="Sync Obsidian lab log to #lab-log-francisco on Slack"
    )
    parser.add_argument(
        "date",
        nargs="?",
        default=date.today().isoformat(),
        help="Date to sync (YYYY-MM-DD), defaults to today",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be posted without actually posting",
    )
    parser.add_argument(
        "--install-cron",
        action="store_true",
        help="Print cron entries for automatic scheduling",
    )
    args = parser.parse_args()

    if args.install_cron:
        install_cron()
    else:
        sync_log(args.date, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
