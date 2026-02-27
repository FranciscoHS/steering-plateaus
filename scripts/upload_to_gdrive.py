#!/usr/bin/env python3
"""Upload a file to Google Drive, optionally converting to Google Slides.

First run opens a browser for OAuth consent. Token is cached for future use.

Usage:
    python upload_to_gdrive.py file.pptx                    # Upload as-is
    python upload_to_gdrive.py file.pptx --as-slides         # Convert to Google Slides
    python upload_to_gdrive.py file.pptx --as-slides --replace EXISTING_FILE_ID
"""

import argparse
import sys
from pathlib import Path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

CONFIG_DIR = Path.home() / ".config" / "google-drive-upload"
TOKEN_FILE = CONFIG_DIR / "token.json"
SCOPES = ["https://www.googleapis.com/auth/drive"]

MIME_TYPES = {
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}

GOOGLE_MIME = {
    ".pptx": "application/vnd.google-apps.presentation",
    ".docx": "application/vnd.google-apps.document",
    ".xlsx": "application/vnd.google-apps.spreadsheet",
}


def find_client_secret():
    for f in CONFIG_DIR.glob("client_secret*.json"):
        return f
    return None


def get_credentials():
    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            client_secret = find_client_secret()
            if not client_secret:
                print(f"Error: No client_secret*.json found in {CONFIG_DIR}")
                sys.exit(1)
            flow = InstalledAppFlow.from_client_secrets_file(str(client_secret), SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN_FILE.write_text(creds.to_json())
    return creds


def upload(filepath, as_google_format=False, replace_id=None):
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"Error: {filepath} not found")
        return

    creds = get_credentials()
    service = build("drive", "v3", credentials=creds)

    mime_type = MIME_TYPES.get(filepath.suffix, "application/octet-stream")
    media = MediaFileUpload(str(filepath), mimetype=mime_type, resumable=True)

    if replace_id:
        # Update existing file
        result = service.files().update(
            fileId=replace_id,
            media_body=media,
        ).execute()
        file_id = result["id"]
        print(f"Updated: https://docs.google.com/presentation/d/{file_id}")
    else:
        metadata = {"name": filepath.stem}
        if as_google_format and filepath.suffix in GOOGLE_MIME:
            metadata["mimeType"] = GOOGLE_MIME[filepath.suffix]
        result = service.files().create(
            body=metadata,
            media_body=media,
            fields="id,webViewLink",
        ).execute()
        file_id = result["id"]
        link = result.get("webViewLink", f"https://drive.google.com/file/d/{file_id}")
        print(f"Uploaded: {link}")

    return file_id


def main():
    parser = argparse.ArgumentParser(description="Upload file to Google Drive")
    parser.add_argument("file", help="File to upload")
    parser.add_argument("--as-slides", action="store_true",
                        help="Convert to Google Slides format")
    parser.add_argument("--replace", metavar="FILE_ID",
                        help="Replace an existing Google Drive file by ID")
    args = parser.parse_args()

    upload(args.file, as_google_format=args.as_slides, replace_id=args.replace)


if __name__ == "__main__":
    main()
