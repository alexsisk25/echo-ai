"""CLI wrapper for the Voice Memos sync. The logic lives in app/sync.py
so the web API can call it too.

Run manually with: .venv/bin/python scripts/sync_voice_memos.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import sync  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        copied = sync.sync_voice_memos()
    except PermissionError as err:
        # Distinct from "nothing new": the folder could not be read at
        # all, which is a setup problem, not an empty inbox.
        print(f"Could not read the Voice Memos folder.\n  {err}\n")
        print(
            "Grant Full Disk Access in System Settings > Privacy &\n"
            "Security > Full Disk Access to the app running this script,\n"
            "then retry."
        )
        return 1
    if copied == 0:
        print("Done. The folder was readable and held nothing new.")
    else:
        print(f"Done. {copied} new recording(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
