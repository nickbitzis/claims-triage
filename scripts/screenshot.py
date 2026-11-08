"""Capture README screenshots with headless Chrome. Start the server first: make run.

Usage: uv run python scripts/screenshot.py --port 8834
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path

CHROME_PATHS = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome",
    "chromium",
]
DOCS = Path(__file__).resolve().parents[1] / "docs"


def chrome() -> str:
    for c in CHROME_PATHS:
        if Path(c).exists() or shutil.which(c):
            return c
    raise SystemExit("Chrome not found")


def shoot(url: str, out: Path, width: int, height: int, dark: bool = False) -> None:
    args = [
        chrome(),
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        f"--window-size={width},{height}",
        "--virtual-time-budget=8000",
        f"--screenshot={out}",
    ]
    if dark:
        args += ["--force-dark-mode", "--blink-settings=preferredColorScheme=0"]
    subprocess.run([*args, url], check=True, capture_output=True, timeout=90)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8834)
    ap.add_argument("--claim", default=None, help="claim id for the detail page; defaults to the top of the SIU queue")
    ap.add_argument("--out", default=str(DOCS))
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"
    claim = a.claim
    if claim is None:
        with urllib.request.urlopen(f"{base}/api/queues/siu?limit=1") as r:
            claim = json.load(r)["items"][0]["claim_id"]
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    shoot(f"{base}/#/policy", out / "screenshot.png", 1400, 1250)
    shoot(f"{base}/#/claim/{claim}", out / "claim-detail.png", 1400, 1900)
    shoot(f"{base}/#/queues/siu", out / "queues.png", 1400, 900, dark=True)


if __name__ == "__main__":
    main()
