"""Download the MediaPipe Hand Landmarker model used by the demo.

The model is not committed to the repository. This script downloads the
official float16 task bundle, verifies its SHA-256 digest, and writes it to
``models/hand_landmarker.task`` atomically.

Source:
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

The model is distributed by Google under the Apache License 2.0. See
``models/README.md`` for details.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "models" / "hand_landmarker.task"
URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(output: Path, force: bool = False) -> int:
    if output.exists() and not force:
        actual = sha256(output)
        if actual == SHA256:
            print(f"Model already present and verified: {output}")
            return 0
        print(f"Existing file has a different checksum: {output}", file=sys.stderr)

    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}")
    with tempfile.NamedTemporaryFile(delete=False, dir=output.parent, suffix=".part") as handle:
        temporary = Path(handle.name)
        try:
            with urllib.request.urlopen(URL, timeout=60) as response:
                total = int(response.headers.get("Content-Length", "0"))
                downloaded = 0
                while True:
                    chunk = response.read(1024 * 256)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        print(
                            f"\r{downloaded / 1_048_576:5.1f} / {total / 1_048_576:5.1f} MiB",
                            end="",
                            flush=True,
                        )
            print()
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    actual = sha256(temporary)
    if actual != SHA256:
        temporary.unlink(missing_ok=True)
        print(
            f"Checksum mismatch: expected {SHA256}, got {actual}",
            file=sys.stderr,
        )
        return 1

    os.replace(temporary, output)
    print(f"Wrote {output} ({output.stat().st_size / 1_048_576:.1f} MiB)")
    print(f"SHA-256 {actual}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the Hand Landmarker model.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true", help="download even if the file exists")
    args = parser.parse_args()
    try:
        return download(args.output.expanduser(), args.force)
    except Exception as error:
        print(f"Download failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
