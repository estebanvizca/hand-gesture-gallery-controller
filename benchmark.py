"""Entry point for the guided live benchmark.

    .venv/bin/python benchmark.py --attempts 20 --idle-minutes 5
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from gesture_gallery.benchmark import main

if __name__ == "__main__":
    raise SystemExit(main())
