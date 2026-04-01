from __future__ import annotations

import sys
from pathlib import Path


def find_project_root(start: Path, marker_dir: str = "src") -> Path:
    """
    Find project root by searching for a marker directory (default: 'src').
    """
    for p in [start] + list(start.parents):
        if (p / marker_dir).exists():
            return p
    raise FileNotFoundError(
        f"Cannot find project root with a '{marker_dir}/' folder from start={start}."
    )


def ensure_project_on_syspath(project_root: Path) -> None:
    """
    Ensure project root is at the front of sys.path.
    """
    pr = str(project_root)
    if pr not in sys.path:
        sys.path.insert(0, pr)