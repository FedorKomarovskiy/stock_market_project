from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: str | Path, overwrite: bool = False) -> dict[str, str]:
    env_path = Path(path)
    loaded: dict[str, str] = {}
    if not env_path.exists():
        return loaded
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lstrip("\ufeff")
        value = value.strip().strip("\"").strip("'")
        if not overwrite and key in os.environ:
            continue
        os.environ[key] = value
        loaded[key] = value
    return loaded
