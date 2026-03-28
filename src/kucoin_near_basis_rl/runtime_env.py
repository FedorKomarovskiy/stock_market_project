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


def resolve_env_path(repo_root: str | Path, env_file: str | Path | None = None) -> Path:
    root = Path(repo_root)
    if env_file is None or str(env_file).strip() == "":
        return (root / ".runtime" / "project.env").resolve()
    candidate = Path(env_file)
    if candidate.is_absolute():
        return candidate.resolve()
    return (root / candidate).resolve()


def load_repo_env(
    repo_root: str | Path,
    env_file: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, str]:
    root = Path(repo_root).resolve()
    primary = resolve_env_path(root, env_file)
    fallback = (root / ".runtime" / "kucoin.env").resolve()

    loaded: dict[str, str] = {}
    seen: set[Path] = set()
    for candidate in (primary, fallback):
        if candidate in seen:
            continue
        seen.add(candidate)
        loaded.update(load_env_file(candidate, overwrite=overwrite))
    return loaded
