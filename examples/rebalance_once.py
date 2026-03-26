from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = str((repo_root / "src").resolve())
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    from kucoin_near_basis_rl.live import run_live
    from kucoin_near_basis_rl.runtime_env import load_env_file

    load_env_file(repo_root / ".runtime" / "kucoin.env", overwrite=False)
    run_live(
        config_path=str(repo_root / "config" / "micro_near_v1_1m.json"),
        model_path=str(repo_root / "models" / "near_basis_qlearning.json"),
        paper=True,
        once=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
