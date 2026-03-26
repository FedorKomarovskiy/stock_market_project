from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def configure_console_utf8() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _ensure_pythonpath(repo_root: Path) -> None:
    src_path = str((repo_root / "src").resolve())
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executor for NEAR spot/futures RL basis strategy on KuCoin."
    )
    parser.add_argument(
        "--config",
        default="config/micro_near_v1_1m.json",
        help="Strategy config path.",
    )
    parser.add_argument(
        "--model-path",
        default="models/near_basis_qlearning.json",
        help="Path to trained RL model artifact.",
    )
    parser.add_argument(
        "--mode",
        choices=["shadow", "live", "train"],
        default="shadow",
        help="shadow=train-disabled paper trading, live=real orders, train=train model only.",
    )
    parser.add_argument(
        "--run-real-order",
        action="store_true",
        help="Allow real order sending (forces live mode).",
    )
    parser.add_argument("--once", action="store_true", help="Run one decision cycle and exit.")
    parser.add_argument(
        "--env-file",
        default=".runtime/kucoin.env",
        help="Env file with KUCOIN_API_KEY / KUCOIN_API_SECRET / KUCOIN_API_PASSPHRASE.",
    )
    parser.add_argument(
        "--features-out",
        default="reports/near_basis_features.csv",
        help="Where to dump engineered features during training.",
    )
    parser.add_argument("--source-csv", default="", help="Optional local source CSV for training.")
    parser.add_argument("--start", default="", help="UTC ISO start for training history.")
    parser.add_argument("--end", default="", help="UTC ISO end for training history.")
    parser.add_argument("--episodes", type=int, default=0, help="Override training episodes for faster runs.")
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Train model before execution even if model already exists.",
    )
    parser.add_argument(
        "--train-if-missing",
        action="store_true",
        help="Train model if model file does not exist.",
    )
    parser.add_argument("--show-command-context", action="store_true", help="Print resolved paths and env source.")
    return parser.parse_args()


def _load_runtime_env(env_file: Path) -> dict[str, str]:
    from kucoin_near_basis_rl.runtime_env import load_env_file

    return load_env_file(env_file, overwrite=False)


def _train_model_if_needed(args: argparse.Namespace, repo_root: Path) -> None:
    from kucoin_near_basis_rl.train import run_training

    model_path = (repo_root / args.model_path).resolve()
    should_train = args.mode == "train" or args.force_train or (
        args.train_if_missing and not model_path.exists()
    )
    if not should_train:
        return

    model_path.parent.mkdir(parents=True, exist_ok=True)
    if args.features_out:
        (repo_root / args.features_out).resolve().parent.mkdir(parents=True, exist_ok=True)

    metrics = run_training(
        config_path=str((repo_root / args.config).resolve()),
        model_out=str(model_path),
        source_csv=str((repo_root / args.source_csv).resolve()) if args.source_csv else None,
        start_iso=args.start or None,
        end_iso=args.end or None,
        features_out=str((repo_root / args.features_out).resolve()) if args.features_out else None,
        episodes_override=(int(args.episodes) if args.episodes and args.episodes > 0 else None),
    )
    print("Training complete:")
    print(json.dumps(metrics, indent=2))


def _run_live_or_shadow(args: argparse.Namespace, repo_root: Path) -> None:
    from kucoin_near_basis_rl.live import run_live

    live_mode = args.run_real_order or args.mode == "live"
    run_live(
        config_path=str((repo_root / args.config).resolve()),
        model_path=str((repo_root / args.model_path).resolve()),
        paper=not live_mode,
        once=args.once,
    )


def main() -> int:
    configure_console_utf8()
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    _ensure_pythonpath(repo_root)

    env_file = (repo_root / args.env_file).resolve()
    loaded = _load_runtime_env(env_file)
    if args.show_command_context:
        print("Repo root:", repo_root)
        print("Config:", (repo_root / args.config).resolve())
        print("Model:", (repo_root / args.model_path).resolve())
        print("Env file:", env_file)
        print("Loaded env vars:", sorted(loaded.keys()))

    if args.mode in {"train", "shadow", "live"}:
        _train_model_if_needed(args, repo_root)

    if args.mode == "train":
        return 0

    # KuCoin SDK auth requires passphrase in addition to key/secret for live orders.
    if args.run_real_order or args.mode == "live":
        required = ["KUCOIN_API_KEY", "KUCOIN_API_SECRET", "KUCOIN_API_PASSPHRASE"]
        missing = [name for name in required if not os.getenv(name)]
        if missing:
            raise RuntimeError(
                f"Missing required env vars for live mode: {', '.join(missing)}. "
                f"Fill {env_file} or set env vars explicitly."
            )

    _run_live_or_shadow(args, repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
