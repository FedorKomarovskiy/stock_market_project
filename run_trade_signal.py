from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def configure_console_utf8() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-platform launcher for trade_signal_executor_kucoin.py. "
            "Supports train / shadow / live for RL basis strategy."
        )
    )
    parser.add_argument(
        "--config",
        default="config/micro_near_v1_1m.json",
        help="Bot config path.",
    )
    parser.add_argument(
        "--model-path",
        default="models/near_basis_qlearning.json",
        help="Trained RL model path.",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "shadow", "live"],
        default="shadow",
        help="train: train only; shadow: paper mode; live: send real orders.",
    )
    parser.add_argument(
        "--run-real-order",
        action="store_true",
        help="Force live mode and order execution.",
    )
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit.")
    parser.add_argument(
        "--env-file",
        default=".runtime/kucoin.env",
        help="Path to local env file with KuCoin credentials.",
    )
    parser.add_argument(
        "--features-out",
        default="reports/near_basis_features.csv",
        help="Path to save engineered features on training.",
    )
    parser.add_argument("--source-csv", default="", help="Optional source CSV for training.")
    parser.add_argument("--start", default="", help="UTC ISO start for training history.")
    parser.add_argument("--end", default="", help="UTC ISO end for training history.")
    parser.add_argument("--episodes", type=int, default=0, help="Optional override for training episodes.")
    parser.add_argument("--force-train", action="store_true", help="Train even if model exists.")
    parser.add_argument("--train-if-missing", action="store_true", help="Train if model is missing.")
    parser.add_argument("--python-exe", default=sys.executable, help="Python executable to use.")
    parser.add_argument("--show-command", action="store_true", help="Print exact command.")
    return parser.parse_args()


def main() -> int:
    configure_console_utf8()
    args = parse_args()

    repo_root = Path(__file__).resolve().parent
    executor = repo_root / "trade_signal_executor_kucoin.py"
    if not executor.exists():
        print(f"Executor script not found: {executor}", file=sys.stderr)
        return 2

    cmd = [
        str(Path(args.python_exe)),
        str(executor),
        "--config",
        str(args.config),
        "--model-path",
        str(args.model_path),
        "--mode",
        str(args.mode),
        "--env-file",
        str(args.env_file),
        "--features-out",
        str(args.features_out),
    ]
    if args.run_real_order:
        cmd.append("--run-real-order")
    if args.once:
        cmd.append("--once")
    if args.source_csv:
        cmd.extend(["--source-csv", args.source_csv])
    if args.start:
        cmd.extend(["--start", args.start])
    if args.end:
        cmd.extend(["--end", args.end])
    if args.episodes and args.episodes > 0:
        cmd.extend(["--episodes", str(args.episodes)])
    if args.force_train:
        cmd.append("--force-train")
    if args.train_if_missing:
        cmd.append("--train-if-missing")
    if args.show_command:
        cmd.append("--show-command-context")

    print("Python      :", args.python_exe)
    print("Executor    :", executor)
    print("Mode        :", args.mode)
    print("RunRealOrder:", bool(args.run_real_order))
    print("Config      :", args.config)
    print("Model       :", args.model_path)
    print("Env file    :", args.env_file)
    if args.show_command:
        print("Command     :", " ".join(cmd))

    completed = subprocess.run(cmd, env=os.environ.copy())
    return int(completed.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
