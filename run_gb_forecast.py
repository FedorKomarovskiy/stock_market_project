from __future__ import annotations

import argparse
import json
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


def _ensure_pythonpath(repo_root: Path) -> None:
    src_path = str((repo_root / "src").resolve())
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train gradient boosting price models on the last 3 years of KuCoin history."
    )
    parser.add_argument("--config", default="config/project_near_hourly.json", help="Path to JSON config.")
    parser.add_argument(
        "--model-out",
        default="models/project_near_hourly_gb.pkl",
        help="Path to save gradient boosting model artifact.",
    )
    parser.add_argument(
        "--report-dir",
        default="reports/project_near_hourly_gb",
        help="Directory for gradient boosting reports.",
    )
    parser.add_argument("--source-csv", default="", help="Optional local CSV instead of downloading from KuCoin.")
    parser.add_argument("--start", default="", help="UTC ISO start timestamp.")
    parser.add_argument("--end", default="", help="UTC ISO end timestamp.")
    parser.add_argument(
        "--raw-out",
        default="data/project_near_hourly_gb_raw.csv",
        help="Optional path to save merged OHLCV data.",
    )
    parser.add_argument(
        "--features-out",
        default="reports/project_near_hourly_gb/features.csv",
        help="Optional path to save model-ready frame.",
    )
    return parser.parse_args()


def main() -> int:
    configure_console_utf8()
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    _ensure_pythonpath(repo_root)

    from kucoin_near_basis_rl.gb_model import run_gradient_boosting_pipeline

    summary = run_gradient_boosting_pipeline(
        config_path=str((repo_root / args.config).resolve()),
        model_out=str((repo_root / args.model_out).resolve()),
        report_dir=str((repo_root / args.report_dir).resolve()),
        source_csv=str((repo_root / args.source_csv).resolve()) if args.source_csv else None,
        start_iso=args.start or None,
        end_iso=args.end or None,
        raw_out=str((repo_root / args.raw_out).resolve()) if args.raw_out else None,
        features_out=str((repo_root / args.features_out).resolve()) if args.features_out else None,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
