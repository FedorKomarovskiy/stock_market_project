#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXE="${PROJECT_DIR}/venv/bin/python"
if [[ ! -x "$PYTHON_EXE" ]]; then
  PYTHON_EXE="${PYTHON_EXE_OVERRIDE:-python3}"
fi

RUNNER_SCRIPT="${PROJECT_DIR}/run_trade_signal.py"
CONFIG_PATH="${PROJECT_DIR}/config/micro_near_v1_1m.json"
MODEL_PATH="${PROJECT_DIR}/models/near_basis_qlearning.json"
ENV_FILE="${PROJECT_DIR}/.runtime/kucoin.env"
MODE="shadow"
RUN_REAL_ORDER=0
ONCE=0
FORCE_TRAIN=0
TRAIN_IF_MISSING=0

usage() {
  cat <<'EOF'
Usage: ./run_kucoin_trade_signal.sh [options]

Options:
  --config PATH
  --model-path PATH
  --env-file PATH
  --python PATH
  --mode train|shadow|live
  --run-real-order
  --once
  --force-train
  --train-if-missing
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG_PATH="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --env-file) ENV_FILE="$2"; shift 2 ;;
    --python) PYTHON_EXE="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --run-real-order) RUN_REAL_ORDER=1; shift ;;
    --once) ONCE=1; shift ;;
    --force-train) FORCE_TRAIN=1; shift ;;
    --train-if-missing) TRAIN_IF_MISSING=1; shift ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ "$MODE" != "train" && "$MODE" != "shadow" && "$MODE" != "live" ]]; then
  echo "Invalid --mode: $MODE" >&2
  exit 2
fi
if [[ ! -f "$RUNNER_SCRIPT" ]]; then
  echo "Runner script not found: $RUNNER_SCRIPT" >&2
  exit 2
fi

mkdir -p "${PROJECT_DIR}/logs"
LOG_PATH="${PROJECT_DIR}/logs/kucoin_trade_signal_$(date +%Y%m%d_%H%M%S).log"

ARGS=(
  "$RUNNER_SCRIPT"
  "--config" "$CONFIG_PATH"
  "--model-path" "$MODEL_PATH"
  "--env-file" "$ENV_FILE"
  "--mode" "$MODE"
)
if [[ $RUN_REAL_ORDER -eq 1 ]]; then
  ARGS+=("--run-real-order")
fi
if [[ $ONCE -eq 1 ]]; then
  ARGS+=("--once")
fi
if [[ $FORCE_TRAIN -eq 1 ]]; then
  ARGS+=("--force-train")
fi
if [[ $TRAIN_IF_MISSING -eq 1 ]]; then
  ARGS+=("--train-if-missing")
fi

echo "Python       : $PYTHON_EXE"
echo "Runner script: $RUNNER_SCRIPT"
echo "Config       : $CONFIG_PATH"
echo "ModelPath    : $MODEL_PATH"
echo "EnvFile      : $ENV_FILE"
echo "Mode         : $MODE"
echo "RunRealOrder : $RUN_REAL_ORDER"
echo "Once         : $ONCE"
echo "ForceTrain   : $FORCE_TRAIN"
echo "TrainIfMissing: $TRAIN_IF_MISSING"
echo "Log file     : $LOG_PATH"

"$PYTHON_EXE" "${ARGS[@]}" 2>&1 | tee "$LOG_PATH"
STATUS=${PIPESTATUS[0]}
if [[ $STATUS -ne 0 ]]; then
  echo "run_trade_signal.py finished with exit code $STATUS" >&2
  exit $STATUS
fi
echo "Done. ExitCode=0"
