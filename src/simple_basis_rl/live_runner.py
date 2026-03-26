from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from stable_baselines3 import PPO

from delta_bot.execution import ExecutionPlanner, PlannedOrder
from delta_bot.kucoin_client import KuCoinApiError, KuCoinRestClient

from .live_policy import build_live_window_dataframe, decide_action
from .positioning import build_base_neutral_position


@dataclass
class RunnerState:
    position: int = 0
    spot_qty: float = 0.0
    perp_contracts: int = 0
    perp_base_qty: float = 0.0
    spot_entry: float = 0.0
    perp_entry: float = 0.0
    last_action: str = "hold"
    last_equity: float = 0.0
    updated_at: str = ""


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_state(path: Path) -> RunnerState:
    if not path.exists():
        return RunnerState()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RunnerState(**payload)


def _save_state(path: Path, state: RunnerState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2), encoding="utf-8")


def _base_ccy(spot_symbol: str) -> str:
    return spot_symbol.split("-")[0]


def _current_live_positions(client: KuCoinRestClient, symbol_spot: str, symbol_perp: str) -> tuple[float, int]:
    if not client.has_auth:
        return 0.0, 0
    base = _base_ccy(symbol_spot)
    spot_qty = client.get_spot_account_balance(base, account_type="trade")
    perp_contracts = client.get_futures_position_contracts(symbol_perp)
    return float(spot_qty), int(perp_contracts)


def _unrealized_from_state(state: RunnerState, spot_price: float, perp_price: float) -> float:
    if state.position == 1:
        return state.spot_qty * (spot_price - state.spot_entry) - state.perp_base_qty * (
            perp_price - state.perp_entry
        )
    if state.position == -1:
        return -state.spot_qty * (spot_price - state.spot_entry) + state.perp_base_qty * (
            perp_price - state.perp_entry
        )
    return 0.0


def _build_planner(capital_usd: float, symbol_spot: str, symbol_perp: str, contract_size: float) -> ExecutionPlanner:
    instr_cfg = SimpleNamespace(
        spot_symbol=symbol_spot,
        futures_symbol=symbol_perp,
        futures_multiplier_base=contract_size,
        spot_base_increment=0.0001,
        spot_min_funds_usdt=0.1,
        futures_contract_step=1,
        spot_min_size_base=0.0001,
    )
    risk_cfg = SimpleNamespace(
        max_single_order_notional_usdt=max(float(capital_usd), 1.0),
    )
    exec_cfg = SimpleNamespace(order_type="market")
    return ExecutionPlanner(instr_cfg=instr_cfg, risk_cfg=risk_cfg, exec_cfg=exec_cfg)


def _target_from_action(
    *,
    action_name: str,
    spot_price: float,
    perp_price: float,
    capital_usd: float,
    contract_size: float,
    allow_short_spot: bool,
    current_spot_qty: float,
    current_perp_contracts: int,
) -> tuple[float, int, int]:
    # Returns (target_spot_qty, target_perp_contracts, target_position_code)
    if action_name == "hold":
        if current_spot_qty > 0 and current_perp_contracts < 0:
            return current_spot_qty, current_perp_contracts, 1
        if current_spot_qty < 0 and current_perp_contracts > 0:
            return current_spot_qty, current_perp_contracts, -1
        return current_spot_qty, current_perp_contracts, 0

    if action_name == "close":
        return 0.0, 0, 0

    if action_name == "open_short_basis" and not allow_short_spot:
        return current_spot_qty, current_perp_contracts, 0

    side = "long_basis" if action_name == "open_long_basis" else "short_basis"
    built = build_base_neutral_position(
        spot_price=spot_price,
        perp_price=perp_price,
        capital_usd=capital_usd,
        contract_size=contract_size,
        side=side,  # type: ignore[arg-type]
    )
    spot_qty = float(built["spot_qty"])
    perp_contracts = int(built["perp_contracts"])
    if side == "long_basis":
        return spot_qty, -perp_contracts, 1
    return -spot_qty, perp_contracts, -1


def _execute_orders(
    client: KuCoinRestClient,
    orders: list[PlannedOrder],
    max_retries: int = 3,
) -> list[dict[str, Any]]:
    sent: list[dict[str, Any]] = []
    for order in orders:
        for attempt in range(1, max_retries + 1):
            try:
                if order.venue == "spot":
                    resp = client.place_spot_market_order(
                        symbol=order.symbol,
                        side=order.side,
                        size=order.size,
                    )
                elif order.venue == "futures":
                    resp = client.place_futures_market_order(
                        symbol=order.symbol,
                        side=order.side,
                        contracts=int(order.size),
                    )
                else:
                    raise KuCoinApiError(f"Unknown venue: {order.venue}")
                sent.append(
                    {
                        "venue": order.venue,
                        "symbol": order.symbol,
                        "side": order.side,
                        "size": float(order.size),
                        "order_id": str(resp.get("orderId", "")),
                        "attempt": attempt,
                    }
                )
                break
            except Exception as exc:  # noqa: BLE001
                if attempt >= max_retries:
                    raise KuCoinApiError(
                        f"Order failed after {max_retries} attempts: {order.venue} {order.side} "
                        f"{order.symbol} size={order.size}. Error: {exc}"
                    ) from exc
                time.sleep(0.25 * attempt)
    return sent


def run_once(
    *,
    mode: str,
    config_path: str | Path,
    state_file: str | Path,
    rl_model_path: str,
) -> dict[str, Any]:
    cfg = _load_json(config_path)
    symbol_spot = cfg.get("symbol_spot", "NEAR-USDT")
    symbol_perp = cfg.get("symbol_perp", "NEAR-USDTM")
    timeframe = cfg.get("timeframe", "1m")
    rolling_window = int(cfg.get("rolling_window", 100))
    lookback_bars = int(cfg.get("history_lookback_bars", 500))
    capital_usd = float(cfg.get("capital_usd", 1000.0))
    contract_size = float(cfg.get("contract_size", 1.0))
    allow_short_spot = bool(cfg.get("allow_short_spot", True))

    model_path = rl_model_path or cfg.get("rl_model_path", "")
    if not model_path:
        raise ValueError("rl_model_path is required")

    state_path = Path(state_file)
    state = _load_state(state_path)

    client = KuCoinRestClient.from_env()
    window = build_live_window_dataframe(
        client=client,
        symbol_spot=symbol_spot,
        symbol_perp=symbol_perp,
        timeframe=timeframe,
        lookback_bars=lookback_bars,
        rolling_window=rolling_window,
    )
    if window.empty:
        raise ValueError("No market data returned from KuCoin")

    latest = window.iloc[-1]
    spot_price = float(latest["spot"])
    perp_price = float(latest["perp"])

    if mode == "live":
        current_spot_qty, current_perp_contracts = _current_live_positions(
            client=client,
            symbol_spot=symbol_spot,
            symbol_perp=symbol_perp,
        )
    else:
        current_spot_qty = float(state.spot_qty if state.position != -1 else -state.spot_qty)
        current_perp_contracts = int(state.perp_contracts if state.position != 1 else -state.perp_contracts)

    unrealized = _unrealized_from_state(state, spot_price=spot_price, perp_price=perp_price)
    model = PPO.load(model_path)
    action = decide_action(
        model=model,
        market_df=window,
        position=state.position,
        unrealized_pnl=unrealized,
    )

    target_spot_qty, target_perp_contracts, target_pos = _target_from_action(
        action_name=action["action_name"],
        spot_price=spot_price,
        perp_price=perp_price,
        capital_usd=capital_usd,
        contract_size=contract_size,
        allow_short_spot=allow_short_spot,
        current_spot_qty=current_spot_qty,
        current_perp_contracts=current_perp_contracts,
    )

    planner = _build_planner(
        capital_usd=capital_usd,
        symbol_spot=symbol_spot,
        symbol_perp=symbol_perp,
        contract_size=contract_size,
    )
    planned_orders = planner.plan_rebalance(
        current_spot_qty=current_spot_qty,
        current_futures_contracts=current_perp_contracts,
        target_spot_qty=target_spot_qty,
        target_futures_contracts=target_perp_contracts,
        spot_price=spot_price,
        futures_price=perp_price,
    )

    sent_orders: list[dict[str, Any]] = []
    if mode == "live" and planned_orders:
        if not client.has_auth:
            raise KuCoinApiError("Live mode requires KUCOIN_API_KEY/SECRET/PASSPHRASE")
        sent_orders = _execute_orders(client=client, orders=planned_orders)

    if target_pos == 0:
        state.position = 0
        state.spot_qty = 0.0
        state.perp_contracts = 0
        state.perp_base_qty = 0.0
        state.spot_entry = 0.0
        state.perp_entry = 0.0
    else:
        rebuilt = build_base_neutral_position(
            spot_price=spot_price,
            perp_price=perp_price,
            capital_usd=capital_usd,
            contract_size=contract_size,
            side="long_basis" if target_pos == 1 else "short_basis",
        )
        state.position = target_pos
        state.spot_qty = float(rebuilt["spot_qty"])
        state.perp_contracts = int(rebuilt["perp_contracts"])
        state.perp_base_qty = float(rebuilt["perp_base_qty"])
        state.spot_entry = spot_price
        state.perp_entry = perp_price

    state.last_action = str(action["action_name"])
    state.last_equity = float(unrealized)
    state.updated_at = datetime.now(timezone.utc).isoformat()
    _save_state(state_path, state)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "action": action,
        "spot_price": spot_price,
        "perp_price": perp_price,
        "current_spot_qty": current_spot_qty,
        "current_perp_contracts": current_perp_contracts,
        "target_spot_qty": target_spot_qty,
        "target_perp_contracts": target_perp_contracts,
        "planned_orders": [asdict(o) for o in planned_orders],
        "sent_orders": sent_orders,
        "state_file": str(state_path),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Live/shadow runner for simple basis RL bot")
    parser.add_argument("--mode", choices=["shadow", "live"], default="shadow")
    parser.add_argument("--config", default="config/simple_basis_rl.json")
    parser.add_argument("--state-file", default=".runtime/simple_basis_rl_state.json")
    parser.add_argument("--rl-model-path", default="")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--sleep-seconds", type=int, default=60)
    args = parser.parse_args()

    def _run() -> None:
        out = run_once(
            mode=args.mode,
            config_path=args.config,
            state_file=args.state_file,
            rl_model_path=args.rl_model_path,
        )
        print(json.dumps(out, ensure_ascii=False, indent=2))

    if not args.loop:
        _run()
        return

    while True:
        try:
            _run()
        except Exception as exc:  # noqa: BLE001
            print(
                json.dumps(
                    {
                        "status": "error",
                        "error": str(exc),
                        "mode": args.mode,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        time.sleep(max(args.sleep_seconds, 1))


if __name__ == "__main__":
    main()

