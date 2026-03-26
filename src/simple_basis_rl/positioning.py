from __future__ import annotations

from typing import Any, Literal


PositionSide = Literal["long_basis", "short_basis"]


def compute_spot_qty(capital_usd: float, spot_price: float) -> float:
    if capital_usd <= 0:
        raise ValueError("capital_usd must be > 0")
    if spot_price <= 0:
        raise ValueError("spot_price must be > 0")
    return capital_usd / spot_price


def compute_perp_contracts(spot_qty: float, contract_size: float) -> tuple[int, float]:
    if spot_qty < 0:
        raise ValueError("spot_qty must be >= 0")
    if contract_size <= 0:
        raise ValueError("contract_size must be > 0")
    perp_contracts = int(round(spot_qty / contract_size))
    perp_base_qty = float(perp_contracts) * contract_size
    return perp_contracts, perp_base_qty


def build_base_neutral_position(
    *,
    spot_price: float,
    perp_price: float,
    capital_usd: float,
    contract_size: float,
    side: PositionSide,
) -> dict[str, Any]:
    if perp_price <= 0:
        raise ValueError("perp_price must be > 0")
    if side not in {"long_basis", "short_basis"}:
        raise ValueError("side must be long_basis or short_basis")

    spot_qty = compute_spot_qty(capital_usd=capital_usd, spot_price=spot_price)
    perp_contracts, perp_base_qty = compute_perp_contracts(
        spot_qty=spot_qty,
        contract_size=contract_size,
    )
    hedge_error = spot_qty - perp_base_qty

    spot_direction = 1 if side == "long_basis" else -1
    perp_direction = -1 if side == "long_basis" else 1

    spot_notional = spot_qty * spot_price
    perp_notional = perp_base_qty * perp_price
    gross_notional = spot_notional + perp_notional

    return {
        "spot_qty": float(spot_qty),
        "perp_contracts": int(perp_contracts),
        "perp_base_qty": float(perp_base_qty),
        "hedge_error": float(hedge_error),
        "spot_notional": float(spot_notional),
        "perp_notional": float(perp_notional),
        "gross_notional": float(gross_notional),
        "spot_direction": int(spot_direction),
        "perp_direction": int(perp_direction),
    }


def validate_position(position: dict[str, Any], tolerance: float = 1e-9) -> None:
    required = {
        "spot_qty",
        "perp_contracts",
        "perp_base_qty",
        "hedge_error",
        "spot_notional",
        "perp_notional",
        "gross_notional",
        "spot_direction",
        "perp_direction",
    }
    missing = required - set(position.keys())
    if missing:
        raise ValueError(f"Position is missing fields: {sorted(missing)}")

    if float(position["spot_qty"]) < 0:
        raise ValueError("spot_qty must be >= 0")
    if int(position["perp_contracts"]) < 0:
        raise ValueError("perp_contracts must be >= 0")
    if float(position["perp_base_qty"]) < 0:
        raise ValueError("perp_base_qty must be >= 0")
    if int(position["spot_direction"]) not in {-1, 1}:
        raise ValueError("spot_direction must be -1 or 1")
    if int(position["perp_direction"]) not in {-1, 1}:
        raise ValueError("perp_direction must be -1 or 1")
    if int(position["spot_direction"]) == int(position["perp_direction"]):
        raise ValueError("spot and perp directions must be opposite")

    expected_error = float(position["spot_qty"]) - float(position["perp_base_qty"])
    if abs(expected_error - float(position["hedge_error"])) > tolerance:
        raise ValueError("hedge_error is inconsistent with spot_qty/perp_base_qty")

