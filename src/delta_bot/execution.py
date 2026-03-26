from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .config import ExecutionConfig, InstrumentsConfig, RiskLimitsConfig
from .math_utils import ceil_to_step, floor_to_step


@dataclass(frozen=True)
class PlannedOrder:
    venue: str
    symbol: str
    side: str
    size: float
    order_type: str


class ExecutionPlanner:
    def __init__(
        self,
        instr_cfg: InstrumentsConfig,
        risk_cfg: RiskLimitsConfig,
        exec_cfg: ExecutionConfig,
    ):
        self.instr_cfg = instr_cfg
        self.risk_cfg = risk_cfg
        self.exec_cfg = exec_cfg

    def plan_rebalance(
        self,
        current_spot_qty: float,
        current_futures_contracts: int,
        target_spot_qty: float,
        target_futures_contracts: int,
        spot_price: float,
        futures_price: float,
    ) -> List[PlannedOrder]:
        orders: List[PlannedOrder] = []

        spot_delta_qty = target_spot_qty - current_spot_qty
        fut_delta_contracts = target_futures_contracts - current_futures_contracts

        orders.extend(
            self._plan_spot_orders(
                spot_delta_qty=spot_delta_qty,
                spot_price=spot_price,
            )
        )
        orders.extend(
            self._plan_futures_orders(
                fut_delta_contracts=fut_delta_contracts,
                futures_price=futures_price,
            )
        )
        return orders

    def _plan_spot_orders(self, spot_delta_qty: float, spot_price: float) -> List[PlannedOrder]:
        if spot_delta_qty == 0:
            return []

        side = "buy" if spot_delta_qty > 0 else "sell"
        abs_qty = abs(spot_delta_qty)
        min_qty_base = max(
            float(getattr(self.instr_cfg, "spot_min_size_base", 0.0)),
            self.instr_cfg.spot_base_increment,
        )
        max_order_qty = self.risk_cfg.max_single_order_notional_usdt / max(spot_price, 1e-12)
        max_order_qty = max(max_order_qty, min_qty_base)
        max_order_qty = floor_to_step(max_order_qty, self.instr_cfg.spot_base_increment)
        float_tol = 1e-9

        chunks: List[float] = []
        remaining = abs_qty
        while remaining > 0:
            chunk = min(remaining, max_order_qty)
            chunk = floor_to_step(chunk, self.instr_cfg.spot_base_increment)
            if chunk <= 0:
                break
            if chunk < min_qty_base:
                if 0 < (min_qty_base - remaining) <= float_tol:
                    chunk = min_qty_base
                else:
                    # Exchange rejects tiny leftovers; leave them as residual delta.
                    break
            if chunk * spot_price < self.instr_cfg.spot_min_funds_usdt:
                min_qty_by_funds = ceil_to_step(
                    self.instr_cfg.spot_min_funds_usdt / max(spot_price, 1e-12),
                    self.instr_cfg.spot_base_increment,
                )
                min_required = max(min_qty_base, min_qty_by_funds)
                if (min_required - remaining) > float_tol or (min_required <= 0):
                    break
                chunk = min_required
            chunks.append(chunk)
            remaining -= chunk

        return [
            PlannedOrder(
                venue="spot",
                symbol=self.instr_cfg.spot_symbol,
                side=side,
                size=c,
                order_type=self.exec_cfg.order_type,
            )
            for c in chunks
        ]

    def _plan_futures_orders(
        self, fut_delta_contracts: int, futures_price: float
    ) -> List[PlannedOrder]:
        if fut_delta_contracts == 0:
            return []

        side = "buy" if fut_delta_contracts > 0 else "sell"
        abs_contracts = abs(fut_delta_contracts)
        max_contracts = int(
            self.risk_cfg.max_single_order_notional_usdt
            / max(futures_price * self.instr_cfg.futures_multiplier_base, 1e-12)
        )
        max_contracts = max(max_contracts, self.instr_cfg.futures_contract_step)

        chunks: List[int] = []
        remaining = abs_contracts
        while remaining > 0:
            chunk = min(remaining, max_contracts)
            chunk = int(
                floor_to_step(float(chunk), float(self.instr_cfg.futures_contract_step))
            )
            if chunk <= 0:
                break
            chunks.append(chunk)
            remaining -= chunk

        return [
            PlannedOrder(
                venue="futures",
                symbol=self.instr_cfg.futures_symbol,
                side=side,
                size=float(c),
                order_type=self.exec_cfg.order_type,
            )
            for c in chunks
        ]
