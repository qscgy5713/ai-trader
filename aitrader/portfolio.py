"""模擬帳戶:現貨、只做多。以「目標持倉比例」再平衡,計入手續費與滑價。"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import List, Optional

DUST = 1e-12


@dataclass
class Trade:
    ts: float
    side: str  # buy / sell
    price: float  # 成交價(含滑價)
    qty: float
    fee: float
    realized_pnl: float  # 賣出才有;已扣該筆手續費
    reason: str = ""


class Portfolio:
    def __init__(self, cash: float, fee_rate: float = 0.001, slippage_bps: float = 5.0):
        if cash <= 0:
            raise ValueError("初始資金必須 > 0")
        if not 0 <= fee_rate < 0.05:
            raise ValueError("fee_rate 需在 [0, 0.05)")
        if not 0 <= slippage_bps < 500:
            raise ValueError("slippage_bps 需在 [0, 500)")
        self.initial = cash
        self.cash = cash
        self.qty = 0.0
        self.avg_cost = 0.0  # 持倉平均成本(含買入手續費)
        self.fee_rate = fee_rate
        self.slip = slippage_bps / 10000
        self.trades: List[Trade] = []

    def equity(self, price: float) -> float:
        return self.cash + self.qty * price

    def position_pct(self, price: float) -> float:
        eq = self.equity(price)
        return self.qty * price / eq * 100 if eq > 0 else 0.0

    def unrealized_pct(self, price: float) -> float:
        return (price / self.avg_cost - 1) * 100 if self.qty > 0 and self.avg_cost > 0 else 0.0

    def rebalance(self, target_pct: float, price: float, ts: float, reason: str = "") -> Optional[Trade]:
        """調整到目標持倉比例。現金不會為負、持倉不會為負。沒有實際變動則回 None。"""
        if not 0 <= target_pct <= 100:
            raise ValueError("target_pct 需在 0~100")
        eq = self.equity(price)
        diff = eq * target_pct / 100 - self.qty * price  # 要增加(正)/ 減少(負)的市值
        if diff > 0:
            px = price * (1 + self.slip)
            spend = min(diff, self.cash / (1 + self.fee_rate))  # 含手續費不超過現金
            qty = spend / px
            if qty * px < DUST:
                return None
            fee = qty * px * self.fee_rate
            self.avg_cost = (self.avg_cost * self.qty + qty * px + fee) / (self.qty + qty)
            self.cash = max(self.cash - qty * px - fee, 0.0)
            self.qty += qty
            t = Trade(ts, "buy", px, qty, fee, 0.0, reason)
        elif diff < 0:
            px = price * (1 - self.slip)
            qty = min(-diff / price, self.qty)
            if target_pct == 0:
                qty = self.qty  # 清倉就全賣,不留零頭
            if qty * px < DUST:
                return None
            fee = qty * px * self.fee_rate
            pnl = (px - self.avg_cost) * qty - fee
            self.cash += qty * px - fee
            self.qty -= qty
            if self.qty * price < DUST:
                self.qty, self.avg_cost = 0.0, 0.0
            t = Trade(ts, "sell", px, qty, fee, pnl, reason)
        else:
            return None
        self.trades.append(t)
        return t

    # ---- 持久化 ----
    def to_dict(self) -> dict:
        return {"initial": self.initial, "cash": self.cash, "qty": self.qty, "avg_cost": self.avg_cost,
                "fee_rate": self.fee_rate, "slippage_bps": self.slip * 10000,
                "trades": [asdict(t) for t in self.trades[-500:]]}

    @classmethod
    def from_dict(cls, d: dict) -> "Portfolio":
        try:
            p = cls(float(d["initial"]), float(d["fee_rate"]), float(d["slippage_bps"]))
            p.cash, p.qty, p.avg_cost = float(d["cash"]), float(d["qty"]), float(d["avg_cost"])
            p.trades = [Trade(**t) for t in d["trades"]]
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"帳戶狀態格式錯誤:{e!r}") from e
        if not all(math.isfinite(x) for x in (p.cash, p.qty, p.avg_cost)) or p.cash < 0 or p.qty < 0 or p.avg_cost < 0:
            raise ValueError("帳戶狀態數值異常")
        return p
