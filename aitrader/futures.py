"""模擬 USDT 永續合約帳戶(單向持倉、全倉保證金)。只模擬,不下真單。

- 持倉 qty 可正(多)可負(空);以「目標曝險 %」(持倉名目價值 / 權益,帶正負號)調整,100% = 1 倍槓桿。
- 權益 = 錢包餘額 + 未實現損益。成交計手續費與滑價;反手(多翻空)一次完成,已實現損益照實結算。
- 資金費:每次結算支付 qty × 價格 × 費率(費率為正時多方付給空方)。
- 強平:權益 <= 維持保證金(名目價值 × mmr)時,以當下價格強制平倉,並扣清算費(名目價值 × liq_fee)。
簡化:以單一價格同時當成交價與標記價格;維持保證金率固定(實際依倉位級距而定)。
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import List, Optional

DUST = 1e-12


@dataclass
class FTrade:
    ts: float
    side: str  # buy / sell
    price: float  # 成交價(含滑價)
    qty: float
    fee: float
    realized_pnl: float  # 這筆平掉的部分的損益(未扣手續費)
    reason: str = ""


class FuturesAccount:
    def __init__(self, cash: float, fee_rate: float = 0.0005, slippage_bps: float = 2.0, mmr: float = 0.005,
                 liq_fee: float = 0.005):
        if cash <= 0:
            raise ValueError("初始資金必須 > 0")
        if not 0 <= fee_rate < 0.05:
            raise ValueError("fee_rate 需在 [0, 0.05)")
        if not 0 <= slippage_bps < 500:
            raise ValueError("slippage_bps 需在 [0, 500)")
        if not 0 < mmr < 0.5 or not 0 <= liq_fee < 0.5:
            raise ValueError("mmr / liq_fee 不合理")
        self.initial = cash
        self.wallet = cash
        self.qty = 0.0
        self.entry = 0.0  # 持倉均價
        self.fee_rate, self.slip, self.mmr, self.liq_fee = fee_rate, slippage_bps / 10000, mmr, liq_fee
        self.funding_paid = 0.0  # 累計支付的資金費(負值 = 收到)
        self.fees_paid = 0.0
        self.liquidated = False  # 曾被強平(帳戶可能已接近歸零)
        self.trades: List[FTrade] = []  # 只保留最近 500 筆;累計數字看下面兩個計數
        self.trade_count = 0
        self.realized_total = 0.0  # 累計已實現損益(未扣手續費與資金費)

    # ---- 查詢 ----
    def upnl(self, price: float) -> float:
        return self.qty * (price - self.entry)

    def equity(self, price: float) -> float:
        return self.wallet + self.upnl(price)

    def exposure_pct(self, price: float) -> float:
        """帶正負號的曝險 %(多 > 0、空 < 0);100 = 1 倍槓桿。"""
        eq = self.equity(price)
        return self.qty * price / eq * 100 if eq > 0 else 0.0

    def position_loss_pct(self, price: float) -> float:
        """持倉未實現虧損占錢包餘額的 %(獲利時為負)。"""
        return -self.upnl(price) / self.wallet * 100 if self.wallet > 0 else 0.0

    def liq_price(self) -> Optional[float]:
        """強平價:equity == mmr × |qty| × p 的價格;空手回 None。"""
        q = self.qty
        if abs(q) < DUST:
            return None
        # wallet + q(p - entry) = mmr |q| p  →  p = (wallet - q·entry) / (mmr|q| - q)
        den = self.mmr * abs(q) - q
        p = (self.wallet - q * self.entry) / den
        return p if p > 0 else None

    def stop_price(self, loss_pct: float) -> Optional[float]:
        """持倉虧損達錢包 loss_pct% 的價格。"""
        if abs(self.qty) < DUST:
            return None
        p = self.entry - loss_pct / 100 * self.wallet / self.qty
        return p if p > 0 else None

    def is_liquidatable(self, price: float) -> bool:
        return abs(self.qty) > DUST and self.equity(price) <= self.mmr * abs(self.qty) * price

    # ---- 交易 ----
    def rebalance(self, target_pct: float, price: float, ts: float, reason: str = "") -> Optional[FTrade]:
        """調整到目標曝險 %(可為負)。沒有實際變動回 None。槓桿上限由風控負責。"""
        if not math.isfinite(target_pct):
            raise ValueError("target_pct 不合法")
        eq = self.equity(price)
        if eq <= 0 and target_pct != 0:
            return None
        target_qty = 0.0 if target_pct == 0 else eq * target_pct / 100 / price
        delta = target_qty - self.qty
        if abs(delta) * price < 1e-8:
            return None
        px = price * (1 + self.slip) if delta > 0 else price * (1 - self.slip)
        fee = abs(delta) * px * self.fee_rate
        realized = 0.0
        q = self.qty
        if q != 0 and (delta > 0) != (q > 0):  # 有減倉 / 平倉的部分
            closed = min(abs(delta), abs(q))
            realized = closed * (px - self.entry) * (1 if q > 0 else -1)
        new_q = q + delta
        if abs(new_q) * price < 1e-8:
            new_q, self.entry = 0.0, 0.0
        elif q == 0 or (new_q > 0) != (q > 0):  # 新開倉或反手:均價 = 成交價
            self.entry = px
        elif abs(new_q) > abs(q):  # 同方向加倉:加權均價
            self.entry = (self.entry * abs(q) + px * abs(delta)) / abs(new_q)
        self.qty = new_q
        self.wallet += realized - fee
        self.fees_paid += fee
        t = FTrade(ts, "buy" if delta > 0 else "sell", px, abs(delta), fee, realized, reason)
        self._record(t)
        return t

    def _record(self, t: FTrade) -> None:
        self.trades.append(t)
        del self.trades[:-500]
        self.trade_count += 1
        self.realized_total += t.realized_pnl

    def liquidate(self, price: float, ts: float) -> FTrade:
        """強制平倉:以 price 平倉並扣清算費;錢包不會低於 0(虧損超過保證金視為由保險基金承擔)。"""
        q = self.qty
        notional = abs(q) * price
        realized = q * (price - self.entry)
        fee = notional * self.liq_fee
        self.wallet = max(0.0, self.wallet + realized - fee)
        self.qty, self.entry = 0.0, 0.0
        self.fees_paid += fee
        self.liquidated = True
        t = FTrade(ts, "sell" if q > 0 else "buy", price, abs(q), fee, realized, "強制平倉")
        self._record(t)
        return t

    def apply_funding(self, rate: float, price: float) -> float:
        """資金費結算;回傳這次支付的金額(負值 = 收到)。"""
        pay = self.qty * price * rate
        self.wallet -= pay
        self.funding_paid += pay
        return pay

    # ---- 持久化 ----
    def to_dict(self) -> dict:
        return {"initial": self.initial, "wallet": self.wallet, "qty": self.qty, "entry": self.entry,
                "fee_rate": self.fee_rate, "slippage_bps": self.slip * 10000, "mmr": self.mmr, "liq_fee": self.liq_fee,
                "funding_paid": self.funding_paid, "fees_paid": self.fees_paid, "liquidated": self.liquidated,
                "trade_count": self.trade_count, "realized_total": self.realized_total,
                "trades": [asdict(t) for t in self.trades[-500:]]}

    @classmethod
    def from_dict(cls, d: dict) -> "FuturesAccount":
        try:
            a = cls(float(d["initial"]), float(d["fee_rate"]), float(d["slippage_bps"]), float(d["mmr"]),
                    float(d["liq_fee"]))
            a.wallet, a.qty, a.entry = float(d["wallet"]), float(d["qty"]), float(d["entry"])
            a.funding_paid, a.fees_paid = float(d["funding_paid"]), float(d["fees_paid"])
            a.liquidated = bool(d["liquidated"])
            a.trades = [FTrade(**t) for t in d["trades"]]
            # 舊版狀態檔沒有累計欄位:以保留的成交紀錄估計
            a.trade_count = int(d.get("trade_count", len(a.trades)))
            a.realized_total = float(d.get("realized_total", sum(t.realized_pnl for t in a.trades)))
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"合約帳戶狀態格式錯誤:{e!r}") from e
        if not all(math.isfinite(x) for x in (a.wallet, a.qty, a.entry)) or a.wallet < 0 or a.entry < 0:
            raise ValueError("合約帳戶狀態數值異常")
        return a
