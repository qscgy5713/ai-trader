"""現貨網格引擎(純邏輯,不碰網路與時間)。移植自 crypto-grid-bot,改為等比網格並支援從既有持倉重建。

規則:
- [lower, upper] 等比切成 grids 格,共 grids+1 條價位線;現價所在處留一個空檔,下方掛買單、上方掛賣單。
- 買單在 level i 成交 -> 在 level i+1 掛賣單;賣單在 level i 成交 -> 在 level i-1 掛買單。
- 成交採「穿價」:價格必須越過掛單價才算成交,剛好碰到不算。
- 建立網格時,以市價把持幣調整到「上方賣單所需數量」,只交易差額(重設網格時不必先全部賣掉),計入手續費與滑價。
- 每格數量固定,資金預先依所有買單分配,保證不透支。
- 停損:連續行情以停損價出場;跳空(gap=True,如停機後接續)以當下價格出場。市價出場都再扣滑價。
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

REL_TOL = 1e-9
STATE_VERSION = 1
MAX_SAVED_FILLS = 500


@dataclass
class GridConfig:
    lower: float
    upper: float
    grids: int
    fee_rate: float = 0.001  # 單邊手續費率
    slippage: float = 0.0005  # 市價單滑價比例(限價網格單不計)
    stop_loss: Optional[float] = None  # 跌破此價全數賣出並停機

    def validate(self) -> None:
        if not (0 < self.lower < self.upper):
            raise ValueError("需滿足 0 < lower < upper")
        if self.grids < 2:
            raise ValueError("grids 至少為 2")
        if not (0 <= self.fee_rate < 0.05):
            raise ValueError("fee_rate 需在 [0, 0.05) 之間")
        if not (0 <= self.slippage < 0.05):
            raise ValueError("slippage 需在 [0, 0.05) 之間")
        if self.stop_loss is not None and not (0 < self.stop_loss < self.lower):
            raise ValueError("stop_loss 必須低於 lower")


@dataclass
class Fill:
    ts: float
    side: str  # buy / sell
    level: int  # -1 = 市價單(建倉調整、停損、暫停)
    price: float
    qty: float
    fee: float
    note: str = ""


def build_levels(cfg: GridConfig) -> List[float]:
    ratio = (cfg.upper / cfg.lower) ** (1.0 / cfg.grids)
    lv = [cfg.lower * ratio**i for i in range(cfg.grids + 1)]
    lv[0], lv[-1] = cfg.lower, cfg.upper
    return lv


def spacing_pct(lower: float, upper: float, grids: int) -> float:
    """等比網格的格距(%)。"""
    return ((upper / lower) ** (1.0 / grids) - 1) * 100


def max_grids_for_spacing(lower: float, upper: float, min_spacing_pct: float) -> int:
    """格距不低於 min_spacing_pct 時最多能切幾格。"""
    return int(math.log(upper / lower) / math.log(1 + min_spacing_pct / 100) + 1e-9)


class GridEngine:
    def __init__(self, cfg: GridConfig, price: float, ts: float = 0.0, quote: float = 0.0, base: float = 0.0):
        """以現有資金(quote 計價幣 + base 持幣)在 price 建立網格。"""
        cfg.validate()
        if not (cfg.lower < price < cfg.upper):
            raise ValueError("建立網格時現價必須落在 (lower, upper) 區間內")
        if quote < 0 or base < 0 or quote + base * price <= 0:
            raise ValueError("沒有可用資金")
        self.cfg = cfg
        self.levels = build_levels(cfg)
        self.last_price = price
        self.halted = False
        self.fills: List[Fill] = []
        self.cycles = 0  # 完成的「買 -> 賣」次數
        self.buy_count = 0  # 網格買單成交次數(不含建倉)
        self.grid_profit = 0.0  # 網格套利利潤(估算,已扣手續費)
        self.fees = 0.0  # 這個網格產生的手續費(含建倉與出場)
        self.orders: Dict[int, str] = {}

        f = cfg.fee_rate
        pb, ps = price * (1 + cfg.slippage), price * (1 - cfg.slippage)
        gap = next(i for i, p in enumerate(self.levels) if p >= price)
        buys, sells = list(range(gap)), list(range(gap + 1, len(self.levels)))
        ns, b_sum = len(sells), sum(self.levels[i] for i in buys)
        # 解每格數量 qty:建倉調整後剩下的計價幣要剛好夠付所有買單(含手續費)。
        # 調整後持幣 = qty*ns;若要買進(qty*ns >= base)與要賣出時的成本不同,分兩段解。
        if ns and quote >= (1 + f) * b_sum * base / ns:
            qty = (quote + base * pb * (1 + f)) / ((1 + f) * (ns * pb + b_sum))
        else:
            qty = (quote + base * ps * (1 - f)) / (ns * ps * (1 - f) + (1 + f) * b_sum)
        self.qty = qty * (1 - REL_TOL)  # 留一點浮點餘裕
        if not (self.qty > 0 and math.isfinite(self.qty)):
            raise ValueError("資金不足以建立網格")

        target = self.qty * ns
        delta = target - base
        self.quote, self.base = quote, base
        if delta * price > 1e-8:
            cost = delta * pb
            fee = cost * f
            self.quote -= cost + fee
            self._record(Fill(ts, "buy", -1, pb, delta, fee, "建倉調整"))
        elif -delta * price > 1e-8:
            proceeds = -delta * ps
            fee = proceeds * f
            self.quote += proceeds - fee
            self._record(Fill(ts, "sell", -1, ps, -delta, fee, "建倉調整"))
        self.base = target
        self.quote = max(self.quote, 0.0)
        for i in buys:
            self.orders[i] = "buy"
        for i in sells:
            self.orders[i] = "sell"

    def _record(self, fl: Fill) -> None:
        self.fees += fl.fee
        self.fills.append(fl)
        if len(self.fills) > 2 * MAX_SAVED_FILLS:
            del self.fills[:-MAX_SAVED_FILLS]

    # ---- 查詢 ----
    def equity(self, price: float) -> float:
        return self.quote + self.base * price

    def position_in_range(self, price: float) -> float:
        """現價在區間中的位置(等比尺度):0 = 下緣,1 = 上緣;區間外會小於 0 或大於 1。"""
        return math.log(price / self.cfg.lower) / math.log(self.cfg.upper / self.cfg.lower)

    # ---- 驅動 ----
    def on_price(self, price: float, ts: float = 0.0, gap: bool = False) -> List[Fill]:
        """價格由 last_price 移動到 price,依序撮合途中被穿越的掛單;回傳新成交。"""
        if self.halted:
            return []
        new: List[Fill] = []
        last, stop = self.last_price, self.cfg.stop_loss
        if stop is not None and price <= stop:
            new.extend(self._fill_range(last, stop, ts))
            new.append(self.liquidate(price if gap else stop, ts, "停損清倉"))
            self.last_price = price
            return new
        new.extend(self._fill_range(last, price, ts))
        self.last_price = price
        return new

    def _fill_range(self, last: float, price: float, ts: float) -> List[Fill]:
        out: List[Fill] = []
        if price < last:
            idxs = sorted((i for i, s in self.orders.items() if s == "buy" and price < self.levels[i] <= last),
                          reverse=True)
            for i in idxs:
                fl = self._fill_buy(i, ts)
                if fl is None:
                    break
                out.append(fl)
        elif price > last:
            idxs = sorted(i for i, s in self.orders.items() if s == "sell" and last <= self.levels[i] < price)
            for i in idxs:
                fl = self._fill_sell(i, ts)
                if fl is None:
                    break
                out.append(fl)
        return out

    def _fill_buy(self, i: int, ts: float) -> Optional[Fill]:
        p = self.levels[i]
        cost = self.qty * p
        fee = cost * self.cfg.fee_rate
        if self.quote < (cost + fee) * (1 - REL_TOL):
            return None  # 資金不足(浮點誤差外不應發生)
        del self.orders[i]
        self.quote = max(self.quote - cost - fee, 0.0)
        self.base += self.qty
        self.orders[i + 1] = "sell"
        self.buy_count += 1
        fl = Fill(ts, "buy", i, p, self.qty, fee)
        self._record(fl)
        return fl

    def _fill_sell(self, i: int, ts: float) -> Optional[Fill]:
        p = self.levels[i]
        if self.base < self.qty * (1 - REL_TOL):
            return None
        proceeds = self.qty * p
        fee = proceeds * self.cfg.fee_rate
        del self.orders[i]
        self.quote += proceeds - fee
        self.base = max(self.base - self.qty, 0.0)
        self.orders[i - 1] = "buy"
        self.cycles += 1
        buy_p = self.levels[i - 1]
        self.grid_profit += self.qty * (p - buy_p) - fee - self.qty * buy_p * self.cfg.fee_rate
        fl = Fill(ts, "sell", i, p, self.qty, fee)
        self._record(fl)
        return fl

    def liquidate(self, price: float, ts: float, note: str) -> Fill:
        """市價賣出全部持幣、撤掉所有掛單並停機(停損 / 暫停 / 單日虧損上限)。"""
        qty, px = self.base, price * (1 - self.cfg.slippage)
        fee = qty * px * self.cfg.fee_rate
        self.quote += qty * px - fee
        self.base = 0.0
        self.orders.clear()
        self.halted = True
        fl = Fill(ts, "sell", -1, px, qty, fee, note)
        self._record(fl)
        return fl

    # ---- 持久化 ----
    def to_dict(self) -> dict:
        return {
            "version": STATE_VERSION, "cfg": asdict(self.cfg), "qty": self.qty, "quote": self.quote,
            "base": self.base, "last_price": self.last_price, "halted": self.halted, "cycles": self.cycles,
            "buy_count": self.buy_count, "grid_profit": self.grid_profit, "fees": self.fees,
            "orders": {str(i): s for i, s in self.orders.items()},
            "fills": [asdict(fl) for fl in self.fills[-MAX_SAVED_FILLS:]],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GridEngine":
        """還原狀態;資料缺漏或內部不一致一律丟 ValueError。"""
        try:
            if d["version"] != STATE_VERSION:
                raise ValueError(f"不支援的網格狀態版本 {d['version']}")
            cfg = GridConfig(**d["cfg"])
            cfg.validate()
            e = object.__new__(cls)
            e.cfg, e.levels = cfg, build_levels(cfg)
            e.qty, e.quote, e.base = float(d["qty"]), float(d["quote"]), float(d["base"])
            e.last_price, e.halted = float(d["last_price"]), bool(d["halted"])
            e.cycles, e.buy_count = int(d["cycles"]), int(d["buy_count"])
            e.grid_profit, e.fees = float(d["grid_profit"]), float(d["fees"])
            e.orders = {int(i): s for i, s in d["orders"].items()}
            e.fills = [Fill(**fl) for fl in d["fills"]]
        except (KeyError, TypeError, AttributeError) as ex:
            raise ValueError(f"網格狀態格式錯誤:{ex!r}") from ex
        nums = (e.qty, e.quote, e.base, e.last_price, e.grid_profit, e.fees)
        if not all(math.isfinite(x) for x in nums) or e.qty <= 0 or e.last_price <= 0:
            raise ValueError("網格狀態數值異常")
        if e.quote < -1e-6 or e.base < -1e-9:
            raise ValueError("網格狀態餘額為負")
        n = len(e.levels)
        if any(not (0 <= i < n) or s not in ("buy", "sell") for i, s in e.orders.items()):
            raise ValueError("網格狀態掛單異常")
        if not e.halted:
            sells = sum(1 for s in e.orders.values() if s == "sell")
            if len(e.orders) != cfg.grids or abs(e.base - sells * e.qty) > 1e-6 * max(1.0, e.base):
                raise ValueError("網格狀態不一致:掛單數或持倉與賣單不符")
        return e
