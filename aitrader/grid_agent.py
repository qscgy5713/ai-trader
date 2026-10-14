"""AI 調整網格:模型每根 K 線決定網格區間 / 格數(或暫停),網格引擎負責高頻成交。

分工:
- 網格引擎(grid.py):每個價格 tick 撮合掛單,處理停損。
- 本模組:強制風控(停損後 / 單日虧損上限觸發後當日停機)、問模型、過濾模型決策(信心、每日重設次數、
  區間上限、最小格距、變動太小不重設)、套用決策、寫決策日誌。
- 模型出錯一律維持現狀;設定錯誤(LLMConfigError)往上丟讓使用者處理。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

from .agent import utc_day
from .grid import Fill, GridConfig, GridEngine, max_grids_for_spacing, spacing_pct
from .llm import GridDecision, LLMConfigError, LLMError
from .market import MIN_BARS, Candle, features
from .prompt import GRID_SYSTEM, build_grid_prompt
from .state import append_jsonl


@dataclass
class GridRiskConfig:
    min_confidence: float = 0.6  # SET / PAUSE 的信心門檻;低於此值維持現狀
    min_spacing_pct: float = 0.5  # 最小格距 %(手續費 0.1% 時,一買一賣打平約 0.2%)
    max_range_pct: float = 15  # 區間上下緣各自最多離現價幾 %
    stop_loss_pct: float = 5  # 區間下緣再往下幾 % 停損;0 = 不設
    daily_loss_limit_pct: float = 3  # 權益較當日(UTC)開始下跌達此 % 清倉,當日停機
    max_reconfigs: int = 4  # 每日最多重設 / 暫停網格次數
    min_change_pct: float = 1.0  # 新舊區間上下緣都差不到此 % 且格數相同時不重設

    def validate(self) -> None:
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence 需在 0~1")
        if not 0 < self.min_spacing_pct < 10:
            raise ValueError("min_spacing 需在 0~10 %")
        if not 0 < self.max_range_pct < 100:
            raise ValueError("max_range 需在 0~100 %")
        if not 0 <= self.stop_loss_pct < 100:
            raise ValueError("stop_loss 需在 0~100 %")
        if not 0 <= self.daily_loss_limit_pct < 100:
            raise ValueError("daily_loss_limit 需在 0~100 %")
        if self.max_reconfigs < 1:
            raise ValueError("max_reconfigs 至少 1")
        if self.min_change_pct < 0:
            raise ValueError("min_change 不可為負")


class GridAgent:
    """帳戶 + 目前的網格 + 風控狀態。沒有網格時資金全為計價幣(暫停 / 停損都會清倉)。"""

    def __init__(self, cash: float, cfg: GridRiskConfig, fee: float = 0.001, slippage: float = 0.0005):
        cfg.validate()
        if cash <= 0:
            raise ValueError("資金必須 > 0")
        self.cfg, self.fee, self.slippage = cfg, fee, slippage
        self.initial = cash
        self.cash = cash  # 沒有網格時的資金
        self.engine: Optional[GridEngine] = None
        self.set_ts, self._set_equity = 0.0, 0.0
        self.day, self.day_start_equity, self.reconfigs_today = "", cash, 0
        self.halted_day = ""  # 觸發停損 / 單日虧損上限的那一天(UTC);當天不再建網格
        # 已結束網格的累計統計(加上目前網格的即為總計)
        self.closed_cycles, self.closed_profit, self.closed_fees = 0, 0.0, 0.0

    # ---- 查詢 ----
    def equity(self, price: float) -> float:
        return self.engine.equity(price) if self.engine else self.cash

    def totals(self) -> dict:
        e = self.engine
        return {
            "cycles": self.closed_cycles + (e.cycles if e else 0),
            "grid_profit": self.closed_profit + (e.grid_profit if e else 0.0),
            "fees": self.closed_fees + (e.fees if e else 0.0),
        }

    def grid_info(self, price: float, ts: float) -> Optional[dict]:
        """給模型與日誌看的網格狀態;沒有網格回 None。"""
        e = self.engine
        if e is None:
            return None
        eq = e.equity(price)
        c = e.cfg
        return {
            "lower": round(c.lower, 6), "upper": round(c.upper, 6), "grids": c.grids,
            "lower_pct_below_price": round((1 - c.lower / price) * 100, 2),
            "upper_pct_above_price": round((c.upper / price - 1) * 100, 2),
            "spacing_pct": round(spacing_pct(c.lower, c.upper, c.grids), 3),
            "stop_loss": round(c.stop_loss, 6) if c.stop_loss else None,
            "position_in_range": round(e.position_in_range(price), 3),
            "coin_value_pct_of_equity": round(e.base * price / eq * 100, 1) if eq > 0 else 0.0,
            "hours_since_set": round(max(0.0, ts - self.set_ts) / 3600, 1),
            "completed_cycles": e.cycles,
            "grid_profit_pct_of_equity": round(e.grid_profit / eq * 100, 3) if eq > 0 else 0.0,
            "pnl_since_set_pct": round((eq / self._set_equity - 1) * 100, 2) if self._set_equity else 0.0,
        }

    # ---- 驅動 ----
    def _new_day(self, ts: float, price: float) -> None:
        day = utc_day(ts)
        if day != self.day:
            self.day, self.day_start_equity, self.reconfigs_today = day, self.equity(price), 0

    def _close_engine(self) -> None:
        e = self.engine
        if e is None:
            return
        self.closed_cycles += e.cycles
        self.closed_profit += e.grid_profit
        self.closed_fees += e.fees
        self.cash = e.quote  # 只在清倉後(base = 0)或被新網格取代時呼叫
        self.engine = None

    def on_price(self, price: float, ts: float, gap: bool = False) -> Tuple[List[Fill], Optional[str]]:
        """每個價格 tick:撮合網格、檢查停損與單日虧損上限。回傳 (新成交, 事件代碼或 None)。"""
        self._new_day(ts, price)
        fills: List[Fill] = []
        event = None
        if self.engine is not None:
            fills = self.engine.on_price(price, ts, gap)
            if self.engine.halted:  # 停損出場
                self._close_engine()
                self.halted_day, event = self.day, "stop_loss"
        lim = self.cfg.daily_loss_limit_pct
        if (lim and self.halted_day != self.day
                and self.equity(price) <= self.day_start_equity * (1 - lim / 100)):
            if self.engine is not None:
                fills.append(self.engine.liquidate(price, ts, "單日虧損上限"))
                self._close_engine()
            self.halted_day, event = self.day, "daily_loss"
        return fills, event

    def decide(self, model, symbol: str, interval: str, candles: List[Candle], price: float, ts: float,
               journal: Optional[str] = None) -> dict:
        """K 線收盤後的決策點:問模型、過濾、套用。停機中不問模型(省費用)。"""
        self._new_day(ts, price)
        rec = {"kind": "grid_decision", "ts": ts, "symbol": symbol, "price": price,
               "equity_before": round(self.equity(price), 4), "grid_before": self.grid_info(price, ts),
               "model": getattr(model, "name", "?"), "decision": None, "meta": None, "error": None}
        fills: List[Fill] = []
        if self.halted_day == self.day:
            why, code = "今日已觸發停損或單日虧損上限,停機到明天(UTC)", "halted"
        else:
            feats = features(candles)
            info = self.grid_info(price, ts)
            prompt = build_grid_prompt(symbol, interval, feats, info, self._constraints())
            decision = None
            try:
                decision, meta = model.decide(GRID_SYSTEM, prompt, dict(feats, grid=info))
                rec["decision"], rec["meta"] = asdict(decision), meta
            except LLMConfigError:
                raise
            except LLMError as e:
                rec["error"] = str(e)
            why, code, fills = self._apply(decision, price, ts)
        rec.update(action_taken=why, code=code, fills=[asdict(f) for f in fills],
                   equity_after=round(self.equity(price), 4), grid_after=self.grid_info(price, ts))
        if journal:
            append_jsonl(journal, rec)
        return rec

    def _constraints(self) -> dict:
        c = self.cfg
        return {"fee_rate_per_side": self.fee, "min_spacing_pct": c.min_spacing_pct,
                "max_range_pct_each_side": c.max_range_pct,
                "stop_loss_pct_below_lower": c.stop_loss_pct or None,
                "reconfigs_left_today": max(0, c.max_reconfigs - self.reconfigs_today)}

    def _apply(self, d: Optional[GridDecision], price: float, ts: float) -> Tuple[str, str, List[Fill]]:
        c = self.cfg
        if d is None:
            return "模型沒有有效決策,維持現狀", "no_decision", []
        if d.action == "KEEP":
            return ("維持目前網格" if self.engine else "維持空手"), "hold", []
        if d.confidence < c.min_confidence:
            return f"信心 {d.confidence:.2f} 低於門檻 {c.min_confidence},維持現狀", "low_confidence", []
        if d.action == "PAUSE" and self.engine is None:
            return "已是空手", "noop", []
        if self.reconfigs_today >= c.max_reconfigs:
            return f"今日已重設 {self.reconfigs_today} 次,達上限", "max_reconfigs", []
        if d.action == "PAUSE":
            fl = self.engine.liquidate(price, ts, "AI 暫停")
            self._close_engine()
            self.reconfigs_today += 1
            return "暫停網格並清倉", "pause", [fl]
        # SET
        lo_pct, up_pct = min(d.lower_pct, c.max_range_pct), min(d.upper_pct, c.max_range_pct)
        lower, upper = price * (1 - lo_pct / 100), price * (1 + up_pct / 100)
        cap = max_grids_for_spacing(lower, upper, c.min_spacing_pct)
        if cap < 2:
            return f"區間 -{lo_pct:g}%/+{up_pct:g}% 太窄,格距無法達到 {c.min_spacing_pct}%", "too_narrow", []
        grids = min(d.grids, cap)
        e = self.engine
        if e is not None and grids == e.cfg.grids and \
                abs(lower / e.cfg.lower - 1) * 100 < c.min_change_pct and abs(upper / e.cfg.upper - 1) * 100 < c.min_change_pct:
            return "新區間與目前網格相近,不重設", "small_change", []
        stop = lower * (1 - c.stop_loss_pct / 100) if c.stop_loss_pct else None
        quote, base = (e.quote, e.base) if e else (self.cash, 0.0)
        new = GridEngine(GridConfig(lower, upper, grids, self.fee, self.slippage, stop), price, ts, quote, base)
        if e is not None:
            self._close_engine()  # 資金已轉入新網格;舊網格只結算統計
        self.engine, self.set_ts, self._set_equity = new, ts, new.equity(price)
        self.reconfigs_today += 1
        notes = []
        if lo_pct < d.lower_pct or up_pct < d.upper_pct:
            notes.append(f"區間截斷在 ±{c.max_range_pct:g}%")
        if grids < d.grids:
            notes.append(f"格數由 {d.grids} 降為 {grids}(最小格距 {c.min_spacing_pct}%)")
        why = (f"重設網格 {lower:.2f} ~ {upper:.2f} × {grids} 格,格距 {spacing_pct(lower, upper, grids):.2f}%"
               + (f"(停損 {stop:.2f})" if stop else "") + ("；" + "、".join(notes) if notes else ""))
        return why, "set", list(new.fills)

    # ---- 持久化 ----
    def to_dict(self) -> dict:
        return {"cfg_fee": self.fee, "cfg_slippage": self.slippage, "initial": self.initial, "cash": self.cash,
                "engine": self.engine.to_dict() if self.engine else None, "set_ts": self.set_ts,
                "set_equity": self._set_equity, "day": self.day, "day_start_equity": self.day_start_equity,
                "reconfigs_today": self.reconfigs_today, "halted_day": self.halted_day,
                "closed": [self.closed_cycles, self.closed_profit, self.closed_fees]}

    @classmethod
    def from_dict(cls, d: dict, cfg: GridRiskConfig) -> "GridAgent":
        try:
            a = cls(float(d["initial"]), cfg, float(d["cfg_fee"]), float(d["cfg_slippage"]))
            a.cash = float(d["cash"])
            a.engine = GridEngine.from_dict(d["engine"]) if d["engine"] else None
            a.set_ts, a._set_equity = float(d["set_ts"]), float(d["set_equity"])
            a.day, a.day_start_equity = str(d["day"]), float(d["day_start_equity"])
            a.reconfigs_today, a.halted_day = int(d["reconfigs_today"]), str(d["halted_day"])
            cc, cp, cf = d["closed"]
            a.closed_cycles, a.closed_profit, a.closed_fees = int(cc), float(cp), float(cf)
        except (KeyError, TypeError, ValueError) as ex:
            raise ValueError(f"網格狀態檔格式錯誤:{ex!r}") from ex
        return a


# ---------------- 歷史回放 ----------------

def candle_path(o: float, h: float, l: float, c: float) -> List[float]:
    """K 線內部路徑未知,採保守假設:陽線先探低再拉高,陰線先拉高再探低。"""
    return [o, l, h, c] if c >= o else [o, h, l, c]


@dataclass
class GridReplayResult:
    bars: int
    start_price: float
    end_price: float
    equity: float
    return_pct: float
    hold_return_pct: float
    max_drawdown_pct: float
    decisions: int
    model_errors: int
    sets: int
    pauses: int
    stops: int  # 停損 + 單日虧損上限
    grid_fills: int  # 網格掛單成交筆數(不含建倉調整 / 出場)
    cycles: int
    grid_profit: float
    fees: float
    stopped_early: bool


def grid_replay(model, symbol: str, interval: str, candles: List[Candle], agent: GridAgent, every: int = 4,
                max_calls: Optional[int] = None, journal: Optional[str] = None,
                calls_used=lambda: 0) -> GridReplayResult:
    """第 i 根收盤後決策、以第 i+1 根開盤價建倉;每根 K 線依保守路徑撮合網格。"""
    if every < 1:
        raise ValueError("every 至少 1")
    if len(candles) < MIN_BARS + 2:
        raise ValueError(f"K 線不足:至少需要 {MIN_BARS + 2} 根")
    start = MIN_BARS - 1
    start_price = end_price = candles[start + 1].open
    peak, max_dd = agent.equity(start_price), 0.0
    decisions = errors = sets = pauses = stops = grid_fills = processed = 0
    stopped = False
    prev_close = candles[start].close

    def tick(p: float, ts: float, gap: bool) -> None:
        nonlocal grid_fills, stops, peak, max_dd
        fills, ev = agent.on_price(p, ts, gap)
        grid_fills += sum(1 for f in fills if f.level >= 0)
        stops += 1 if ev else 0
        if journal and (fills or ev):
            append_jsonl(journal, {"kind": "grid_fill", "ts": ts, "price": p, "event": ev,
                                   "fills": [asdict(f) for f in fills], "equity_after": round(agent.equity(p), 4)})
        eq = agent.equity(p)
        peak = max(peak, eq)
        max_dd = max(max_dd, (peak - eq) / peak)

    for i in range(start, len(candles) - 1):
        nxt = candles[i + 1]
        if (i - start) % every == 0 and max_calls is not None and calls_used() >= max_calls:
            stopped = True
            break
        tick(nxt.open, nxt.ts, nxt.open != prev_close)  # 上一根收盤到這根開盤之間的走勢看不到,視為跳空
        if (i - start) % every == 0:
            rec = agent.decide(model, symbol, interval, candles[:i + 1], nxt.open, nxt.ts, journal)
            decisions += 0 if rec["code"] == "halted" else 1
            errors += 1 if rec["error"] else 0
            sets += rec["code"] == "set"
            pauses += rec["code"] == "pause"
        for p in candle_path(nxt.open, nxt.high, nxt.low, nxt.close)[1:]:
            tick(p, nxt.ts, False)
        prev_close = end_price = nxt.close
        processed += 1
    eq = agent.equity(end_price)
    t = agent.totals()
    return GridReplayResult(processed, start_price, end_price, eq, (eq / agent.initial - 1) * 100,
                            (end_price / start_price - 1) * 100, max_dd * 100, decisions, errors, sets, pauses,
                            stops, grid_fills, t["cycles"], t["grid_profit"], t["fees"], stopped)
