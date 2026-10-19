"""合約模式:AI 決定做多 / 做空 / 空手與槓桿,程式強制風控。只模擬。

每個價格檢查點先跑強制規則(優先順序依「價格先碰到哪個」):
  強平(liquidation)> 停損(持倉虧損達錢包 X%)> 單日虧損上限(% 或金額,先到者為準;之後當日停機)。
K 線收盤後問模型,決策再經信心門檻、槓桿上限、最小調整幅度、每日交易次數過濾。

倉位控管(依當日剩餘虧損額度):
  每筆最多可虧 = min(錢包 × 停損 %, 剩餘額度 × risk_per_trade %),這就是這筆倉位的停損金額;
  停損距離(價格 %)= 停損金額 / 錢包 / 槓桿,不得小於 min_stop_distance(太近會被正常波動掃掉),
  因此槓桿上限 = 停損 % / min_stop_distance。今天虧越多,能開的倉位越小。

學習(learn=True):每次決策附上最近決策的結果;UTC 換日後請模型檢討前一天並更新心得(見 memory.py)。
"""
from __future__ import annotations

import bisect
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

from .agent import utc_day
from .futures import DUST, FuturesAccount
from .grid_agent import candle_path
from .llm import FuturesDecision, LLMConfigError, LLMError
from .market import MIN_BARS, Candle, features
from .memory import LearningMemory, daily_reflection
from .prompt import FUTURES_REFLECTION_SYSTEM, build_futures_prompt, futures_system
from .state import append_jsonl

BANKRUPT_PCT = 1.0  # 權益低於初始資金此 % 視為爆倉,停止交易
MIN_LEVERAGE = 0.1  # 額度限制後可開槓桿低於此值就不開倉


@dataclass
class FuturesRiskConfig:
    max_leverage: float = 2.0  # 槓桿上限(曝險 / 權益)
    min_confidence: float = 0.6
    min_change_pct: float = 20.0  # 同方向調整的曝險差距小於此 % 不交易(100% = 1 倍)
    stop_loss_pct: float = 3.0  # 持倉虧損達錢包此 % 強制平倉
    daily_loss_limit_pct: float = 5.0  # 權益較當日(UTC)開始下跌達此 % 平倉並停到隔天
    max_trades_per_day: int = 6
    daily_loss_amount: float = 0.0  # 單日最多虧損金額(USDT);0 = 只用 %。與 % 先到者為準
    risk_per_trade_pct: float = 50.0  # 每筆倉位停損時最多用掉當日剩餘額度的 %
    min_stop_distance_pct: float = 0.5  # 停損距離(價格 %)下限
    trail_activate_pct: float = 2.0  # 持倉獲利達錢包此 % 啟動追蹤停損;0 = 不啟用
    trail_distance_pct: float = 1.5  # 從獲利高點回吐超過錢包此 % 就平倉(最低約保本,未計手續費)

    def validate(self) -> None:
        if not 0 < self.max_leverage <= 10:
            raise ValueError("max_leverage 需在 (0, 10]")
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence 需在 [0, 1]")
        if not 0 <= self.min_change_pct < 1000:
            raise ValueError("min_change 需在 [0, 1000)")
        if not 0 < self.stop_loss_pct < 100:
            raise ValueError("stop_loss 需在 (0, 100)")
        if not 0 < self.daily_loss_limit_pct < 100:
            raise ValueError("daily_loss_limit 需在 (0, 100)")
        if self.max_trades_per_day < 1:
            raise ValueError("max_trades 至少 1")
        if self.daily_loss_amount < 0:
            raise ValueError("daily_loss_amount 不可為負")
        if not 0 < self.risk_per_trade_pct <= 100:
            raise ValueError("risk_per_trade 需在 (0, 100]")
        if not 0 < self.min_stop_distance_pct < 50:
            raise ValueError("min_stop_distance 需在 (0, 50)")
        if self.trail_activate_pct < 0 or not 0 < self.trail_distance_pct < 100:
            raise ValueError("trail_activate 不可為負;trail_distance 需在 (0, 100)")


class FuturesAgent:
    def __init__(self, acct: FuturesAccount, cfg: FuturesRiskConfig, learn: bool = True, style: str = "auto"):
        cfg.validate()
        self.acct, self.cfg = acct, cfg
        self.style, self.system = style, futures_system(style)  # 交易風格只影響提示詞,不影響風控
        self.day, self.day_start_equity, self.trades_today = "", acct.wallet, 0
        self.halted_day = ""
        self.pos_stop_pct = cfg.stop_loss_pct  # 目前倉位的停損(錢包 %),開倉時依剩餘額度決定
        self.trail_peak: Optional[float] = None  # 追蹤停損啟動後的最高獲利(錢包 %);None = 未啟動
        self.learn = learn
        self.memory = LearningMemory(fill_label=None)
        self.day_start_costs = 0.0  # 當日開始時的累計手續費 + 資金費

    # ---- 狀態 ----
    def _new_day(self, ts: float, price: float) -> None:
        day = utc_day(ts)
        if day != self.day:
            eq, costs = self.acct.equity(price), self._costs()
            self.memory.roll_day(self.day, ts, price, eq, self.day_start_equity, costs - self.day_start_costs)
            self.day, self.day_start_equity, self.trades_today = day, eq, 0
            self.day_start_costs = costs

    def _costs(self) -> float:
        return self.acct.fees_paid + self.acct.funding_paid

    def day_floor(self) -> float:
        """當日權益下限:跌到這裡觸發單日虧損上限(% 與金額取較嚴格者)。"""
        c = self.cfg
        floor = self.day_start_equity * (1 - c.daily_loss_limit_pct / 100)
        if c.daily_loss_amount > 0:
            floor = max(floor, self.day_start_equity - c.daily_loss_amount)
        return floor

    def budget_left(self, price: float) -> float:
        """今天還能虧多少(USDT)。"""
        return max(0.0, self.acct.equity(price) - self.day_floor())

    def sizing(self, price: float) -> Tuple[float, float]:
        """依剩餘額度決定 (這筆的停損 % of 錢包, 目前可開的槓桿上限)。"""
        c, w = self.cfg, self.acct.wallet
        if w <= 0:
            return 0.0, 0.0
        risk_amt = min(w * c.stop_loss_pct / 100, self.budget_left(price) * c.risk_per_trade_pct / 100)
        stop_pct = risk_amt / w * 100
        return stop_pct, min(c.max_leverage, stop_pct / c.min_stop_distance_pct)

    def bankrupt(self, price: float) -> bool:
        return self.acct.equity(price) <= self.acct.initial * BANKRUPT_PCT / 100

    def position_info(self, price: float) -> dict:
        a = self.acct
        side = "FLAT" if abs(a.qty) < DUST else ("LONG" if a.qty > 0 else "SHORT")
        info = {"side": side, "leverage": round(abs(a.exposure_pct(price)) / 100, 2),
                "equity": round(a.equity(price), 2)}
        if side != "FLAT":
            liq, stop = a.liq_price(), a.stop_price(self.pos_stop_pct)
            trail = self._trail_price()
            info.update(entry_price=round(a.entry, 6),
                        unrealized_pnl_pct_of_wallet=round(-a.position_loss_pct(price), 2),
                        liquidation_price=round(liq, 2) if liq else None,
                        stop_loss_price=round(stop, 2) if stop else None,
                        trailing_stop_price=round(trail, 2) if trail else None)
        return info

    def _update_trail(self, price: float) -> None:
        """依目前價格更新追蹤停損的獲利高點(獲利達門檻才啟動)。"""
        c, a = self.cfg, self.acct
        if not c.trail_activate_pct or abs(a.qty) < DUST:
            return
        profit = -a.position_loss_pct(price)  # 獲利占錢包 %
        if self.trail_peak is None:
            if profit >= c.trail_activate_pct:
                self.trail_peak = profit
        else:
            self.trail_peak = max(self.trail_peak, profit)

    def _trail_price(self) -> Optional[float]:
        """追蹤停損價:獲利回落到 max(高點 − 回吐距離, 0) 的價格。"""
        a = self.acct
        if self.trail_peak is None or abs(a.qty) < DUST or a.wallet <= 0:
            return None
        floor = max(self.trail_peak - self.cfg.trail_distance_pct, 0.0)
        p = a.entry + floor / 100 * a.wallet / a.qty
        return p if p > 0 else None

    def _triggers(self) -> List[Tuple[float, str]]:
        a, c = self.acct, self.cfg
        if abs(a.qty) < DUST:
            return []
        out = []
        liq = a.liq_price()
        if liq:
            out.append((liq, "liquidation"))
        stop = a.stop_price(self.pos_stop_pct)
        if stop:
            out.append((stop, "stop_loss"))
        trail = self._trail_price()
        if trail:
            out.append((trail, "trailing_stop"))
        if self.halted_day != self.day and self.day_start_equity > 0:
            # equity == 當日權益下限 的價格
            p = a.entry + (self.day_floor() - a.wallet) / a.qty
            if p > 0:
                out.append((p, "daily_loss"))
        return out

    # ---- 強制規則 ----
    def check(self, price: float, ts: float, prev: Optional[float] = None, journal: Optional[str] = None) -> Optional[dict]:
        """價格檢查點。prev 為同一段連續走勢的上一個價格(回放用):連續穿越觸發價時以觸發價出場;
        否則(模擬盤輪詢、跳空)以當下價格出場。強平一律以強平價成交(由交易所執行)。"""
        self._new_day(ts, price)
        a = self.acct
        if abs(a.qty) < DUST:
            return None
        long = a.qty > 0
        self._update_trail(price)
        hit = [(p, code) for p, code in self._triggers() if (price <= p if long else price >= p)]
        if not hit:
            return None
        p, code = max(hit) if long else min(hit)  # 價格往不利方向走時最先碰到的
        continuous = prev is not None and (prev > p if long else prev < p)
        liq = next((x for x in hit if x[1] == "liquidation"), None)
        if liq and not continuous:
            # 輪詢 / 跳空時已越過強平價:交易所會先在強平價強平,停損來不及以當下價格出場(否則錢包會變負)
            p, code = liq
        rec = self._rec(price, ts, None)
        if code == "liquidation":
            t = a.liquidate(p, ts)
            why = f"權益低於維持保證金,於 {p:.2f} 強制平倉"
            self.halted_day = self.day
        else:
            px = p if continuous else price
            if code == "stop_loss":
                why = f"持倉虧損達錢包 {self.pos_stop_pct:.2f}%,停損平倉"
            elif code == "trailing_stop":
                why = (f"追蹤停損:獲利從高點 {self.trail_peak:.2f}% 回吐超過 {self.cfg.trail_distance_pct:g}%,"
                       "平倉保住獲利")
            else:
                why = "權益觸及今日虧損上限,平倉並停止交易至明天(UTC)"
                self.halted_day = self.day
            t = a.rebalance(0, px, ts, why)  # 原因寫中文,畫面「最近成交」直接顯示
        self.trail_peak = None
        self.memory.on_tick(0, code)
        rec.update(forced=True, action_taken=why, code=code, trade=asdict(t) if t else None)
        return self._finish(rec, price, journal)

    def funding(self, rate: float, price: float, ts: float, journal: Optional[str] = None) -> Optional[dict]:
        if abs(self.acct.qty) < DUST:
            return None
        paid = self.acct.apply_funding(rate, price)
        rec = {"kind": "futures_funding", "ts": ts, "rate": rate, "price": price, "paid": round(paid, 6),
               "equity_after": round(self.acct.equity(price), 4)}
        if journal:
            append_jsonl(journal, rec)
        return rec

    # ---- 模型決策 ----
    def decide(self, model, symbol: str, interval: str, candles: List[Candle], price: float, ts: float,
               funding_rate: Optional[float] = None, journal: Optional[str] = None) -> dict:
        self._new_day(ts, price)
        eq0 = self.acct.equity(price)
        self.memory.settle(ts, price, eq0)
        rec = self._rec(price, ts, getattr(model, "name", "?"))
        rec["reflection"] = (daily_reflection(self.memory, model, FUTURES_REFLECTION_SYSTEM, symbol, ts, journal,
                                              "futures_reflection") if self.learn else None)
        if self.halted_day != self.day and eq0 <= self.day_floor():
            # 空手時也要檢查(例如同一天連續停損):否則開倉後觸發價已在有利側,會立刻被平倉、白付手續費
            self.halted_day = self.day
        if self.bankrupt(price):
            why, code, target = "帳戶已爆倉(權益低於初始資金 1%),停止交易", "bankrupt", None
        elif self.halted_day == self.day:
            why, code, target = "今日已觸發強平或單日虧損上限,停止交易至明天(UTC)", "halted", None
        else:
            feats = features(candles)
            prompt = build_futures_prompt(symbol, interval, feats, self.position_info(price), funding_rate,
                                          self._constraints(price), self.memory.prompt_view() if self.learn else None)
            decision = None
            try:
                decision, meta = model.decide(self.system, prompt, feats)
                rec["decision"], rec["meta"] = asdict(decision), meta
            except LLMConfigError:
                raise
            except LLMError as e:
                rec["error"] = str(e)
            target, why, code = self._filter(decision, price)
        stop_pct = self.sizing(price)[0]
        t = self.acct.rebalance(target, price, ts, why) if target is not None else None
        if t:
            self.trades_today += 1
            self.trail_peak = None  # 倉位變了:追蹤停損重新計算
            if target:
                self.pos_stop_pct = stop_pct  # 新倉位的停損依開倉當下的剩餘額度
        if rec["decision"] is not None:  # 模型失敗(稍後重試)不記,避免成績單被錯誤紀錄擠掉
            d = rec["decision"]
            side = self.position_info(price)
            self.memory.record_decision(
                ts, price, eq0, d, code, f"{d['leverage']:g}x" if d["action"] in ("LONG", "SHORT") else None,
                {"position_after": side["side"] + (f" {side['leverage']:g}x" if side["side"] != "FLAT" else "")})
        rec.update(action_taken=why, code=code, trade=asdict(t) if t else None)
        return self._finish(rec, price, journal)

    def _constraints(self, price: float) -> dict:
        c = self.cfg
        stop_pct, cap = self.sizing(price)
        return {"max_leverage_now": round(cap, 2), "max_leverage_config": c.max_leverage,
                "taker_fee_rate": self.acct.fee_rate,
                "stop_loss_pct_of_wallet_for_new_position": round(stop_pct, 3),
                "trailing_stop": ({"activate_at_profit_pct_of_wallet": c.trail_activate_pct,
                                   "giveback_pct_of_wallet": c.trail_distance_pct} if c.trail_activate_pct else None),
                "daily_loss_budget_left_pct_of_equity": round(self.budget_left(price) / self.acct.equity(price) * 100, 2)
                if self.acct.equity(price) > 0 else 0.0,
                "trades_left_today": max(0, c.max_trades_per_day - self.trades_today)}

    def _filter(self, d: Optional[FuturesDecision], price: float) -> Tuple[Optional[float], str, str]:
        c = self.cfg
        if d is None:
            return None, "模型無有效決策,不交易", "no_decision"
        if d.action == "HOLD":
            return None, "模型建議 HOLD", "hold"
        cur = self.acct.exposure_pct(price)
        cap = self.sizing(price)[1]
        if d.action == "FLAT":
            target = 0.0
        else:
            lev = min(d.leverage, cap)
            target = lev * 100 * (1 if d.action == "LONG" else -1)
        if d.action == "FLAT" and abs(self.acct.qty) < DUST:
            return None, "已是空手", "noop"
        if d.action != "FLAT" and cap < MIN_LEVERAGE:
            return None, "今日剩餘虧損額度不足,不開新倉", "budget_exhausted"
        if d.confidence < c.min_confidence:
            return None, f"信心 {d.confidence:.2f} 低於門檻 {c.min_confidence:g}", "low_confidence"
        same_side = target != 0 and cur != 0 and (target > 0) == (cur > 0)
        if same_side and abs(target - cur) < c.min_change_pct:
            return None, f"已接近目標:目標 {target / 100:+.2f} 倍、目前 {cur / 100:+.2f} 倍", "noop"
        if self.trades_today >= c.max_trades_per_day:
            return None, f"今日交易已達 {c.max_trades_per_day} 次上限", "max_trades"
        note = ""
        if d.action != "FLAT" and d.leverage > cap:
            note = (f"(槓桿上限 {c.max_leverage:g} 倍)" if cap >= c.max_leverage
                    else f"(依今日剩餘額度,上限 {cap:.2f} 倍)")
        label = {"LONG": "做多", "SHORT": "做空", "FLAT": "平倉"}[d.action]
        return target, f"執行 {label} {abs(target) / 100:.2f} 倍{note}", "execute"

    # ---- 日誌 ----
    def _rec(self, price: float, ts: float, model_name: Optional[str]) -> dict:
        return {"kind": "futures", "ts": ts, "price": price, "model": model_name,
                "equity_before": round(self.acct.equity(price), 4),
                "exposure_before": round(self.acct.exposure_pct(price), 2),
                "decision": None, "meta": None, "error": None, "forced": False}

    def _finish(self, rec: dict, price: float, journal: Optional[str]) -> dict:
        rec.update(equity_after=round(self.acct.equity(price), 4),
                   exposure_after=round(self.acct.exposure_pct(price), 2))
        if journal:
            append_jsonl(journal, rec)
        return rec

    # ---- 持久化 ----
    def to_dict(self) -> dict:
        return {"acct": self.acct.to_dict(), "day": self.day, "day_start_equity": self.day_start_equity,
                "trades_today": self.trades_today, "halted_day": self.halted_day, "pos_stop_pct": self.pos_stop_pct,
                "trail_peak": self.trail_peak,
                "day_start_costs": self.day_start_costs, "memory": self.memory.to_dict()}

    @classmethod
    def from_dict(cls, d: dict, cfg: FuturesRiskConfig, learn: bool = True, style: str = "auto") -> "FuturesAgent":
        try:
            a = cls(FuturesAccount.from_dict(d["acct"]), cfg, learn, style)
            a.day, a.day_start_equity = str(d["day"]), float(d["day_start_equity"])
            a.trades_today, a.halted_day = int(d["trades_today"]), str(d["halted_day"])
            # 舊版狀態檔沒有以下欄位
            a.pos_stop_pct = float(d.get("pos_stop_pct", cfg.stop_loss_pct))
            a.trail_peak = None if d.get("trail_peak") is None else float(d["trail_peak"])
            a.day_start_costs = float(d.get("day_start_costs", a._costs()))
            a.memory = LearningMemory.from_dict(d.get("memory"), fill_label=None)
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"合約狀態檔格式錯誤:{e!r}") from e
        return a


# ---------------- 歷史回放 ----------------

@dataclass
class FuturesReplayResult:
    bars: int
    start_price: float
    end_price: float
    equity: float
    return_pct: float
    hold_return_pct: float
    max_drawdown_pct: float
    decisions: int
    model_errors: int
    trades: int
    stop_losses: int
    trailing_stops: int
    liquidations: int
    daily_halts: int
    funding_paid: float
    fees_paid: float
    reflections: int  # 成功的每日檢討次數
    stopped_early: bool  # 達到 max_calls 或爆倉而提前結束


def futures_replay(model, symbol: str, interval: str, candles: List[Candle], agent: FuturesAgent,
                   funding: List[Tuple[float, float]], every: int = 4, max_calls: Optional[int] = None,
                   journal: Optional[str] = None, calls_used=lambda: 0) -> FuturesReplayResult:
    """第 i 根收盤後決策、第 i+1 根開盤成交;每根 K 線依保守路徑檢查強平 / 停損;資金費依歷史費率在結算時間收付。"""
    if every < 1:
        raise ValueError("every 至少 1")
    if len(candles) < MIN_BARS + 2:
        raise ValueError(f"K 線不足:至少需要 {MIN_BARS + 2} 根")
    start = MIN_BARS - 1
    a = agent.acct
    start_price = end_price = candles[start + 1].open
    peak, max_dd = a.equity(start_price), 0.0
    counts = {"decisions": 0, "errors": 0, "trades": 0, "stop_loss": 0, "trailing_stop": 0, "liquidation": 0,
              "daily_loss": 0, "reflections": 0}
    stopped = False
    prev_close = candles[start].close
    times = [t for t, _ in funding]
    fi = bisect.bisect_right(times, candles[start].ts)

    def forced(p: float, ts: float, prev: Optional[float]) -> None:
        nonlocal peak, max_dd
        r = agent.check(p, ts, prev, journal)
        if r:
            counts[r["code"]] += 1
            counts["trades"] += 1 if r["trade"] else 0
        eq = a.equity(p)
        peak = max(peak, eq)
        max_dd = max(max_dd, (peak - eq) / peak if peak > 0 else 0.0)

    processed = 0
    for i in range(start, len(candles) - 1):
        nxt = candles[i + 1]
        while fi < len(funding) and funding[fi][0] <= nxt.ts:  # 這根開盤前的資金費結算
            agent.funding(funding[fi][1], prev_close, funding[fi][0], journal)
            fi += 1
        forced(nxt.open, nxt.ts, None)  # 收盤到開盤之間看不到走勢:視為跳空
        if (i - start) % every == 0:
            if agent.bankrupt(nxt.open):
                stopped = True
                break
            if max_calls is not None and calls_used() >= max_calls:
                stopped = True
                break
            rate = funding[fi - 1][1] if fi > 0 else None
            r = agent.decide(model, symbol, interval, candles[:i + 1], nxt.open, nxt.ts, rate, journal)
            if r["code"] not in ("halted", "bankrupt"):
                counts["decisions"] += 1
            counts["errors"] += 1 if r["error"] else 0
            counts["trades"] += 1 if r["trade"] else 0
            counts["reflections"] += 1 if r["reflection"] and not r["reflection"]["error"] else 0
        path = candle_path(nxt.open, nxt.high, nxt.low, nxt.close)
        for prev, p in zip(path, path[1:]):
            forced(p, nxt.ts, prev)
        prev_close = end_price = nxt.close
        processed += 1
    eq = a.equity(end_price)
    return FuturesReplayResult(processed, start_price, end_price, eq, (eq / a.initial - 1) * 100,
                               (end_price / start_price - 1) * 100, max_dd * 100, counts["decisions"],
                               counts["errors"], counts["trades"], counts["stop_loss"], counts["trailing_stop"],
                               counts["liquidation"],
                               counts["daily_loss"], a.funding_paid, a.fees_paid, counts["reflections"], stopped)
