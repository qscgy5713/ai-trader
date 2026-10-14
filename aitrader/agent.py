"""單步決策流程與歷史回放。

一步 = 強制風控檢查 →(必要時)問模型 → 風控過濾 → 模擬成交 → 寫決策日誌。
模型只提供建議;模型出錯(LLMError)一律不交易,設定錯誤(LLMConfigError)直接往上丟讓使用者處理。
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import List, Optional

from .llm import LLMConfigError, LLMError
from .market import MIN_BARS, Candle, features
from .portfolio import Portfolio
from .prompt import SYSTEM, build_prompt
from .risk import RiskManager
from .state import append_jsonl


def utc_day(ts: float) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(ts))


def step(model, symbol: str, interval: str, candles: List[Candle], price: float, ts: float,
         pf: Portfolio, risk: RiskManager, ask_model: bool = True,
         journal: Optional[str] = None) -> Optional[dict]:
    """執行一個決策點。ask_model=False 時只做強制風控檢查(例如停損),沒觸發就回 None。"""
    risk.new_bar(utc_day(ts), pf.equity(price))
    forced = risk.forced(pf, price)
    if forced is None and not ask_model:
        return None
    rec = {"ts": ts, "symbol": symbol, "price": price, "equity_before": round(pf.equity(price), 4),
           "position_pct_before": round(pf.position_pct(price), 2),
           "model": getattr(model, "name", "?"), "decision": None, "meta": None, "error": None}
    if forced is not None:
        target, why, code = forced
        rec["forced"] = True
    else:
        feats = features(candles)
        prompt = build_prompt(symbol, interval, feats, pf.position_pct(price), pf.unrealized_pct(price),
                              risk.cfg.max_position_pct)
        decision = None
        try:
            decision, meta = model.decide(SYSTEM, prompt, feats)
            rec["decision"], rec["meta"] = asdict(decision), meta
        except LLMConfigError:
            raise
        except LLMError as e:
            rec["error"] = str(e)
        target, why, code = risk.filter(decision, pf, price)
    trade = pf.rebalance(target, price, ts, why) if target is not None else None
    if trade:
        risk.record_trade()
    rec.update(action_taken=why, code=code, trade=asdict(trade) if trade else None,
               equity_after=round(pf.equity(price), 4), position_pct_after=round(pf.position_pct(price), 2))
    if journal:
        append_jsonl(journal, rec)
    return rec


@dataclass
class ReplayResult:
    bars: int
    start_price: float
    end_price: float
    equity: float
    return_pct: float
    hold_return_pct: float
    max_drawdown_pct: float
    trades: int
    decisions: int
    model_errors: int
    forced_exits: int
    stopped_early: bool  # 達到 max_calls 提前結束


def replay(model, symbol: str, interval: str, candles: List[Candle], pf: Portfolio, risk: RiskManager,
           every: int = 4, max_calls: Optional[int] = None, journal: Optional[str] = None,
           calls_used=lambda: 0) -> ReplayResult:
    """歷史回放:第 i 根收盤後做決策,以第 i+1 根「開盤價」成交(決策時看不到的價格,避免偷看未來)。

    每根 K 線都做強制風控檢查(停損 / 單日虧損),模型每 every 根才問一次。
    max_calls:模型實際呼叫(不含快取命中)達上限就提前結束,控制費用。
    """
    if every < 1:
        raise ValueError("every 至少 1")
    if len(candles) < MIN_BARS + 2:
        raise ValueError(f"K 線不足:至少需要 {MIN_BARS + 2} 根")
    start = MIN_BARS - 1
    start_price = candles[start + 1].open
    peak, max_dd = pf.equity(start_price), 0.0
    trades = decisions = errors = forced = processed = 0
    stopped = False
    end_price = start_price
    for i in range(start, len(candles) - 1):
        nxt = candles[i + 1]
        ask = (i - start) % every == 0
        if ask and max_calls is not None and calls_used() >= max_calls:
            stopped = True
            break
        rec = step(model, symbol, interval, candles[:i + 1], nxt.open, nxt.ts, pf, risk, ask_model=ask, journal=journal)
        if rec:
            decisions += 0 if rec.get("forced") else 1
            forced += 1 if rec.get("forced") else 0
            errors += 1 if rec["error"] else 0
            trades += 1 if rec["trade"] else 0
        processed += 1
        end_price = nxt.close
        eq = pf.equity(end_price)
        peak = max(peak, eq)
        max_dd = max(max_dd, (peak - eq) / peak)
    eq = pf.equity(end_price)
    return ReplayResult(processed, start_price, end_price, eq, (eq / pf.initial - 1) * 100,
                        (end_price / start_price - 1) * 100, max_dd * 100, trades, decisions, errors, forced, stopped)
