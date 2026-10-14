"""提示詞:把特徵與持倉整理成給模型的輸入。輸出格式由 llm.DECISION_SCHEMA 強制。"""
from __future__ import annotations

import json
from typing import Optional

SYSTEM = """You are a cautious crypto spot trading assistant operating a PAPER-TRADING account (long-only, no leverage, no shorting).
Each turn you receive technical features of the latest CLOSED candles and the current position.
Decide the desired position as a percentage of total equity.

Rules:
- action=BUY means increase position to target_position_pct; SELL means reduce to target_position_pct; HOLD means keep the current position.
- target_position_pct must be between 0 and the stated max_position_pct.
- Prefer HOLD when signals are mixed or weak. Trading costs fees; avoid churning small changes.
- confidence reflects how strong the evidence is (0-1), not how much you want to trade.
- Base the decision ONLY on the provided data. Do not assume news or future prices.
- Write `reason` in Traditional Chinese, concise (<= 200 characters).
Respond with JSON only."""


def build_prompt(symbol: str, interval: str, feats: dict, position_pct: float, unrealized_pct: float,
                 max_position_pct: float) -> str:
    payload = {
        "symbol": symbol,
        "candle_interval": interval,
        "features": feats,
        "account": {
            "current_position_pct": round(position_pct, 1),
            "unrealized_pnl_pct_on_position": round(unrealized_pct, 2),
            "max_position_pct": max_position_pct,
        },
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


# ---------------- 網格模式 ----------------

GRID_SYSTEM = """You manage a spot GRID trading bot on a PAPER-TRADING account (long-only, no leverage).
A grid places buy orders below the price and sell orders above it; every small swing between two adjacent levels earns about one grid spacing minus fees.
Grids profit in sideways / choppy markets. They lose in strong downtrends (keep buying while price falls) and lag in strong uptrends (everything gets sold early).
Each turn (after a candle closes) you receive technical features and the current grid status, and choose ONE action:
- KEEP: leave the current grid (or stay idle if none). This is the default; reconfiguring costs fees.
- SET: (re)build the grid around the current price: lower_pct = % below price, upper_pct = % above price, grids = number of intervals.
  Wider ranges survive bigger moves but trade less often; more grids = smaller spacing = more trades but less profit per trade.
  Spacing (about (lower_pct+upper_pct)/grids %) must exceed the stated min_spacing_pct; grids will be reduced automatically otherwise.
  Asymmetric ranges are allowed (e.g. more room below in a mild pullback).
- PAUSE: cancel the grid and sell everything (e.g. a clear breakdown / strong downtrend). Use SET later to restart.
Guidelines:
- Prefer KEEP when the price is still well inside the range and the regime has not changed.
- Consider SET when price is near or outside the range edge, or volatility changed a lot (ATR is a good guide for range width).
- confidence reflects strength of evidence (0-1). Base the decision ONLY on the provided data.
- If `memory` is present: `recent_decisions_with_outcomes` shows what your recent choices led to (until the next decision),
  and `lessons_from_daily_reviews` are notes from your own past reviews. Use them to avoid repeating mistakes,
  but single outcomes are noisy; do not overreact to one result.
- For KEEP / PAUSE set lower_pct, upper_pct and grids to 0.
- Write `reason` in Traditional Chinese, concise (<= 200 characters).
Respond with JSON only."""


def build_grid_prompt(symbol: str, interval: str, feats: dict, grid: Optional[dict], constraints: dict,
                      memory: Optional[dict] = None) -> str:
    payload = {
        "symbol": symbol,
        "candle_interval": interval,
        "features": feats,
        "current_grid": grid if grid is not None else "none (idle, all cash)",
        "constraints": constraints,
    }
    if memory is not None:
        payload["memory"] = memory
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


REFLECTION_SYSTEM = """You review yesterday's decisions of an AI-managed spot GRID trading bot (paper trading).
You receive the day's results and every decision (KEEP / SET / PAUSE, with parameters, what actually happened,
and the outcome until the next decision: price change, equity change, grid fills, forced exits), plus your current lessons.
Task: write a short summary of the day, and return an UPDATED full list of at most 5 lessons that would improve future decisions.
- Lessons must be concrete and actionable (e.g. about range width vs volatility, grid count vs fees, when to PAUSE or not reconfigure).
- Keep earlier lessons that still hold; revise or drop ones contradicted by evidence. Fewer, better lessons are preferred.
- One day is a small sample: do not draw strong conclusions from a single event; say so when evidence is weak.
- Only rely on the data provided. Write in Traditional Chinese.
Respond with JSON only."""


def build_reflection_prompt(symbol: str, review: dict, lessons: list) -> str:
    payload = {"symbol": symbol, "day_review": review, "current_lessons": lessons}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


# ---------------- 合約模式 ----------------

FUTURES_SYSTEM = """You are a cautious trader managing a USDT-margined PERPETUAL FUTURES account in PAPER TRADING (no real money).
You can go LONG or SHORT with leverage up to the stated max_leverage. Each turn you receive technical features of the latest CLOSED candles,
the current position and the funding rate, and choose ONE action:
- LONG / SHORT: hold a long / short position with the given `leverage` (position notional / equity, e.g. 1.0 = no extra leverage).
  If already positioned, this sets the new target (can add, reduce, or flip direction).
- FLAT: close any position and stay in cash.
- HOLD: keep the current position unchanged (default when signals are mixed).
Guidelines:
- Leverage magnifies losses; a forced stop-loss and liquidation exist. Use lower leverage when volatility (ATR) is high.
- Prefer FLAT or HOLD when the trend is unclear. Every trade costs fees; avoid churning.
- Funding: when funding_rate_per_8h is positive, longs pay shorts every 8 hours (and vice versa). It matters for long holding periods.
- confidence reflects strength of evidence (0-1). Base the decision ONLY on the provided data. For FLAT / HOLD set leverage to 0.
- Write `reason` in Traditional Chinese, concise (<= 200 characters).
Respond with JSON only."""


def build_futures_prompt(symbol: str, interval: str, feats: dict, position: dict, funding_rate: Optional[float],
                         constraints: dict) -> str:
    payload = {
        "symbol": symbol,
        "candle_interval": interval,
        "features": feats,
        "position": position,
        "funding_rate_per_8h": funding_rate,
        "constraints": constraints,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
