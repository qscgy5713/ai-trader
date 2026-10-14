"""提示詞:把特徵與持倉整理成給模型的輸入。輸出格式由 llm.DECISION_SCHEMA 強制。"""
from __future__ import annotations

import json

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
