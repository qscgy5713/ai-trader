"""讓 AI 從過去的結果學習(不是重新訓練模型,而是把成績單放進提示詞)。

兩層:
1. 短期:最近幾次決策與「到下一次決策為止」的結果(價格、權益變化、網格成交、停損等事件),每次決策都附上。
2. 長期:每天(UTC 換日)把前一天的決策紀錄交給模型檢討,整理成最多 5 條心得;之後每次決策都附上,下次檢討時更新。

樣本少時容易學錯,所以心得只是參考;停損、上限等規則仍由程式強制執行。
"""
from __future__ import annotations

import time
from typing import List, Optional

from .llm import MAX_LESSONS

HISTORY_LEN = 8  # 決策時附上最近幾次的結果
MAX_DAY_LOG = 48  # 一天最多保留幾筆給檢討(防止短週期時提示詞過長)
MAX_REVIEW_ATTEMPTS = 3  # 檢討呼叫失敗最多重試幾次(之後放棄這一天)


def _utc(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.gmtime(ts))


class LearningMemory:
    def __init__(self) -> None:
        self.history: List[dict] = []  # 已有結果的最近決策
        self.day_log: List[dict] = []  # 今天已有結果的決策(給明天檢討)
        self.pending: Optional[dict] = None  # 最近一次決策,結果尚未結算
        self.lessons: List[str] = []
        self.lessons_updated = ""  # 最近一次檢討的日期
        self.review: Optional[dict] = None  # 等待檢討的前一天資料
        self.review_attempts = 0
        self.grid_fills = 0  # 累計網格掛單成交筆數
        self.day_fills0 = 0  # 當日開始時的 grid_fills
        self.day_events: List[str] = []

    # ---- 記錄 ----
    def on_tick(self, grid_fills: int, event: Optional[str]) -> None:
        self.grid_fills += grid_fills
        if event:
            self.day_events.append(event)
            if self.pending is not None:
                self.pending["events"].append(event)

    def record_decision(self, ts: float, price: float, equity: float, decision: Optional[dict], code: str) -> None:
        """記錄一次決策點(問了模型才記;停機中不記)。結果在下一次決策或換日時結算。"""
        d = decision or {}
        params = (f"-{d['lower_pct']:g}%/+{d['upper_pct']:g}% {d['grids']} grids"
                  if d.get("action") == "SET" else None)
        self.pending = {"time_utc": _utc(ts), "action": d.get("action", "ERROR"), "params": params,
                        "confidence": d.get("confidence"), "result": code, "reason": (d.get("reason") or "")[:80],
                        "events": [], "_ts": ts, "_price": price, "_equity": equity, "_fills": self.grid_fills}

    def settle(self, ts: float, price: float, equity: float) -> None:
        """結算最近一次決策的結果。"""
        p = self.pending
        if p is None:
            return
        entry = {k: v for k, v in p.items() if not k.startswith("_")}
        entry["outcome"] = {
            "hours": round((ts - p["_ts"]) / 3600, 1),
            "price_change_pct": round((price / p["_price"] - 1) * 100, 2),
            "equity_change_pct": round((equity / p["_equity"] - 1) * 100, 3) if p["_equity"] else 0.0,
            "grid_fills": self.grid_fills - p["_fills"],
        }
        self.history = (self.history + [entry])[-HISTORY_LEN:]
        self.day_log = (self.day_log + [entry])[-MAX_DAY_LOG:]
        self.pending = None

    def roll_day(self, old_day: str, ts: float, price: float, equity: float, day_start_equity: float,
                 fees_today: float) -> None:
        """UTC 換日:結算未完成的決策,把前一天整理成待檢討資料。"""
        self.settle(ts, price, equity)
        if old_day and (self.day_log or self.day_events):
            self.review = {
                "day_utc": old_day,
                "equity_change_pct": round((equity / day_start_equity - 1) * 100, 3) if day_start_equity else 0.0,
                "grid_fills": self.grid_fills - self.day_fills0,
                "fees_pct_of_equity": round(fees_today / day_start_equity * 100, 3) if day_start_equity else 0.0,
                "forced_exits": list(self.day_events),
                "decisions": list(self.day_log),
            }
            self.review_attempts = 0
        self.day_log, self.day_events, self.day_fills0 = [], [], self.grid_fills

    # ---- 給模型看 ----
    def prompt_view(self) -> Optional[dict]:
        """沒有任何紀錄與心得時回 None(不浪費 tokens)。"""
        view = {}
        if self.history:
            view["recent_decisions_with_outcomes"] = [{k: v for k, v in e.items() if k != "reason"} for e in self.history]
        if self.lessons:
            view["lessons_from_daily_reviews"] = self.lessons
        return view or None

    def set_lessons(self, lessons: List[str], day: str) -> None:
        clean = [x.strip()[:200] for x in lessons if isinstance(x, str) and x.strip()]  # 快取讀回的也要清理
        self.lessons, self.lessons_updated, self.review = clean[:MAX_LESSONS], day, None

    def review_failed(self) -> None:
        self.review_attempts += 1
        if self.review_attempts >= MAX_REVIEW_ATTEMPTS:
            self.review = None

    # ---- 持久化 ----
    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("history", "day_log", "pending", "lessons", "lessons_updated", "review",
                                              "review_attempts", "grid_fills", "day_fills0", "day_events")}

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "LearningMemory":
        m = cls()
        if not d:
            return m  # 舊版狀態檔沒有學習紀錄:從空白開始
        try:
            m.history, m.day_log = list(d["history"]), list(d["day_log"])
            m.pending, m.review = d["pending"], d["review"]
            m.lessons, m.lessons_updated = [str(x) for x in d["lessons"]], str(d["lessons_updated"])
            m.review_attempts, m.grid_fills, m.day_fills0 = int(d["review_attempts"]), int(d["grid_fills"]), int(d["day_fills0"])
            m.day_events = [str(x) for x in d["day_events"]]
        except (KeyError, TypeError, ValueError) as ex:
            raise ValueError(f"學習紀錄格式錯誤:{ex!r}") from ex
        return m
