"""急跌檢查成效檢驗(report --alerts):觸發後的實際處理 vs 不減倉的反事實。

只從觸發當下起算(觸發前已經虧的部分兩邊相同,不列入),比較三種結果:
1. 實際:急跌檢查後的減倉 / 平倉 / 續抱,直到部位結束。
2. 不減倉、同時出場:原倉位抱到實際的出場時間;期間 1 分鐘 K 線碰到觸發時的停損價就以停損價出場。
3. 不減倉、抱到停損或 24 小時:不管 AI 之後怎麼做,只看停損會不會先到。
反事實不含追蹤停損、資金費與滑價,手續費以實際費率計;只是估計,不是精確回放。
"""
from __future__ import annotations

import re
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .currency import Money
from .market import Candle

HOLD_SECS = 24 * 3600  # 第 3 種:最多抱多久
NAME_RE = re.compile(r"futures_([A-Z0-9]+)_")


def symbol_of(path: str) -> Optional[str]:
    """從預設檔名(journal/futures_SOLUSDT_gemini.jsonl)取出交易對;回放日誌與其他檔名回 None。"""
    m = NAME_RE.search(path.rsplit("/", 1)[-1])
    return m.group(1) if m and "replay" not in path else None


def alert_positions(recs: Sequence[dict]) -> Tuple[List[dict], int]:
    """把成交拆成一個個部位,回傳(有急跌檢查且已結束的部位, 仍持有中的急跌檢查部位數)。"""
    out, pos, cur = [], 0.0, None
    for e in recs:
        if e.get("kind") != "futures" or not e.get("trade"):
            continue
        t = e["trade"]
        dq = t["qty"] if t["side"] == "buy" else -t["qty"]
        if cur is None:
            cur = {"trades": [], "alert": None}
        cur["trades"].append((e, pos, dq))
        if e.get("alert") and cur["alert"] is None and (e["alert"] or {}).get("stop_price"):
            cur["alert"] = (e, pos)  # 第一次觸發;舊版紀錄沒有停損價,無法比較
        new = pos + dq
        if abs(new) < 1e-9 or pos * new < 0:  # 平倉或反手:部位結束
            if cur["alert"]:
                out.append(cur)
            cur = {"trades": [], "alert": None} if abs(new) > 1e-9 else None
        pos = new
    return out, int(bool(cur and cur["alert"]))


def actual_pnl(p: dict) -> float:
    """觸發當下(以觸發價估值)到部位結束的實際損益,含手續費(USDT)。"""
    e0, q0 = p["alert"]
    cash, started = 0.0, False
    for e, pos, dq in p["trades"]:
        started = started or e is e0
        if not started:
            continue
        t = e["trade"]
        close = dq if pos * (pos + dq) >= 0 else -pos  # 反手只算平掉舊倉的部分
        cash += -close * t["price"] - t["fee"] * abs(close) / abs(dq)
    return cash - q0 * e0["price"]


def fee_rate(p: dict) -> float:
    t = p["trades"][0][0]["trade"]
    return t["fee"] / (t["qty"] * t["price"]) if t["qty"] and t["price"] else 0.0


def counterfactual(q0: float, ref: float, stop: float, t0: float, t_end: float, bars: Sequence[Candle],
                   fee: float) -> Tuple[Optional[float], Optional[float]]:
    """原倉位 q0 自 t0(價格 ref)抱到 t_end,1 分鐘 K 線碰到 stop 就以停損價出場。
    回傳(損益 USDT, 停損時間;沒停損為 None);沒有 K 線資料時損益為 None。"""
    long, exit_px = q0 > 0, None
    if not bars or bars[0].ts > t0 + 60:  # K 線沒涵蓋觸發時間(太久以前):無法比較
        return None, None
    for c in bars:
        if c.ts + 60 <= t0:
            continue
        if c.ts > t_end:
            break
        if (c.low <= stop) if long else (c.high >= stop):
            return (stop - ref) * q0 - abs(q0) * stop * fee, c.ts
        exit_px = c.close
    if exit_px is None:
        return None, None
    return (exit_px - ref) * q0 - abs(q0) * exit_px * fee, None


def review(recs: Sequence[dict], bars: Sequence[Candle], now: float) -> Tuple[List[dict], int]:
    """逐筆比較;回傳(各次急跌檢查的結果, 仍持有中的筆數)。"""
    positions, open_n = alert_positions(recs)
    rows = []
    for p in positions:
        e0, q0 = p["alert"]
        stop, last = e0["alert"]["stop_price"], p["trades"][-1][0]
        fee = fee_rate(p)
        same, same_stop = counterfactual(q0, e0["price"], stop, e0["ts"], last["ts"], bars, fee)
        hold, hold_stop = counterfactual(q0, e0["price"], stop, e0["ts"], min(e0["ts"] + HOLD_SECS, now), bars, fee)
        rows.append({"ts": e0["ts"], "side": "多" if q0 > 0 else "空", "actual": actual_pnl(p),
                     "same": same, "same_stop": same_stop, "hold": hold, "hold_stop": hold_stop,
                     "hold_full": e0["ts"] + HOLD_SECS <= now})
    return rows, open_n


def bars_needed(recs_list: Sequence[Sequence[dict]], now: float) -> int:
    """涵蓋所有急跌檢查(含之後 24 小時)需要的 1 分鐘 K 線根數;沒有急跌檢查回 0。"""
    first = [p["alert"][0]["ts"] for recs in recs_list for p in alert_positions(recs)[0]]
    return int((now - min(first)) / 60) + 10 if first else 0


def print_review(groups: Dict[str, List[dict]], open_n: int, money: Money,
                 out: Callable[[str], None] = print) -> None:
    """groups:{名稱: review() 的結果}。"""
    rows = [(name, r) for name, rs in groups.items() for r in rs]
    if not rows:
        out("沒有可比較的急跌檢查(需要已結束的部位,且紀錄含停損價)"
            + (f";另有 {open_n} 筆仍持有中" if open_n else ""))
        return

    def f(x: Optional[float]) -> str:
        return "?" if x is None else money.fmt(x, sign=True)

    def how(stop_ts: Optional[float], full: bool = True) -> str:
        return f"停損 {time.strftime('%m-%d %H:%M', time.localtime(stop_ts))}" if stop_ts else ("未停損" if full else "未停損(未滿 24h)")

    out("急跌檢查成效(自觸發當下起算,含手續費;反事實不含追蹤停損與資金費,僅供估計)")
    out("  觸發時間     方向  實際          不減倉同時出場            不減倉抱到停損 / 24h")
    for name, r in rows:
        tag = f"  [{name}]" if len(groups) > 1 else ""
        out(f"  {time.strftime('%m-%d %H:%M', time.localtime(r['ts']))}  {r['side']}   {f(r['actual']):>12}  "
            f"{f(r['same']):>12} {how(r['same_stop'], True) if r['same_stop'] else '':<14}  "
            f"{f(r['hold']):>12} {how(r['hold_stop'], r['hold_full'])}{tag}")
    ok = [r for _, r in rows if r["same"] is not None and r["hold"] is not None]
    skip = len(rows) - len(ok)
    tot = {k: sum(r[k] for r in ok) for k in ("actual", "same", "hold")}
    out(f"  合計 {len(ok)} 次{f'(另 {skip} 次沒有 K 線資料,未列入)' if skip else ''}:實際 {f(tot['actual'])};不減倉同時出場 {f(tot['same'])};"
        f"不減倉抱到停損 / 24h {f(tot['hold'])}")
    saved = [r["same"] - r["actual"] for r in ok]
    better = sum(1 for x in saved if x < 0)
    out(f"  與「不減倉同時出場」相比:急跌檢查{'少虧' if sum(saved) <= 0 else '多虧'} {money.fmt(abs(sum(saved)))}"
        f"(做得比較好 {better} 次 / 比較差 {len(saved) - better} 次)"
        + (f";另有 {open_n} 筆仍持有中,未列入" if open_n else ""))
