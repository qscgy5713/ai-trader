"""行情:Binance 公開 API(免金鑰)與技術指標。"""
from __future__ import annotations

import json
import math
import urllib.request
from typing import Dict, List, NamedTuple, Tuple

BINANCE = "https://api.binance.com"
BINANCE_FUTURES = "https://fapi.binance.com"  # USDT 永續合約


class Candle(NamedTuple):
    ts: float  # 開盤時間(秒)
    open: float
    high: float
    low: float
    close: float
    volume: float


def _get(url: str):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.load(r)


def fetch_klines(symbol: str, interval: str = "1h", limit: int = 200, futures: bool = False) -> List[Candle]:
    """最近 limit 根「已收盤」K 線(>1000 自動往回分頁)。futures=True 取 USDT 永續合約的 K 線。

    不帶 endTime 的第一頁最後一根是形成中的 K 線,直接丟掉,並多要一根補回。"""
    if limit < 1:
        raise ValueError("limit 必須 >= 1")
    out: List[Candle] = []
    end, first = "", True
    while len(out) < limit:
        need = limit - len(out)
        n = min(1000, need + 1 if first else need)
        url = (f"{BINANCE_FUTURES}/fapi/v1/klines" if futures else f"{BINANCE}/api/v3/klines")
        data = _get(f"{url}?symbol={symbol}&interval={interval}&limit={n}{end}")
        if not data:
            break
        rows = data[:-1] if first else data
        out = [Candle(k[0] / 1000, *(float(x) for x in k[1:6])) for k in rows] + out
        end = f"&endTime={data[0][0] - 1}"
        if len(data) < n:
            break
        first = False
    return out[-limit:]


def fetch_price(symbol: str) -> float:
    return float(_get(f"{BINANCE}/api/v3/ticker/price?symbol={symbol}")["price"])


def fetch_premium(symbol: str) -> Tuple[float, float, float]:
    """合約 (標記價格, 目前預估資金費率, 下次結算時間秒)。"""
    d = _get(f"{BINANCE_FUTURES}/fapi/v1/premiumIndex?symbol={symbol}")
    return float(d["markPrice"]), float(d["lastFundingRate"]), d["nextFundingTime"] / 1000


def fetch_funding_history(symbol: str, start_ts: float, end_ts: float) -> List[Tuple[float, float]]:
    """[start_ts, end_ts] 之間的歷史資金費率 [(結算時間秒, 費率)],依時間排序。"""
    out: List[Tuple[float, float]] = []
    start = int(start_ts * 1000)
    end = int(end_ts * 1000)
    while start <= end:
        rows = _get(f"{BINANCE_FUTURES}/fapi/v1/fundingRate?symbol={symbol}&startTime={start}&endTime={end}&limit=1000")
        if not rows:
            break
        out += [(r["fundingTime"] / 1000, float(r["fundingRate"])) for r in rows]
        if len(rows) < 1000:
            break
        start = rows[-1]["fundingTime"] + 1
    return out


# ---------------- 指標 ----------------

def sma(xs: List[float], n: int) -> float:
    return sum(xs[-n:]) / n


def ema(xs: List[float], n: int) -> float:
    k = 2 / (n + 1)
    e = xs[0]
    for x in xs[1:]:
        e = x * k + e * (1 - k)
    return e


def rsi(xs: List[float], n: int = 14) -> float:
    """Wilder RSI。"""
    gains = losses = 0.0
    for a, b in zip(xs[-n - 1:-1], xs[-n:]):
        d = b - a
        gains += max(d, 0)
        losses += max(-d, 0)
    if losses == 0:
        return 100.0 if gains > 0 else 50.0
    rs = (gains / n) / (losses / n)
    return 100 - 100 / (1 + rs)


def atr_pct(cs: List[Candle], n: int = 14) -> float:
    trs = [max(c.high - c.low, abs(c.high - p.close), abs(c.low - p.close)) for p, c in zip(cs[-n - 1:-1], cs[-n:])]
    return sum(trs) / n / cs[-1].close * 100


MIN_BARS = 60  # 計算指標所需的最少 K 線數


def features(cs: List[Candle]) -> Dict:
    """把 K 線整理成給模型看的精簡特徵(數字取到合理位數,控制 token)。"""
    if len(cs) < MIN_BARS:
        raise ValueError(f"至少需要 {MIN_BARS} 根 K 線,目前 {len(cs)} 根")
    closes = [c.close for c in cs]
    last = closes[-1]

    def ret(k: int) -> float:
        return round((last / closes[-1 - k] - 1) * 100, 2)

    vols = [c.volume for c in cs]
    hi20, lo20 = max(c.high for c in cs[-20:]), min(c.low for c in cs[-20:])
    log_rets = [math.log(b / a) for a, b in zip(closes[-25:-1], closes[-24:])]
    mean = sum(log_rets) / len(log_rets)
    vol24 = math.sqrt(sum((r - mean) ** 2 for r in log_rets) / (len(log_rets) - 1)) * 100
    return {
        "last_close": round(last, 6),
        "return_pct": {"1_bar": ret(1), "6_bars": ret(6), "24_bars": ret(24), "48_bars": ret(48)},
        "sma20_vs_price_pct": round((sma(closes, 20) / last - 1) * 100, 2),
        "sma50_vs_price_pct": round((sma(closes, 50) / last - 1) * 100, 2),
        "macd_hist_pct": round((ema(closes[-60:], 12) - ema(closes[-60:], 26)) / last * 100, 3),
        "rsi14": round(rsi(closes, 14), 1),
        "atr14_pct": round(atr_pct(cs, 14), 2),
        "volatility_24_bars_pct": round(vol24, 2),
        "position_in_20_bar_range": round((last - lo20) / (hi20 - lo20), 2) if hi20 > lo20 else 0.5,
        "volume_vs_20_bar_avg": round(vols[-1] / (sum(vols[-20:]) / 20), 2) if sum(vols[-20:]) > 0 else 1.0,
        "recent_closes": [round(x, 6) for x in closes[-24:]],
    }


MAX_EXCHANGE = "https://max-api.maicoin.com"  # 台灣交易所 MAX,用來取 USDT/TWD 匯率


def fetch_usdt_twd() -> float:
    """1 USDT 等於多少台幣(MAX 最新成交價)。"""
    rate = float(_get(f"{MAX_EXCHANGE}/api/v2/tickers/usdttwd")["last"])
    if not rate > 0:
        raise ValueError(f"USDT/TWD 匯率異常:{rate}")
    return rate
