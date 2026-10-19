"""決策模型:Gemini(REST,僅標準庫)、離線規則基準、快取包裝。

所有模型都實作 decide(system, prompt, feats) -> (Decision, meta)。
模型輸出一律經 parse_decision 嚴格驗證;任何不合格的輸出都丟 LLMError,呼叫端視為「不交易」。
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Tuple

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
ACTIONS = ("BUY", "SELL", "HOLD")


class LLMError(Exception):
    """模型呼叫或輸出有問題;呼叫端應視為 HOLD,不交易。"""


class LLMConfigError(LLMError):
    """設定 / 帳務錯誤(金鑰無效、模型不存在、請求格式錯、額度用完);重試無用,應停下來讓使用者處理。"""


@dataclass
class Decision:
    action: str  # BUY / SELL / HOLD
    target_position_pct: float  # 期望持倉占總權益的百分比 0~100
    confidence: float  # 0~1
    reason: str


# Gemini responseSchema(OpenAPI 子集)
DECISION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "action": {"type": "STRING", "enum": list(ACTIONS)},
        "target_position_pct": {"type": "NUMBER", "description": "0-100, desired position as % of total equity"},
        "confidence": {"type": "NUMBER", "description": "0-1"},
        "reason": {"type": "STRING", "description": "brief rationale in Traditional Chinese, <= 200 chars"},
    },
    "required": ["action", "target_position_pct", "confidence", "reason"],
    "propertyOrdering": ["action", "target_position_pct", "confidence", "reason"],
}


def parse_decision(text: str) -> Decision:
    """嚴格解析模型輸出。欄位缺漏、型別錯、超出範圍都丟 LLMError,不做「猜測修正」。"""
    try:
        d = json.loads(text)
    except (ValueError, TypeError) as e:
        raise LLMError(f"模型輸出不是合法 JSON:{str(text)[:120]!r}") from e
    if not isinstance(d, dict):
        raise LLMError("模型輸出不是 JSON 物件")
    action = d.get("action")
    if action not in ACTIONS:
        raise LLMError(f"action 不合法:{action!r}")
    try:
        target = float(d["target_position_pct"])
        conf = float(d["confidence"])
    except (KeyError, TypeError, ValueError) as e:
        raise LLMError(f"數值欄位缺漏或格式錯誤:{e!r}") from e
    if not 0 <= target <= 100:  # NaN 也會在這裡被擋下
        raise LLMError(f"target_position_pct 超出 0~100:{target}")
    if not 0 <= conf <= 1:
        raise LLMError(f"confidence 超出 0~1:{conf}")
    reason = d.get("reason")
    if not isinstance(reason, str):
        raise LLMError("reason 必須是字串")
    return Decision(action, target, conf, reason.strip()[:500])


class GeminiModel:
    """Gemini generateContent(JSON 結構化輸出)。金鑰放 header,不出現在網址或錯誤訊息。"""

    can_reflect = True  # 支援每日檢討(reflect)
    can_search_news = True  # 支援以 Google 搜尋整理每日新聞(search_news)

    def __init__(self, api_key: str, model: str, timeout: float = 60.0, max_retries: int = 3,
                 temperature: float = 0.2, schema: Optional[dict] = None,
                 parser: Optional[Callable[[str], object]] = None, reflect_model: Optional[str] = None):
        if not api_key:
            raise LLMConfigError("缺少 GEMINI_API_KEY:請在專案根目錄建立 .env(參考 .env.example)或設定環境變數")
        self.api_key, self.model = api_key, model
        # 每日檢討一天只呼叫一次,可改用較強(較貴)的模型;None = 與決策同一個
        self.reflect_model = reflect_model or model
        self.reflect_name = f"gemini:{self.reflect_model}"
        # 預設輸出「目標持倉」決策;網格模式改用 GRID_SCHEMA / parse_grid_decision
        self.schema, self.parser = schema or DECISION_SCHEMA, parser or parse_decision
        self.name = f"gemini:{model}"
        self.timeout, self.max_retries, self.temperature = timeout, max_retries, temperature

    def _request(self, body: dict, model: str) -> dict:
        req = urllib.request.Request(
            GEMINI_URL.format(model=model),
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
            method="POST",
        )
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                detail = _error_message(e)
                if e.code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    retry_after = e.headers.get("Retry-After") if e.headers else None
                    wait = float(retry_after) if retry_after and retry_after.isdigit() else delay
                    time.sleep(min(wait, 60) + random.uniform(0, 0.5))
                    delay *= 2
                    continue
                if e.code == 402:  # 額度 / 帳務問題:重試無用,必須由使用者處理
                    raise LLMConfigError(f"Gemini API 402 帳務問題:{detail}") from None
                if e.code in (400, 401, 403, 404):
                    raise LLMConfigError(f"Gemini API {e.code}:{detail}(請檢查金鑰、模型名稱 {model})") from None
                raise LLMError(f"Gemini API {e.code}:{detail}") from None
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                if attempt < self.max_retries:
                    time.sleep(delay + random.uniform(0, 0.5))
                    delay *= 2
                    continue
                raise LLMError(f"無法連線 Gemini:{e}") from None
        raise LLMError("Gemini 重試次數用盡")  # pragma: no cover

    def decide(self, system: str, prompt: str, feats: Optional[dict] = None) -> Tuple[object, Dict]:
        return self._generate(system, prompt, self.schema, self.parser, self.model)

    def reflect(self, system: str, prompt: str) -> Tuple["Reflection", Dict]:
        """每日檢討:回顧過去決策的結果,整理成心得。"""
        return self._generate(system, prompt, REFLECTION_SCHEMA, parse_reflection, self.reflect_model)

    def search_news(self, system: str, prompt: str) -> Tuple["NewsBrief", Dict]:
        """以 Google 搜尋整理新聞(用檢討模型,一天一次);meta["sources"] 是實際引用的網頁。
        不指定 responseSchema:實測搭配結構化輸出時常不附引用來源,改由提示詞規定格式、從文字取出 JSON。"""
        return self._generate(system, prompt, None, parse_news, self.reflect_model, tools=[{"google_search": {}}])

    def _generate(self, system: str, prompt: str, schema: Optional[dict], parser: Callable[[str], object],
                  model: str, tools: Optional[list] = None) -> Tuple[object, Dict]:
        """schema=None:不要求結構化輸出(由 parser 自行從文字取出 JSON)。"""
        config = {"temperature": self.temperature}
        if schema is not None:
            config.update(responseMimeType="application/json", responseSchema=schema)
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": config,
        }
        if tools:
            body["tools"] = tools
        t0 = time.monotonic()
        resp = self._request(body, model)
        latency = time.monotonic() - t0
        cands = resp.get("candidates") or []
        if not cands:
            block = (resp.get("promptFeedback") or {}).get("blockReason")
            raise LLMError("Gemini 沒有回傳結果" + (f"(被擋:{block})" if block else ""))
        cand = cands[0]
        finish = cand.get("finishReason")
        if finish not in (None, "STOP"):
            raise LLMError(f"Gemini 未正常結束:finishReason={finish}")
        parts = (cand.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        usage = resp.get("usageMetadata") or {}
        meta = {
            "model": f"gemini:{model}",
            "latency_s": round(latency, 2),
            "tokens_in": usage.get("promptTokenCount", 0),
            "tokens_out": usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0),
        }
        if tools:
            g = cand.get("groundingMetadata") or {}
            meta["search_queries"] = list(g.get("webSearchQueries") or [])[:5]
            meta["sources"] = [{"title": str(w.get("title", ""))[:100], "uri": str(w.get("uri", ""))[:300]}
                               for w in ((c.get("web") or {}) for c in g.get("groundingChunks") or []) if w][:8]
        return parser(text), meta


def _error_message(e: urllib.error.HTTPError) -> str:
    try:
        return str(json.load(e).get("error", {}).get("message", ""))[:300] or e.reason
    except Exception:
        return str(e.reason)


class RuleModel:
    """離線基準模型(不花錢):均線趨勢 + RSI 過熱 / 超賣。用來和 Gemini 比較、以及測試。"""

    name = "rule"

    def decide(self, system: str, prompt: str, feats: Optional[dict] = None) -> Tuple[Decision, Dict]:
        if feats is None:
            raise LLMError("RuleModel 需要 feats")
        trend_up = feats["sma20_vs_price_pct"] < 0 and feats["sma50_vs_price_pct"] < 0  # 價格在兩條均線之上
        r = feats["rsi14"]
        if trend_up and r < 70:
            d = Decision("BUY", 50, 0.7, "價格在 20/50 均線之上且 RSI 未過熱")
        elif not trend_up and feats["sma20_vs_price_pct"] > 0 and r > 30:
            d = Decision("SELL", 0, 0.7, "價格跌破 20 均線且未超賣")
        else:
            d = Decision("HOLD", 0, 0.5, "訊號不明確")
        return d, {"model": self.name, "latency_s": 0.0, "tokens_in": 0, "tokens_out": 0}


def model_label(model) -> str:
    """顯示用的模型名稱;每日檢討用不同模型時一併標示。"""
    inner = getattr(model, "inner", model)
    name, refl = getattr(model, "name", "?"), getattr(inner, "reflect_name", None)
    return f"{name}(每日檢討 {refl})" if refl and refl != name else name


class CachedModel:
    """以 (模型, system, prompt) 的雜湊快取決策到磁碟,重跑回放時不重複付費。

    decision_cls:決策型別(Decision 或 GridDecision),讀回快取時用來還原。"""

    def __init__(self, inner, cache_dir: str = "cache", decision_cls=Decision):
        self.inner, self.cache_dir, self.decision_cls = inner, cache_dir, decision_cls
        self.name = inner.name
        self.hits = self.misses = 0
        os.makedirs(cache_dir, exist_ok=True)

    @property
    def can_reflect(self) -> bool:
        return getattr(self.inner, "can_reflect", False)

    def _path(self, system: str, prompt: str, name: str) -> str:
        h = hashlib.sha256(f"{name}\n{system}\n{prompt}".encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{h}.json")

    def decide(self, system: str, prompt: str, feats: Optional[dict] = None) -> Tuple[object, Dict]:
        return self._cached(system, prompt, lambda: self.inner.decide(system, prompt, feats), self.decision_cls,
                            self.inner.name)

    def reflect(self, system: str, prompt: str) -> Tuple["Reflection", Dict]:
        # 檢討模型不同時用它的名稱當快取鍵;相同時與舊快取一致
        return self._cached(system, prompt, lambda: self.inner.reflect(system, prompt), Reflection,
                            getattr(self.inner, "reflect_name", self.inner.name))

    def _cached(self, system: str, prompt: str, call, cls, name: str) -> Tuple[object, Dict]:
        path = self._path(system, prompt, name)
        if os.path.exists(path):
            try:
                with open(path) as fh:
                    d = json.load(fh)
                self.hits += 1
                return cls(**d["decision"]), dict(d["meta"], cached=True)
            except (ValueError, KeyError, TypeError):
                pass  # 快取壞了就重新呼叫
        decision, meta = call()
        self.misses += 1
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"decision": asdict(decision), "meta": meta}, fh, ensure_ascii=False)
        os.replace(tmp, path)
        return decision, meta


# ---------------- 網格模式:AI 決定網格區間 ----------------

GRID_ACTIONS = ("KEEP", "SET", "PAUSE")
MAX_GRIDS = 200


@dataclass
class GridDecision:
    action: str  # KEEP 維持 / SET 重設網格 / PAUSE 暫停並清倉
    lower_pct: float  # SET:區間下緣在現價下方幾 %
    upper_pct: float  # SET:區間上緣在現價上方幾 %
    grids: int  # SET:格數
    confidence: float
    reason: str


GRID_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "action": {"type": "STRING", "enum": list(GRID_ACTIONS)},
        "lower_pct": {"type": "NUMBER", "description": "SET only: range lower bound, % below current price (e.g. 4). 0 otherwise"},
        "upper_pct": {"type": "NUMBER", "description": "SET only: range upper bound, % above current price (e.g. 4). 0 otherwise"},
        "grids": {"type": "INTEGER", "description": f"SET only: number of grid intervals, 2-{MAX_GRIDS}. 0 otherwise"},
        "confidence": {"type": "NUMBER", "description": "0-1"},
        "reason": {"type": "STRING", "description": "brief rationale in Traditional Chinese, <= 200 chars"},
    },
    "required": ["action", "lower_pct", "upper_pct", "grids", "confidence", "reason"],
    "propertyOrdering": ["action", "lower_pct", "upper_pct", "grids", "confidence", "reason"],
}


def parse_grid_decision(text: str) -> GridDecision:
    """嚴格解析網格決策。SET 的區間 / 格數必須合法;KEEP / PAUSE 的數值欄位忽略(一律歸零)。"""
    try:
        d = json.loads(text)
    except (ValueError, TypeError) as e:
        raise LLMError(f"模型輸出不是合法 JSON:{str(text)[:120]!r}") from e
    if not isinstance(d, dict):
        raise LLMError("模型輸出不是 JSON 物件")
    action = d.get("action")
    if action not in GRID_ACTIONS:
        raise LLMError(f"action 不合法:{action!r}")
    try:
        conf = float(d["confidence"])
    except (KeyError, TypeError, ValueError) as e:
        raise LLMError(f"confidence 缺漏或格式錯誤:{e!r}") from e
    if not 0 <= conf <= 1:
        raise LLMError(f"confidence 超出 0~1:{conf}")
    reason = d.get("reason")
    if not isinstance(reason, str):
        raise LLMError("reason 必須是字串")
    lo = up = 0.0
    grids = 0
    if action == "SET":
        try:
            lo, up, g = float(d["lower_pct"]), float(d["upper_pct"]), float(d["grids"])
        except (KeyError, TypeError, ValueError) as e:
            raise LLMError(f"SET 的區間 / 格數缺漏或格式錯誤:{e!r}") from e
        if not (0 < lo < 100 and 0 < up <= 100):  # NaN 也會被擋下
            raise LLMError(f"SET 區間不合法:下 {lo}% / 上 {up}%")
        if not (2 <= g <= MAX_GRIDS) or g != int(g):
            raise LLMError(f"grids 必須是 2~{MAX_GRIDS} 的整數:{g}")
        grids = int(g)
    return GridDecision(action, lo, up, grids, conf, reason.strip()[:500])


class RuleGridModel:
    """網格模式的離線基準(不花錢):明顯下跌趨勢暫停;價格還在區間中段就維持;否則以 ATR 決定區間重設。"""

    name = "rule"

    def decide(self, system: str, prompt: str, feats: Optional[dict] = None) -> Tuple[GridDecision, Dict]:
        if feats is None:
            raise LLMError("RuleGridModel 需要 feats")
        meta = {"model": self.name, "latency_s": 0.0, "tokens_in": 0, "tokens_out": 0}
        grid = feats.get("grid")
        down = feats["sma20_vs_price_pct"] > 0 and feats["sma50_vs_price_pct"] > 0 and feats["macd_hist_pct"] < 0
        if down:
            if grid is None:
                return GridDecision("KEEP", 0, 0, 0, 0.7, "下跌趨勢,維持空手"), meta
            return GridDecision("PAUSE", 0, 0, 0, 0.7, "價格在 20/50 均線之下且 MACD 轉負,暫停網格"), meta
        if grid is not None and 0.2 <= grid["position_in_range"] <= 0.8:
            return GridDecision("KEEP", 0, 0, 0, 0.7, "價格仍在區間中段"), meta
        half = min(max(3 * feats["atr14_pct"], 2.0), 12.0)
        grids = max(2, int(2 * half / 0.8))  # 目標格距約 0.8%
        return GridDecision("SET", half, half, grids, 0.7, f"以 ATR 設定 ±{half:.1f}% 區間"), meta


# ---------------- 每日檢討(讓 AI 從過去的結果學習) ----------------

MAX_LESSONS = 5


@dataclass
class Reflection:
    summary: str  # 對前一天的簡短總結
    lessons: List[str]  # 更新後的完整心得清單(取代舊的)


REFLECTION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "summary": {"type": "STRING", "description": "brief review of the day in Traditional Chinese, <= 200 chars"},
        "lessons": {"type": "ARRAY", "items": {"type": "STRING"},
                    "description": f"updated full list of at most {MAX_LESSONS} lessons, Traditional Chinese, each <= 120 chars"},
    },
    "required": ["summary", "lessons"],
    "propertyOrdering": ["summary", "lessons"],
}


def parse_reflection(text: str) -> Reflection:
    try:
        d = json.loads(text)
    except (ValueError, TypeError) as e:
        raise LLMError(f"模型輸出不是合法 JSON:{str(text)[:120]!r}") from e
    if not isinstance(d, dict) or not isinstance(d.get("summary"), str) or not isinstance(d.get("lessons"), list):
        raise LLMError("檢討輸出格式錯誤:需要 summary 字串與 lessons 陣列")
    lessons = [x.strip()[:200] for x in d["lessons"] if isinstance(x, str) and x.strip()]
    return Reflection(d["summary"].strip()[:500], lessons[:MAX_LESSONS])


# ---------------- 每日新聞(Google 搜尋) ----------------

NEWS_SENTIMENTS = ("BULLISH", "BEARISH", "NEUTRAL")
MAX_NEWS_EVENTS = 4


@dataclass
class NewsEvent:
    date: str  # 事件日期(YYYY-MM-DD,來源標示的日期)
    source: str  # 來源名稱(例如 Reuters、CoinDesk、官方公告)
    event: str  # 一句話說明


@dataclass
class NewsBrief:
    summary: str  # 整體市場背景摘要
    sentiment: str  # 消息面整體偏多 / 偏空 / 中性
    events: List[NewsEvent]


def _json_in_text(text: str) -> str:
    """取出文字中的 JSON 物件(容許 ```json 區塊或前後說明文字)。"""
    s = str(text or "")
    i, j = s.find("{"), s.rfind("}")
    return s[i:j + 1] if 0 <= i < j else s


def parse_news(text: str) -> NewsBrief:
    try:
        d = json.loads(_json_in_text(text))
    except (ValueError, TypeError) as e:
        raise LLMError(f"模型輸出不是合法 JSON:{str(text)[:120]!r}") from e
    if not isinstance(d, dict) or not isinstance(d.get("summary"), str) or not isinstance(d.get("events"), list):
        raise LLMError("新聞輸出格式錯誤:需要 summary 字串與 events 陣列")
    sentiment = str(d.get("sentiment", "")).upper()
    if sentiment not in NEWS_SENTIMENTS:
        raise LLMError(f"新聞 sentiment 不合法:{sentiment!r}")
    events = []
    for e in d["events"]:
        if isinstance(e, dict) and all(isinstance(e.get(k), str) and e[k].strip() for k in ("date", "source", "event")):
            events.append(NewsEvent(e["date"].strip()[:20], e["source"].strip()[:60], e["event"].strip()[:200]))
    return NewsBrief(d["summary"].strip()[:500], sentiment, events[:MAX_NEWS_EVENTS])


# ---------------- 合約模式:AI 決定多空方向與槓桿 ----------------

FUTURES_ACTIONS = ("LONG", "SHORT", "FLAT", "HOLD")
MAX_MODEL_LEVERAGE = 10.0  # 模型輸出的格式上限;實際上限由風控 --max-leverage 決定


@dataclass
class FuturesDecision:
    action: str  # LONG 做多 / SHORT 做空 / FLAT 平倉空手 / HOLD 維持
    leverage: float  # LONG / SHORT 的目標槓桿(曝險 / 權益),例如 1.5
    confidence: float
    reason: str


FUTURES_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "action": {"type": "STRING", "enum": list(FUTURES_ACTIONS)},
        "leverage": {"type": "NUMBER", "description": "LONG/SHORT only: target leverage (position notional / equity), "
                                                      "must not exceed max_leverage. 0 for FLAT/HOLD"},
        "confidence": {"type": "NUMBER", "description": "0-1"},
        "reason": {"type": "STRING", "description": "brief rationale in Traditional Chinese, <= 200 chars"},
    },
    "required": ["action", "leverage", "confidence", "reason"],
    "propertyOrdering": ["action", "leverage", "confidence", "reason"],
}


def parse_futures_decision(text: str) -> FuturesDecision:
    """嚴格解析合約決策。LONG / SHORT 的槓桿必須 > 0;FLAT / HOLD 的槓桿忽略。"""
    try:
        d = json.loads(text)
    except (ValueError, TypeError) as e:
        raise LLMError(f"模型輸出不是合法 JSON:{str(text)[:120]!r}") from e
    if not isinstance(d, dict):
        raise LLMError("模型輸出不是 JSON 物件")
    action = d.get("action")
    if action not in FUTURES_ACTIONS:
        raise LLMError(f"action 不合法:{action!r}")
    try:
        conf = float(d["confidence"])
        lev = float(d.get("leverage", 0)) if action in ("LONG", "SHORT") else 0.0
    except (KeyError, TypeError, ValueError) as e:
        raise LLMError(f"數值欄位缺漏或格式錯誤:{e!r}") from e
    if not 0 <= conf <= 1:
        raise LLMError(f"confidence 超出 0~1:{conf}")
    if action in ("LONG", "SHORT") and not 0 < lev <= MAX_MODEL_LEVERAGE:
        raise LLMError(f"leverage 需在 (0, {MAX_MODEL_LEVERAGE:g}]:{lev}")
    reason = d.get("reason")
    if not isinstance(reason, str):
        raise LLMError("reason 必須是字串")
    return FuturesDecision(action, lev, conf, reason.strip()[:500])


class RuleFuturesModel:
    """合約模式的離線基準(不花錢):均線 + MACD 同向時順勢 1 倍,否則空手。"""

    name = "rule"

    def decide(self, system: str, prompt: str, feats: Optional[dict] = None) -> Tuple[FuturesDecision, Dict]:
        if feats is None:
            raise LLMError("RuleFuturesModel 需要 feats")
        meta = {"model": self.name, "latency_s": 0.0, "tokens_in": 0, "tokens_out": 0}
        up = feats["sma20_vs_price_pct"] < 0 and feats["sma50_vs_price_pct"] < 0 and feats["macd_hist_pct"] > 0
        down = feats["sma20_vs_price_pct"] > 0 and feats["sma50_vs_price_pct"] > 0 and feats["macd_hist_pct"] < 0
        if up:
            return FuturesDecision("LONG", 1.0, 0.7, "價格在 20/50 均線之上且 MACD 為正"), meta
        if down:
            return FuturesDecision("SHORT", 1.0, 0.7, "價格在 20/50 均線之下且 MACD 為負"), meta
        return FuturesDecision("FLAT", 0.0, 0.7, "趨勢不明確,空手"), meta
