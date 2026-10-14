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
from typing import Dict, Optional, Tuple

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

    def __init__(self, api_key: str, model: str, timeout: float = 60.0, max_retries: int = 3,
                 temperature: float = 0.2):
        if not api_key:
            raise LLMConfigError("缺少 GEMINI_API_KEY:請在專案根目錄建立 .env(參考 .env.example)或設定環境變數")
        self.api_key, self.model = api_key, model
        self.name = f"gemini:{model}"
        self.timeout, self.max_retries, self.temperature = timeout, max_retries, temperature

    def _request(self, body: dict) -> dict:
        req = urllib.request.Request(
            GEMINI_URL.format(model=self.model),
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
                    raise LLMConfigError(f"Gemini API {e.code}:{detail}(請檢查金鑰、模型名稱 {self.model})") from None
                raise LLMError(f"Gemini API {e.code}:{detail}") from None
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                if attempt < self.max_retries:
                    time.sleep(delay + random.uniform(0, 0.5))
                    delay *= 2
                    continue
                raise LLMError(f"無法連線 Gemini:{e}") from None
        raise LLMError("Gemini 重試次數用盡")  # pragma: no cover

    def decide(self, system: str, prompt: str, feats: Optional[dict] = None) -> Tuple[Decision, Dict]:
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "responseMimeType": "application/json",
                "responseSchema": DECISION_SCHEMA,
            },
        }
        t0 = time.monotonic()
        resp = self._request(body)
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
            "model": self.name,
            "latency_s": round(latency, 2),
            "tokens_in": usage.get("promptTokenCount", 0),
            "tokens_out": usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0),
        }
        return parse_decision(text), meta


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


class CachedModel:
    """以 (模型, system, prompt) 的雜湊快取決策到磁碟,重跑回放時不重複付費。"""

    def __init__(self, inner, cache_dir: str = "cache"):
        self.inner, self.cache_dir = inner, cache_dir
        self.name = inner.name
        self.hits = self.misses = 0
        os.makedirs(cache_dir, exist_ok=True)

    def _path(self, system: str, prompt: str) -> str:
        h = hashlib.sha256(f"{self.inner.name}\n{system}\n{prompt}".encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{h}.json")

    def decide(self, system: str, prompt: str, feats: Optional[dict] = None) -> Tuple[Decision, Dict]:
        path = self._path(system, prompt)
        if os.path.exists(path):
            try:
                with open(path) as fh:
                    d = json.load(fh)
                self.hits += 1
                return Decision(**d["decision"]), dict(d["meta"], cached=True)
            except (ValueError, KeyError, TypeError):
                pass  # 快取壞了就重新呼叫
        decision, meta = self.inner.decide(system, prompt, feats)
        self.misses += 1
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"decision": asdict(decision), "meta": meta}, fh, ensure_ascii=False)
        os.replace(tmp, path)
        return decision, meta
