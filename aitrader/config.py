"""設定:從環境變數讀取,可放在專案根目錄的 .env(已 gitignore)。"""
from __future__ import annotations

import os
from typing import Tuple

DEFAULT_MODEL = "gemini-3.8-flash"


def load_dotenv(path: str = ".env") -> None:
    """極簡 .env 解析:KEY=VALUE、# 註解、可加引號。已存在的環境變數優先,不會被覆蓋。"""
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
                v = v[1:-1]
            if k:
                os.environ.setdefault(k, v)


def gemini_settings() -> Tuple[str, str]:
    """(API key, 模型名稱)。key 可能是空字串,由呼叫端決定是否需要。"""
    return os.environ.get("GEMINI_API_KEY", "").strip(), os.environ.get("GEMINI_MODEL", "").strip() or DEFAULT_MODEL


def gemini_reflect_model() -> str:
    """每日檢討用的模型(GEMINI_REFLECT_MODEL);空字串 = 與決策用同一個模型。"""
    return os.environ.get("GEMINI_REFLECT_MODEL", "").strip()
