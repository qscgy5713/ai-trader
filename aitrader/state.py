"""狀態存檔:JSON 原子寫入(暫存檔 + fsync + replace)與檔案鎖(避免兩個行程同時寫)。"""
from __future__ import annotations

import json
import os
import tempfile
import time
from typing import Optional, Tuple

try:
    import fcntl
except ImportError:  # Windows:不加鎖
    fcntl = None  # type: ignore[assignment]


def save_json(path: str, payload: dict) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".state-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def load_json(path: str) -> dict:
    try:
        with open(path) as fh:
            d = json.load(fh)
    except ValueError as e:
        raise ValueError(f"狀態檔 {path} 損毀:{e}") from e
    if not isinstance(d, dict):
        raise ValueError(f"狀態檔 {path} 格式錯誤")
    return d


def acquire_lock(path: str):
    """非阻塞獨占鎖;回傳的檔案物件要持有到結束(close 即釋放)。已被占用則丟 ValueError。"""
    if fcntl is None:
        return None
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fh = open(path + ".lock", "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        raise ValueError(f"狀態檔 {path} 正被另一個行程使用;請先結束它,或用 --state 指定其他檔案")
    return fh


def append_jsonl(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def archive_journal(path: str) -> Optional[str]:
    """開新帳戶時把舊日誌改名封存(檔名加上時間),讓 report 只看到目前帳戶。回傳封存後的路徑;沒有舊日誌回 None。"""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    base, ext = os.path.splitext(path)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(os.path.getmtime(path)))
    dest = f"{base}.{stamp}{ext}"
    n = 1
    while os.path.exists(dest):  # 同一秒內多次封存
        dest = f"{base}.{stamp}-{n}{ext}"
        n += 1
    os.replace(path, dest)
    return dest


def old_lessons(path: str, a) -> Tuple[list, str]:
    """開新帳戶時要沿用的心得與其檢討日期:舊狀態檔裡的心得;--reset-memory 時不沿用。讀不到就回 ([], "")。"""
    if getattr(a, "reset_memory", False) or not os.path.exists(path):
        return [], ""
    try:
        mem = (load_json(path).get("agent") or {}).get("memory") or {}
        return [str(x) for x in mem.get("lessons") or []], str(mem.get("lessons_updated") or "")
    except (ValueError, OSError, AttributeError):
        return [], ""


def wait_lock(path: str, timeout: float, poll: float = 1.0):
    """等待取得獨占鎖(最多 timeout 秒);逾時回 False,不支援檔案鎖的平台回 None。"""
    end = time.monotonic() + timeout
    while True:
        fh = try_lock(path)
        if fh is not False or time.monotonic() >= end:
            return fh
        time.sleep(poll)


def try_lock(path: str):
    """和 acquire_lock 相同,但已被占用時回 False(而不是丟例外);不支援檔案鎖的平台回 None。"""
    try:
        return acquire_lock(path)
    except ValueError:
        return False
