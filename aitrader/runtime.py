"""長時間執行的共用工具:安全停止(SIGINT / SIGTERM / SIGHUP)與可中斷的等待。"""
from __future__ import annotations

import contextlib
import os
import signal
import sys
import time
from typing import Optional

INTERVAL_SECS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400}


class StopFlag:
    """SIGINT / SIGTERM / SIGHUP(關閉視窗):第一次只設旗標,在安全點存檔結束;第二次才強制中斷。"""

    def __init__(self, tag: str = "paper") -> None:
        self.requested = False
        self.tag = tag

    def __call__(self, signum, frame) -> None:
        if signum == getattr(signal, "SIGHUP", None):
            # 終端機視窗已關閉:之後印字會出錯(EIO),導到 /dev/null,確保能存檔結束
            sys.stdout = sys.stderr = open(os.devnull, "w")
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        try:
            print(f"\n[{self.tag}] 收到停止訊號,處理完目前這一步就存檔結束(再按一次強制結束)", flush=True)
        except OSError:
            pass


@contextlib.contextmanager
def handle_signals(stop: StopFlag):
    old = {}
    # SIGHUP:關閉終端機視窗時也存檔再結束(預設行為是直接終止)
    for sig in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGHUP", None)):
        if sig is None:
            continue
        try:
            old[sig] = signal.signal(sig, stop)
        except ValueError:  # 非主執行緒
            pass
    try:
        yield stop
    finally:
        for sig, h in old.items():
            signal.signal(sig, h)


def wait(seconds: float, stop: StopFlag, clock=time) -> bool:
    """可被停止訊號打斷的等待;回傳 False 表示被要求停止。clock 可換成假時鐘(測試用)。"""
    end = clock.monotonic() + seconds
    while clock.monotonic() < end:
        if stop.requested:
            return False
        clock.sleep(min(1.0, end - clock.monotonic()))
    return not stop.requested


def parse_duration(text: str) -> float:
    """「5s」「5m」「1h」或純數字(秒)→ 秒數。"""
    t = str(text).strip().lower()
    mult = {"s": 1, "m": 60, "h": 3600}.get(t[-1:], None)
    try:
        v = float(t[:-1]) * mult if mult else float(t)
    except ValueError:
        raise ValueError(f"時間格式錯誤:{text!r}(例如 5s、5m、1h 或秒數 300)") from None
    if v < 0:
        raise ValueError("時間不可為負")
    return v


DOWNTIME_SECS = 600  # 兩輪之間超過這麼久,視為程式中斷(電腦睡眠、程式未執行)


class DowntimeWatch:
    """偵測執行迴圈的中斷:以牆上時間計算(macOS 睡眠時 monotonic 時鐘不前進,不能用)。"""

    def __init__(self, last_ts: Optional[float] = None):
        self.last = last_ts

    def tick(self, now: float) -> Optional[float]:
        """每輪開始時呼叫:回傳距上次 mark 的中斷秒數(超過門檻才回);否則 None。"""
        gap = now - self.last if self.last else 0.0
        self.last = now
        return gap if gap > DOWNTIME_SECS else None

    def mark(self, now: float) -> None:
        """每輪結束、開始等待前呼叫:只計算等待期間的空檔,程式自己處理的時間(如 AI 呼叫重試)不算中斷。"""
        self.last = now


def fmt_hours(seconds: float) -> str:
    return f"{seconds / 3600:.1f} 小時" if seconds >= 3600 else f"{seconds / 60:.0f} 分鐘"
