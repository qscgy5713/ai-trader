"""長時間執行的共用工具:安全停止(SIGINT / SIGTERM)與可中斷的等待。"""
from __future__ import annotations

import contextlib
import signal
import time

INTERVAL_SECS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400}


class StopFlag:
    """SIGINT / SIGTERM:第一次只設旗標,在安全點存檔結束;第二次才強制中斷。"""

    def __init__(self, tag: str = "paper") -> None:
        self.requested = False
        self.tag = tag

    def __call__(self, signum, frame) -> None:
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        print(f"\n[{self.tag}] 收到停止訊號,處理完目前這一步就存檔結束(再按一次強制結束)", flush=True)


@contextlib.contextmanager
def handle_signals(stop: StopFlag):
    old = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
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
