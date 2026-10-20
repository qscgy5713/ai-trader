"""一次跑多個交易對:每個交易對一個子程式(帳戶、心得、日誌各自獨立),輸出加上幣別前綴。

用子程式而不是執行緒:每個交易對沿用單一交易對的完整流程(檔案鎖、停止訊號、存檔),互不影響;
其中一個出錯結束,其他照常執行。Ctrl+C 時終端機會把訊號送給整個程序群組,子程式各自存檔後結束。
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import threading
import time
from typing import Callable, List, Sequence

SYMBOL_RE = re.compile(r"^[A-Z0-9]{2,20}$")
MAX_SYMBOLS = 8  # 避免一次開太多(每個都會呼叫 Gemini)


def parse_symbols(text: str) -> List[str]:
    """「SOLUSDT,ethusdt, SUIUSDT」→ ['SOLUSDT', 'ETHUSDT', 'SUIUSDT'](去重、保留順序)。"""
    out: List[str] = []
    for s in str(text).split(","):
        s = s.strip().upper()
        if not s:
            continue
        if not SYMBOL_RE.match(s):
            raise ValueError(f"交易對格式錯誤:{s!r}(例如 SOLUSDT,多個用逗號分隔)")
        if s not in out:
            out.append(s)
    if not out:
        raise ValueError("--symbol 不可為空")
    if len(out) > MAX_SYMBOLS:
        raise ValueError(f"--symbol 最多 {MAX_SYMBOLS} 個")
    return out


def replace_symbol(argv: Sequence[str], symbol: str) -> List[str]:
    """把參數列中的 --symbol 值換成單一交易對(支援 `--symbol X` 與 `--symbol=X`)。"""
    out, i, found = [], 0, False
    while i < len(argv):
        x = argv[i]
        if x == "--symbol" and i + 1 < len(argv):
            out += ["--symbol", symbol]
            i, found = i + 2, True
            continue
        if x.startswith("--symbol="):
            out.append(f"--symbol={symbol}")
            found = True
        else:
            out.append(x)
        i += 1
    if not found:
        raise ValueError("參數列中找不到 --symbol")
    return out


def drop_option(argv: Sequence[str], name: str) -> List[str]:
    """移除選項與它的值(支援 `--x`、`--x VALUE`、`--x=VALUE`;值不是以 - 開頭才算)。"""
    out, i = [], 0
    while i < len(argv):
        x = argv[i]
        if x == name:
            i += 2 if i + 1 < len(argv) and not argv[i + 1].startswith("-") else 1
            continue
        if not x.startswith(name + "="):
            out.append(x)
        i += 1
    return out


def short_name(symbol: str) -> str:
    return symbol[:-4] if symbol.endswith("USDT") and len(symbol) > 4 else symbol


def run_multi(symbols: Sequence[str], command_for: Callable[[str], List[str]], out=None) -> int:
    """每個交易對執行 command_for(symbol) 這個子程式,輸出逐行加上 [幣別] 前綴;等全部結束,回傳最大的結束代碼。"""
    out = out or sys.stdout
    env = dict(os.environ, PYTHONUNBUFFERED="1")  # 子程式輸出不緩衝,才能即時顯示
    lock = threading.Lock()
    procs, threads = [], []

    def pump(sym: str, p: subprocess.Popen) -> None:
        tag = f"[{short_name(sym)}] "
        for line in p.stdout:  # 一定要讀完:不讀的話子程式的輸出管道塞滿就會卡住
            with lock:
                try:
                    out.write(tag + line.rstrip("\n") + "\n")
                    out.flush()
                except (OSError, ValueError):  # 終端機已關閉:丟掉輸出,繼續讀
                    pass

    def forward(signum, frame) -> None:  # 視窗關閉(SIGHUP)或 kill(SIGTERM):轉給子程式,讓它們存檔後結束
        nonlocal out
        if signum == getattr(signal, "SIGHUP", None):
            out = sys.stdout = sys.stderr = open(os.devnull, "w")  # 終端機已關閉(結束時 flush 才不會失敗)
        for p in procs:
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)

    old = {}
    for sig in (signal.SIGTERM, getattr(signal, "SIGHUP", None)):
        if sig is not None:
            try:
                old[sig] = signal.signal(sig, forward)
            except ValueError:  # 非主執行緒(測試)
                pass
    try:
        for sym in symbols:
            try:
                p = subprocess.Popen(command_for(sym), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     env=env, bufsize=1)
            except OSError:
                forward(None, None)  # 已啟動的子程式不能變成沒人管:請它們存檔結束
                for q in procs:
                    q.wait()
                raise
            procs.append(p)
            t = threading.Thread(target=pump, args=(sym, p), daemon=True)
            t.start()
            threads.append(t)
        codes = {}
        while len(codes) < len(procs):  # 哪個先結束就先顯示(不要等前面的交易對)
            try:
                for sym, p, t in zip(symbols, procs, threads):
                    if sym in codes or p.poll() is None:
                        continue
                    t.join(timeout=5)  # 先把它剩下的輸出(例如錯誤訊息)印完
                    p.stdout.close()
                    codes[sym] = p.returncode
                    left = [s for s in symbols if s not in codes]
                    note = "" if p.returncode == 0 or not left else f";⚠ 其他交易對繼續執行:{'、'.join(left)}"
                    with lock:
                        try:
                            out.write(f"[{short_name(sym)}] 已結束(代碼 {p.returncode}){note}\n")
                            out.flush()
                        except (OSError, ValueError):
                            pass
                time.sleep(0.2)
            except KeyboardInterrupt:  # Ctrl+C:子程式也收到了,等它們存檔結束
                continue
        return max(codes.values()) if codes else 0
    finally:
        for sig, h in old.items():
            signal.signal(sig, h)
