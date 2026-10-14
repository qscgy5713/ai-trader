"""顯示與輸入幣別:帳戶實際以 USDT 計價;選 TWD 時,輸入的金額換算成 USDT,顯示時換回台幣。"""
from __future__ import annotations

from typing import Optional

from .market import fetch_usdt_twd


class Money:
    def __init__(self, currency: str = "USDT", rate: float = 1.0):
        if currency not in ("USDT", "TWD"):
            raise ValueError("幣別只支援 USDT 或 TWD")
        if not rate > 0:
            raise ValueError("匯率必須 > 0")
        self.currency, self.rate = currency, rate if currency == "TWD" else 1.0

    def to_usdt(self, amount: float) -> float:
        return amount / self.rate

    def from_usdt(self, usdt: float) -> float:
        return usdt * self.rate

    def fmt(self, usdt: float, sign: bool = False) -> str:
        v = self.from_usdt(usdt)
        if self.currency == "TWD":
            return f"{v:+,.0f} 元" if sign else f"{v:,.0f} 元"
        return f"{v:+,.2f} USDT" if sign else f"{v:,.2f} USDT"

    def describe(self) -> str:
        return f"金額以台幣顯示(1 USDT ≈ {self.rate:.3f} 元)" if self.currency == "TWD" else "金額以 USDT 顯示"


def make_money(currency: str, twd_rate: Optional[float]) -> Money:
    """TWD 且沒指定 --twd-rate 時,從 MAX 取即時匯率。"""
    if currency != "TWD":
        return Money()
    if twd_rate is not None:
        return Money("TWD", twd_rate)
    try:
        return Money("TWD", fetch_usdt_twd())
    except (OSError, ValueError, KeyError) as e:
        raise ValueError(f"無法取得 USDT/TWD 匯率({e});請用 --twd-rate 手動指定,例如 --twd-rate 32") from None
