"""風控:AI 只給建議,真正能不能交易、交易多少由這裡決定。

每個判斷都回傳 (目標持倉 % 或 None, 原因文字, 原因代碼);代碼供報表統計:
  強制:daily_loss / stop_loss / take_profit / halted
  過濾:no_decision / hold / noop(已在目標)/ low_confidence / small_change / max_trades / execute
  其中 low_confidence / small_change / max_trades / halted 代表「模型想交易但被風控擋下」。

順序:強制規則(單日虧損上限、停損、停利)優先於模型;模型決策再經信心門檻、持倉上限、
最小調整幅度、每日交易次數限制過濾。模型出錯一律不交易。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple

from .llm import Decision
from .portfolio import Portfolio


@dataclass
class RiskConfig:
    max_position_pct: float = 50.0  # 持倉上限(占總權益 %)
    min_confidence: float = 0.6  # 低於此信心不交易
    min_trade_pct: float = 5.0  # 目標與目前差距小於此值不交易(避免手續費磨損)
    stop_loss_pct: float = 5.0  # 持倉虧損達此 % 強制清倉
    take_profit_pct: float = 0.0  # 持倉獲利達此 % 強制清倉;0 = 不啟用
    daily_loss_limit_pct: float = 3.0  # 權益較當日開始下跌達此 % 清倉並停到隔天
    max_trades_per_day: int = 6

    def validate(self) -> None:
        if not 0 < self.max_position_pct <= 100:
            raise ValueError("max_position_pct 需在 (0, 100]")
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence 需在 [0, 1]")
        if not 0 <= self.min_trade_pct < 100:
            raise ValueError("min_trade_pct 需在 [0, 100)")
        if not 0 < self.stop_loss_pct < 100:
            raise ValueError("stop_loss_pct 需在 (0, 100)")
        if self.take_profit_pct < 0:
            raise ValueError("take_profit_pct 不可為負")
        if not 0 < self.daily_loss_limit_pct < 100:
            raise ValueError("daily_loss_limit_pct 需在 (0, 100)")
        if self.max_trades_per_day < 1:
            raise ValueError("max_trades_per_day 至少 1")


@dataclass
class RiskState:
    day: str = ""
    day_start_equity: float = 0.0
    trades_today: int = 0
    halted_today: bool = False


class RiskManager:
    def __init__(self, cfg: RiskConfig, state: Optional[RiskState] = None):
        cfg.validate()
        self.cfg = cfg
        self.state = state or RiskState()

    def new_bar(self, day: str, equity: float) -> None:
        """每個決策點呼叫;換日時重置當日統計。"""
        if day != self.state.day:
            self.state = RiskState(day=day, day_start_equity=equity)

    def forced(self, pf: Portfolio, price: float) -> Optional[Tuple[float, str, str]]:
        """強制規則;觸發時回傳 (目標持倉 %, 原因, 代碼),不需要問模型。"""
        s, c = self.state, self.cfg
        if s.halted_today:
            return (0.0, "今日已觸發單日虧損上限,停止交易至明天", "halted") if pf.qty > 0 else None
        if s.day_start_equity > 0 and pf.equity(price) <= s.day_start_equity * (1 - c.daily_loss_limit_pct / 100):
            s.halted_today = True
            return 0.0, f"權益較今日開始下跌超過 {c.daily_loss_limit_pct:g}%,清倉並停止交易至明天", "daily_loss"
        if pf.qty > 0:
            u = pf.unrealized_pct(price)
            if u <= -c.stop_loss_pct:
                return 0.0, f"持倉虧損 {u:.2f}% 觸發停損 {c.stop_loss_pct:g}%", "stop_loss"
            if c.take_profit_pct and u >= c.take_profit_pct:
                return 0.0, f"持倉獲利 {u:.2f}% 觸發停利 {c.take_profit_pct:g}%", "take_profit"
        return None

    def filter(self, d: Optional[Decision], pf: Portfolio, price: float) -> Tuple[Optional[float], str, str]:
        """把模型決策轉成可執行的目標持倉 %;None 表示不交易。另回傳原因與代碼。"""
        c, s = self.cfg, self.state
        if s.halted_today:
            return None, "今日已停止交易", "halted"
        if d is None:
            return None, "模型無有效決策,不交易", "no_decision"
        if d.action == "HOLD":
            return None, "模型建議 HOLD", "hold"
        cur = pf.position_pct(price)
        target = min(d.target_position_pct, c.max_position_pct)
        if d.action == "BUY" and target <= cur:
            return None, f"已在目標:BUY 目標 {target:.1f}% 不高於目前 {cur:.1f}%", "noop"
        if d.action == "SELL" and target >= cur:
            return None, f"已在目標:SELL 目標 {target:.1f}% 不低於目前 {cur:.1f}%", "noop"
        if d.confidence < c.min_confidence:
            return None, f"信心 {d.confidence:.2f} 低於門檻 {c.min_confidence:g}", "low_confidence"
        if abs(target - cur) < c.min_trade_pct and target != 0:
            return None, f"調整幅度 {abs(target - cur):.1f}% 小於 {c.min_trade_pct:g}%", "small_change"
        if s.trades_today >= c.max_trades_per_day:
            return None, f"今日交易已達 {c.max_trades_per_day} 次上限", "max_trades"
        note = f"(上限 {c.max_position_pct:g}%)" if d.target_position_pct > c.max_position_pct else ""
        return target, f"執行模型決策 {d.action} → {target:.1f}%{note}", "execute"

    def record_trade(self) -> None:
        self.state.trades_today += 1

    def to_dict(self) -> dict:
        return {"cfg": asdict(self.cfg), "state": asdict(self.state)}

    @classmethod
    def from_dict(cls, d: dict) -> "RiskManager":
        try:
            return cls(RiskConfig(**d["cfg"]), RiskState(**d["state"]))
        except (KeyError, TypeError) as e:
            raise ValueError(f"風控狀態格式錯誤:{e!r}") from e
