import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock

from aitrader.futures import FuturesAccount
from aitrader.futures_agent import FuturesAgent, FuturesRiskConfig, futures_replay
from aitrader.llm import FuturesDecision, LLMError, RuleFuturesModel, parse_futures_decision
from aitrader.market import Candle
from tests.test_aitrader import FakeTime, candles

META = {"model": "f", "latency_s": 0, "tokens_in": 1, "tokens_out": 1}
TS = 1_700_000_000


class SeqModel:
    name = "seq"

    def __init__(self, *decisions):
        self.decisions, self.calls = list(decisions), []

    def decide(self, system, prompt, feats=None):
        self.calls.append(prompt)
        d = self.decisions.pop(0) if self.decisions else FuturesDecision("HOLD", 0, 0.9, "hold")
        if isinstance(d, Exception):
            raise d
        return d, META


def D(action, lev=1.0, conf=0.9):
    return FuturesDecision(action, lev, conf, action)


def acct(**kw):
    return FuturesAccount(10000, **{"fee_rate": 0, "slippage_bps": 0, **kw})


class AccountTest(unittest.TestCase):
    def test_long_short_pnl_and_flip(self):
        a = acct()
        a.rebalance(200, 100, TS)  # 多 2 倍
        self.assertAlmostEqual(a.qty, 200)
        self.assertAlmostEqual(a.equity(110), 12000)
        a.rebalance(-100, 110, TS)  # 反手做空 1 倍:先實現多單獲利
        self.assertAlmostEqual(a.wallet, 12000)
        self.assertAlmostEqual(a.qty, -12000 / 110)
        self.assertAlmostEqual(a.entry, 110)
        self.assertAlmostEqual(a.equity(99), 12000 + 12000 / 110 * 11)  # 空單下跌獲利
        a.rebalance(0, 99, TS)
        self.assertEqual(a.qty, 0)
        self.assertAlmostEqual(a.wallet, a.equity(1))

    def test_add_and_reduce_keep_average_entry(self):
        a = acct()
        a.rebalance(100, 100, TS)
        a.rebalance(200, 120, TS)  # 加倉
        self.assertGreater(a.entry, 100)
        self.assertLess(a.entry, 120)
        e = a.entry
        a.rebalance(50, 120, TS)  # 減倉:均價不變、實現部分損益
        self.assertAlmostEqual(a.entry, e)
        self.assertGreater(a.trades[-1].realized_pnl, 0)

    def test_fees_slippage_and_funding(self):
        a = FuturesAccount(10000, fee_rate=0.0005, slippage_bps=2)
        t = a.rebalance(100, 100, TS)
        self.assertAlmostEqual(t.price, 100.02)
        self.assertAlmostEqual(t.fee, t.qty * t.price * 0.0005)
        w = a.wallet
        paid = a.apply_funding(0.0001, 100)  # 費率為正:多方支付
        self.assertGreater(paid, 0)
        self.assertAlmostEqual(a.wallet, w - paid)
        a.rebalance(-100, 100, TS)
        self.assertLess(a.apply_funding(0.0001, 100), 0)  # 空方收到

    def test_liquidation_and_stop_prices(self):
        a = acct(mmr=0.005)
        a.rebalance(300, 100, TS)  # 多 3 倍
        liq = a.liq_price()
        self.assertTrue(a.is_liquidatable(liq * 0.999))
        self.assertFalse(a.is_liquidatable(liq * 1.01))
        self.assertAlmostEqual(a.equity(liq), 0.005 * abs(a.qty) * liq)
        stop = a.stop_price(3)
        self.assertAlmostEqual(a.position_loss_pct(stop), 3)
        s = acct()
        s.rebalance(-200, 100, TS)
        self.assertGreater(s.liq_price(), 100)
        self.assertGreater(s.stop_price(3), 100)
        low = acct()
        low.rebalance(50, 100, TS)
        self.assertIsNone(low.liq_price())  # 0.5 倍多單不會被強平
        t = a.liquidate(50, TS)  # 遠低於強平價:錢包不會為負
        self.assertEqual(a.wallet, 0)
        self.assertTrue(a.liquidated)
        self.assertEqual(t.reason, "強制平倉")

    def test_state_roundtrip(self):
        a = FuturesAccount(10000)
        a.rebalance(-150, 100, TS)
        a.apply_funding(0.0002, 100)
        b = FuturesAccount.from_dict(json.loads(json.dumps(a.to_dict())))
        self.assertAlmostEqual(b.equity(95), a.equity(95))
        self.assertAlmostEqual(b.funding_paid, a.funding_paid)
        d = a.to_dict()
        d["wallet"] = -1
        with self.assertRaises(ValueError):
            FuturesAccount.from_dict(d)


class ParseTest(unittest.TestCase):
    def test_parse(self):
        d = parse_futures_decision('{"action":"SHORT","leverage":1.5,"confidence":0.7,"reason":"跌破"}')
        self.assertEqual((d.action, d.leverage), ("SHORT", 1.5))
        self.assertEqual(parse_futures_decision('{"action":"FLAT","leverage":99,"confidence":0.5,"reason":""}').leverage, 0)
        for bad in ['{"action":"BUY","leverage":1,"confidence":0.5,"reason":""}',
                    '{"action":"LONG","leverage":0,"confidence":0.5,"reason":""}',
                    '{"action":"LONG","leverage":20,"confidence":0.5,"reason":""}',
                    '{"action":"LONG","leverage":1,"confidence":1.5,"reason":""}', "x"]:
            with self.assertRaises(LLMError, msg=bad):
                parse_futures_decision(bad)


class AgentTest(unittest.TestCase):
    def agent(self, **kw):
        return FuturesAgent(acct(), FuturesRiskConfig(**kw))

    def test_filters_and_leverage_cap(self):
        a, cs = self.agent(max_leverage=2, max_trades_per_day=2), candles()
        self.assertEqual(a.decide(SeqModel(D("FLAT")), "X", "1h", cs, 100, TS)["code"], "noop")
        self.assertEqual(a.decide(SeqModel(D("LONG", 1, 0.3)), "X", "1h", cs, 100, TS)["code"], "low_confidence")
        r = a.decide(SeqModel(D("LONG", 5)), "X", "1h", cs, 100, TS)
        self.assertEqual(r["code"], "execute")
        self.assertIn("槓桿上限", r["action_taken"])
        self.assertAlmostEqual(a.acct.exposure_pct(100), 200)
        self.assertEqual(a.decide(SeqModel(D("LONG", 1.9)), "X", "1h", cs, 100, TS)["code"], "noop")  # 差距太小
        self.assertEqual(a.decide(SeqModel(), "X", "1h", cs, 100, TS)["code"], "hold")
        self.assertEqual(a.decide(SeqModel(LLMError("x")), "X", "1h", cs, 100, TS)["code"], "no_decision")
        self.assertEqual(a.decide(SeqModel(D("SHORT", 1)), "X", "1h", cs, 100, TS)["code"], "execute")
        self.assertEqual(a.decide(SeqModel(D("FLAT")), "X", "1h", cs, 100, TS)["code"], "execute")  # 平倉不受次數限制
        self.assertEqual(a.decide(SeqModel(D("LONG", 1)), "X", "1h", cs, 100, TS)["code"], "max_trades")  # 開倉仍受限
        self.assertEqual(a.decide(SeqModel(D("LONG", 1)), "X", "1h", cs, 100, TS + 86400)["code"], "execute")  # 隔天重置

    def test_prompt_has_position_and_funding(self):
        a, cs = self.agent(), candles()
        m = SeqModel(D("SHORT", 1))
        a.decide(m, "X", "1h", cs, 100, TS, funding_rate=0.0001)
        a.decide(m, "X", "1h", cs, 100, TS + 3600, funding_rate=0.0001)
        self.assertIn('"side":"FLAT"', m.calls[0])
        self.assertIn('"side":"SHORT"', m.calls[1])
        self.assertIn("liquidation_price", m.calls[1])
        self.assertIn("funding_rate_per_8h", m.calls[1])

    def test_stop_loss_continuous_vs_gap(self):
        a, cs = self.agent(stop_loss_pct=3, daily_loss_limit_pct=50), candles()
        a.decide(SeqModel(D("LONG", 2)), "X", "1h", cs, 100, TS)
        stop = a.acct.stop_price(3)
        self.assertIsNone(a.check(stop + 0.01, TS + 1))
        r = a.check(stop - 1, TS + 2, prev=stop + 0.5)  # 連續走勢穿越:以停損價出場
        self.assertEqual(r["code"], "stop_loss")
        self.assertAlmostEqual(r["trade"]["price"], stop)
        self.assertIn("停損", r["trade"]["reason"])
        b = self.agent(stop_loss_pct=3, daily_loss_limit_pct=50)
        b.decide(SeqModel(D("SHORT", 2)), "X", "1h", cs, 100, TS)
        r = b.check(110, TS + 1)  # 輪詢 / 跳空:以當下價格出場
        self.assertEqual(r["code"], "stop_loss")
        self.assertEqual(r["trade"]["price"], 110)

    def test_stop_sized_by_daily_budget(self):
        a, cs = self.agent(stop_loss_pct=10, daily_loss_limit_pct=2, risk_per_trade_pct=50), candles()
        stop_pct, cap = a.sizing(100)
        self.assertAlmostEqual(stop_pct, 1.0)  # min(錢包 10%, 剩餘額度 2% × 50%)
        self.assertAlmostEqual(cap, 2.0)  # 1% / 最小停損距離 0.5%
        r = a.decide(SeqModel(D("LONG", 5)), "X", "1h", cs, 100, TS)
        self.assertIn("槓桿上限", r["action_taken"])
        r = a.check(a.acct.stop_price(a.pos_stop_pct) - 0.01, TS + 1, prev=100)
        self.assertEqual(r["code"], "stop_loss")
        self.assertGreater(a.acct.equity(99), a.day_floor())  # 停損在當日額度內,不會停機
        self.assertNotEqual(a.halted_day, a.day)

    def test_daily_amount_shrinks_position_then_exhausts(self):
        a, cs = self.agent(stop_loss_pct=3, daily_loss_limit_pct=50, daily_loss_amount=100), candles()
        a.check(100, TS)  # 進入當天
        self.assertAlmostEqual(a.day_floor(), 9900)  # 金額比 % 嚴格
        self.assertAlmostEqual(a.sizing(100)[0], 0.5)  # 100 × 50% / 10000
        a.acct.wallet -= 99.5  # 今天已虧掉幾乎全部額度
        r = a.decide(SeqModel(D("LONG", 1)), "X", "1h", cs, 100, TS)
        self.assertEqual(r["code"], "budget_exhausted")
        self.assertEqual(a.acct.qty, 0)
        self.assertEqual(a.decide(SeqModel(D("FLAT")), "X", "1h", cs, 100, TS)["code"], "noop")  # 平倉不受影響

    def test_daily_limit_trigger_when_costs_eat_budget(self):
        a, cs = self.agent(stop_loss_pct=3, daily_loss_limit_pct=2), candles()
        a.decide(SeqModel(D("LONG", 1)), "X", "1h", cs, 100, TS)
        a.acct.apply_funding(0.019, 100)  # 資金費吃掉大部分額度
        r = a.check(99.5, TS + 1, prev=100)
        self.assertEqual(r["code"], "daily_loss")
        m = SeqModel(D("LONG", 1))
        self.assertEqual(a.decide(m, "X", "1h", cs, 99.5, TS + 2)["code"], "halted")
        self.assertEqual(m.calls, [])  # 停機中不問模型

    def test_daily_limit_checked_when_flat(self):
        a, cs = self.agent(stop_loss_pct=3, daily_loss_limit_pct=5), candles()
        for i in range(2):  # 同一天連續停損兩次,權益累計跌超過 5%
            a.decide(SeqModel(D("LONG", 2)), "X", "1h", cs, 100, TS + i * 10)
            a.check(a.acct.stop_price(3) - 0.01, TS + i * 10 + 1)
        self.assertEqual(a.acct.qty, 0)
        self.assertLess(a.acct.equity(100), a.day_start_equity * 0.95)
        m = SeqModel(D("LONG", 2))
        r = a.decide(m, "X", "1h", cs, 100, TS + 100)
        self.assertEqual(r["code"], "halted")
        self.assertEqual(m.calls, [])

    def test_liquidation_when_stop_is_behind(self):
        a, cs = self.agent(max_leverage=10, stop_loss_pct=99, daily_loss_limit_pct=99, risk_per_trade_pct=100), candles()
        a.decide(SeqModel(D("LONG", 10)), "X", "1h", cs, 100, TS)
        liq = a.acct.liq_price()
        r = a.check(liq - 5, TS + 1)
        self.assertEqual(r["code"], "liquidation")
        self.assertAlmostEqual(r["trade"]["price"], liq)  # 交易所以強平價成交
        self.assertLess(a.acct.wallet, 10000 * 0.1)

    def test_gap_beyond_liquidation_is_liquidated_not_stopped(self):
        a, cs = self.agent(max_leverage=3, stop_loss_pct=3, daily_loss_limit_pct=50), candles()
        a.decide(SeqModel(D("LONG", 3)), "X", "1h", cs, 100, TS)
        r = a.check(a.acct.liq_price() * 0.8, TS + 1)  # 跳空越過停損與強平價
        self.assertEqual(r["code"], "liquidation")
        self.assertGreaterEqual(a.acct.wallet, 0)
        b = self.agent(max_leverage=3, stop_loss_pct=3, daily_loss_limit_pct=50)
        b.decide(SeqModel(D("LONG", 3)), "X", "1h", cs, 100, TS)
        stop, liq = b.acct.stop_price(3), b.acct.liq_price()
        r = b.check(liq * 0.8, TS + 1, prev=stop + 0.1)  # 連續走勢:先碰到停損價
        self.assertEqual(r["code"], "stop_loss")
        self.assertAlmostEqual(r["trade"]["price"], stop)

    def test_state_roundtrip(self):
        a, cs = self.agent(), candles()
        a.decide(SeqModel(D("LONG", 1)), "X", "1h", cs, 100, TS)
        b = FuturesAgent.from_dict(json.loads(json.dumps(a.to_dict())), FuturesRiskConfig())
        self.assertAlmostEqual(b.acct.equity(101), a.acct.equity(101))
        self.assertEqual(b.trades_today, 1)


class ReplayTest(unittest.TestCase):
    def test_replay_with_funding_and_rule_model(self):
        cs = candles(300, drift=0.004)
        funding = [(cs[0].ts + 8 * 3600 * k, 0.0001) for k in range(1, 37)]
        a = FuturesAgent(FuturesAccount(10000), FuturesRiskConfig())
        r = futures_replay(RuleFuturesModel(), "X", "1h", cs, a, funding, every=4)
        self.assertEqual(r.bars, 300 - 60)
        self.assertGreater(r.trades, 0)
        self.assertNotEqual(r.funding_paid, 0)
        self.assertAlmostEqual(r.equity, a.acct.equity(cs[-1].close))

    def test_replay_stops_when_bankrupt(self):
        cs = candles(100, drift=0.0)
        crash = [Candle(c.ts, c.open, c.high, c.low, c.close, c.volume) for c in cs[:62]]
        crash += [Candle(cs[0].ts + i * 3600, 100, 100, 50, 50, 1) for i in range(62, 100)]
        a = FuturesAgent(FuturesAccount(10000), FuturesRiskConfig(max_leverage=10, stop_loss_pct=99,
                                                                  daily_loss_limit_pct=99, risk_per_trade_pct=100))
        r = futures_replay(SeqModel(D("LONG", 10)), "X", "1h", crash, a, [], every=1)
        self.assertEqual(r.liquidations, 1)
        self.assertTrue(r.stopped_early)


class CliTest(unittest.TestCase):
    def test_paper_funding_resume_report_and_help(self):
        from aitrader.cli import main
        ft = FakeTime()
        settle = [ft.now + 40]  # 第一次結算在啟動 40 秒後,之後每 8 小時

        def prem(symbol):
            if ft.now >= settle[0]:
                settle[0] += 8 * 3600
            return 100.0, 0.0001, settle[0]
        with tempfile.TemporaryDirectory() as t, \
                mock.patch("aitrader.futures_cli.fetch_klines", return_value=candles()), \
                mock.patch("aitrader.futures_cli.fetch_premium", side_effect=prem), \
                mock.patch("aitrader.futures_cli.time", ft), \
                mock.patch("aitrader.futures_cli._make_model", return_value=SeqModel(D("SHORT", 1))), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            state, journal = os.path.join(t, "s.json"), os.path.join(t, "j.jsonl")
            args = ["futures", "--state", state, "--journal", journal, "--max-steps", "2"]
            self.assertEqual(main(args), 0)
            self.assertIn("資金費結算", out.getvalue())
            self.assertIn("做空", out.getvalue())
            self.assertEqual(main(args[:-1] + ["1"]), 0)
            self.assertIn("接續", out.getvalue())
            self.assertEqual(main(["report", journal]), 0)
            self.assertIn("合約日誌", out.getvalue())
            self.assertIn("資金費淨支付", out.getvalue())
        for cmd in ("futures", "futures-replay"):
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as cm:
                main([cmd, "-h"])
            self.assertEqual(cm.exception.code, 0)

    def test_replay_cli(self):
        from aitrader.cli import main
        with tempfile.TemporaryDirectory() as t, \
                mock.patch("aitrader.futures_cli.fetch_klines", return_value=candles(200, drift=0.004)), \
                mock.patch("aitrader.futures_cli.fetch_funding_history", return_value=[]), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(["futures-replay", "--model", "rule", "--limit", "200",
                                   "--journal", os.path.join(t, "r.jsonl")]), 0)
        self.assertIn("期末權益", out.getvalue())


if __name__ == "__main__":
    unittest.main()


class CurrencyTest(unittest.TestCase):
    def test_money(self):
        from aitrader.currency import Money, make_money
        m = Money("TWD", 32)
        self.assertAlmostEqual(m.to_usdt(32000), 1000)
        self.assertEqual(m.fmt(100), "3,200 元")
        self.assertEqual(m.fmt(-1.5, sign=True), "-48 元")
        self.assertEqual(Money().fmt(12.345), "12.35 USDT")
        self.assertEqual(make_money("TWD", 31.5).rate, 31.5)
        with mock.patch("aitrader.currency.fetch_usdt_twd", return_value=32.1):
            self.assertEqual(make_money("TWD", None).rate, 32.1)
        with mock.patch("aitrader.currency.fetch_usdt_twd", side_effect=OSError("down")):
            with self.assertRaisesRegex(ValueError, "--twd-rate"):
                make_money("TWD", None)
        with self.assertRaises(ValueError):
            Money("JPY")


class FuturesLearningTest(unittest.TestCase):
    class Learner(SeqModel):
        can_reflect = True

        def __init__(self, *d):
            super().__init__(*d)
            self.reflect_calls = []

        def reflect(self, system, prompt):
            from aitrader.llm import Reflection
            self.reflect_calls.append((system, prompt))
            return Reflection("今天方向判斷錯", ["低波動時別反手"]), META

    def test_memory_prompt_and_daily_reflection(self):
        a, cs = FuturesAgent(acct(), FuturesRiskConfig()), candles()
        m = self.Learner(D("LONG", 1))
        day0 = TS - TS % 86400 + 3600
        a.decide(m, "X", "1h", cs, 100, day0)
        a.check(a.acct.stop_price(a.pos_stop_pct) - 0.01, day0 + 60)  # 停損事件記進這筆決策
        a.decide(m, "X", "1h", cs, 99, day0 + 3600)
        self.assertNotIn("memory", m.calls[0])
        self.assertIn('"position_after":"LONG 1x"', m.calls[1])
        self.assertIn("stop_loss", m.calls[1])
        r = a.decide(m, "X", "1h", cs, 99, day0 + 86400)
        self.assertEqual(r["reflection"]["lessons"], ["低波動時別反手"])
        system, prompt = m.reflect_calls[0]
        self.assertIn("FUTURES", system)
        self.assertNotIn("grid_fills", prompt)
        self.assertIn("低波動時別反手", m.calls[-1])
        b = FuturesAgent(acct(), FuturesRiskConfig(), learn=False)
        m2 = self.Learner(D("LONG", 1))
        b.decide(m2, "X", "1h", cs, 100, day0)
        b.decide(m2, "X", "1h", cs, 100, day0 + 86400)
        self.assertEqual(m2.reflect_calls, [])
        self.assertNotIn("memory", m2.calls[-1])

    def test_old_state_without_new_fields(self):
        a = FuturesAgent(acct(), FuturesRiskConfig())
        a.decide(SeqModel(D("LONG", 1)), "X", "1h", candles(), 100, TS)
        d = json.loads(json.dumps(a.to_dict()))
        for k in ("pos_stop_pct", "day_start_costs", "memory"):
            del d[k]
        b = FuturesAgent.from_dict(d, FuturesRiskConfig(stop_loss_pct=4))
        self.assertEqual(b.pos_stop_pct, 4)
        self.assertEqual(b.memory.history, [])


class FuturesCliCurrencyTest(unittest.TestCase):
    def test_twd_paper_and_report(self):
        from aitrader.cli import main
        ft = FakeTime()
        with tempfile.TemporaryDirectory() as t, \
                mock.patch("aitrader.futures_cli.fetch_klines", return_value=candles()), \
                mock.patch("aitrader.futures_cli.fetch_premium", return_value=(100.0, 0.0001, ft.now + 10 ** 6)), \
                mock.patch("aitrader.futures_cli.time", ft), \
                mock.patch("aitrader.futures_cli._make_model", return_value=SeqModel(D("LONG", 1))), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            state, journal = os.path.join(t, "s.json"), os.path.join(t, "j.jsonl")
            self.assertEqual(main(["futures", "--currency", "TWD", "--twd-rate", "32", "--cash", "32000",
                                   "--daily-loss-amount", "640", "--state", state, "--journal", journal,
                                   "--max-steps", "1"]), 0)
            with open(state) as fh:
                agent = json.load(fh)["agent"]
            self.assertAlmostEqual(agent["acct"]["initial"], 1000)  # 32000 元 / 32
            self.assertIn("32,000 元", out.getvalue())
            self.assertIn("或虧 640 元", out.getvalue())
            self.assertIn("今日額度剩", out.getvalue())
            self.assertEqual(main(["report", journal, "--currency", "TWD", "--twd-rate", "32"]), 0)
            self.assertIn("最後權益 3", out.getvalue())


class StyleTest(unittest.TestCase):
    def test_style_changes_system_prompt_only(self):
        from aitrader.prompt import futures_system
        seen = {}
        for style in ("auto", "conservative", "balanced", "aggressive"):
            class M(SeqModel):
                def decide(self, system, prompt, feats=None):
                    seen[style] = system
                    return super().decide(system, prompt, feats)
            a = FuturesAgent(acct(), FuturesRiskConfig(), style=style)
            a.decide(M(D("LONG", 5)), "X", "1h", candles(), 100, TS)
            self.assertLessEqual(a.acct.exposure_pct(100), 200 + 1e-6)  # 風控不受風格影響
        self.assertIn("CONSERVATIVE", seen["conservative"])
        self.assertIn("slightly proactive", seen["balanced"])
        self.assertIn("AGGRESSIVE", seen["aggressive"])
        self.assertNotIn("{style}", seen["balanced"])
        self.assertIn("none imposed", seen["auto"])  # auto:不指定風格,也不叫它 cautious
        self.assertNotIn("cautious", seen["auto"])
        self.assertIn("a cautious trader", seen["balanced"])
        self.assertNotIn("{", seen["auto"] + seen["balanced"])  # 佔位符都有替換
        for biased in ("default when signals are mixed", "Use lower leverage", "do not overreact"):
            self.assertNotIn(biased, seen["auto"])  # auto 用中性說法
            self.assertIn(biased, seen["balanced"])
        self.assertEqual(futures_system(), seen["auto"])  # 預設 auto
        with self.assertRaises(ValueError):
            futures_system("yolo")

    def test_style_sets_default_min_confidence(self):
        import argparse
        from aitrader.currency import Money
        from aitrader.futures_cli import _add_common, _risk_cfg
        ap = argparse.ArgumentParser()
        _add_common(ap)
        self.assertEqual(_risk_cfg(ap.parse_args([]), Money()).min_confidence, 0.5)
        self.assertEqual(_risk_cfg(ap.parse_args(["--style", "conservative"]), Money()).min_confidence, 0.6)
        self.assertEqual(_risk_cfg(ap.parse_args(["--style", "aggressive", "--min-confidence", "0.7"]),
                                   Money()).min_confidence, 0.7)


class DisplayTest(unittest.TestCase):
    def test_counters_survive_trade_trim_and_old_state(self):
        a = acct()
        for i in range(600):
            a.rebalance(100 if i % 2 == 0 else 0, 100 + i % 3, TS + i)
        self.assertEqual(len(a.trades), 500)
        self.assertEqual(a.trade_count, 600)
        d = json.loads(json.dumps(a.to_dict()))
        self.assertEqual(FuturesAccount.from_dict(d).trade_count, 600)
        del d["trade_count"], d["realized_total"]
        self.assertEqual(FuturesAccount.from_dict(d).trade_count, 500)

    def test_paper_shows_summary_and_recent_trades_on_resume(self):
        from aitrader.cli import main
        ft = FakeTime()
        with tempfile.TemporaryDirectory() as t, \
                mock.patch("aitrader.futures_cli.fetch_klines", return_value=candles()), \
                mock.patch("aitrader.futures_cli.fetch_premium", return_value=(100.0, 0.0, ft.now + 10 ** 6)), \
                mock.patch("aitrader.futures_cli.time", ft), \
                mock.patch("aitrader.futures_cli._make_model", return_value=SeqModel(D("LONG", 1))), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            args = ["futures", "--state", os.path.join(t, "s.json"), "--journal", os.path.join(t, "j.jsonl"),
                    "--max-steps", "1"]
            self.assertEqual(main(args), 0)
            self.assertIn("累計:成交 1 筆", out.getvalue())
            self.assertEqual(main(args), 0)  # 接續:啟動時列出最近成交
            self.assertIn("最近 1 筆成交", out.getvalue())


class StatusTest(unittest.TestCase):
    def test_parse_duration(self):
        from aitrader.runtime import parse_duration
        self.assertEqual(parse_duration("5s"), 5)
        self.assertEqual(parse_duration("5m"), 300)
        self.assertEqual(parse_duration("1h"), 3600)
        self.assertEqual(parse_duration("90"), 90)
        for bad in ("abc", "5x", "-5s"):
            with self.assertRaises(ValueError):
                parse_duration(bad)

    def test_status_line_between_decisions(self):
        from aitrader.cli import main
        ft = FakeTime()
        ft.now = (ft.now // 3600) * 3600 + 60  # 整點過 1 分鐘:離下一根 1h K 線還很久
        with tempfile.TemporaryDirectory() as t, \
                mock.patch("aitrader.futures_cli.fetch_klines", return_value=candles()), \
                mock.patch("aitrader.futures_cli.fetch_premium", return_value=(100.0, 0.0, ft.now + 10 ** 6)), \
                mock.patch("aitrader.futures_cli.time", ft), \
                mock.patch("aitrader.futures_cli._make_model", return_value=SeqModel(D("LONG", 1))), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(["futures", "--status-every", "10m", "--state", os.path.join(t, "s.json"),
                                   "--journal", os.path.join(t, "j.jsonl"), "--max-steps", "2"]), 0)
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["futures", "--status-every", "2s", "--max-steps", "1"]), 2)
        lines = [l for l in out.getvalue().splitlines() if "下次問 AI" in l and "標記價" in l]
        self.assertGreaterEqual(len(lines), 4)  # 一小時內約每 10 分鐘一行
        self.assertLessEqual(len(lines), 7)
        self.assertIn("今日額度剩", lines[0])


class BudgetWarningTest(unittest.TestCase):
    def test_warns_when_amount_overridden_by_pct(self):
        from aitrader.currency import Money
        from aitrader.futures_cli import _budget_warning
        m = Money("TWD", 30)
        rc = FuturesRiskConfig(daily_loss_limit_pct=5, daily_loss_amount=10000 / 30)
        w = _budget_warning(rc, 1000, m)  # 本金 30,000 元
        self.assertIn("1,500 元", w)
        self.assertIn("--daily-loss-limit 34", w)
        self.assertEqual(_budget_warning(FuturesRiskConfig(daily_loss_limit_pct=40, daily_loss_amount=10000 / 30), 1000, m), "")
        self.assertEqual(_budget_warning(FuturesRiskConfig(), 1000, m), "")


class ArchiveJournalTest(unittest.TestCase):
    def test_archive_helper(self):
        from aitrader.state import archive_journal
        with tempfile.TemporaryDirectory() as t:
            j = os.path.join(t, "j.jsonl")
            self.assertIsNone(archive_journal(j))  # 不存在
            open(j, "w").close()
            self.assertIsNone(archive_journal(j))  # 空檔不封存
            for _ in range(2):
                with open(j, "w") as fh:
                    fh.write("{}\n")
                dest = archive_journal(j)
                self.assertTrue(os.path.exists(dest))
                self.assertFalse(os.path.exists(j))
            self.assertEqual(len(os.listdir(t)), 2)  # 同一秒兩次封存不互相覆蓋

    def test_reset_archives_but_resume_keeps_journal(self):
        from aitrader.cli import main
        ft = FakeTime()
        with tempfile.TemporaryDirectory() as t, \
                mock.patch("aitrader.futures_cli.fetch_klines", return_value=candles()), \
                mock.patch("aitrader.futures_cli.fetch_premium", return_value=(100.0, 0.0, ft.now + 10 ** 6)), \
                mock.patch("aitrader.futures_cli.time", ft), \
                mock.patch("aitrader.futures_cli._make_model", return_value=SeqModel(D("LONG", 1))), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            j = os.path.join(t, "j.jsonl")
            args = ["futures", "--state", os.path.join(t, "s.json"), "--journal", j, "--max-steps", "1"]
            main(args)
            main(args)  # 接續:日誌保留並繼續寫
            self.assertEqual(len([f for f in os.listdir(t) if f.startswith("j.")]), 1)
            n_before = sum(1 for _ in open(j))
            main(args + ["--reset"])
            files = [f for f in os.listdir(t) if f.startswith("j.")]
            self.assertEqual(len(files), 2)  # 新日誌 + 封存檔
            self.assertLess(sum(1 for _ in open(j)), n_before)
            self.assertIn("已封存", out.getvalue())


class KeepLessonsTest(unittest.TestCase):
    def _run(self, t, extra, model=None):
        from aitrader.cli import main
        ft = FakeTime()
        with mock.patch("aitrader.futures_cli.fetch_klines", return_value=candles()), \
                mock.patch("aitrader.futures_cli.fetch_premium", return_value=(100.0, 0.0, ft.now + 10 ** 6)), \
                mock.patch("aitrader.futures_cli.time", ft), \
                mock.patch("aitrader.futures_cli._make_model", return_value=model or SeqModel(D("LONG", 1))), \
                contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
            rc = main(["futures", "--state", os.path.join(t, "s.json"), "--journal", os.path.join(t, "j.jsonl"),
                       "--max-steps", "1"] + extra)
        return rc, out.getvalue()

    def _set_lessons(self, t, lessons, day="2026-01-02"):
        p = os.path.join(t, "s.json")
        d = json.load(open(p))
        d["agent"]["memory"]["lessons"], d["agent"]["memory"]["lessons_updated"] = lessons, day
        json.dump(d, open(p, "w"))

    def _lessons(self, t):
        return json.load(open(os.path.join(t, "s.json")))["agent"]["memory"]["lessons"]

    def test_reset_keeps_lessons_unless_reset_memory(self):
        with tempfile.TemporaryDirectory() as t:
            self._run(t, [])
            self._set_lessons(t, ["槓桿別太高"])
            rc, out = self._run(t, ["--reset"])
            self.assertEqual(rc, 0)
            self.assertIn("沿用先前的 1 條心得", out)
            self.assertEqual(self._lessons(t), ["槓桿別太高"])
            d = json.load(open(os.path.join(t, "s.json")))
            self.assertEqual(d["agent"]["memory"]["lessons_updated"], "2026-01-02")  # 檢討日期跟著沿用
            self.assertEqual(len(d["agent"]["memory"]["history"]), 0)  # 成績單重來(只有 pending)
            self.assertEqual(self._run(t, ["--reset-memory"])[0], 2)  # 需搭配 --reset
            self._run(t, ["--reset", "--reset-memory"])
            self.assertEqual(self._lessons(t), [])

    def test_lessons_command_show_and_import(self):
        from aitrader.cli import main
        with tempfile.TemporaryDirectory() as t:
            self._run(t, [])
            j = os.path.join(t, "old.jsonl")
            with open(j, "w") as fh:
                fh.write(json.dumps({"kind": "futures_reflection", "ts": TS, "day": "2026-10-14", "summary": "s",
                                     "lessons": ["舊的"], "error": None}) + "\n")
                fh.write(json.dumps({"kind": "futures_reflection", "ts": TS, "day": "2026-10-15", "summary": "s",
                                     "lessons": ["少反手", "槓桿 1.5 倍內"], "error": None}) + "\n")
                fh.write(json.dumps({"kind": "futures_reflection", "ts": TS, "day": "2026-10-16", "summary": None,
                                     "lessons": None, "error": "x"}) + "\n")
            s = os.path.join(t, "s.json")
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(main(["lessons", s, "--import", j]), 0)
            self.assertEqual(self._lessons(t), ["少反手", "槓桿 1.5 倍內"])  # 最後一次「成功」的檢討
            self.assertIn("2026-10-15", out.getvalue())
            self.assertIn("成績單", out.getvalue())
            from aitrader.memory import pending_lessons_path
            from aitrader.state import acquire_lock
            self._set_lessons(t, [])
            lk = acquire_lock(s)  # 模擬盤執行中:排入暫存檔,不直接改狀態檔
            try:
                with contextlib.redirect_stdout(io.StringIO()) as out2:
                    self.assertEqual(main(["lessons", s, "--import", j]), 0)
                self.assertIn("已排入 2 條心得", out2.getvalue())
                self.assertTrue(os.path.exists(pending_lessons_path(s)))
                self.assertEqual(self._lessons(t), [])
            finally:
                lk.close()
            rc, out3 = self._run(t, [])  # 執行中的程式(這裡用接續模擬)下一輪套用
            self.assertIn("已套用匯入的心得 2 條", out3)
            self.assertEqual(self._lessons(t), ["少反手", "槓桿 1.5 倍內"])
            self.assertFalse(os.path.exists(pending_lessons_path(s)))

    def test_apply_pending_keeps_review_and_rejects_bad_file(self):
        from aitrader.memory import LearningMemory, apply_pending_lessons, pending_lessons_path
        with tempfile.TemporaryDirectory() as t:
            s = os.path.join(t, "s.json")
            m = LearningMemory()
            m.review = {"day_utc": "2026-10-16"}
            self.assertIsNone(apply_pending_lessons(s, m))
            with open(pending_lessons_path(s), "w") as fh:
                json.dump({"lessons": ["a", "", "b"], "day": "2026-10-15"}, fh)
            self.assertEqual(apply_pending_lessons(s, m), 2)
            self.assertEqual(m.lessons, ["a", "b"])
            self.assertEqual(m.review, {"day_utc": "2026-10-16"})  # 待檢討不受影響
            with open(pending_lessons_path(s), "w") as fh:
                fh.write("{bad")
            with self.assertRaises(ValueError):
                apply_pending_lessons(s, m)
            self.assertFalse(os.path.exists(pending_lessons_path(s)))


class TrailingStopTest(unittest.TestCase):
    def agent(self, **kw):
        return FuturesAgent(acct(), FuturesRiskConfig(**{"trail_activate_pct": 2, "trail_distance_pct": 1, **kw}))

    def test_long_trailing_locks_profit(self):
        a, cs = self.agent(), candles()
        a.decide(SeqModel(D("LONG", 1)), "X", "1h", cs, 100, TS)
        self.assertIsNone(a.check(101.5, TS + 1))  # 獲利 1.5% < 2%:未啟動
        self.assertIsNone(a.trail_peak)
        self.assertIsNone(a.check(104, TS + 2))  # 獲利 4%:啟動,高點 4%
        self.assertAlmostEqual(a.trail_peak, 4, places=6)
        self.assertAlmostEqual(a.position_info(104)["trailing_stop_price"], 103, places=2)  # 回吐 1% → 3%
        r = a.check(102.5, TS + 3, prev=103.5)  # 連續走勢穿越 103:以追蹤停損價出場
        self.assertEqual(r["code"], "trailing_stop")
        self.assertAlmostEqual(r["trade"]["price"], 103, places=6)
        self.assertGreater(a.acct.wallet, 10000)
        self.assertIsNone(a.trail_peak)
        self.assertIn("追蹤停損", r["trade"]["reason"])

    def test_short_trailing_never_below_breakeven_and_off_switch(self):
        a, cs = self.agent(trail_activate_pct=2, trail_distance_pct=5), candles()
        a.decide(SeqModel(D("SHORT", 1)), "X", "1h", cs, 100, TS)
        a.check(97, TS + 1)  # 空單獲利 3%:啟動
        self.assertAlmostEqual(a.position_info(97)["trailing_stop_price"], 100, places=6)  # 3% − 5% < 0 → 保本
        r = a.check(100.5, TS + 2)
        self.assertEqual(r["code"], "trailing_stop")
        off = self.agent(trail_activate_pct=0)
        off.decide(SeqModel(D("LONG", 1)), "X", "1h", cs, 100, TS)
        off.check(110, TS + 1)
        self.assertIsNone(off.trail_peak)

    def test_new_trade_resets_trail_and_state_roundtrip(self):
        a, cs = self.agent(), candles()
        a.decide(SeqModel(D("LONG", 1)), "X", "1h", cs, 100, TS)
        a.check(104, TS + 1)
        b = FuturesAgent.from_dict(json.loads(json.dumps(a.to_dict())), a.cfg)
        self.assertAlmostEqual(b.trail_peak, a.trail_peak)
        a.decide(SeqModel(D("SHORT", 1)), "X", "1h", cs, 104, TS + 2)  # 反手
        self.assertIsNone(a.trail_peak)


class DowntimeTest(unittest.TestCase):
    def test_watch(self):
        from aitrader.runtime import DowntimeWatch, fmt_hours
        w = DowntimeWatch()
        self.assertIsNone(w.tick(1000))
        self.assertIsNone(w.tick(1030))
        self.assertEqual(w.tick(1030 + 7200), 7200)
        self.assertEqual(fmt_hours(7200), "2.0 小時")
        self.assertEqual(fmt_hours(900), "15 分鐘")
        self.assertEqual(DowntimeWatch(0.0 + 1).tick(1 + 3600), 3600)  # 接續:距上次存檔
        w2 = DowntimeWatch(100)
        self.assertIsNone(w2.tick(130))
        w2.mark(130 + 900)  # 這一輪處理了 15 分鐘(例如 AI 逾時重試):不算中斷
        self.assertIsNone(w2.tick(130 + 930))

    def test_sleep_detected_in_loop_and_report(self):
        from aitrader.cli import main
        class SleepyTime(FakeTime):
            """第 3 次等待時「睡著」3 小時(真實的睡眠發生在等待期間)。"""
            n = 0

            def sleep(self, sec):
                SleepyTime.n += 1
                self.now += sec + (3 * 3600 if SleepyTime.n == 3 else 0)

        ft = SleepyTime()
        ft.now = (ft.now // 3600) * 3600 + 60

        def prem(symbol):
            return 100.0, 0.0, ft.now + 10 ** 6

        with tempfile.TemporaryDirectory() as t, \
                mock.patch("aitrader.futures_cli.fetch_klines", return_value=candles()), \
                mock.patch("aitrader.futures_cli.fetch_premium", side_effect=prem), \
                mock.patch("aitrader.futures_cli.time", ft), \
                mock.patch("aitrader.futures_cli._make_model", return_value=SeqModel(D("LONG", 1))), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            j = os.path.join(t, "j.jsonl")
            args = ["futures", "--state", os.path.join(t, "s.json"), "--journal", j, "--max-steps", "2"]
            self.assertEqual(main(args), 0)
            self.assertIn("偵測到程式中斷 3.0 小時", out.getvalue())
            ft.now += 2 * 3600  # 程式停了 2 小時後再接續
            self.assertEqual(main(args[:-1] + ["1"]), 0)
            self.assertEqual(out.getvalue().count("偵測到程式中斷"), 2)
            self.assertEqual(main(["report", j]), 0)
        self.assertIn("程式中斷 2 次", out.getvalue())


class NewsModel(SeqModel):
    """可搜尋新聞、可檢討的假模型。"""
    can_reflect = can_search_news = True

    def __init__(self, *decisions, news_errors=()):
        super().__init__(*decisions)
        self.news_calls, self.reflect_prompts, self.news_errors = [], [], list(news_errors)

    def search_news(self, system, prompt):
        from aitrader.llm import NewsBrief, NewsEvent
        self.news_calls.append(prompt)
        if self.news_errors:
            raise self.news_errors.pop(0)
        return NewsBrief("ETF 資金流入", "BULLISH", [NewsEvent("2023-11-14", "Reuters", "某事件")]), \
            dict(META, sources=[{"title": "reuters.com", "uri": "https://x"}], search_queries=["sol news"])

    def reflect(self, system, prompt):
        from aitrader.llm import Reflection
        self.reflect_prompts.append(prompt)
        return Reflection("s", ["l"]), META


class NewsTest(unittest.TestCase):
    def test_reputable_source(self):
        from aitrader.prompt import is_reputable_source
        for ok in ("coindesk.com", "www.Reuters.com", "theblock.co"):
            self.assertTrue(is_reputable_source(ok))
        for bad in ("pluang.com", "notreuters.com", "coindesk.com.evil.io", ""):
            self.assertFalse(is_reputable_source(bad))

    def test_parse_news(self):
        from aitrader.llm import parse_news
        n = parse_news(json.dumps({"summary": " 無重大消息 ", "sentiment": "neutral", "events": [
            {"date": "2026-10-19", "source": "CoinDesk", "event": "e"}, {"date": "", "source": "x", "event": "缺日期"}]}))
        self.assertEqual((n.summary, n.sentiment, len(n.events)), ("無重大消息", "NEUTRAL", 1))  # 缺欄位的事件丟掉
        fenced = parse_news('```json\n{"summary":"s","sentiment":"BULLISH","events":[]}\n```')
        self.assertEqual(fenced.sentiment, "BULLISH")  # 容許 markdown 區塊
        for bad in ('{"summary":"s","events":[]}', '{"summary":"s","sentiment":"MOON","events":[]}', "not json"):
            with self.assertRaises(LLMError):
                parse_news(bad)

    def test_gemini_search_news_request(self):
        from aitrader.llm import FUTURES_SCHEMA, GeminiModel
        from tests.test_aitrader import _Resp, _ok
        resp = _ok('{"summary":"s","sentiment":"BEARISH","events":[]}')
        resp["candidates"][0]["groundingMetadata"] = {
            "webSearchQueries": ["sol"], "groundingChunks": [{"web": {"title": "coindesk.com", "uri": "https://c"}}, {}]}
        m = GeminiModel("K", "flash-x", reflect_model="pro-x", schema=FUTURES_SCHEMA, parser=parse_futures_decision)
        with mock.patch("aitrader.llm.urllib.request.urlopen", return_value=_Resp(resp)) as u:
            n, meta = m.search_news("SYS", "P")
        req = u.call_args[0][0]
        body = json.loads(req.data)
        self.assertIn("models/pro-x:generateContent", req.full_url)  # 用檢討模型
        self.assertEqual(body["tools"], [{"google_search": {}}])
        self.assertNotIn("responseSchema", body["generationConfig"])  # 搭配結構化輸出常不附來源,改由文字取 JSON
        self.assertEqual(n.sentiment, "BEARISH")
        self.assertEqual(meta["sources"], [{"title": "coindesk.com", "uri": "https://c"}])
        self.assertEqual(meta["search_queries"], ["sol"])
        with mock.patch("aitrader.llm.urllib.request.urlopen", return_value=_Resp(_ok('{"action":"HOLD","leverage":0,"confidence":0.5,"reason":"r"}'))) as u:
            m.decide("S", "P")
        self.assertNotIn("tools", json.loads(u.call_args[0][0].data))  # 一般決策不搜尋

    def test_news_off_by_default_and_once_per_day(self):
        c = candles()
        m = NewsModel()
        a = FuturesAgent(acct(), FuturesRiskConfig())
        a.decide(m, "SOLUSDT", "1h", c, 100, TS)
        self.assertEqual(m.news_calls, [])  # 預設關閉(回放也是)
        self.assertNotIn("market_news", m.calls[-1])
        a.news_enabled = True
        with tempfile.TemporaryDirectory() as t:
            j = os.path.join(t, "j.jsonl")
            rec = a.decide(m, "SOLUSDT", "1h", c, 100, TS + 3600, journal=j)
            self.assertEqual(rec["news"]["sentiment"], "BULLISH")
            self.assertIn('"asset":"SOL"', m.news_calls[0])
            payload = json.loads(m.calls[-1])
            self.assertEqual(payload["market_news"]["summary"], "ETF 資金流入")
            self.assertIn("not as a trade signal", payload["market_news"]["note"])
            self.assertNotIn("sources", payload["market_news"])  # 來源網址給人查核,不塞給 AI
            self.assertEqual(payload["market_news"]["source_quality"], {"reputable_sites": 1, "total_sites": 1})
            a.decide(m, "SOLUSDT", "1h", c, 100, TS + 3700, journal=j)  # TS 是 UTC 22:13,同一天內
            self.assertEqual(len(m.news_calls), 1)  # 同一天只搜一次
            kinds = [json.loads(l)["kind"] for l in open(j)]
            self.assertEqual(kinds.count("futures_news"), 1)
        a.decide(m, "SOLUSDT", "1h", c, 100, TS + 86400)  # 隔天:先檢討(附上昨天的消息面)再搜新的
        self.assertEqual(len(m.news_calls), 2)
        self.assertIn("ETF 資金流入", m.reflect_prompts[-1])
        self.assertIsNone(a.news_view(TS + 86400 + 40 * 3600))  # 太舊就不附

    def test_news_failures_do_not_stop_trading(self):
        from aitrader.llm import LLMConfigError
        c = candles()
        m = NewsModel(D("LONG", 1), news_errors=[LLMError("timeout")] * 3)
        a = FuturesAgent(acct(), FuturesRiskConfig())
        a.news_enabled = True
        rec = a.decide(m, "SOLUSDT", "1h", c, 100, TS)
        self.assertIn("timeout", rec["news"]["error"])
        self.assertIsNotNone(rec["trade"])  # 新聞失敗照常交易
        for i in range(1, 4):
            a.decide(m, "SOLUSDT", "1h", c, 100, TS + i * 60)
        self.assertEqual(len(m.news_calls), 2)  # 當天最多 2 輪(逾時等錯誤不在輪內重試)
        m3 = NewsModel()
        m3.search_news = lambda system, prompt: (NewsModel().search_news(system, prompt)[0], dict(META, sources=[]))
        c3 = FuturesAgent(acct(), FuturesRiskConfig())
        c3.news_enabled = True
        self.assertIn("沒有附上引用來源", c3.decide(m3, "SOLUSDT", "1h", c, 100, TS)["news"]["error"])
        self.assertIsNone(c3.news)  # 無來源的結果不採用
        self.assertEqual(c3.decide(m3, "SOLUSDT", "1h", c, 100, TS + 60)["news"]["meta"]["tokens_out"], 3)  # 輪內重試 3 次,仍記用量
        self.assertIsNone(c3.decide(m3, "SOLUSDT", "1h", c, 100, TS + 120)["news"])  # 2 輪用完,當天不再搜
        # API 偶爾漏回來源:同一輪立即重試,第 2 次有來源就採用
        m4, seq = NewsModel(), [[], [{"title": "coindesk.com", "uri": "u"}]]
        m4.search_news = lambda system, prompt: (NewsModel().search_news(system, prompt)[0], dict(META, sources=seq.pop(0)))
        c4 = FuturesAgent(acct(), FuturesRiskConfig())
        c4.news_enabled = True
        r4 = c4.decide(m4, "SOLUSDT", "1h", c, 100, TS)["news"]
        self.assertIsNone(r4["error"])
        self.assertEqual((r4["meta"]["calls"], r4["sources"][0]["reputable"]), (2, True))
        m2 = NewsModel(news_errors=[LLMConfigError("400 不支援搜尋")])
        b = FuturesAgent(acct(), FuturesRiskConfig())
        b.news_enabled = True
        b.decide(m2, "SOLUSDT", "1h", c, 100, TS)
        b.decide(m2, "SOLUSDT", "1h", c, 100, TS + 60)
        self.assertEqual(len(m2.news_calls), 1)  # 設定錯誤:當天不再重試,也不中斷

    def test_news_persisted(self):
        a = FuturesAgent(acct(), FuturesRiskConfig())
        a.news_enabled = True
        a.decide(NewsModel(), "SOLUSDT", "1h", candles(), 100, TS)
        b = FuturesAgent.from_dict(json.loads(json.dumps(a.to_dict())), FuturesRiskConfig())
        self.assertEqual(b.news, a.news)
        self.assertFalse(b.news_enabled)  # 是否開啟以這次啟動參數為準
        d = a.to_dict()
        del d["news"], d["news_attempt_day"], d["news_attempts"]  # 舊版狀態檔
        self.assertIsNone(FuturesAgent.from_dict(d, FuturesRiskConfig()).news)

    def test_news_requires_gemini(self):
        from aitrader.cli import main
        with tempfile.TemporaryDirectory() as t, contextlib.redirect_stderr(io.StringIO()) as err, \
                contextlib.redirect_stdout(io.StringIO()):
            rc = main(["futures", "--model", "rule", "--news", "--state", os.path.join(t, "s.json")])
        self.assertEqual(rc, 2)
        self.assertIn("--news", err.getvalue())

    def test_news_shared_between_accounts_of_same_symbol(self):
        with tempfile.TemporaryDirectory() as t:
            cache = os.path.join(t, "news_SOLUSDT.json")
            m1, m2 = NewsModel(), NewsModel()
            a, b = (FuturesAgent(acct(), FuturesRiskConfig()) for _ in range(2))
            for x in (a, b):
                x.news_enabled, x.news_cache = True, cache
            a.decide(m1, "SOLUSDT", "1h", candles(), 100, TS)
            rb = b.decide(m2, "SOLUSDT", "1h", candles(), 100, TS + 30)["news"]
            self.assertEqual((len(m1.news_calls), len(m2.news_calls)), (1, 0))  # 第二個帳戶不再呼叫
            self.assertTrue(rb["shared"])
            self.assertIsNone(rb["meta"])  # 沒花錢
            self.assertEqual(b.news, a.news)
            self.assertIn("market_news", m2.calls[-1])
            # 隔天:共用檔是前一天的,不能用,要重新搜
            b.decide(m2, "SOLUSDT", "1h", candles(), 100, TS + 86400)
            self.assertEqual(len(m2.news_calls), 1)
            # 搜尋失敗不寫入共用檔,其他帳戶要自己搜
            c = FuturesAgent(acct(), FuturesRiskConfig())
            c.news_enabled, c.news_cache = True, os.path.join(t, "news_ETHUSDT.json")
            c.decide(NewsModel(news_errors=[LLMError("x")]), "ETHUSDT", "1h", candles(), 100, TS)
            self.assertFalse(os.path.exists(c.news_cache))

    def test_news_waits_for_other_account_searching(self):
        from aitrader.state import acquire_lock, save_json
        import threading
        with tempfile.TemporaryDirectory() as t:
            cache = os.path.join(t, "news_SOLUSDT.json")
            lock = acquire_lock(cache)  # 另一個帳戶正在搜尋
            a = FuturesAgent(acct(), FuturesRiskConfig())
            a.news_enabled, a.news_cache = True, cache
            a._new_day(TS, 100)
            news = {"day": a.day, "ts": TS, "summary": "別人搜的", "sentiment": "NEUTRAL", "events": [],
                    "sources": [{"title": "coindesk.com", "uri": "u", "reputable": True}]}

            def finish():  # 1 秒後對方搜完、寫入並釋放鎖
                save_json(cache, {"symbol": "SOLUSDT", "news": news})
                lock.close()
            threading.Timer(1.0, finish).start()
            m = NewsModel()
            r = a.refresh_news(m, "SOLUSDT", TS, None)
            self.assertEqual((r["summary"], len(m.news_calls)), ("別人搜的", 0))

    def test_report_shows_news_and_tokens(self):
        from aitrader.futures_cli import report_futures
        m = NewsModel()
        a = FuturesAgent(acct(), FuturesRiskConfig())
        a.news_enabled = True
        with tempfile.TemporaryDirectory() as t:
            j = os.path.join(t, "j.jsonl")
            a.decide(m, "SOLUSDT", "1h", candles(), 100, TS, journal=j)
            recs = [json.loads(l) for l in open(j)]
            with contextlib.redirect_stdout(io.StringIO()) as out:
                report_futures(j, recs)
        s = out.getvalue()
        self.assertIn("每日新聞 1 次", s)
        self.assertIn("消息面偏多", s)
        self.assertIn("2023-11-14 Reuters:某事件", s)
        self.assertIn("引用網站(主流來源 1/1):reuters.com ✓", s)
        self.assertIn(";每日新聞 輸入 1 / 輸出 1", s)


class ProcessReviewTest(unittest.TestCase):
    """檢討要能分辨「險勝」:記錄區間內最差權益與距停損最近的距離。"""

    def test_lucky_win_is_visible_in_outcome_and_review(self):
        from aitrader.memory import LearningMemory
        a = FuturesAgent(acct(), FuturesRiskConfig(max_leverage=2, stop_loss_pct=3))
        c = candles()
        a.decide(SeqModel(D("LONG", 1)), "X", "1h", c, 100, TS)  # 做多 1 倍,停損約在 97(錢包 3% 以內)
        stop = a.acct.stop_price(a.pos_stop_pct)
        for p in (99.0, stop + 0.3, 99.5, 103.0):  # 一度接近停損,最後漲上去
            self.assertIsNone(a.check(p, TS + 600))
        a.decide(SeqModel(D("HOLD")), "X", "1h", c, 103, TS + 3600)
        out = a.memory.history[-1]["outcome"]
        self.assertGreater(out["equity_change_pct"], 0)  # 結果是賺
        self.assertLess(out["worst_equity_pct"], -2)  # 但中途虧超過 2%
        self.assertAlmostEqual(out["closest_to_stop_pct"], 0.3 / (stop + 0.3) * 100, places=1)
        self.assertGreater(out["best_equity_pct"], out["equity_change_pct"] - 1e-9)
        a.decide(SeqModel(D("HOLD")), "X", "1h", c, 103, TS + 86400)  # 換日:前一天整理成待檢討
        rv = a.memory.review
        self.assertLess(rv["intraday_worst_equity_pct"], -2)
        self.assertLess(rv["closest_to_stop_pct"], 1)
        m = LearningMemory.from_dict(json.loads(json.dumps(a.memory.to_dict())), fill_label=None)
        self.assertEqual((m.day_eq_lo, m.day_stop_min), (a.memory.day_eq_lo, a.memory.day_stop_min))
        d = a.memory.to_dict()
        del d["day_eq_lo"], d["day_stop_min"]  # 舊版狀態檔
        self.assertIsNone(LearningMemory.from_dict(d, fill_label=None).day_eq_lo)

    def test_flat_has_no_stop_distance(self):
        a = FuturesAgent(acct(), FuturesRiskConfig())
        a.decide(SeqModel(D("FLAT")), "X", "1h", candles(), 100, TS)
        a.check(95, TS + 60)
        a.decide(SeqModel(D("HOLD")), "X", "1h", candles(), 100, TS + 3600)
        out = a.memory.history[-1]["outcome"]
        self.assertNotIn("closest_to_stop_pct", out)
        self.assertEqual(out["worst_equity_pct"], 0)

    def test_reflection_prompts_judge_process(self):
        from aitrader.prompt import FUTURES_REFLECTION_SYSTEM, REFLECTION_SYSTEM
        for s in (FUTURES_REFLECTION_SYSTEM, REFLECTION_SYSTEM):
            self.assertIn("Judge the PROCESS", s)
            self.assertIn("LUCKY", s)
            self.assertIn("do not take credit", s)


class UrgentCheckTest(unittest.TestCase):
    """急跌檢查:持倉往不利方向走超過門檻時,不等整點立刻多問一次 AI;可部分減倉,減倉不占交易次數。"""

    def test_alert_move_trigger_and_once_per_regular_decision(self):
        a = FuturesAgent(acct(), FuturesRiskConfig(alert_move_pct=1.5))
        self.assertIsNone(a.alert_move(90))  # 空手不觸發
        a.decide(SeqModel(D("LONG", 1)), "X", "1h", candles(), 100, TS)
        self.assertIsNone(a.alert_move(99))  # -1%:未達門檻
        self.assertIsNone(a.alert_move(103))  # 有利方向不觸發
        self.assertEqual(a.alert_move(98.4), 1.6)
        m = SeqModel(D("LONG", 0.5))
        r = a.decide(m, "X", "1h", candles(), 98.4, TS + 600, alert=1.6)
        payload = json.loads(m.calls[-1])
        self.assertIn("urgent_check", payload["constraints"])
        self.assertIn("reduce part", payload["constraints"]["urgent_check"]["note"])
        self.assertEqual(r["alert"], {"move_pct": 1.6, "from_price": 100})
        self.assertEqual(r["code"], "execute")
        self.assertAlmostEqual(a.acct.exposure_pct(98.4), 50, delta=1)  # 減倉一半
        self.assertIsNone(a.alert_move(90))  # 急跌檢查後要等下一次例行決策才會再觸發
        a.decide(SeqModel(D("HOLD")), "X", "1h", candles(), 98, TS + 3600)
        self.assertIsNotNone(a.alert_move(96))  # 例行決策後重新啟用,以新的決策價計算
        b = FuturesAgent.from_dict(json.loads(json.dumps(a.to_dict())), FuturesRiskConfig(alert_move_pct=1.5))
        self.assertEqual((b.last_decision_price, b.alert_armed), (98, True))
        self.assertIsNone(FuturesAgent(acct(), FuturesRiskConfig(alert_move_pct=0)).alert_move(1))  # 0 = 關閉
        short = FuturesAgent(acct(), FuturesRiskConfig(alert_move_pct=1.5))
        short.decide(SeqModel(D("SHORT", 1)), "X", "1h", candles(), 100, TS)
        self.assertEqual(short.alert_move(102), 2.0)  # 空單:上漲是不利方向

    def test_reduction_not_counted_toward_daily_trades(self):
        a = FuturesAgent(acct(), FuturesRiskConfig(max_trades_per_day=1, max_leverage=2))
        a.decide(SeqModel(D("LONG", 2)), "X", "1h", candles(), 100, TS)
        self.assertEqual(a.trades_today, 1)
        self.assertEqual(a.decide(SeqModel(D("LONG", 1)), "X", "1h", candles(), 100, TS + 60)["code"], "execute")  # 減倉
        self.assertEqual(a.trades_today, 1)  # 不占次數
        self.assertEqual(a.decide(SeqModel(D("LONG", 2)), "X", "1h", candles(), 100, TS + 120)["code"], "max_trades")  # 加倉受限
        self.assertEqual(a.decide(SeqModel(D("SHORT", 0.5)), "X", "1h", candles(), 100, TS + 180)["code"], "max_trades")  # 反手不算減倉

    def test_replay_runs_alert_checks(self):
        drop = [Candle(TS + i * 3600, 100, 100.5, 99.5, 100, 1) for i in range(70)]
        drop += [Candle(TS + 70 * 3600, 100, 100.2, 97, 97.5, 1), Candle(TS + 71 * 3600, 97.5, 98, 97, 97.5, 1)]
        m = SeqModel(D("LONG", 1), D("FLAT"))
        agent = FuturesAgent(acct(), FuturesRiskConfig(alert_move_pct=1.5, stop_loss_pct=10, max_leverage=2))
        r = futures_replay(m, "X", "1h", drop, agent, [], every=100)
        self.assertEqual(r.alerts, 1)  # 第 70 根下探 97 → 觸發一次並平倉
        self.assertLess(abs(agent.acct.qty), 1e-9)

    def test_urgent_check_failure_keeps_armed_and_budget_allows_reduction(self):
        a = FuturesAgent(acct(), FuturesRiskConfig(alert_move_pct=1.5, max_leverage=2))
        a.decide(SeqModel(D("LONG", 2)), "X", "1h", candles(), 100, TS)
        r = a.decide(SeqModel(LLMError("timeout")), "X", "1h", candles(), 98, TS + 60, alert=2.0)
        self.assertIsNone(r["decision"])
        self.assertEqual((a.alert_armed, a.last_decision_price), (True, 100))  # 沒拿到決策:保留,稍後可再試
        self.assertEqual(a.alert_move(98), 2.0)
        a.day_start_equity = a.acct.equity(98) * 1.0001  # 讓今日額度幾乎用完(可開槓桿 < 最低)
        a.cfg.daily_loss_amount = a.acct.initial * 0.0001
        self.assertLess(a.sizing(98)[1], 0.1)
        self.assertEqual(a.decide(SeqModel(D("LONG", 1)), "X", "1h", candles(), 98, TS + 120, alert=2.0)["code"], "execute")  # 減倉放行
        self.assertEqual(a.decide(SeqModel(D("LONG", 2)), "X", "1h", candles(), 98, TS + 180)["code"], "budget_exhausted")  # 加倉不行
