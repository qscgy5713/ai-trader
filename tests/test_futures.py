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
        self.assertEqual(a.decide(SeqModel(D("FLAT")), "X", "1h", cs, 100, TS)["code"], "max_trades")
        self.assertEqual(a.decide(SeqModel(D("FLAT")), "X", "1h", cs, 100, TS + 86400)["code"], "execute")

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
        b = self.agent(stop_loss_pct=3, daily_loss_limit_pct=50)
        b.decide(SeqModel(D("SHORT", 2)), "X", "1h", cs, 100, TS)
        r = b.check(110, TS + 1)  # 輪詢 / 跳空:以當下價格出場
        self.assertEqual(r["code"], "stop_loss")
        self.assertEqual(r["trade"]["price"], 110)

    def test_daily_limit_before_stop_and_halts(self):
        a, cs = self.agent(stop_loss_pct=10, daily_loss_limit_pct=2), candles()
        a.decide(SeqModel(D("LONG", 2)), "X", "1h", cs, 100, TS)
        r = a.check(98, TS + 1, prev=100)
        self.assertEqual(r["code"], "daily_loss")
        m = SeqModel(D("LONG", 1))
        self.assertEqual(a.decide(m, "X", "1h", cs, 98, TS + 2)["code"], "halted")
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
        a, cs = self.agent(max_leverage=10, stop_loss_pct=99, daily_loss_limit_pct=99), candles()
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
                                                                  daily_loss_limit_pct=99))
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
