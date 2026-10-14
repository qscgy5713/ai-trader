import contextlib
import io
import itertools
import json
import os
import tempfile
import unittest
from unittest import mock

from aitrader.grid import GridConfig, GridEngine, max_grids_for_spacing, spacing_pct
from aitrader.grid_agent import GridAgent, GridRiskConfig, grid_replay
from aitrader.llm import (GRID_SCHEMA, MAX_LESSONS, REFLECTION_SCHEMA, CachedModel, GeminiModel, GridDecision,
                          LLMError, Reflection, RuleGridModel, parse_grid_decision, parse_reflection)
from aitrader.memory import HISTORY_LEN, LearningMemory
from tests.test_aitrader import FakeTime, _ok, _Resp, candles

META = {"model": "f", "latency_s": 0, "tokens_in": 1, "tokens_out": 1}
TS = 1_700_000_000


class SeqModel:
    """依序回傳決策;用完後一律 KEEP。"""

    name = "seq"

    def __init__(self, *decisions):
        self.decisions, self.calls = list(decisions), []

    def decide(self, system, prompt, feats=None):
        self.calls.append(prompt)
        d = self.decisions.pop(0) if self.decisions else GridDecision("KEEP", 0, 0, 0, 0.9, "keep")
        if isinstance(d, Exception):
            raise d
        return d, META


def SET(lo=5, up=5, grids=10, conf=0.9):
    return GridDecision("SET", lo, up, grids, conf, "set")


class GridEngineTest(unittest.TestCase):
    def test_build_from_cash_conserves_equity_without_costs(self):
        e = GridEngine(GridConfig(90, 110, 10, fee_rate=0, slippage=0), 100, quote=1000)
        self.assertAlmostEqual(e.equity(100), 1000, places=4)
        sells = sum(1 for s in e.orders.values() if s == "sell")
        self.assertAlmostEqual(e.base, sells * e.qty)
        self.assertEqual(len(e.orders), 10)
        # 所有買單都成交也不透支
        e.on_price(80)
        self.assertGreaterEqual(e.quote, 0)
        self.assertTrue(all(s == "sell" for s in e.orders.values()))

    def test_rebuild_trades_only_the_difference(self):
        cfg = GridConfig(90, 110, 10, fee_rate=0.001, slippage=0.0005)
        e = GridEngine(cfg, 100, quote=500, base=5)  # 已有一半資金是幣
        adj = [f for f in e.fills if f.note == "建倉調整"]
        self.assertLessEqual(len(adj), 1)
        traded = adj[0].qty if adj else 0
        self.assertLess(traded, 5)  # 只交易差額,不是全部賣掉再買回
        self.assertGreater(e.equity(100), 1000 * 0.998)
        e.on_price(80)  # 買單全成交不透支
        self.assertGreaterEqual(e.quote, 0)

    def test_rebuild_from_all_coin_sells_excess(self):
        e = GridEngine(GridConfig(90, 110, 10, fee_rate=0.001, slippage=0), 100, quote=0, base=10)
        self.assertEqual(e.fills[0].side, "sell")
        e.on_price(80)
        self.assertGreaterEqual(e.quote, -1e-9)

    def test_cycle_profit_and_stop(self):
        e = GridEngine(GridConfig(90, 110, 10, fee_rate=0.001, slippage=0, stop_loss=85), 100, quote=1000)
        e.on_price(97)
        e.on_price(101)
        self.assertGreaterEqual(e.cycles, 1)
        self.assertGreater(e.grid_profit, 0)
        e.on_price(70, gap=True)  # 跳空:以當下價格停損
        self.assertTrue(e.halted)
        self.assertEqual(e.base, 0)
        self.assertEqual(e.fills[-1].price, 70)

    def test_state_roundtrip_and_tamper(self):
        e = GridEngine(GridConfig(90, 110, 8), 100, quote=1000)
        e.on_price(95)
        d = json.loads(json.dumps(e.to_dict()))
        r = GridEngine.from_dict(d)
        self.assertAlmostEqual(r.equity(95), e.equity(95))
        self.assertEqual(r.orders, e.orders)
        d["base"] = d["base"] * 2
        with self.assertRaises(ValueError):
            GridEngine.from_dict(d)

    def test_spacing_helpers(self):
        n = max_grids_for_spacing(95, 105, 0.5)
        self.assertGreaterEqual(spacing_pct(95, 105, n), 0.5 - 1e-9)
        self.assertLess(spacing_pct(95, 105, n + 1), 0.5)


class ParseGridDecisionTest(unittest.TestCase):
    def test_valid(self):
        d = parse_grid_decision('{"action":"SET","lower_pct":4,"upper_pct":3.5,"grids":12,"confidence":0.7,"reason":"盤整"}')
        self.assertEqual((d.action, d.lower_pct, d.upper_pct, d.grids), ("SET", 4, 3.5, 12))
        k = parse_grid_decision('{"action":"KEEP","lower_pct":99,"upper_pct":-1,"grids":0,"confidence":0.5,"reason":""}')
        self.assertEqual((k.lower_pct, k.upper_pct, k.grids), (0, 0, 0))  # KEEP 的數值忽略

    def test_invalid(self):
        bad = [
            '{"action":"BUY","lower_pct":1,"upper_pct":1,"grids":5,"confidence":0.5,"reason":""}',
            '{"action":"SET","lower_pct":0,"upper_pct":1,"grids":5,"confidence":0.5,"reason":""}',
            '{"action":"SET","lower_pct":1,"upper_pct":1,"grids":1,"confidence":0.5,"reason":""}',
            '{"action":"SET","lower_pct":1,"upper_pct":1,"grids":5.5,"confidence":0.5,"reason":""}',
            '{"action":"SET","lower_pct":1,"upper_pct":1,"grids":500,"confidence":0.5,"reason":""}',
            '{"action":"SET","lower_pct":1,"upper_pct":1,"grids":5,"confidence":2,"reason":""}',
            '{"action":"PAUSE","confidence":0.5}',
            "not json",
        ]
        for t in bad:
            with self.assertRaises(LLMError, msg=t):
                parse_grid_decision(t)

    def test_gemini_uses_grid_schema(self):
        text = '{"action":"PAUSE","lower_pct":0,"upper_pct":0,"grids":0,"confidence":0.8,"reason":"跌破"}'
        m = GeminiModel("k", "m", schema=GRID_SCHEMA, parser=parse_grid_decision)
        with mock.patch("aitrader.llm.urllib.request.urlopen", return_value=_Resp(_ok(text))) as u:
            d, _ = m.decide("sys", "prompt")
        body = json.loads(u.call_args[0][0].data)
        self.assertEqual(body["generationConfig"]["responseSchema"], GRID_SCHEMA)
        self.assertEqual(d.action, "PAUSE")


class GridAgentTest(unittest.TestCase):
    def agent(self, **kw):
        return GridAgent(10000, GridRiskConfig(**kw), fee=0.001, slippage=0.0005)

    def test_set_keep_pause_flow(self):
        a, cs = self.agent(), candles()
        r = a.decide(SeqModel(SET()), "X", "1h", cs, 100, TS)
        self.assertEqual(r["code"], "set")
        self.assertIsNotNone(a.engine)
        self.assertEqual(json.loads(json.dumps(r))["grid_after"]["grids"], 10)  # 日誌可序列化
        r = a.decide(SeqModel(), "X", "1h", cs, 100, TS + 3600)
        self.assertEqual(r["code"], "hold")
        r = a.decide(SeqModel(GridDecision("PAUSE", 0, 0, 0, 0.9, "p")), "X", "1h", cs, 100, TS + 7200)
        self.assertEqual(r["code"], "pause")
        self.assertIsNone(a.engine)
        self.assertGreater(a.cash, 9900)

    def test_filters(self):
        a, cs = self.agent(max_reconfigs=2, max_range_pct=10, min_spacing_pct=0.5), candles()
        self.assertEqual(a.decide(SeqModel(SET(conf=0.3)), "X", "1h", cs, 100, TS)["code"], "low_confidence")
        self.assertEqual(a.decide(SeqModel(SET(0.1, 0.1, 2)), "X", "1h", cs, 100, TS)["code"], "too_narrow")
        r = a.decide(SeqModel(SET(50, 50, 200)), "X", "1h", cs, 100, TS)
        self.assertEqual(r["code"], "set")
        self.assertAlmostEqual(a.engine.cfg.lower, 90)  # 截斷在 max_range
        self.assertGreaterEqual(spacing_pct(a.engine.cfg.lower, a.engine.cfg.upper, a.engine.cfg.grids), 0.5)
        self.assertIn("降為", r["action_taken"])
        self.assertEqual(a.decide(SeqModel(SET(10, 10, a.engine.cfg.grids)), "X", "1h", cs, 100.3, TS)["code"],
                         "small_change")
        self.assertEqual(a.decide(SeqModel(SET(3, 3, 6)), "X", "1h", cs, 100, TS)["code"], "set")
        self.assertEqual(a.decide(SeqModel(SET(4, 4, 6)), "X", "1h", cs, 100, TS)["code"], "max_reconfigs")
        self.assertEqual(a.decide(SeqModel(LLMError("x")), "X", "1h", cs, 100, TS)["code"], "no_decision")
        # 隔天(UTC)次數重算
        self.assertEqual(a.decide(SeqModel(SET(4, 4, 6)), "X", "1h", cs, 100, TS + 86400)["code"], "set")

    def test_stop_loss_halts_for_the_day(self):
        a, cs = self.agent(stop_loss_pct=2, daily_loss_limit_pct=0), candles()
        a.decide(SeqModel(SET(5, 5, 10)), "X", "1h", cs, 100, TS)
        fills, ev = a.on_price(80, TS + 60, gap=True)
        self.assertEqual(ev, "stop_loss")
        self.assertIsNone(a.engine)
        m = SeqModel(SET())
        r = a.decide(m, "X", "1h", cs, 80, TS + 120)
        self.assertEqual(r["code"], "halted")
        self.assertEqual(m.calls, [])  # 停機中不花錢問模型
        self.assertEqual(a.decide(m, "X", "1h", cs, 80, TS + 86400)["code"], "set")

    def test_daily_loss_limit(self):
        a, cs = self.agent(stop_loss_pct=0, daily_loss_limit_pct=1), candles()
        a.decide(SeqModel(SET(10, 10, 10)), "X", "1h", cs, 100, TS)
        _, ev = a.on_price(95, TS + 60)
        self.assertEqual(ev, "daily_loss")
        self.assertIsNone(a.engine)

    def test_resetting_keeps_value_and_stats(self):
        a, cs = self.agent(), candles()
        a.decide(SeqModel(SET(5, 5, 10)), "X", "1h", cs, 100, TS)
        a.on_price(97, TS + 1)
        a.on_price(101, TS + 2)
        cycles = a.totals()["cycles"]
        before = a.equity(101)
        a.decide(SeqModel(SET(3, 3, 6)), "X", "1h", cs, 101, TS + 3)
        self.assertGreater(a.equity(101), before * 0.999)  # 重設只付差額的成本
        self.assertEqual(a.totals()["cycles"], cycles)

    def test_state_roundtrip(self):
        a, cs = self.agent(), candles()
        a.decide(SeqModel(SET()), "X", "1h", cs, 100, TS)
        a.on_price(97, TS + 1)
        b = GridAgent.from_dict(json.loads(json.dumps(a.to_dict())), GridRiskConfig())
        self.assertAlmostEqual(b.equity(97), a.equity(97))
        self.assertEqual(b.engine.orders, a.engine.orders)
        self.assertEqual(b.reconfigs_today, 1)

    def test_prompt_includes_grid_state_and_rule_model(self):
        a, cs = self.agent(), candles()
        m = SeqModel(SET())
        a.decide(m, "X", "1h", cs, 100, TS)
        a.decide(m, "X", "1h", cs, 100, TS + 3600)
        self.assertIn("none", m.calls[0])
        self.assertIn("position_in_range", m.calls[1])
        r = a.decide(RuleGridModel(), "X", "1h", cs, 100, TS + 7200)
        self.assertIsNotNone(r["decision"])

    def test_replay(self):
        cs = candles(300, drift=0.004)
        a = self.agent()
        r = grid_replay(RuleGridModel(), "X", "1h", cs, a, every=4)
        self.assertEqual(r.bars, 300 - 60)
        self.assertGreater(r.sets, 0)
        self.assertGreater(r.grid_fills, 0)
        self.assertAlmostEqual(r.equity, a.equity(cs[-1].close))
        calls = {"n": 0}

        class Counting(RuleGridModel):
            def decide(self, *x):
                calls["n"] += 1
                return super().decide(*x)

        r = grid_replay(Counting(), "X", "1h", cs, self.agent(), every=1, max_calls=5, calls_used=lambda: calls["n"])
        self.assertTrue(r.stopped_early)
        self.assertEqual(calls["n"], 5)


class GridCliTest(unittest.TestCase):
    def _patch(self, prices):
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch("aitrader.grid_cli.fetch_klines", return_value=candles()))
        stack.enter_context(mock.patch("aitrader.grid_cli.fetch_price", side_effect=prices))
        stack.enter_context(mock.patch("aitrader.grid_cli.time", FakeTime()))
        return stack

    def test_grid_paper_resume_and_report(self):
        from aitrader.cli import main
        prices = itertools.cycle([100, 99, 97.5, 101, 102])
        with tempfile.TemporaryDirectory() as t, self._patch(lambda *a: next(prices)), \
                mock.patch("aitrader.grid_cli._make_model", return_value=SeqModel(SET(5, 5, 10))), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            state, journal = os.path.join(t, "s.json"), os.path.join(t, "j.jsonl")
            args = ["grid", "--state", state, "--journal", journal, "--max-steps", "2"]
            self.assertEqual(main(args), 0)
            self.assertIn("重設網格", out.getvalue())
            self.assertIn("成交", out.getvalue())
            self.assertEqual(main(args[:-1] + ["1"]), 0)  # 接續
            self.assertIn("接續", out.getvalue())
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(args[:-2] + ["--interval", "4h", "--max-steps", "1"]), 2)
            self.assertEqual(main(["report", journal]), 0)
            self.assertIn("網格日誌", out.getvalue())
            self.assertIn("網格成交", out.getvalue())

    def test_grid_paper_retries_model_error(self):
        from aitrader.cli import main
        m = SeqModel(LLMError("暫時失敗"), SET())
        with tempfile.TemporaryDirectory() as t, self._patch(lambda *a: 100.0), \
                mock.patch("aitrader.grid_cli._make_model", return_value=m), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            rc = main(["grid", "--state", os.path.join(t, "s.json"), "--journal", os.path.join(t, "j.jsonl"),
                       "--max-steps", "1"])
        self.assertEqual(rc, 0)
        self.assertEqual(len(m.calls), 2)
        self.assertIn("後重試", out.getvalue())

    def test_grid_replay_cli_and_help(self):
        from aitrader.cli import main
        with tempfile.TemporaryDirectory() as t, \
                mock.patch("aitrader.grid_cli.fetch_klines", return_value=candles(200, drift=0.004)), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            j = os.path.join(t, "r.jsonl")
            self.assertEqual(main(["grid-replay", "--model", "rule", "--limit", "200", "--journal", j]), 0)
            self.assertEqual(main(["report", j]), 0)
        self.assertIn("網格成交", out.getvalue())
        for cmd in ("grid", "grid-replay"):
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as cm:
                main([cmd, "-h"])
            self.assertEqual(cm.exception.code, 0)


if __name__ == "__main__":
    unittest.main()


class LearningModel(SeqModel):
    """可做每日檢討的假模型。"""

    can_reflect = True

    def __init__(self, *decisions, reflections=None):
        super().__init__(*decisions)
        self.reflections = list(reflections or [])
        self.reflect_calls = []

    def reflect(self, system, prompt):
        self.reflect_calls.append(prompt)
        r = self.reflections.pop(0) if self.reflections else Reflection("ok", ["盤整時維持網格"])
        if isinstance(r, Exception):
            raise r
        return r, META


class LearningTest(unittest.TestCase):
    def agent(self, learn=True, **kw):
        return GridAgent(10000, GridRiskConfig(**kw), fee=0.001, slippage=0.0005, learn=learn)

    def test_memory_outcome_and_day_review(self):
        m = LearningMemory()
        m.record_decision(TS, 100, 1000, {"action": "SET", "lower_pct": 3, "upper_pct": 2, "grids": 8,
                                           "confidence": 0.8, "reason": "r"}, "set")
        m.on_tick(3, None)
        m.on_tick(0, "stop_loss")
        m.settle(TS + 7200, 98, 990)
        e = m.history[-1]
        self.assertEqual(e["params"], "-3%/+2% 8 grids")
        self.assertEqual(e["events"], ["stop_loss"])
        self.assertEqual(e["outcome"], {"hours": 2.0, "price_change_pct": -2.0, "equity_change_pct": -1.0,
                                        "grid_fills": 3})
        m.roll_day("2023-11-14", TS + 9000, 98, 990, 1000, 2.0)
        self.assertEqual(m.review["grid_fills"], 3)
        self.assertEqual(m.review["forced_exits"], ["stop_loss"])
        self.assertEqual(len(m.review["decisions"]), 1)
        self.assertEqual(m.day_log, [])
        for i in range(20):
            m.record_decision(TS + i, 100, 1000, None, "no_decision")
            m.settle(TS + i + 1, 100, 1000)
        self.assertEqual(len(m.history), HISTORY_LEN)
        r = LearningMemory.from_dict(json.loads(json.dumps(m.to_dict())))
        self.assertEqual(r.to_dict(), m.to_dict())
        self.assertEqual(LearningMemory.from_dict(None).history, [])  # 舊版狀態檔

    def test_model_errors_not_recorded_in_history(self):
        a, cs = self.agent(), candles()
        m = LearningModel(LLMError("x"), LLMError("x"), SET())
        for i in range(4):
            a.decide(m, "X", "1h", cs, 100, TS + i * 60)
        self.assertEqual([e["action"] for e in a.memory.history], ["SET"])

    def test_prompt_contains_memory_only_when_learning(self):
        cs = candles()
        for learn in (True, False):
            a, m = self.agent(learn), LearningModel(SET())
            a.decide(m, "X", "1h", cs, 100, TS)
            a.decide(m, "X", "1h", cs, 100.5, TS + 3600)
            self.assertNotIn("memory", m.calls[0])  # 第一次還沒有紀錄
            self.assertEqual("recent_decisions_with_outcomes" in m.calls[1], learn)

    def test_daily_reflection_updates_lessons(self):
        cs = candles()
        a = self.agent()
        m = LearningModel(SET(), reflections=[LLMError("暫時"), Reflection("今天盤整", ["格數別太多", "", 5])])
        day0 = TS - TS % 86400 + 3600
        a.decide(m, "X", "1h", cs, 100, day0)
        a.on_price(97, day0 + 60)
        a.decide(m, "X", "1h", cs, 99, day0 + 3600)
        self.assertEqual(m.reflect_calls, [])  # 同一天不檢討
        r = a.decide(m, "X", "1h", cs, 99, day0 + 86400)  # 換日:檢討失敗,不影響決策
        self.assertTrue(r["reflection"]["error"])
        self.assertIsNotNone(r["decision"])
        self.assertIn("grid_fills", m.reflect_calls[0])
        r = a.decide(m, "X", "1h", cs, 99, day0 + 86400 + 3600)  # 下一次決策重試成功
        self.assertEqual(r["reflection"]["lessons"], ["格數別太多"])  # 空字串與非字串被濾掉
        self.assertEqual(a.memory.lessons, ["格數別太多"])
        r = a.decide(m, "X", "1h", cs, 99, day0 + 86400 + 7200)
        self.assertIsNone(r["reflection"])  # 同一天只檢討一次
        self.assertIn("格數別太多", m.calls[-1])  # 心得放進之後的決策提示詞

    def test_reflection_gives_up_after_retries_and_skips_without_support(self):
        cs = candles()
        a = self.agent()
        m = LearningModel(reflections=[LLMError("x")] * 5)
        day0 = TS - TS % 86400
        a.decide(m, "X", "1h", cs, 100, day0)
        for i in range(5):
            a.decide(m, "X", "1h", cs, 100, day0 + 86400 + i)
        self.assertEqual(len(m.reflect_calls), 3)
        b, rule = self.agent(), RuleGridModel()  # 規則模型不支援檢討:略過
        b.decide(rule, "X", "1h", cs, 100, day0)
        self.assertIsNone(b.decide(rule, "X", "1h", cs, 100, day0 + 86400)["reflection"])
        c, m2 = self.agent(learn=False), LearningModel()
        c.decide(m2, "X", "1h", cs, 100, day0)
        c.decide(m2, "X", "1h", cs, 100, day0 + 86400)
        self.assertEqual(m2.reflect_calls, [])

    def test_agent_state_keeps_memory_and_reads_old_state(self):
        cs = candles()
        a = self.agent()
        a.decide(LearningModel(SET()), "X", "1h", cs, 100, TS)
        a.memory.lessons = ["心得"]
        d = json.loads(json.dumps(a.to_dict()))
        b = GridAgent.from_dict(d, GridRiskConfig())
        self.assertEqual(b.memory.lessons, ["心得"])
        self.assertIsNotNone(b.memory.pending)
        del d["memory"], d["day_start_fees"]
        self.assertEqual(GridAgent.from_dict(d, GridRiskConfig()).memory.lessons, [])

    def test_parse_reflection_and_gemini_and_cache(self):
        with self.assertRaises(LLMError):
            parse_reflection('{"summary":"x"}')
        r = parse_reflection(json.dumps({"summary": "s", "lessons": [f"l{i}" for i in range(9)]}))
        self.assertEqual(len(r.lessons), MAX_LESSONS)
        g = GeminiModel("k", "m", schema=GRID_SCHEMA, parser=parse_grid_decision)
        with mock.patch("aitrader.llm.urllib.request.urlopen",
                        return_value=_Resp(_ok('{"summary":"s","lessons":["a"]}'))) as u:
            r, _ = g.reflect("sys", "p")
        self.assertEqual(json.loads(u.call_args[0][0].data)["generationConfig"]["responseSchema"], REFLECTION_SCHEMA)
        self.assertEqual(r.lessons, ["a"])
        with tempfile.TemporaryDirectory() as t:
            inner = LearningModel()
            c = CachedModel(inner, t, GridDecision)
            self.assertTrue(c.can_reflect)
            c.reflect("s", "p")
            r, meta = c.reflect("s", "p")
            self.assertEqual(len(inner.reflect_calls), 1)
            self.assertTrue(meta["cached"])
            self.assertIsInstance(r, Reflection)
            self.assertFalse(CachedModel(RuleGridModel(), t).can_reflect)

    def test_cli_no_learn_uses_separate_files_and_report_shows_lessons(self):
        from aitrader.cli import main
        with tempfile.TemporaryDirectory() as t:
            cwd = os.getcwd()
            os.chdir(t)
            try:
                with mock.patch("aitrader.grid_cli.fetch_klines", return_value=candles()), \
                        mock.patch("aitrader.grid_cli.fetch_price", return_value=100.0), \
                        mock.patch("aitrader.grid_cli.time", FakeTime()), \
                        mock.patch("aitrader.grid_cli._make_model", return_value=LearningModel(SET())), \
                        contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main(["grid", "--no-learn", "--max-steps", "1"]), 0)
                self.assertTrue(os.path.exists("state/grid_BTCUSDT_gemini_nolearn.json"))
                j = "journal/grid_BTCUSDT_gemini_nolearn.jsonl"
                with open(j, "a") as fh:
                    fh.write(json.dumps({"kind": "grid_reflection", "ts": TS, "day": "2023-11-14", "summary": "盤整",
                                         "lessons": ["別太常重設"], "meta": META, "error": None}) + "\n")
                with contextlib.redirect_stdout(io.StringIO()) as out:
                    self.assertEqual(main(["report", j]), 0)
            finally:
                os.chdir(cwd)
        self.assertIn("別太常重設", out.getvalue())
        self.assertIn("最後權益", out.getvalue())
