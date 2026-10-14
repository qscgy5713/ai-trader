import contextlib
import io
import json
import os
import tempfile
import unittest
import urllib.error
from unittest import mock

from aitrader import llm
from aitrader.agent import replay, step
from aitrader.config import gemini_settings, load_dotenv
from aitrader.llm import (CachedModel, Decision, GeminiModel, LLMConfigError, LLMError, RuleModel,
                          parse_decision)
from aitrader.market import MIN_BARS, Candle, features, rsi, sma
from aitrader.portfolio import Portfolio
from aitrader.risk import RiskConfig, RiskManager


def candles(n=120, start=100.0, drift=0.001, ts0=1_700_000_000, step_s=3600):
    out, p = [], start
    for i in range(n):
        o = p
        c = o * (1 + drift * (1 if i % 3 else -1))
        out.append(Candle(ts0 + i * step_s, o, max(o, c) * 1.002, min(o, c) * 0.998, c, 10 + i % 5))
        p = c
    return out


class FakeModel:
    """固定回傳某個決策,並記錄被呼叫時看到的資料。"""

    name = "fake"

    def __init__(self, decision=None, exc=None):
        self.decision, self.exc, self.calls = decision, exc, []

    def decide(self, system, prompt, feats=None):
        self.calls.append((prompt, feats))
        if self.exc:
            raise self.exc
        return self.decision, {"model": "fake", "latency_s": 0, "tokens_in": 1, "tokens_out": 1}


class ConfigTest(unittest.TestCase):
    def test_dotenv(self):
        with tempfile.TemporaryDirectory() as t:
            p = os.path.join(t, ".env")
            with open(p, "w") as fh:
                fh.write("# comment\nGEMINI_API_KEY='abc'\nGEMINI_MODEL=\nOTHER_X=1\nbad line\n")
            with mock.patch.dict(os.environ, {"OTHER_X": "keep"}, clear=False):
                os.environ.pop("GEMINI_API_KEY", None)
                os.environ.pop("GEMINI_MODEL", None)
                load_dotenv(p)
                self.assertEqual(os.environ["GEMINI_API_KEY"], "abc")
                self.assertEqual(os.environ["OTHER_X"], "keep")  # 已存在的環境變數優先
                self.assertEqual(gemini_settings(), ("abc", llm_default()))  # 空模型名稱用預設
                os.environ.pop("GEMINI_API_KEY")


def llm_default():
    from aitrader.config import DEFAULT_MODEL
    return DEFAULT_MODEL


class MarketTest(unittest.TestCase):
    def test_indicators(self):
        self.assertEqual(sma([1, 2, 3, 4], 2), 3.5)
        self.assertEqual(rsi(list(range(1, 30))), 100.0)
        self.assertEqual(rsi(list(range(30, 1, -1))), 0.0)
        self.assertEqual(rsi([5.0] * 30), 50.0)

    def test_features(self):
        f = features(candles())
        for k in ("last_close", "return_pct", "rsi14", "atr14_pct", "recent_closes", "sma20_vs_price_pct"):
            self.assertIn(k, f)
        self.assertEqual(len(f["recent_closes"]), 24)
        with self.assertRaises(ValueError):
            features(candles(MIN_BARS - 1))

    def test_fetch_klines_drops_forming_candle(self):
        from aitrader import market
        rows = [[i * 3600_000, "1", "2", "0.5", str(i), "3"] for i in range(5)]
        with mock.patch.object(market, "_get", return_value=rows) as g:
            k = market.fetch_klines("X", "1h", 4)
        self.assertEqual([c.close for c in k], [0.0, 1.0, 2.0, 3.0])  # 不含形成中的第 5 根
        self.assertIn("limit=5", g.call_args[0][0])


class ParseDecisionTest(unittest.TestCase):
    def test_valid(self):
        d = parse_decision('{"action":"BUY","target_position_pct":30,"confidence":0.8,"reason":" 趨勢向上 "}')
        self.assertEqual(d, Decision("BUY", 30.0, 0.8, "趨勢向上"))

    def test_invalid(self):
        for bad in ("not json", "[]", '{"action":"SHORT","target_position_pct":1,"confidence":1,"reason":""}',
                    '{"action":"BUY","target_position_pct":101,"confidence":0.5,"reason":""}',
                    '{"action":"BUY","target_position_pct":"NaN","confidence":0.5,"reason":""}',
                    '{"action":"BUY","target_position_pct":10,"confidence":1.5,"reason":""}',
                    '{"action":"BUY","target_position_pct":10,"confidence":0.5}',
                    '{"action":"BUY","confidence":0.5,"reason":"x"}'):
            with self.assertRaises(LLMError, msg=bad):
                parse_decision(bad)


class _Resp:
    def __init__(self, payload):
        self.data = json.dumps(payload).encode()

    def read(self, *a):
        return self.data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _http_error(code, msg="boom", headers=None):
    return urllib.error.HTTPError("https://x", code, "err", headers or {},
                                  io.BytesIO(json.dumps({"error": {"message": msg}}).encode()))


def _ok(text, finish="STOP", thought=None):
    parts = ([{"text": thought, "thought": True}] if thought else []) + [{"text": text}]
    return {"candidates": [{"content": {"parts": parts}, "finishReason": finish}],
            "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 20, "thoughtsTokenCount": 5}}


GOOD = '{"action":"HOLD","target_position_pct":0,"confidence":0.4,"reason":"訊號不明"}'


class GeminiTest(unittest.TestCase):
    def setUp(self):
        self.m = GeminiModel("SECRET-KEY", "gemini-test", max_retries=2)
        p = mock.patch("aitrader.llm.time.sleep")
        self.sleep = p.start()
        self.addCleanup(p.stop)

    def test_success_request_format_and_key_not_in_url(self):
        with mock.patch("aitrader.llm.urllib.request.urlopen", return_value=_Resp(_ok(GOOD, thought="思考中"))) as u:
            d, meta = self.m.decide("SYS", "PROMPT")
        req = u.call_args[0][0]
        self.assertNotIn("SECRET-KEY", req.full_url)
        self.assertIn("models/gemini-test:generateContent", req.full_url)
        self.assertEqual(req.get_header("X-goog-api-key"), "SECRET-KEY")
        body = json.loads(req.data)
        self.assertEqual(body["generationConfig"]["responseMimeType"], "application/json")
        self.assertEqual(body["systemInstruction"]["parts"][0]["text"], "SYS")
        self.assertEqual(d.action, "HOLD")  # 思考內容(thought)不會混進輸出
        self.assertEqual((meta["tokens_in"], meta["tokens_out"]), (100, 25))

    def test_retry_on_429_then_success(self):
        seq = [_http_error(429, headers={"Retry-After": "2"}), _http_error(503), _Resp(_ok(GOOD))]
        with mock.patch("aitrader.llm.urllib.request.urlopen", side_effect=seq):
            d, _ = self.m.decide("S", "P")
        self.assertEqual(d.action, "HOLD")
        self.assertEqual(self.sleep.call_count, 2)
        self.assertGreaterEqual(self.sleep.call_args_list[0][0][0], 2)  # 遵守 Retry-After

    def test_retries_exhausted(self):
        with mock.patch("aitrader.llm.urllib.request.urlopen", side_effect=[_http_error(500)] * 3):
            with self.assertRaises(LLMError) as cm:
                self.m.decide("S", "P")
        self.assertNotIsInstance(cm.exception, LLMConfigError)
        with mock.patch("aitrader.llm.urllib.request.urlopen",
                        side_effect=urllib.error.URLError("no network")):
            with self.assertRaises(LLMError):
                self.m.decide("S", "P")

    def test_config_errors_not_retried_and_no_key_leak(self):
        for code in (400, 401, 402, 403, 404):  # 402 = 額度用完(帳務),同樣不重試、直接停
            with mock.patch("aitrader.llm.urllib.request.urlopen", side_effect=_http_error(code, "API key not valid")) as u:
                with self.assertRaises(LLMConfigError) as cm:
                    self.m.decide("S", "P")
            self.assertEqual(u.call_count, 1)
            self.assertNotIn("SECRET-KEY", str(cm.exception))
        with self.assertRaises(LLMConfigError):
            GeminiModel("", "m")

    def test_bad_outputs(self):
        cases = [_ok(GOOD, finish="SAFETY"), _ok("not json"), {"candidates": []},
                 {"promptFeedback": {"blockReason": "SAFETY"}},
                 _ok('{"action":"BUY","target_position_pct":500,"confidence":1,"reason":"x"}')]
        for payload in cases:
            with mock.patch("aitrader.llm.urllib.request.urlopen", return_value=_Resp(payload)):
                with self.assertRaises(LLMError, msg=str(payload)[:80]):
                    self.m.decide("S", "P")


class CacheTest(unittest.TestCase):
    def test_cache_hit_skips_inner(self):
        inner = FakeModel(Decision("BUY", 20, 0.9, "x"))
        with tempfile.TemporaryDirectory() as t:
            c = CachedModel(inner, t)
            a, _ = c.decide("S", "P")
            b, meta = c.decide("S", "P")
            c.decide("S", "P2")
        self.assertEqual(a, b)
        self.assertTrue(meta["cached"])
        self.assertEqual(len(inner.calls), 2)
        self.assertEqual((c.hits, c.misses), (1, 2))


class PortfolioTest(unittest.TestCase):
    def test_buy_sell_accounting(self):
        p = Portfolio(1000, fee_rate=0.001, slippage_bps=10)
        t = p.rebalance(100, 100, 0)
        self.assertGreaterEqual(p.cash, 0)
        self.assertAlmostEqual(t.price, 100.1)
        self.assertAlmostEqual(p.cash + t.qty * t.price + t.fee, 1000)
        s = p.rebalance(0, 110, 1)
        self.assertEqual(p.qty, 0)
        self.assertAlmostEqual(s.price, 109.89)
        self.assertAlmostEqual(s.realized_pnl, (109.89 - p.trades[0].price * 1.001) * s.qty - s.fee, places=6)
        self.assertIsNone(p.rebalance(0, 110, 2))  # 已空手

    def test_partial_and_roundtrip_state(self):
        p = Portfolio(1000, 0, 0)
        p.rebalance(50, 100, 0)
        self.assertAlmostEqual(p.position_pct(100), 50)
        p.rebalance(20, 100, 1)
        self.assertAlmostEqual(p.position_pct(100), 20)
        q = Portfolio.from_dict(json.loads(json.dumps(p.to_dict())))
        self.assertAlmostEqual(q.equity(100), p.equity(100))
        for bad in ({}, dict(p.to_dict(), cash=-1), dict(p.to_dict(), qty=float("nan"))):
            with self.assertRaises(ValueError):
                Portfolio.from_dict(bad)
        with self.assertRaises(ValueError):
            p.rebalance(120, 100, 2)


class RiskTest(unittest.TestCase):
    def setUp(self):
        self.r = RiskManager(RiskConfig(max_position_pct=50, min_confidence=0.6, min_trade_pct=5, stop_loss_pct=5,
                                        daily_loss_limit_pct=3, max_trades_per_day=2))
        self.p = Portfolio(1000, 0, 0)
        self.r.new_bar("d1", 1000)

    def test_filter_codes(self):
        f = lambda a, t, c: self.r.filter(Decision(a, t, c, ""), self.p, 100)
        self.assertEqual(self.r.filter(None, self.p, 100)[2], "no_decision")
        self.assertEqual(f("HOLD", 0, 1)[2], "hold")
        self.assertEqual(f("SELL", 0, 1)[2], "noop")
        self.assertEqual(f("BUY", 30, 0.5)[2], "low_confidence")
        self.assertEqual(f("BUY", 3, 0.9)[2], "small_change")
        self.assertEqual(f("BUY", 90, 0.9)[:1], (50,))  # 被持倉上限截斷
        self.r.state.trades_today = 2
        self.assertEqual(f("BUY", 30, 0.9)[2], "max_trades")

    def test_stop_loss_and_daily_limit(self):
        self.p.rebalance(50, 100, 0)
        self.assertIsNone(self.r.forced(self.p, 99))
        self.assertEqual(self.r.forced(self.p, 95)[2], "stop_loss")
        # 權益 1000 -> 下跌 3%:持倉 500 要跌 30 = 價格 94
        self.assertEqual(self.r.forced(self.p, 94)[2], "daily_loss")
        self.assertTrue(self.r.state.halted_today)
        self.p.rebalance(0, 94, 1)
        self.assertEqual(self.r.filter(Decision("BUY", 30, 0.9, ""), self.p, 94)[2], "halted")
        self.r.new_bar("d2", self.p.equity(94))  # 隔天重置
        self.assertFalse(self.r.state.halted_today)

    def test_config_validation_and_state_roundtrip(self):
        for bad in (dict(max_position_pct=0), dict(min_confidence=2), dict(stop_loss_pct=0), dict(max_trades_per_day=0)):
            with self.assertRaises(ValueError):
                RiskConfig(**bad).validate()
        r2 = RiskManager.from_dict(json.loads(json.dumps(self.r.to_dict())))
        self.assertEqual(r2.state, self.r.state)


class AgentTest(unittest.TestCase):
    def _env(self):
        return Portfolio(1000, 0, 0), RiskManager(RiskConfig())

    def test_model_error_means_no_trade(self):
        p, r = self._env()
        rec = step(FakeModel(exc=LLMError("bad output")), "X", "1h", candles(), 100, 0, p, r)
        self.assertIsNone(rec["trade"])
        self.assertEqual(rec["code"], "no_decision")
        self.assertIn("bad output", rec["error"])

    def test_config_error_propagates(self):
        p, r = self._env()
        with self.assertRaises(LLMConfigError):
            step(FakeModel(exc=LLMConfigError("key")), "X", "1h", candles(), 100, 0, p, r)

    def test_forced_exit_skips_model(self):
        p, r = self._env()
        p.rebalance(50, 100, 0)
        m = FakeModel(Decision("BUY", 50, 1, ""))
        rec = step(m, "X", "1h", candles(), 90, 0, p, r)
        self.assertEqual(m.calls, [])
        self.assertEqual(rec["code"], "stop_loss")
        self.assertEqual(p.qty, 0)

    def test_journal_written(self):
        p, r = self._env()
        with tempfile.TemporaryDirectory() as t:
            j = os.path.join(t, "j", "x.jsonl")
            step(FakeModel(Decision("BUY", 30, 0.9, "ok")), "X", "1h", candles(), 100, 0, p, r, journal=j)
            with open(j) as fh:
                rec = json.loads(fh.readline())
        self.assertEqual(rec["code"], "execute")
        self.assertEqual(rec["decision"]["reason"], "ok")

    def test_replay_executes_at_next_open_no_lookahead(self):
        cs = candles(MIN_BARS + 5)
        m = FakeModel(Decision("BUY", 40, 0.9, ""))
        p, r = Portfolio(1000, 0, 0), RiskManager(RiskConfig(max_position_pct=40))
        res = replay(m, "X", "1h", cs, p, r, every=1)
        first_seen = m.calls[0][1]["last_close"]
        self.assertAlmostEqual(first_seen, cs[MIN_BARS - 1].close, places=5)  # 第一次決策只看到第 MIN_BARS 根(特徵取 6 位小數)
        self.assertAlmostEqual(p.trades[0].price, cs[MIN_BARS].open)  # 以下一根開盤價成交
        self.assertEqual(res.bars, 5)

    def test_replay_max_calls(self):
        cs = candles(MIN_BARS + 20)
        m = FakeModel(Decision("HOLD", 0, 0.5, ""))
        used = lambda: len(m.calls)
        res = replay(m, "X", "1h", cs, Portfolio(1000), RiskManager(RiskConfig()), every=1, max_calls=3, calls_used=used)
        self.assertTrue(res.stopped_early)
        self.assertEqual(len(m.calls), 3)


class FakeTime:
    """可控時鐘:sleep 只推進時間;同時提供 time / monotonic / strftime / gmtime。"""

    def __init__(self, now=1_700_000_000.0):
        import time as _t
        self._t, self.now = _t, now

    def time(self):
        return self.now

    def monotonic(self):
        return self.now

    def sleep(self, s):
        self.now += s

    def strftime(self, fmt, t=None):
        return self._t.strftime(fmt, self._t.gmtime(self.now) if t is None else t)

    def gmtime(self, *a):
        return self._t.gmtime(*a)

    def localtime(self, *a):
        return self._t.localtime(*a)


class CliTest(unittest.TestCase):
    def _patch_market(self):
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch("aitrader.cli.fetch_klines", return_value=candles()))
        stack.enter_context(mock.patch("aitrader.cli.fetch_price", return_value=100.0))
        stack.enter_context(mock.patch("aitrader.cli.time", FakeTime()))  # 接續時要等下一根 K 線:用假時鐘快轉
        return stack

    def test_once_and_paper_resume_and_report(self):
        from aitrader.cli import main
        with tempfile.TemporaryDirectory() as t, self._patch_market(), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(["once", "--model", "rule"]), 0)
            state, journal = os.path.join(t, "s.json"), os.path.join(t, "j.jsonl")
            args = ["paper", "--model", "rule", "--state", state, "--journal", journal, "--max-steps", "1"]
            self.assertEqual(main(args), 0)
            self.assertTrue(os.path.exists(state))
            self.assertEqual(main(args), 0)  # 接續
            self.assertIn("接續", out.getvalue())
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(args[:-2] + ["--interval", "4h", "--max-steps", "1"]), 2)  # 週期不同:拒絕
            self.assertEqual(main(["report", journal]), 0)

    def test_missing_key_is_clear_error(self):
        from aitrader.cli import main
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": ""}), mock.patch("aitrader.cli.load_dotenv"), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(main(["once"]), 2)
        self.assertIn("GEMINI_API_KEY", err.getvalue())

    def test_paper_retries_same_bar_after_model_error(self):
        from aitrader.cli import main

        class Flaky(FakeModel):
            def decide(self, system, prompt, feats=None):
                self.calls.append(prompt)
                if len(self.calls) <= 2:
                    raise LLMError("暫時失敗")
                return Decision("BUY", 30, 0.9, "ok"), {"model": "f", "latency_s": 0, "tokens_in": 0, "tokens_out": 0}

        m = Flaky()
        with tempfile.TemporaryDirectory() as t, self._patch_market(), mock.patch("aitrader.cli._make_model", return_value=m), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            j = os.path.join(t, "j.jsonl")
            rc = main(["paper", "--state", os.path.join(t, "s.json"), "--journal", j, "--max-steps", "1"])
            with open(j) as fh:
                codes = [json.loads(l)["code"] for l in fh]
        self.assertEqual(rc, 0)
        self.assertEqual(len(m.calls), 3)  # 同一根 K 線重試到成功,而不是等下一根
        self.assertEqual(codes, ["no_decision", "no_decision", "execute"])
        self.assertIn("後重試", out.getvalue())

    def test_report_without_decisions_shows_errors(self):
        from aitrader.cli import main
        with tempfile.TemporaryDirectory() as t:
            j = os.path.join(t, "j.jsonl")
            with open(j, "w") as fh:
                fh.write(json.dumps({"ts": 0, "error": "Gemini API 402:depleted", "code": "no_decision", "decision": None,
                                     "trade": None, "equity_after": 10000, "position_pct_after": 0}) + "\n")
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(main(["report", j]), 0)
        self.assertIn("402", out.getvalue())
        self.assertIn("還沒有任何有效的模型決策", out.getvalue())

    def test_once_exit_code_reflects_model_error(self):
        from aitrader.cli import main
        with self._patch_market(), mock.patch("aitrader.cli._make_model", return_value=FakeModel(exc=LLMError("x"))), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["once"]), 1)

    def test_help_renders(self):
        from aitrader.cli import main
        for cmd in ("once", "paper", "replay", "report"):
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as cm:
                main([cmd, "-h"])
            self.assertEqual(cm.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
