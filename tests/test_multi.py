import contextlib
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

from aitrader.multi import parse_symbols, replace_symbol, run_multi, short_name
from tests.test_aitrader import candles


class MultiSymbolTest(unittest.TestCase):
    def test_parse_symbols(self):
        self.assertEqual(parse_symbols("solusdt, ETHUSDT,,SOLUSDT"), ["SOLUSDT", "ETHUSDT"])  # 去重、保留順序
        for bad in ("", " , ", "SOL-USDT", "SOL USDT", ",".join(f"A{i}USDT" for i in range(9))):
            with self.assertRaises(ValueError):
                parse_symbols(bad)

    def test_replace_symbol(self):
        self.assertEqual(replace_symbol(["futures", "--symbol", "A,B", "--news"], "A"), ["futures", "--symbol", "A", "--news"])
        self.assertEqual(replace_symbol(["futures", "--symbol=A,B"], "B"), ["futures", "--symbol=B"])
        with self.assertRaises(ValueError):
            replace_symbol(["futures"], "A")
        self.assertEqual((short_name("SOLUSDT"), short_name("USDT")), ("SOL", "USDT"))

    def test_run_multi_prefixes_output_and_keeps_others_running(self):
        def cmd(sym):
            code = 3 if sym == "BADUSDT" else 0
            return [sys.executable, "-c", f"print('hello {sym}'); print('line2'); raise SystemExit({code})"]
        out = io.StringIO()
        rc = run_multi(["SOLUSDT", "BADUSDT"], cmd, out)
        s = out.getvalue()
        self.assertEqual(rc, 3)  # 回傳最大的結束代碼
        self.assertIn("[SOL] hello SOLUSDT", s)
        self.assertIn("[BAD] line2", s)
        self.assertIn("[SOL] 已結束(代碼 0)", s)
        self.assertIn("[BAD] 已結束(代碼 3)", s)

    def test_run_multi_reports_early_exit_without_waiting_for_others(self):
        def cmd(sym):  # SLOW 跑 2 秒;BAD 立刻失敗,應該先顯示
            src = "import time; time.sleep(2)" if sym == "SLOWUSDT" else "print('錯誤:狀態檔不同'); raise SystemExit(2)"
            return [sys.executable, "-c", src]
        out = io.StringIO()
        run_multi(["SLOWUSDT", "BADUSDT"], cmd, out)
        lines = out.getvalue().splitlines()
        self.assertLess(lines.index("[BAD] 已結束(代碼 2);⚠ 其他交易對繼續執行:SLOWUSDT"), lines.index("[SLOW] 已結束(代碼 0)"))
        self.assertEqual(lines[0], "[BAD] 錯誤:狀態檔不同")

    def test_reset_only_listed_symbols(self):
        import argparse
        from aitrader.futures_cli import _resolve_reset
        for reset, sym, want in ((None, "SOLUSDT", False), ("*", "SOLUSDT", True), ("ETHUSDT", "SOLUSDT", False),
                                 ("ethusdt", "ETHUSDT", True)):
            a = argparse.Namespace(reset=reset, symbol=sym)
            _resolve_reset(a, ["SOLUSDT", "ETHUSDT"])
            self.assertIs(a.reset, want, (reset, sym))
        with self.assertRaises(ValueError):
            _resolve_reset(argparse.Namespace(reset="SUIUSDT", symbol="SOLUSDT"), ["SOLUSDT", "ETHUSDT"])

    def test_reset_symbol_cli(self):
        from aitrader.cli import main
        from tests.test_aitrader import FakeTime
        from tests.test_futures import SeqModel
        with tempfile.TemporaryDirectory() as t, \
                mock.patch("aitrader.futures_cli.fetch_klines", return_value=candles()), \
                mock.patch("aitrader.futures_cli.fetch_premium", return_value=(100.0, 0.0, FakeTime().now + 10 ** 6)), \
                mock.patch("aitrader.futures_cli.time", FakeTime()), \
                mock.patch("aitrader.futures_cli._make_model", side_effect=lambda *a, **k: SeqModel()), \
                contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
            base = ["futures", "--state", os.path.join(t, "s.json"), "--journal", os.path.join(t, "j.jsonl"), "--max-steps", "1"]
            main(base)
            self.assertEqual(main(base + ["--interval", "5m"]), 2)  # 週期不同:拒絕接續
            self.assertEqual(main(base + ["--interval", "5m", "--reset", "SOLUSDT"]), 2)  # 不在 --symbol(BTCUSDT)之中
            self.assertEqual(main(base + ["--interval", "5m", "--reset", "BTCUSDT"]), 0)  # 只重設這個交易對
            self.assertEqual(main(base + ["--interval", "5m"]), 0)  # 之後同一個指令(不加 --reset)就能接續
        self.assertIn("新帳戶", out.getvalue())

    def test_futures_multi_spawns_one_child_per_symbol(self):
        from aitrader.cli import main
        with mock.patch("aitrader.futures_cli.run_multi", return_value=0) as rm, \
                contextlib.redirect_stdout(io.StringIO()) as out:
            argv = ["futures", "--model", "rule", "--symbol", "SOLUSDT,ETHUSDT", "--max-steps", "1"]
            self.assertEqual(main(argv), 0)
        symbols, command_for = rm.call_args[0]
        self.assertEqual(symbols, ["SOLUSDT", "ETHUSDT"])
        self.assertEqual(command_for("ETHUSDT")[1:], ["-m", "aitrader", "futures", "--model", "rule", "--symbol", "ETHUSDT",
                                                      "--max-steps", "1"])
        self.assertIn("同時跑 2 個交易對", out.getvalue())

    def test_child_argv_reset_only_listed(self):
        from aitrader.cli import main
        for extra, want in ((["--reset", "ETHUSDT"], {"SOLUSDT": False, "ETHUSDT": True}),
                            (["--reset=ETHUSDT,SOLUSDT", "--no-learn"], {"SOLUSDT": True, "ETHUSDT": True}),
                            (["--reset"], {"SOLUSDT": True, "ETHUSDT": True}),
                            ([], {"SOLUSDT": False, "ETHUSDT": False}),
                            (["--reset", "ETHUSDT", "--reset-memory"], {"SOLUSDT": False, "ETHUSDT": True})):
            with mock.patch("aitrader.futures_cli.run_multi", return_value=0) as rm, \
                    contextlib.redirect_stdout(io.StringIO()):
                main(["futures", "--model", "rule", "--symbol", "SOLUSDT,ETHUSDT"] + extra)
            command_for = rm.call_args[0][1]
            for sym, reset in want.items():
                child = command_for(sym)[3:]
                self.assertEqual(child.count("--reset"), int(reset), (extra, sym, child))
                self.assertNotIn("ETHUSDT,SOLUSDT", " ".join(child))
                # 子程式的參數本身要能通過檢查(之前把 --reset ETHUSDT 原樣傳給 SOL,SOL 啟動即失敗)
                from aitrader.futures_cli import _resolve_reset, _symbols
                import argparse
                ns = argparse.Namespace(symbol=sym, state=None, journal=None,
                                        reset="*" if "--reset" in child else None)
                _resolve_reset(ns, _symbols(ns))
                self.assertIs(ns.reset, reset)
                self.assertEqual("--reset-memory" in child, reset and "--reset-memory" in extra)  # 單獨出現會被判錯

    def test_spawn_failure_stops_started_children(self):
        def cmd(sym):
            if sym == "BADUSDT":
                return ["/nonexistent/python"]
            return [sys.executable, "-c", "import signal, time; signal.signal(signal.SIGTERM, lambda *a: exit(0)); time.sleep(30)"]
        with self.assertRaises(OSError):
            run_multi(["SOLUSDT", "BADUSDT"], cmd, io.StringIO())  # SOL 已啟動:要被收掉,不能留下孤兒程式

    def test_drop_option(self):
        from aitrader.multi import drop_option
        self.assertEqual(drop_option(["a", "--reset", "X", "--news"], "--reset"), ["a", "--news"])
        self.assertEqual(drop_option(["a", "--reset", "--news"], "--reset"), ["a", "--news"])
        self.assertEqual(drop_option(["--reset=X", "--reset-memory"], "--reset"), ["--reset-memory"])

    def test_multi_rejects_shared_state_and_bad_args_before_spawning(self):
        from aitrader.cli import main
        with mock.patch("aitrader.futures_cli.run_multi") as rm, contextlib.redirect_stderr(io.StringIO()) as err, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["futures", "--symbol", "SOLUSDT,ETHUSDT", "--state", "x.json"]), 2)
            self.assertEqual(main(["futures", "--symbol", "SOLUSDT,ETHUSDT", "--check-every", "1"]), 2)
        rm.assert_not_called()
        self.assertIn("--state", err.getvalue())

    def test_replay_multi_runs_sequentially_with_summary(self):
        from aitrader.cli import main
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as t, \
                mock.patch("aitrader.futures_cli.fetch_klines", return_value=candles(200, drift=0.004)), \
                mock.patch("aitrader.futures_cli.fetch_funding_history", return_value=[]), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            os.chdir(t)  # 預設日誌路徑是相對路徑:在暫存目錄執行,不弄髒專案
            try:
                self.assertEqual(main(["futures-replay", "--model", "rule", "--limit", "200",
                                       "--symbol", "SOLUSDT,ETHUSDT"]), 0)
                self.assertTrue(os.path.exists(os.path.join("journal", "futures_replay_ETHUSDT_rule.jsonl")))  # 依交易對分開
            finally:
                os.chdir(cwd)
        s = out.getvalue()
        self.assertIn("===== SOLUSDT", s)
        self.assertIn("===== 對照", s)
        self.assertEqual(s.count("報酬 "), 2 + 2)  # 每個交易對各自的結果 + 對照表兩列


if __name__ == "__main__":
    unittest.main()
