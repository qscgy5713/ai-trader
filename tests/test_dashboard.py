import json
import os
import tempfile
import threading
import unittest
import urllib.request
from unittest import mock

from aitrader.dashboard import Dashboard, account_view, discover, serve
from aitrader.futures_agent import FuturesAgent, FuturesRiskConfig
from aitrader.state import acquire_lock, save_json
from tests.test_aitrader import candles
from tests.test_futures import TS, D, SeqModel, acct


def _make_account(t: str, name: str, decisions) -> str:
    """建立一個有成交紀錄的帳戶:回傳狀態檔路徑。"""
    agent = FuturesAgent(acct(), FuturesRiskConfig())
    j = os.path.join(t, "journal", name + ".jsonl")
    ts = TS
    for d, price in decisions:
        agent.decide(SeqModel(d), "SOLUSDT", "1h", candles(), price, ts, journal=j)
        ts += 3600
    path = os.path.join(t, "state", name + ".json")
    save_json(path, {"symbol": "SOLUSDT", "interval": "1h", "model": "gemini", "agent": agent.to_dict(), "saved_ts": ts})
    return path


class DashboardTest(unittest.TestCase):
    def test_discover_accounts(self):
        with tempfile.TemporaryDirectory() as t:
            for n in ("futures_SOLUSDT_gemini.json", "futures_SOLUSDT_gemini_balanced.json",
                      "futures_SUIUSDT_rule_nolearn.json", "futures_replay_SOLUSDT_rule.json",
                      "futures_SOLUSDT_gemini.json.lock", "news_SOLUSDT.json", "grid_BTCUSDT_gemini.json"):
                open(os.path.join(t, n), "w").close()
            got = {(a["id"], a["style"], a["learn"]) for a in discover(t)}
        self.assertEqual(got, {("futures_SOLUSDT_gemini", "auto", True), ("futures_SOLUSDT_gemini_balanced", "balanced", True),
                               ("futures_SUIUSDT_rule_nolearn", "auto", False)})  # 不含回放、鎖、新聞、網格

    def test_account_view(self):
        with tempfile.TemporaryDirectory() as t:
            path = _make_account(t, "futures_SOLUSDT_gemini", [(D("LONG", 1), 100), (D("FLAT"), 110), (D("LONG", 1), 110)])
            acc = discover(os.path.join(t, "state"))[0]
            v = account_view(acc, os.path.join(t, "journal"), 121.0, TS + 4 * 3600)
            self.assertFalse(v["running"])  # 沒有被鎖住
            self.assertEqual((v["trade_count"], v["closes"], v["wins"]), (3, 1, 1))
            self.assertIsNone(v["position"]["upnl"])  # 已停止:不用即時價格
            lock = acquire_lock(path)  # 模擬執行中
            try:
                v = account_view(acc, os.path.join(t, "journal"), 121.0, TS + 4 * 3600)
            finally:
                lock.close()
            self.assertTrue(v["running"])
            self.assertGreater(v["position"]["upnl"], 0)
            self.assertEqual(v["curve"][-1][0], TS + 4 * 3600)  # 執行中:曲線最後一點是即時權益
            self.assertGreater(v["return_pct"], 10)
            self.assertEqual(v["usage_24h"]["decide"][0], 3)  # 近 24 小時的決策
            later = account_view(acc, os.path.join(t, "journal"), 121.0, TS + 30 * 3600)
            self.assertEqual(later["usage_24h"]["decide"][0], 0)  # 超過 24 小時就不算
            self.assertGreater(v["position"]["stop_distance_pct"], 0)

    def test_server_endpoints(self):
        with tempfile.TemporaryDirectory() as t:
            _make_account(t, "futures_SOLUSDT_gemini", [(D("LONG", 1), 100)])
            dash = Dashboard(os.path.join(t, "state"), os.path.join(t, "journal"), "TWD", twd_rate=32.0)
            with mock.patch("aitrader.dashboard.fetch_premium", return_value=(105.0, 0.0, 0)), \
                    mock.patch("aitrader.market._get", return_value=[[i * 3600_000, "1", "2", "0.5", "1.5", "9", 0] for i in range(10)]):
                srv = serve(dash, port=0)
                threading.Thread(target=srv.serve_forever, daemon=True).start()
                base = f"http://127.0.0.1:{srv.server_address[1]}"
                try:
                    d = json.load(urllib.request.urlopen(base + "/api/data"))
                    page = urllib.request.urlopen(base + "/").read().decode()
                    k = json.load(urllib.request.urlopen(base + "/api/klines?symbol=SOLUSDT&interval=5m"))
                    lv = json.load(urllib.request.urlopen(base + "/api/live?q=SOLUSDT:1m,SUIUSDT:1h"))
                    for bad in ("/../state/x.json", "/api/klines?symbol=../x&interval=1h", "/api/klines?symbol=SOLUSDT&interval=7m",
                                "/api/live?q=sol;rm:1m"):
                        with self.assertRaises(urllib.error.HTTPError, msg=bad):
                            urllib.request.urlopen(base + bad)
                finally:
                    srv.shutdown()
                    srv.server_close()
        self.assertEqual(d["rate"], 32.0)
        self.assertEqual(len(d["accounts"]), 1)
        self.assertEqual(d["prices"]["SOLUSDT"]["mark"], 105.0)
        self.assertEqual((k["secs"], len(k["klines"]), k["klines"][-1][4]), (300, 10, 1.5))
        self.assertEqual(set(lv["items"]), {"SOLUSDT:1m", "SUIUSDT:1h"})
        self.assertEqual(lv["items"]["SOLUSDT:1m"]["mark"], 105.0)
        self.assertIn('id="equity"', page)

    def test_futures_dashboard_flag_not_passed_to_children(self):
        import contextlib
        import io
        from aitrader.cli import main
        with mock.patch("aitrader.futures_cli.run_multi", return_value=0) as rm, \
                mock.patch("aitrader.dashboard.start_background") as sb, contextlib.redirect_stdout(io.StringIO()):
            main(["futures", "--model", "rule", "--symbol", "SOLUSDT,SUIUSDT", "--dashboard"])
        sb.assert_called_once()
        self.assertEqual(sb.call_args[0][0], 8765)
        for s in ("SOLUSDT", "SUIUSDT"):
            self.assertNotIn("--dashboard", rm.call_args[0][1](s))  # 只由主程式開一次

    def test_background_port_in_use_does_not_raise(self):
        import contextlib
        import io
        from aitrader.dashboard import start_background
        srv = serve(Dashboard(), port=0)
        try:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertIsNone(start_background(srv.server_address[1], "USDT", None))
        finally:
            srv.server_close()
        self.assertIn("已被使用", out.getvalue())


if __name__ == "__main__":
    unittest.main()
