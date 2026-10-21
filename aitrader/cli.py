from __future__ import annotations

import argparse
import calendar
import json
import os
import sys
import time
from collections import Counter

from .agent import replay, step
from .config import gemini_settings, load_dotenv
from .llm import CachedModel, GeminiModel, LLMConfigError, LLMError, RuleModel
from .market import MIN_BARS, fetch_klines, fetch_price
from .portfolio import Portfolio
from .risk import RiskConfig, RiskManager
from .runtime import INTERVAL_SECS, StopFlag, handle_signals, wait
from .memory import pending_lessons_path
from .state import acquire_lock, archive_journal, load_json, save_json, try_lock

FETCH_BARS = 120  # 每次決策抓的 K 線數(>= MIN_BARS)
BAR_RETRIES = 3  # 模型暫時失敗時,同一根 K 線最多重試幾次(間隔 --check-every)


def _make_model(name: str, cache: bool = False, cache_dir: str = "cache"):
    if name == "rule":
        return RuleModel()
    key, model = gemini_settings()
    m = GeminiModel(key, model)
    return CachedModel(m, cache_dir) if cache else m


def _risk_cfg(a) -> RiskConfig:
    c = RiskConfig(a.max_position, a.min_confidence, a.min_trade, a.stop_loss, a.take_profit,
                   a.daily_loss_limit, a.max_trades)
    c.validate()
    return c


def _new_portfolio(a) -> Portfolio:
    return Portfolio(a.cash, a.fee, a.slippage_bps)


def _fmt_decision(rec: dict) -> str:
    d = rec.get("decision")
    head = (f"AI:{d['action']} 目標 {d['target_position_pct']:.0f}% 信心 {d['confidence']:.2f}|{d['reason']}"
            if d else ("強制規則" if rec.get("forced") else f"AI 錯誤:{rec.get('error')}"))
    t = rec.get("trade")
    trade = f"  成交 {t['side']} {t['qty']:.6f} @ {t['price']:.2f}" if t else ""
    return f"{head}\n    → {rec['action_taken']}{trade}  權益 {rec['equity_after']:.2f}  持倉 {rec['position_pct_after']:.1f}%"


# ------------------------------ once ------------------------------

def cmd_once(a) -> int:
    """用即時行情問一次模型,只顯示決策與風控結果,不改動任何狀態。"""
    rc = _risk_cfg(a)
    model = _make_model(a.model)
    candles = fetch_klines(a.symbol, a.interval, FETCH_BARS)
    price = fetch_price(a.symbol)
    pf, risk = _new_portfolio(a), RiskManager(rc)
    rec = step(model, a.symbol, a.interval, candles, price, time.time(), pf, risk)
    print(f"{a.symbol} 現價 {price}(空手、模擬資金 {a.cash:g},僅試算不存檔)")
    print(_fmt_decision(rec))
    if rec.get("meta"):
        m = rec["meta"]
        print(f"    模型 {m['model']}  延遲 {m['latency_s']}s  tokens 輸入 {m['tokens_in']} / 輸出 {m['tokens_out']}")
    return 1 if rec.get("error") else 0  # 模型沒給出有效決策:回傳非 0,方便腳本判斷


# ------------------------------ replay ------------------------------

def cmd_replay(a) -> int:
    rc = _risk_cfg(a)
    if a.every < 1:
        raise ValueError("--every 至少 1")
    if a.max_calls < 0:
        raise ValueError("--max-calls 不可為負")
    if a.limit < MIN_BARS + 2:
        raise ValueError(f"--limit 至少 {MIN_BARS + 2}")
    model = _make_model(a.model, cache=a.model == "gemini", cache_dir=a.cache_dir)
    calls = (lambda: model.misses) if isinstance(model, CachedModel) else (lambda: 0)
    candles = fetch_klines(a.symbol, a.interval, a.limit)
    est = (len(candles) - MIN_BARS) // a.every + 1
    print(f"{a.symbol} {a.interval} × {len(candles)} 根,每 {a.every} 根問一次模型(約 {est} 次)"
          + (f",實際呼叫上限 {a.max_calls} 次(快取命中不計)" if a.model == "gemini" else ""))
    if a.model == "gemini":
        print("⚠ 回放期間若早於模型訓練資料截止日,模型可能「記得」當時行情,績效會偏樂觀;請以模擬盤結果為準")
    journal = a.journal or os.path.join("journal", f"replay_{a.symbol}_{a.model}.jsonl")
    if os.path.exists(journal):
        os.remove(journal)  # 每次回放重寫日誌
    pf = _new_portfolio(a)
    r = replay(model, a.symbol, a.interval, candles, pf, RiskManager(rc), a.every,
               a.max_calls if a.model == "gemini" else None, journal, calls)
    print(f"價格 {r.start_price:.2f} -> {r.end_price:.2f}({r.bars} 根)" + ("  [已達呼叫上限,提前結束]" if r.stopped_early else ""))
    print(f"期末權益 {r.equity:.2f} / 本金 {a.cash:g}  報酬 {r.return_pct:+.2f}%  同期持有 {r.hold_return_pct:+.2f}%  最大回撤 {r.max_drawdown_pct:.2f}%")
    print(f"模型決策 {r.decisions} 次(錯誤 {r.model_errors})  強制出場 {r.forced_exits} 次  成交 {r.trades} 筆")
    if isinstance(model, CachedModel):
        print(f"Gemini 實際呼叫 {model.misses} 次、快取命中 {model.hits} 次")
    print(f"決策日誌:{journal}(可用 report 檢視)")
    return 0


# ------------------------------ paper ------------------------------

def _wait(seconds: float, stop: StopFlag) -> bool:
    return wait(seconds, stop, time)  # 以本模組的 time 計時,測試可替換成假時鐘


def cmd_paper(a) -> int:
    rc = _risk_cfg(a)
    if a.interval not in INTERVAL_SECS:
        raise ValueError(f"--interval 只支援 {', '.join(INTERVAL_SECS)}")
    if a.check_every < 5:
        raise ValueError("--check-every 至少 5 秒")
    model = _make_model(a.model)
    path = a.state or os.path.join("state", f"paper_{a.symbol}_{a.model}.json")
    journal = a.journal or os.path.join("journal", f"paper_{a.symbol}_{a.model}.jsonl")
    lock = acquire_lock(path)
    try:
        with handle_signals(StopFlag("paper")) as stop:
            return _paper_loop(a, rc, model, path, journal, stop)
    finally:
        if lock:
            lock.close()


def _paper_loop(a, rc: RiskConfig, model, path: str, journal: str, stop: StopFlag) -> int:
    if os.path.exists(path) and not a.reset:
        d = load_json(path)
        if d.get("symbol") != a.symbol or d.get("interval") != a.interval or d.get("model") != a.model:
            raise ValueError(f"狀態檔 {path} 的交易對 / 週期 / 模型與這次不同;加 --reset 重新開始或用 --state 指定其他檔案")
        pf, risk = Portfolio.from_dict(d["portfolio"]), RiskManager.from_dict(d["risk"])
        risk.cfg = rc  # 風控參數以這次啟動為準(可隨時調整),狀態沿用
        last_bar = d.get("last_bar_ts", 0)
        print(f"[paper] 接續 {path}:現金 {pf.cash:.2f}  持倉 {pf.qty:.6f}  成交 {len(pf.trades)} 筆")
    else:
        pf, risk, last_bar = _new_portfolio(a), RiskManager(rc), 0
        archived = archive_journal(journal)  # 新帳戶:舊日誌封存,report 只看目前帳戶
        if archived:
            print(f"[paper] 舊的決策日誌已封存為 {archived}")
        print(f"[paper] 新帳戶:{a.symbol} 模擬資金 {a.cash:g},模型 {getattr(model, 'name', a.model)}")
    print(f"[paper] 每根 {a.interval} K 線收盤後問模型,每 {a.check_every:g} 秒檢查停損;決策日誌 {journal};Ctrl+C 結束", flush=True)

    def save(last_bar_ts: float) -> None:
        save_json(path, {"symbol": a.symbol, "interval": a.interval, "model": a.model, "last_bar_ts": last_bar_ts,
                         "portfolio": pf.to_dict(), "risk": risk.to_dict()})

    save(last_bar)
    secs = INTERVAL_SECS[a.interval]
    if (time.time() // secs) * secs <= last_bar:  # 這根 K 線已決策過(接續時常見)
        nxt = last_bar + secs
        print(f"[paper] 目前這根 K 線已決策過,下一次問模型約在 {time.strftime('%H:%M', time.localtime(nxt))};"
              "期間每分鐘檢查停損,畫面安靜是正常的", flush=True)
    steps = 0
    attempts = 0  # 這根 K 線已失敗幾次
    try:
        while not stop.requested:
            now = time.time()
            bar_close = (now // secs) * secs  # 最近一根已收盤 K 線的收盤時刻
            try:
                if bar_close > last_bar:  # 有新的收盤 K 線:問模型
                    candles = fetch_klines(a.symbol, a.interval, FETCH_BARS)
                    price = fetch_price(a.symbol)
                    rec = step(model, a.symbol, a.interval, candles, price, now, pf, risk, True, journal)
                    print(f"[paper] {time.strftime('%m-%d %H:%M')} 現價 {price}  " + _fmt_decision(rec), flush=True)
                    if rec.get("error") and attempts + 1 < BAR_RETRIES:  # 暫時失敗:不跳過這根,稍後重試
                        attempts += 1
                        print(f"[paper] 這根 K 線第 {attempts} 次失敗,{a.check_every:g} 秒後重試", flush=True)
                    else:
                        if rec.get("error"):
                            print(f"[paper] 這根 K 線已失敗 {BAR_RETRIES} 次,放棄並等下一根", flush=True)
                        last_bar, attempts = bar_close, 0
                        steps += 1
                        nxt = bar_close + secs
                        print(f"[paper] 下一次問模型約在 {time.strftime('%H:%M', time.localtime(nxt))}", flush=True)
                else:  # 兩根之間:只檢查強制規則(停損等),不花模型費用
                    price = fetch_price(a.symbol)
                    rec = step(model, a.symbol, a.interval, [], price, now, pf, risk, False, journal)
                    if rec:
                        print(f"[paper] {time.strftime('%m-%d %H:%M')} 現價 {price}  " + _fmt_decision(rec), flush=True)
                save(last_bar)
            except LLMConfigError:
                raise
            except (OSError, ValueError, LLMError) as e:  # 網路瞬斷等:記錄後下一輪再試
                print(f"[paper] 本輪失敗,稍後重試:{e}", file=sys.stderr, flush=True)
            if a.max_steps and steps >= a.max_steps:
                break
            next_bar = (time.time() // secs + 1) * secs + 5  # 收盤後留 5 秒讓交易所結算 K 線
            if not _wait(min(a.check_every, max(1.0, next_bar - time.time())), stop):
                break
    except KeyboardInterrupt:  # 第二次 Ctrl+C:強制結束,不存可能不一致的狀態
        print("\n[paper] 強制結束,未存最後狀態", flush=True)
        return 0
    save(last_bar)
    try:
        price = fetch_price(a.symbol)
        print(f"\n[paper] 結束:權益 {pf.equity(price):.2f}  報酬 {(pf.equity(price) / pf.initial - 1) * 100:+.2f}%")
    except OSError:
        print("\n[paper] 結束(無法取得現價計算權益)")
    return 0


# ------------------------------ report ------------------------------

def cmd_report(a) -> int:
    if not os.path.exists(a.journal):
        raise ValueError(f"找不到決策日誌 {a.journal}")
    recs = []
    with open(a.journal) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except ValueError:
                    continue
    if not recs:
        raise ValueError("日誌是空的")
    if any(str(r.get("kind", "")).startswith("grid") for r in recs):
        from .grid_cli import report_grid
        return report_grid(a.journal, recs)
    if any(str(r.get("kind", "")).startswith("futures") for r in recs):
        from .currency import make_money
        from .futures_cli import report_futures
        return report_futures(a.journal, recs, make_money(a.currency, a.twd_rate))
    decided = [r for r in recs if r.get("decision")]
    actions = Counter(r["decision"]["action"] for r in decided)
    trades = [r["trade"] for r in recs if r.get("trade")]
    realized = sum(t["realized_pnl"] for t in trades if t["side"] == "sell")
    fees = sum(t["fee"] for t in trades)
    tok_in = sum((r.get("meta") or {}).get("tokens_in", 0) for r in recs)
    tok_out = sum((r.get("meta") or {}).get("tokens_out", 0) for r in recs)
    print(f"日誌 {a.journal}:{len(recs)} 筆紀錄({time.strftime('%m-%d %H:%M', time.localtime(recs[0]['ts']))} ~ "
          f"{time.strftime('%m-%d %H:%M', time.localtime(recs[-1]['ts']))})")
    errors = [r for r in recs if r.get("error")]
    detail = ("(" + "、".join(f"{k} {v}" for k, v in actions.most_common()) + ")") if actions else ""
    print(f"模型決策 {len(decided)} 次{detail};模型錯誤 {len(errors)} 次;強制出場 {sum(1 for r in recs if r.get('forced'))} 次")
    print(f"成交 {len(trades)} 筆,已實現損益 {realized:+.2f},手續費 {fees:.2f}")
    print(f"最後權益 {recs[-1]['equity_after']:.2f}  持倉 {recs[-1]['position_pct_after']:.1f}%")
    if tok_in or tok_out:
        print(f"tokens:輸入 {tok_in} / 輸出 {tok_out}")
    labels = {"low_confidence": "信心不足", "small_change": "調整幅度太小", "max_trades": "達每日交易上限",
              "halted": "單日虧損停機中"}
    blocked = Counter(labels[r["code"]] for r in decided if r.get("code") in labels)
    noop = sum(1 for r in decided if r.get("code") == "noop")
    if blocked:
        print("模型想交易但被風控擋下:" + "、".join(f"{k} {v} 次" for k, v in blocked.most_common()))
    if noop:
        print(f"已在目標持倉、不需交易:{noop} 次")
    forced = Counter(r.get("code") for r in recs if r.get("forced"))
    if forced:
        names = {"stop_loss": "停損", "take_profit": "停利", "daily_loss": "單日虧損上限", "halted": "停機清倉"}
        print("強制出場:" + "、".join(f"{names.get(k, k)} {v} 次" for k, v in forced.most_common()))
    if errors:
        print("最近的模型錯誤:")
        for r in errors[-3:]:
            print(f"  {time.strftime('%m-%d %H:%M', time.localtime(r['ts']))} {r['error'][:100]}")
    if not decided:
        print("還沒有任何有效的模型決策")
        return 0
    print("最近 5 次決策:")
    for r in decided[-5:]:
        d = r["decision"]
        print(f"  {time.strftime('%m-%d %H:%M', time.localtime(r['ts']))} {d['action']:4} {d['target_position_pct']:>5.1f}% "
              f"信心 {d['confidence']:.2f}  {d['reason'][:60]}")
    return 0


# ------------------------------ lessons ------------------------------

def _last_reflection(journal: str) -> dict:
    """日誌裡最後一次成功(有心得)的每日檢討。"""
    refl = []
    with open(journal) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if str(r.get("kind", "")).endswith("_reflection") and not r.get("error") and r.get("lessons"):
                refl.append(r)
    if not refl:
        raise ValueError(f"{journal} 裡沒有成功的每日檢討紀錄")
    return refl[-1]


def cmd_dashboard(a) -> int:
    from .dashboard import Dashboard, serve
    srv = serve(Dashboard(a.state_dir, a.journal_dir, a.currency, a.twd_rate), a.host, a.port)
    print(f"儀表板:http://localhost:{a.port}(Ctrl+C 結束;只讀取 {a.state_dir}/ 與 {a.journal_dir}/,不影響模擬盤)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


def cmd_lessons(a) -> int:
    """查看 AI 的學習狀況;--import 從日誌最後一次成功的每日檢討匯入心得。"""
    if not os.path.exists(a.state):
        raise ValueError(f"找不到狀態檔 {a.state}")
    d = load_json(a.state)
    agent = d.get("agent")
    if not isinstance(agent, dict) or "memory" not in agent:
        raise ValueError("這個狀態檔沒有學習紀錄(只有 grid / futures 模式有學習)")
    if a.import_from:
        last = _last_reflection(a.import_from)
        lock = try_lock(a.state)
        if lock is False:  # 模擬盤執行中:排入暫存檔,由執行中的程式套用(直接改狀態檔會被定時存檔蓋掉)
            save_json(pending_lessons_path(a.state), {"lessons": last["lessons"], "day": last["day"],
                                                     "source": a.import_from})
            print(f"模擬盤執行中:已排入 {len(last['lessons'])} 條心得({last['day']} 的檢討),"
                  "執行中的程式下一輪檢查(數十秒內)會自動套用,畫面會顯示「已套用匯入的心得」")
            for i, x in enumerate(last["lessons"], 1):
                print(f"  {i}. {x}")
            return 0
        try:
            mem = agent["memory"] or {}
            mem["lessons"], mem["lessons_updated"] = list(last["lessons"]), last["day"]
            agent["memory"] = mem
            save_json(a.state, d)
            print(f"已從 {a.import_from}({last['day']} 的檢討)匯入 {len(last['lessons'])} 條心得")
        finally:
            if lock:
                lock.close()
    mem = agent["memory"] or {}
    lessons = mem.get("lessons") or []
    print(f"{a.state}:{d.get('symbol')} {d.get('interval')} {d.get('model')}")
    print(f"心得 {len(lessons)} 條" + (f"(最近一次檢討:{mem.get('lessons_updated')})" if mem.get("lessons_updated") else ""))
    for i, x in enumerate(lessons, 1):
        print(f"  {i}. {x}")
    hist = mem.get("history") or []
    print(f"成績單 {len(hist)} 筆(每次決策附給 AI 參考)" + ("" if not hist else ":"))
    for e in hist:
        o = e.get("outcome", {})
        ev = f"  事件 {','.join(e['events'])}" if e.get("events") else ""
        try:  # 紀錄存 UTC,顯示本地時間
            when = time.strftime("%m-%d %H:%M", time.localtime(calendar.timegm(time.strptime(e["time_utc"], "%Y-%m-%d %H:%M"))))
        except (ValueError, KeyError):
            when = e.get("time_utc", "?")
        print(f"  {when} {e['action']:5} {e.get('params') or '':10} → {o.get('hours')}h 價格 {o.get('price_change_pct'):+}%"
              f" 權益 {o.get('equity_change_pct'):+}%{ev}")
    if os.path.exists(pending_lessons_path(a.state)):
        print("另有待套用的匯入心得(執行中的模擬盤下一輪會套用)")
    review = mem.get("review")
    print("待檢討:" + (f"{review['day_utc']}(下一次決策時進行)" if review else "無(UTC 換日後的第一次決策會檢討前一天)"))
    return 0


# ------------------------------ main ------------------------------

def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--symbol", default="BTCUSDT", help="Binance 交易對(預設 BTCUSDT)")
    p.add_argument("--interval", default="1h", help="K 線週期(預設 1h;paper 支援 " + "/".join(INTERVAL_SECS) + ")")
    p.add_argument("--model", choices=["gemini", "rule"], default="gemini",
                   help="gemini=Gemini API(需 GEMINI_API_KEY);rule=離線規則基準,不花錢(預設 gemini)")
    p.add_argument("--cash", type=float, default=10000, help="模擬資金(預設 10000)")
    p.add_argument("--fee", type=float, default=0.001, help="單邊手續費率(預設 0.001)")
    p.add_argument("--slippage-bps", type=float, default=5, help="滑價,基點(預設 5 = 0.05%%)")
    g = p.add_argument_group("風控")
    g.add_argument("--max-position", type=float, default=50, help="持倉上限,占總權益 %%(預設 50)")
    g.add_argument("--min-confidence", type=float, default=0.6, help="模型信心低於此值不交易(預設 0.6)")
    g.add_argument("--min-trade", type=float, default=5, help="持倉調整幅度小於此 %% 不交易(預設 5)")
    g.add_argument("--stop-loss", type=float, default=5, help="持倉虧損達此 %% 強制清倉(預設 5)")
    g.add_argument("--take-profit", type=float, default=0, help="持倉獲利達此 %% 強制清倉,0=不啟用(預設 0)")
    g.add_argument("--daily-loss-limit", type=float, default=3, help="權益單日下跌達此 %% 清倉並停到隔天(預設 3)")
    g.add_argument("--max-trades", type=int, default=6, help="每日最多成交筆數(預設 6)")


def main(argv=None) -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(prog="aitrader", description="用 Gemini 做交易決策的模擬交易(不下真單)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("once", help="用即時行情問一次模型(試算,不存檔);可用來確認 API 金鑰")
    _add_common(p)
    p.set_defaults(fn=cmd_once)

    p = sub.add_parser("paper", help="模擬盤:每根 K 線收盤後問模型,即時模擬成交")
    _add_common(p)
    p.add_argument("--check-every", type=float, default=60, help="兩根 K 線之間檢查停損的間隔秒數(預設 60)")
    p.add_argument("--state", help="狀態檔(預設 state/paper_<SYMBOL>_<MODEL>.json)")
    p.add_argument("--journal", help="決策日誌(預設 journal/paper_<SYMBOL>_<MODEL>.jsonl)")
    p.add_argument("--reset", action="store_true", help="忽略既有狀態,重新開始")
    p.add_argument("--max-steps", type=int, default=0, help="問模型幾次後結束,0=持續執行")
    p.set_defaults(fn=cmd_paper)

    p = sub.add_parser("replay", help="歷史回放(注意:模型可能記得過去行情,結果偏樂觀)")
    _add_common(p)
    p.add_argument("--limit", type=int, default=500, help="K 線根數(預設 500)")
    p.add_argument("--every", type=int, default=4, help="每幾根 K 線問一次模型(預設 4)")
    p.add_argument("--max-calls", type=int, default=100, help="Gemini 實際呼叫上限,快取命中不計(預設 100)")
    p.add_argument("--cache-dir", default="cache", help="模型回覆快取目錄(預設 cache)")
    p.add_argument("--journal", help="決策日誌(預設 journal/replay_<SYMBOL>_<MODEL>.jsonl)")
    p.set_defaults(fn=cmd_replay)

    from .grid_cli import add_parsers as add_grid_parsers  # 延遲載入,避免循環匯入
    add_grid_parsers(sub)
    from .futures_cli import add_parsers as add_futures_parsers
    add_futures_parsers(sub)

    p = sub.add_parser("lessons", help="查看 AI 的學習狀況(心得、成績單);可從日誌匯入心得")
    p.add_argument("state", help="狀態檔,如 state/futures_SOLUSDT_gemini.json")
    p.add_argument("--import", dest="import_from", metavar="JOURNAL",
                   help="從日誌最後一次成功的每日檢討匯入心得;模擬盤執行中也可以(會在下一輪自動套用)")
    p.set_defaults(fn=cmd_lessons)

    p = sub.add_parser("dashboard", help="本機儀表板:在瀏覽器看合約模擬盤的權益曲線、持倉、成交與 AI 決策")
    p.add_argument("--port", type=int, default=8765, help="埠號(預設 8765)")
    p.add_argument("--host", default="127.0.0.1", help="綁定位址(預設只限本機 127.0.0.1)")
    p.add_argument("--state-dir", default="state")
    p.add_argument("--journal-dir", default="journal")
    p.add_argument("--currency", choices=["USDT", "TWD"], default="TWD", help="金額幣別(預設 TWD)")
    p.add_argument("--twd-rate", type=float, help="USDT/TWD 匯率;不指定則從 MAX 取即時匯率")
    p.set_defaults(fn=cmd_dashboard)

    p = sub.add_parser("report", help="檢視決策日誌(一般、網格、合約模式皆可)")
    p.add_argument("journal", help="日誌路徑,如 journal/paper_BTCUSDT_gemini.jsonl")
    p.add_argument("--currency", choices=["USDT", "TWD"], default="USDT", help="合約報表的金額幣別(預設 USDT)")
    p.add_argument("--twd-rate", type=float, help="USDT/TWD 匯率;不指定則從 MAX 取即時匯率")
    p.set_defaults(fn=cmd_report)

    a = ap.parse_args(argv)
    a.argv = list(sys.argv[1:] if argv is None else argv)  # 多交易對時用來組子程式的參數
    try:
        return a.fn(a)
    except LLMConfigError as e:
        print(f"設定錯誤:{e}", file=sys.stderr)
        return 2
    except (ValueError, OSError, LLMError) as e:
        print(f"錯誤:{e}", file=sys.stderr)
        return 2
