"""網格模式的命令列:grid(即時模擬盤)、grid-replay(歷史回放),以及網格決策日誌的報表。"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from dataclasses import asdict
from typing import List

from .config import gemini_settings
from .grid_agent import GridAgent, GridRiskConfig, grid_replay
from .llm import (GRID_SCHEMA, CachedModel, GeminiModel, GridDecision, LLMConfigError, LLMError, RuleGridModel,
                  parse_grid_decision)
from .market import MIN_BARS, fetch_klines, fetch_price
from .memory import format_reflection
from .runtime import INTERVAL_SECS, StopFlag, handle_signals, wait
from .state import acquire_lock, archive_journal, old_lessons, append_jsonl, load_json, save_json

FETCH_BARS = 120
BAR_RETRIES = 3  # 模型暫時失敗時,同一根 K 線最多重試幾次
SAVE_EVERY = 60  # 沒有成交時,至少每幾秒存一次檔
RETRY_DELAY = 60  # 模型失敗後隔幾秒重試


def _make_model(name: str, cache: bool = False, cache_dir: str = "cache"):
    if name == "rule":
        return RuleGridModel()
    key, model = gemini_settings()
    m = GeminiModel(key, model, schema=GRID_SCHEMA, parser=parse_grid_decision)
    return CachedModel(m, cache_dir, GridDecision) if cache else m


def _risk_cfg(a) -> GridRiskConfig:
    c = GridRiskConfig(a.min_confidence, a.min_spacing, a.max_range, a.stop_loss, a.daily_loss_limit,
                       a.max_reconfigs, a.min_change)
    c.validate()
    return c


def _new_agent(a, rc: GridRiskConfig) -> GridAgent:
    return GridAgent(a.cash, rc, a.fee, a.slippage_bps / 10000, learn=not a.no_learn)


def _default_name(a, prefix: str) -> str:
    """預設檔名;--no-learn 加後綴,方便同時跑「有學習 / 無學習」兩個版本比較。"""
    return f"{prefix}_{a.symbol}_{a.model}" + ("_nolearn" if a.no_learn else "")



def _hm(ts: float) -> str:
    return time.strftime("%m-%d %H:%M", time.localtime(ts))


def _fmt_grid(g) -> str:
    if not g:
        return "無網格(空手)"
    return (f"{g['lower']:.2f} ~ {g['upper']:.2f} × {g['grids']} 格(格距 {g['spacing_pct']:.2f}%,"
            f"位置 {g['position_in_range']:.0%})")


def _fmt_decision(rec: dict) -> str:
    d = rec.get("decision")
    if d:
        rng = f" -{d['lower_pct']:g}%/+{d['upper_pct']:g}% {d['grids']} 格" if d["action"] == "SET" else ""
        head = f"AI:{d['action']}{rng} 信心 {d['confidence']:.2f}|{d['reason']}"
    elif rec.get("error"):
        head = f"AI 錯誤:{rec['error']}"
    else:
        head = "未問 AI"
    return f"{head}\n    → {rec['action_taken']}  權益 {rec['equity_after']:.2f}  網格 {_fmt_grid(rec['grid_after'])}"


def _fmt_fill(f) -> str:
    side = "買" if f.side == "buy" else "賣"
    where = f"第 {f.level} 線" if f.level >= 0 else f.note
    return f"{side} {f.qty:.6f} @ {f.price:.2f}({where})"


EVENT_NAMES = {"stop_loss": "觸發停損,清倉並停機到明天(UTC)", "daily_loss": "觸發單日虧損上限,清倉並停機到明天(UTC)"}


# ------------------------------ grid(模擬盤) ------------------------------

def cmd_grid(a) -> int:
    if a.reset_memory and not a.reset:
        raise ValueError("--reset-memory 需搭配 --reset 使用")
    rc = _risk_cfg(a)
    if a.interval not in INTERVAL_SECS:
        raise ValueError(f"--interval 只支援 {', '.join(INTERVAL_SECS)}")
    if a.tick < 1:
        raise ValueError("--tick 至少 1 秒")
    model = _make_model(a.model)
    path = a.state or os.path.join("state", _default_name(a, "grid") + ".json")
    journal = a.journal or os.path.join("journal", _default_name(a, "grid") + ".jsonl")
    lock = acquire_lock(path)
    try:
        with handle_signals(StopFlag("grid")) as stop:
            return _grid_loop(a, rc, model, path, journal, stop)
    finally:
        if lock:
            lock.close()


def _grid_loop(a, rc: GridRiskConfig, model, path: str, journal: str, stop: StopFlag) -> int:
    if os.path.exists(path) and not a.reset:
        d = load_json(path)
        if d.get("symbol") != a.symbol or d.get("interval") != a.interval or d.get("model") != a.model:
            raise ValueError(f"狀態檔 {path} 的交易對 / 週期 / 模型與這次不同;加 --reset 重新開始或用 --state 指定其他檔案")
        agent = GridAgent.from_dict(d["agent"], rc, not a.no_learn)  # 風控與學習開關以這次啟動為準,帳戶與網格沿用
        last_bar = d.get("last_bar_ts", 0)
        print(f"[grid] 接續 {path}:網格 {_fmt_grid(agent.grid_info(agent.engine.last_price, time.time()) if agent.engine else None)}")
    else:
        carried = old_lessons(path, a)  # 心得是對行情與策略的經驗,重開帳戶預設沿用
        agent, last_bar = _new_agent(a, rc), 0
        if carried:
            agent.memory.lessons = carried
            print(f"[grid] 沿用先前的 {len(carried)} 條心得(要連心得一起清除請加 --reset-memory)")
        archived = archive_journal(journal)  # 新帳戶:舊日誌封存,report 只看目前帳戶
        if archived:
            print(f"[grid] 舊的決策日誌已封存為 {archived}")
        print(f"[grid] 新帳戶:{a.symbol} 模擬資金 {a.cash:g},模型 {getattr(model, 'name', a.model)}")
    if not a.no_learn and getattr(model, "can_reflect", False):
        n = len(agent.memory.lessons)
        print(f"[grid] 學習:每次決策附上最近的結果;每天(UTC 換日)檢討一次並更新心得(目前 {n} 條)")
    print(f"[grid] 每 {a.tick:g} 秒更新價格撮合網格;每根 {a.interval} K 線收盤後問 AI 是否調整網格;"
          f"決策日誌 {journal};Ctrl+C 結束", flush=True)

    def save() -> None:
        save_json(path, {"symbol": a.symbol, "interval": a.interval, "model": a.model, "last_bar_ts": last_bar,
                         "agent": agent.to_dict()})

    save()
    secs = INTERVAL_SECS[a.interval]
    if (time.time() // secs) * secs <= last_bar:
        print(f"[grid] 這根 K 線已決策過,下一次問 AI 約在 {time.strftime('%H:%M', time.localtime(last_bar + secs))};"
              "期間網格照常撮合", flush=True)
    steps = attempts = 0
    retry_at = 0.0  # 模型失敗後,下一次重試的時刻(網格照常每 tick 撮合)
    last_tick = None if agent.engine is None else 0.0  # 接續既有網格:第一個 tick 視為跳空
    last_save = time.time()
    try:
        while not stop.requested:
            now = time.time()
            try:
                price = fetch_price(a.symbol)
                gap = last_tick is not None and now - last_tick > max(3 * a.tick, 30)
                fills, ev = agent.on_price(price, now, gap)
                last_tick = now
                changed = bool(fills or ev)
                for f in fills:
                    print(f"[grid] {_hm(now)} 成交 {_fmt_fill(f)}  權益 {agent.equity(price):.2f}", flush=True)
                if ev:
                    print(f"[grid] {_hm(now)} 現價 {price}:{EVENT_NAMES[ev]}", flush=True)
                if changed:
                    append_jsonl(journal, {"kind": "grid_fill", "ts": now, "price": price, "event": ev,
                                           "fills": [asdict(f) for f in fills],
                                           "equity_after": round(agent.equity(price), 4)})
                bar_close = (now // secs) * secs
                if bar_close > last_bar and now >= retry_at:  # 有新的收盤 K 線:問 AI
                    candles = fetch_klines(a.symbol, a.interval, FETCH_BARS)
                    rec = agent.decide(model, a.symbol, a.interval, candles, price, now, journal)
                    changed = True
                    if rec.get("reflection"):
                        print(f"[grid] {_hm(now)} " + format_reflection(rec["reflection"]), flush=True)
                    print(f"[grid] {_hm(now)} 現價 {price}  " + _fmt_decision(rec), flush=True)
                    if rec.get("error") and attempts + 1 < BAR_RETRIES:
                        attempts += 1
                        retry_at = now + RETRY_DELAY
                        print(f"[grid] 這根 K 線第 {attempts} 次失敗,{RETRY_DELAY} 秒後重試", flush=True)
                    else:
                        if rec.get("error"):
                            print(f"[grid] 這根 K 線已失敗 {BAR_RETRIES} 次,放棄並等下一根", flush=True)
                        last_bar, attempts = bar_close, 0
                        steps += 1
                        print(f"[grid] 下一次問 AI 約在 {time.strftime('%H:%M', time.localtime(bar_close + secs))}", flush=True)
                if changed or now - last_save >= SAVE_EVERY:
                    save()
                    last_save = now
            except LLMConfigError:
                raise
            except (OSError, ValueError, LLMError) as e:
                print(f"[grid] 本輪失敗,稍後重試:{e}", file=sys.stderr, flush=True)
            if a.max_steps and steps >= a.max_steps:
                break
            if not wait(a.tick, stop, time):
                break
    except KeyboardInterrupt:
        print("\n[grid] 強制結束,未存最後狀態", flush=True)
        return 0
    save()
    try:
        price = fetch_price(a.symbol)
        eq, t = agent.equity(price), agent.totals()
        print(f"\n[grid] 結束:權益 {eq:.2f}  報酬 {(eq / agent.initial - 1) * 100:+.2f}%  "
              f"網格套利 {t['cycles']} 次 / {t['grid_profit']:+.2f}  手續費 {t['fees']:.2f}")
    except OSError:
        print("\n[grid] 結束(無法取得現價計算權益)")
    return 0


# ------------------------------ grid-replay ------------------------------

def cmd_grid_replay(a) -> int:
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
    print(f"{a.symbol} {a.interval} × {len(candles)} 根,每 {a.every} 根問一次 AI(約 {est} 次)"
          + (f",實際呼叫上限 {a.max_calls} 次(快取命中不計)" if a.model == "gemini" else ""))
    print("K 線內部走勢未知,以保守路徑(陽線先探低、陰線先拉高)撮合網格;實際成交次數會比回放多")
    if a.model == "gemini":
        print("⚠ 回放期間若早於模型訓練資料截止日,模型可能「記得」當時行情,績效會偏樂觀;請以模擬盤結果為準")
    journal = a.journal or os.path.join("journal", _default_name(a, "grid_replay") + ".jsonl")
    if os.path.exists(journal):
        os.remove(journal)
    agent = _new_agent(a, rc)
    r = grid_replay(model, a.symbol, a.interval, candles, agent, a.every,
                    a.max_calls if a.model == "gemini" else None, journal, calls)
    print(f"價格 {r.start_price:.2f} -> {r.end_price:.2f}({r.bars} 根)" + ("  [已達呼叫上限,提前結束]" if r.stopped_early else ""))
    print(f"期末權益 {r.equity:.2f} / 本金 {a.cash:g}  報酬 {r.return_pct:+.2f}%  同期持有 {r.hold_return_pct:+.2f}%  最大回撤 {r.max_drawdown_pct:.2f}%")
    print(f"AI 決策 {r.decisions} 次(錯誤 {r.model_errors}):重設網格 {r.sets} 次、暫停 {r.pauses} 次;停損 / 單日上限 {r.stops} 次")
    print(f"網格成交 {r.grid_fills} 筆,套利 {r.cycles} 次 / {r.grid_profit:+.2f};手續費共 {r.fees:.2f}")
    if r.reflections:
        print(f"每日檢討 {r.reflections} 次;最後的心得:")
        for i, x in enumerate(agent.memory.lessons, 1):
            print(f"  {i}. {x}")
    if isinstance(model, CachedModel):
        print(f"Gemini 實際呼叫 {model.misses} 次、快取命中 {model.hits} 次")
    print(f"決策日誌:{journal}(可用 report 檢視)")
    return 0


# ------------------------------ report ------------------------------

def report_grid(path: str, recs: List[dict]) -> int:
    decs = [r for r in recs if r.get("kind") == "grid_decision"]
    refl = [r for r in recs if r.get("kind") == "grid_reflection"]
    fill_recs = [r for r in recs if r.get("kind") == "grid_fill"]
    print(f"網格日誌 {path}:{len(recs)} 筆紀錄({_hm(recs[0]['ts'])} ~ {_hm(recs[-1]['ts'])})")
    decided = [r for r in decs if r.get("decision")]
    actions = Counter(r["decision"]["action"] for r in decided)
    errors = [r for r in decs if r.get("error")]
    detail = ("(" + "、".join(f"{k} {v}" for k, v in actions.most_common()) + ")") if actions else ""
    print(f"AI 決策 {len(decided)} 次{detail};AI 錯誤 {len(errors)} 次")
    codes = Counter(r.get("code") for r in decs)
    names = {"set": "重設網格", "pause": "暫停清倉", "hold": "維持", "noop": "已是空手",
             "low_confidence": "信心不足", "max_reconfigs": "達每日重設上限", "too_narrow": "區間太窄",
             "small_change": "變動太小不重設", "halted": "停機中", "no_decision": "無有效決策"}
    print("結果:" + "、".join(f"{names.get(k, k)} {v} 次" for k, v in codes.most_common()))
    fills = [f for r in fill_recs for f in r["fills"]] + [f for r in decs for f in r.get("fills", [])]
    grid_fills = [f for f in fills if f["level"] >= 0]
    sells = sum(1 for f in grid_fills if f["side"] == "sell")
    fees = sum(f["fee"] for f in fills)
    events = Counter(r["event"] for r in fill_recs if r.get("event"))
    print(f"網格成交 {len(grid_fills)} 筆(買 {len(grid_fills) - sells} / 賣 {sells});手續費共 {fees:.2f}")
    if events:
        print("強制出場:" + "、".join(f"{ {'stop_loss': '停損', 'daily_loss': '單日虧損上限'}.get(k, k)} {v} 次"
                                   for k, v in events.items()))
    with_eq = [r for r in recs if "equity_after" in r]  # 檢討紀錄沒有權益
    if with_eq:
        print(f"最後權益 {with_eq[-1]['equity_after']:.2f}")
    if decs:
        print(f"目前網格:{_fmt_grid(decs[-1].get('grid_after'))}(以最後一次決策時為準)")
    tok_in = sum((r.get("meta") or {}).get("tokens_in", 0) for r in decs)
    tok_out = sum((r.get("meta") or {}).get("tokens_out", 0) for r in decs)
    if tok_in or tok_out:
        print(f"tokens:輸入 {tok_in} / 輸出 {tok_out}")
    if errors:
        print("最近的 AI 錯誤:")
        for r in errors[-3:]:
            print(f"  {_hm(r['ts'])} {r['error'][:100]}")
    if refl:
        ok = [r for r in refl if not r.get("error")]
        print(f"每日檢討 {len(ok)} 次(失敗 {len(refl) - len(ok)} 次)")
        if ok:
            print("  " + format_reflection(ok[-1]).replace("\n", "\n  "))
    if decided:
        print("最近 5 次決策:")
        for r in decided[-5:]:
            d = r["decision"]
            rng = f"-{d['lower_pct']:g}%/+{d['upper_pct']:g}% {d['grids']}格" if d["action"] == "SET" else ""
            print(f"  {_hm(r['ts'])} {d['action']:5} {rng:18} 信心 {d['confidence']:.2f}  {d['reason'][:60]}")
    else:
        print("還沒有任何有效的 AI 決策")
    return 0


# ------------------------------ 參數 ------------------------------

def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--symbol", default="BTCUSDT", help="Binance 交易對(預設 BTCUSDT)")
    p.add_argument("--interval", default="1h", help="多久問一次 AI(K 線週期,預設 1h;支援 " + "/".join(INTERVAL_SECS) + ")")
    p.add_argument("--model", choices=["gemini", "rule"], default="gemini",
                   help="gemini=Gemini API;rule=離線規則基準,不花錢(預設 gemini)")
    p.add_argument("--cash", type=float, default=10000, help="模擬資金(預設 10000)")
    p.add_argument("--fee", type=float, default=0.001, help="單邊手續費率(預設 0.001)")
    p.add_argument("--slippage-bps", type=float, default=5, help="市價單滑價,基點(預設 5;網格限價單不計)")
    g = p.add_argument_group("網格風控")
    g.add_argument("--min-confidence", type=float, default=0.6, help="重設 / 暫停的信心門檻(預設 0.6)")
    g.add_argument("--min-spacing", type=float, default=0.5, help="最小格距 %%,AI 要求的格數太多會自動減少(預設 0.5)")
    g.add_argument("--max-range", type=float, default=15, help="區間上下緣各自最多離現價幾 %%(預設 15)")
    g.add_argument("--stop-loss", type=float, default=5, help="區間下緣再往下幾 %% 停損,0=不設(預設 5)")
    g.add_argument("--daily-loss-limit", type=float, default=3, help="權益單日下跌達此 %% 清倉停機到隔天(預設 3)")
    g.add_argument("--max-reconfigs", type=int, default=4, help="每日最多重設 / 暫停網格次數(預設 4)")
    g.add_argument("--min-change", type=float, default=1.0, help="新舊區間上下緣都差不到此 %% 且格數相同就不重設(預設 1)")
    p.add_argument("--no-learn", action="store_true",
                   help="關閉學習(不附最近結果與心得、不做每日檢討);預設檔名加 _nolearn,可與學習版同時跑來比較")


def add_parsers(sub) -> None:
    p = sub.add_parser("grid", help="AI 調整網格的模擬盤:網格即時撮合,每根 K 線收盤問 AI 是否調整")
    _add_common(p)
    p.add_argument("--tick", type=float, default=5, help="更新價格、撮合網格的間隔秒數(預設 5)")
    p.add_argument("--state", help="狀態檔(預設 state/grid_<SYMBOL>_<MODEL>[_nolearn].json)")
    p.add_argument("--journal", help="決策日誌(預設 journal/grid_<SYMBOL>_<MODEL>[_nolearn].jsonl)")
    p.add_argument("--reset", action="store_true", help="忽略既有狀態,重新開始(預設沿用 AI 的心得)")
    p.add_argument("--reset-memory", action="store_true", help="搭配 --reset:連 AI 的心得也一起清除")
    p.add_argument("--max-steps", type=int, default=0, help="問 AI 幾次後結束,0=持續執行")
    p.set_defaults(fn=cmd_grid)

    p = sub.add_parser("grid-replay", help="AI 調整網格的歷史回放")
    _add_common(p)
    p.add_argument("--limit", type=int, default=500, help="K 線根數(預設 500)")
    p.add_argument("--every", type=int, default=4, help="每幾根 K 線問一次 AI(預設 4)")
    p.add_argument("--max-calls", type=int, default=100, help="Gemini 實際呼叫上限,快取命中不計(預設 100)")
    p.add_argument("--cache-dir", default="cache", help="模型回覆快取目錄(預設 cache)")
    p.add_argument("--journal", help="決策日誌(預設 journal/grid_replay_<SYMBOL>_<MODEL>[_nolearn].jsonl)")
    p.set_defaults(fn=cmd_grid_replay)
