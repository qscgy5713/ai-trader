"""合約模式的命令列:futures(即時模擬盤)、futures-replay(歷史回放),以及合約日誌的報表。只模擬,不下真單。"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from typing import List

from .config import gemini_settings
from .currency import Money, make_money
from .futures import FuturesAccount
from .futures_agent import FuturesAgent, FuturesRiskConfig, futures_replay
from .llm import (FUTURES_SCHEMA, CachedModel, FuturesDecision, GeminiModel, LLMConfigError, LLMError,
                  RuleFuturesModel, parse_futures_decision)
from .market import MIN_BARS, fetch_funding_history, fetch_klines, fetch_premium
from .memory import format_reflection
from .runtime import INTERVAL_SECS, StopFlag, handle_signals, parse_duration, wait
from .state import acquire_lock, archive_journal, load_json, save_json

FETCH_BARS = 120
BAR_RETRIES = 3
RETRY_DELAY = 60
STYLE_MIN_CONFIDENCE = {"conservative": 0.6, "balanced": 0.5, "aggressive": 0.45}  # 未指定 --min-confidence 時依風格
WARN = "⚠ 合約有槓桿與強平風險;這是模擬盤,不下真單。模擬結果不代表實盤(實盤另有深度、插針、維持保證金級距等差異)"


def _make_model(name: str, cache: bool = False, cache_dir: str = "cache"):
    if name == "rule":
        return RuleFuturesModel()
    key, model = gemini_settings()
    m = GeminiModel(key, model, schema=FUTURES_SCHEMA, parser=parse_futures_decision)
    return CachedModel(m, cache_dir, FuturesDecision) if cache else m


def _risk_cfg(a, money: Money) -> FuturesRiskConfig:
    if a.daily_loss_amount < 0:
        raise ValueError("--daily-loss-amount 不可為負")
    min_conf = a.min_confidence if a.min_confidence is not None else STYLE_MIN_CONFIDENCE[a.style]
    c = FuturesRiskConfig(a.max_leverage, min_conf, a.min_change, a.stop_loss, a.daily_loss_limit,
                          a.max_trades, money.to_usdt(a.daily_loss_amount), a.risk_per_trade, a.min_stop_distance)
    c.validate()
    return c


def _new_agent(a, rc: FuturesRiskConfig, money: Money) -> FuturesAgent:
    return FuturesAgent(FuturesAccount(money.to_usdt(a.cash), a.fee, a.slippage_bps), rc, learn=not a.no_learn,
                        style=a.style)


def _default_name(a, prefix: str) -> str:
    """預設檔名;--no-learn 加後綴,方便同時跑「有學習 / 無學習」兩個版本比較。"""
    return f"{prefix}_{a.symbol}_{a.model}" + ("_nolearn" if a.no_learn else "")


def _budget_warning(rc: FuturesRiskConfig, equity: float, money: Money) -> str:
    """--daily-loss-amount 比 % 上限寬時(會被 % 蓋過),回傳提醒文字;否則空字串。"""
    if not rc.daily_loss_amount:
        return ""
    by_pct = equity * rc.daily_loss_limit_pct / 100
    if rc.daily_loss_amount <= by_pct:
        return ""
    need = rc.daily_loss_amount / equity * 100
    return (f"⚠ 每日金額上限 {money.fmt(rc.daily_loss_amount)} 比 --daily-loss-limit {rc.daily_loss_limit_pct:g}%"
            f"({money.fmt(by_pct)})寬,實際以 {money.fmt(by_pct)} 為準(先到為準);"
            f"要用金額上限,請加 --daily-loss-limit {min(99, int(need) + 1)}")


def _risk_line(a, rc: FuturesRiskConfig, money: Money) -> str:
    amt = f"、或虧 {money.fmt(rc.daily_loss_amount)}" if rc.daily_loss_amount else ""
    style = {"conservative": "保守", "balanced": "平衡偏積極", "aggressive": "積極"}[a.style]
    return (f"風格 {style}(信心門檻 {rc.min_confidence:g});槓桿上限 {rc.max_leverage:g} 倍;單日虧損上限 {rc.daily_loss_limit_pct:g}%{amt}(先到為準);"
            f"每筆最多用掉剩餘額度 {rc.risk_per_trade_pct:g}%")


def _hm(ts: float) -> str:
    return time.strftime("%m-%d %H:%M", time.localtime(ts))


def _fmt_pos(agent: FuturesAgent, price: float, money: Money) -> str:
    i = agent.position_info(price)
    tail = f"  權益 {money.fmt(agent.acct.equity(price))}  今日額度剩 {money.fmt(agent.budget_left(price))}"
    if i["side"] == "FLAT":
        return "空手" + tail
    side = "多" if i["side"] == "LONG" else "空"
    liq = f"  強平價 {i['liquidation_price']:.2f}" if i.get("liquidation_price") else ""
    stop = f"  停損價 {i['stop_loss_price']:.2f}" if i.get("stop_loss_price") else ""
    return (f"{side} {i['leverage']:.2f} 倍 @ {i['entry_price']:.2f}  未實現 {i['unrealized_pnl_pct_of_wallet']:+.2f}%"
            f"{stop}{liq}" + tail)


def _fmt_summary(agent: FuturesAgent, price: float, money: Money) -> str:
    """累計成績(一行)。"""
    a = agent.acct
    eq = a.equity(price)
    return (f"累計:成交 {a.trade_count} 筆  已實現 {money.fmt(a.realized_total, sign=True)}  "
            f"手續費 {money.fmt(a.fees_paid)}  資金費淨支付 {money.fmt(a.funding_paid, sign=True)}  "
            f"報酬 {(eq / a.initial - 1) * 100:+.2f}%")


def _fmt_trades(agent: FuturesAgent, money: Money, n: int = 5) -> str:
    """最近 n 筆成交(多行);沒有則回空字串。"""
    ts = agent.acct.trades[-n:]
    if not ts:
        return ""
    lines = [f"[futures] 最近 {len(ts)} 筆成交:"]
    for t in ts:
        side = "買" if t.side == "buy" else "賣"
        pnl = f"  實現 {money.fmt(t.realized_pnl, sign=True)}" if t.realized_pnl else ""
        lines.append(f"  {_hm(t.ts)} {side} {t.qty:.6f} @ {t.price:,.2f}  手續費 {money.fmt(t.fee)}{pnl}  {t.reason[:40]}")
    return "\n".join(lines)


def _fmt_rec(rec: dict) -> str:
    d = rec.get("decision")
    if d:
        lev = f" {d['leverage']:g} 倍" if d["action"] in ("LONG", "SHORT") else ""
        head = f"AI:{d['action']}{lev} 信心 {d['confidence']:.2f}|{d['reason']}"
    elif rec.get("forced"):
        head = "強制規則"
    elif rec.get("error"):
        head = f"AI 錯誤:{rec['error']}"
    else:
        head = "未問 AI"
    t = rec.get("trade")
    trade = f"  成交 {t['side']} {t['qty']:.6f} @ {t['price']:.2f}" if t else ""
    return f"{head}\n    → {rec['action_taken']}{trade}"


# ------------------------------ futures(模擬盤) ------------------------------

def cmd_futures(a) -> int:
    a.status_secs = parse_duration(a.status_every)
    if 0 < a.status_secs < 5:
        raise ValueError("--status-every 最短 5 秒(設 0 關閉)")
    money = make_money(a.currency, a.twd_rate)
    rc = _risk_cfg(a, money)
    if a.interval not in INTERVAL_SECS:
        raise ValueError(f"--interval 只支援 {', '.join(INTERVAL_SECS)}")
    if a.check_every < 5:
        raise ValueError("--check-every 至少 5 秒")
    model = _make_model(a.model)
    path = a.state or os.path.join("state", _default_name(a, "futures") + ".json")
    journal = a.journal or os.path.join("journal", _default_name(a, "futures") + ".jsonl")
    lock = acquire_lock(path)
    try:
        with handle_signals(StopFlag("futures")) as stop:
            return _futures_loop(a, rc, model, path, journal, stop, money)
    finally:
        if lock:
            lock.close()


def _futures_loop(a, rc: FuturesRiskConfig, model, path: str, journal: str, stop: StopFlag, money: Money) -> int:
    print(WARN)
    print(f"[futures] {money.describe()}")
    if os.path.exists(path) and not a.reset:
        d = load_json(path)
        if d.get("symbol") != a.symbol or d.get("interval") != a.interval or d.get("model") != a.model:
            raise ValueError(f"狀態檔 {path} 的交易對 / 週期 / 模型與這次不同;加 --reset 重新開始或用 --state 指定其他檔案")
        agent = FuturesAgent.from_dict(d["agent"], rc, not a.no_learn, a.style)  # 風控、學習、風格以這次啟動為準
        last_bar, next_funding = d.get("last_bar_ts", 0), d.get("next_funding_ts", 0)
        print(f"[futures] 接續 {path}")
    else:
        agent, last_bar, next_funding = _new_agent(a, rc, money), 0, 0
        archived = archive_journal(journal)  # 新帳戶:舊日誌封存,report 只看目前帳戶
        if archived:
            print(f"[futures] 舊的決策日誌已封存為 {archived}")
        print(f"[futures] 新帳戶:{a.symbol} 模擬保證金 {money.fmt(agent.acct.initial)},"
              f"模型 {getattr(model, 'name', a.model)}")
    print(f"[futures] {_risk_line(a, rc, money)}")
    warn = _budget_warning(rc, agent.acct.wallet, money)
    if warn:
        print(f"[futures] {warn}")
    if agent.acct.trade_count:
        try:
            p0 = fetch_premium(a.symbol)[0]
            print(f"[futures] {_fmt_pos(agent, p0, money)}\n[futures] {_fmt_summary(agent, p0, money)}")
        except OSError:
            pass
        print(_fmt_trades(agent, money))
    if not a.no_learn and getattr(model, "can_reflect", False):
        print(f"[futures] 學習:每次決策附上最近的結果;每天(UTC 換日)檢討一次並更新心得(目前 {len(agent.memory.lessons)} 條)")
    print(f"[futures] 每根 {a.interval} K 線收盤後問 AI;每 {a.check_every:g} 秒檢查強平 / 停損與資金費;"
          f"決策日誌 {journal};Ctrl+C 結束", flush=True)

    def save() -> None:
        save_json(path, {"symbol": a.symbol, "interval": a.interval, "model": a.model, "last_bar_ts": last_bar,
                         "next_funding_ts": next_funding, "agent": agent.to_dict()})

    save()
    secs = INTERVAL_SECS[a.interval]
    steps = attempts = 0
    retry_at = 0.0
    rate = None
    last_status = time.time()
    try:
        while not stop.requested:
            now = time.time()
            try:
                price, new_rate, nft = fetch_premium(a.symbol)
                if next_funding and now >= next_funding and rate is not None:
                    # 已過結算時間:以結算前最後看到的預估費率收付(接續時錯過的結算無法得知費率,略過)
                    f = agent.funding(rate, price, next_funding, journal)
                    if f:
                        who = "支付" if f["paid"] > 0 else "收到"
                        print(f"[futures] {_hm(now)} 資金費結算 費率 {rate:+.4%}:{who} {money.fmt(abs(f['paid']))}",
                              flush=True)
                rate, next_funding = new_rate, nft
                r = agent.check(price, now, journal=journal)
                if r:
                    print(f"[futures] {_hm(now)} 標記價 {price}  " + _fmt_rec(r) + f"\n    {_fmt_pos(agent, price, money)}"
                          f"\n    {_fmt_summary(agent, price, money)}", flush=True)
                bar_close = (now // secs) * secs
                if bar_close > last_bar and now >= retry_at:
                    candles = fetch_klines(a.symbol, a.interval, FETCH_BARS, futures=True)
                    rec = agent.decide(model, a.symbol, a.interval, candles, price, now, rate, journal)
                    if rec.get("reflection"):
                        print(f"[futures] {_hm(now)} " + format_reflection(rec["reflection"]), flush=True)
                    print(f"[futures] {_hm(now)} 標記價 {price}  " + _fmt_rec(rec) + f"\n    {_fmt_pos(agent, price, money)}"
                          f"\n    {_fmt_summary(agent, price, money)}", flush=True)
                    last_status = now
                    if rec.get("error") and attempts + 1 < BAR_RETRIES:
                        attempts += 1
                        retry_at = now + RETRY_DELAY
                        print(f"[futures] 這根 K 線第 {attempts} 次失敗,{RETRY_DELAY} 秒後重試", flush=True)
                    else:
                        last_bar, attempts = bar_close, 0
                        steps += 1
                        print(f"[futures] 下一次問 AI 約在 {time.strftime('%H:%M', time.localtime(bar_close + secs))}", flush=True)
                    if rec["code"] == "bankrupt":
                        print("[futures] 帳戶已爆倉,結束。用 --reset 重新開始", flush=True)
                        break
                elif a.status_secs and not r and now - last_status >= a.status_secs - 0.5:
                    # 兩次決策之間的狀態列:價格、持倉、未實現損益、剩餘額度(只顯示,不交易)
                    nxt = time.strftime('%H:%M', time.localtime(last_bar + secs)) if last_bar else "-"
                    print(f"[futures] {time.strftime('%H:%M:%S', time.localtime(now))} 標記價 {price:,.2f}  "
                          f"{_fmt_pos(agent, price, money)}  下次問 AI {nxt}", flush=True)
                    last_status = now
                save()
            except LLMConfigError:
                raise
            except (OSError, ValueError, LLMError) as e:
                print(f"[futures] 本輪失敗,稍後重試:{e}", file=sys.stderr, flush=True)
            if a.max_steps and steps >= a.max_steps:
                break
            next_bar = (time.time() // secs + 1) * secs + 5
            gap = min(a.check_every, a.status_secs) if a.status_secs else a.check_every
            if not wait(min(gap, max(1.0, next_bar - time.time())), stop, time):
                break
    except KeyboardInterrupt:
        print("\n[futures] 強制結束,未存最後狀態", flush=True)
        return 0
    save()
    try:
        price = fetch_premium(a.symbol)[0]
        print(f"\n[futures] 結束:{_fmt_pos(agent, price, money)}\n[futures] {_fmt_summary(agent, price, money)}")
    except OSError:
        print("\n[futures] 結束(無法取得現價)")
    return 0


# ------------------------------ futures-replay ------------------------------

def cmd_futures_replay(a) -> int:
    money = make_money(a.currency, a.twd_rate)
    rc = _risk_cfg(a, money)
    if a.every < 1:
        raise ValueError("--every 至少 1")
    if a.max_calls < 0:
        raise ValueError("--max-calls 不可為負")
    if a.limit < MIN_BARS + 2:
        raise ValueError(f"--limit 至少 {MIN_BARS + 2}")
    model = _make_model(a.model, cache=a.model == "gemini", cache_dir=a.cache_dir)
    calls = (lambda: model.misses) if isinstance(model, CachedModel) else (lambda: 0)
    candles = fetch_klines(a.symbol, a.interval, a.limit, futures=True)
    funding = fetch_funding_history(a.symbol, candles[0].ts, candles[-1].ts + 1)
    est = (len(candles) - MIN_BARS) // a.every + 1
    print(WARN)
    print(f"{money.describe()}(回放全程用目前匯率換算);{_risk_line(a, rc, money)}")
    warn = _budget_warning(rc, money.to_usdt(a.cash), money)
    if warn:
        print(warn)
    print(f"{a.symbol} 永續 {a.interval} × {len(candles)} 根,資金費 {len(funding)} 次;每 {a.every} 根問一次 AI(約 {est} 次)"
          + (f",實際呼叫上限 {a.max_calls} 次(快取命中不計)" if a.model == "gemini" else ""))
    if a.model == "gemini":
        print("⚠ 回放期間若早於模型訓練資料截止日,模型可能「記得」當時行情,績效會偏樂觀;請以模擬盤結果為準")
    journal = a.journal or os.path.join("journal", _default_name(a, "futures_replay") + ".jsonl")
    if os.path.exists(journal):
        os.remove(journal)
    agent = _new_agent(a, rc, money)
    r = futures_replay(model, a.symbol, a.interval, candles, agent, funding, a.every,
                       a.max_calls if a.model == "gemini" else None, journal, calls)
    print(f"價格 {r.start_price:.2f} -> {r.end_price:.2f}({r.bars} 根)" + ("  [提前結束]" if r.stopped_early else ""))
    print(f"期末權益 {money.fmt(r.equity)} / 本金 {money.fmt(agent.acct.initial)}  報酬 {r.return_pct:+.2f}%"
          f"  同期持有 {r.hold_return_pct:+.2f}%  最大回撤 {r.max_drawdown_pct:.2f}%")
    print(f"AI 決策 {r.decisions} 次(錯誤 {r.model_errors})  成交 {r.trades} 筆;停損 {r.stop_losses} 次、"
          f"單日上限 {r.daily_halts} 次、強平 {r.liquidations} 次")
    print(f"手續費 {money.fmt(r.fees_paid)}  資金費淨支付 {money.fmt(r.funding_paid, sign=True)}")
    if r.reflections:
        print(f"每日檢討 {r.reflections} 次;最後的心得:")
        for i, x in enumerate(agent.memory.lessons, 1):
            print(f"  {i}. {x}")
    if isinstance(model, CachedModel):
        print(f"Gemini 實際呼叫 {model.misses} 次、快取命中 {model.hits} 次")
    print(f"決策日誌:{journal}(可用 report 檢視)")
    return 0


# ------------------------------ report ------------------------------

CODE_NAMES = {"execute": "執行", "hold": "維持", "noop": "已在目標", "low_confidence": "信心不足",
              "max_trades": "達每日交易上限", "halted": "停機中", "bankrupt": "爆倉", "no_decision": "無有效決策",
              "stop_loss": "停損", "daily_loss": "單日虧損上限", "liquidation": "強平",
              "budget_exhausted": "今日額度不足"}


def report_futures(path: str, recs: List[dict], money: Money = Money()) -> int:
    decs = [r for r in recs if r.get("kind") == "futures"]
    fund = [r for r in recs if r.get("kind") == "futures_funding"]
    refl = [r for r in recs if r.get("kind") == "futures_reflection"]
    print(f"合約日誌 {path}:{len(recs)} 筆紀錄({_hm(recs[0]['ts'])} ~ {_hm(recs[-1]['ts'])})")
    decided = [r for r in decs if r.get("decision")]
    actions = Counter(r["decision"]["action"] for r in decided)
    errors = [r for r in decs if r.get("error")]
    detail = ("(" + "、".join(f"{k} {v}" for k, v in actions.most_common()) + ")") if actions else ""
    print(f"AI 決策 {len(decided)} 次{detail};AI 錯誤 {len(errors)} 次")
    codes = Counter(r.get("code") for r in decs if not r.get("forced"))
    if codes:
        print("決策結果:" + "、".join(f"{CODE_NAMES.get(k, k)} {v} 次" for k, v in codes.most_common()))
    forced = Counter(r.get("code") for r in decs if r.get("forced"))
    if forced:
        print("強制出場:" + "、".join(f"{CODE_NAMES.get(k, k)} {v} 次" for k, v in forced.most_common()))
    trades = [r["trade"] for r in decs if r.get("trade")]
    pnl = sum(t["realized_pnl"] for t in trades)
    fees = sum(t["fee"] for t in trades)
    paid = sum(r["paid"] for r in fund)
    print(f"成交 {len(trades)} 筆,已實現損益(未扣費){money.fmt(pnl, sign=True)},手續費 {money.fmt(fees)},"
          f"資金費淨支付 {money.fmt(paid, sign=True)}")
    with_eq = [r for r in recs if "equity_after" in r and r.get("kind") == "futures"]
    if with_eq:
        last = with_eq[-1]
        print(f"最後權益 {money.fmt(last['equity_after'])}  曝險 {last.get('exposure_after', 0) / 100:+.2f} 倍")
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
            lev = f"{d['leverage']:g}x" if d["action"] in ("LONG", "SHORT") else ""
            print(f"  {_hm(r['ts'])} {d['action']:5} {lev:5} 信心 {d['confidence']:.2f}  {d['reason'][:60]}")
    else:
        print("還沒有任何有效的 AI 決策")
    return 0


# ------------------------------ 參數 ------------------------------

def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--symbol", default="BTCUSDT", help="Binance USDT 永續合約(預設 BTCUSDT)")
    p.add_argument("--interval", default="1h", help="多久問一次 AI(K 線週期,預設 1h;支援 " + "/".join(INTERVAL_SECS) + ")")
    p.add_argument("--model", choices=["gemini", "rule"], default="gemini",
                   help="gemini=Gemini API;rule=離線規則基準(均線 + MACD 順勢),不花錢(預設 gemini)")
    p.add_argument("--style", choices=["conservative", "balanced", "aggressive"], default="balanced",
                   help="交易風格:conservative 保守(趨勢明確才進場)/ balanced 平衡偏積極 / aggressive 積極;"
                        "只影響 AI 進場意願,風控不變(預設 balanced)")
    p.add_argument("--currency", choices=["USDT", "TWD"], default="USDT",
                   help="輸入與顯示的幣別;TWD 時 --cash、--daily-loss-amount 以台幣計(帳戶實際仍為 USDT,預設 USDT)")
    p.add_argument("--twd-rate", type=float, help="USDT/TWD 匯率;不指定則從 MAX 取即時匯率")
    p.add_argument("--cash", type=float, default=10000, help="模擬保證金,以 --currency 計(預設 10000)")
    p.add_argument("--fee", type=float, default=0.0005, help="單邊手續費率,合約吃單(預設 0.0005)")
    p.add_argument("--slippage-bps", type=float, default=2, help="滑價,基點(預設 2)")
    g = p.add_argument_group("合約風控")
    g.add_argument("--max-leverage", type=float, default=2, help="槓桿上限,AI 要求更高會被截斷(預設 2,最多 10)")
    g.add_argument("--min-confidence", type=float, default=None,
                   help="AI 信心低於此值不交易(預設依風格:保守 0.6 / 平衡 0.5 / 積極 0.45)")
    g.add_argument("--min-change", type=float, default=20, help="同方向調整的曝險差距小於此 %% 不交易,100 = 1 倍(預設 20)")
    g.add_argument("--stop-loss", type=float, default=3, help="持倉虧損達錢包此 %% 強制平倉(預設 3)")
    g.add_argument("--daily-loss-limit", type=float, default=5, help="權益單日下跌達此 %% 平倉並停到隔天(預設 5)")
    g.add_argument("--max-trades", type=int, default=6, help="每日最多成交筆數(不含強制出場,預設 6)")
    g.add_argument("--daily-loss-amount", type=float, default=0,
                   help="單日最多虧損金額,以 --currency 計;與 --daily-loss-limit 先到為準,0=不用(預設 0)")
    g.add_argument("--risk-per-trade", type=float, default=50,
                   help="每筆倉位停損時最多用掉當日剩餘額度的 %%;今天虧越多倉位越小(預設 50)")
    g.add_argument("--min-stop-distance", type=float, default=0.5,
                   help="停損距離(價格 %%)下限,用來決定可開槓桿上限(預設 0.5)")
    p.add_argument("--no-learn", action="store_true",
                   help="關閉學習(不附最近結果與心得、不做每日檢討);預設檔名加 _nolearn,可與學習版同時跑來比較")


def add_parsers(sub) -> None:
    p = sub.add_parser("futures", help="合約模擬盤:AI 決定做多 / 做空 / 空手與槓桿(不下真單)")
    _add_common(p)
    p.add_argument("--check-every", type=float, default=30, help="檢查強平 / 停損 / 資金費的間隔秒數(預設 30)")
    p.add_argument("--status-every", default="5m",
                   help="兩次決策之間顯示狀態列(價格、持倉、未實現損益、剩餘額度)的間隔,如 5s、5m、1h;0=不顯示(預設 5m)")
    p.add_argument("--state", help="狀態檔(預設 state/futures_<SYMBOL>_<MODEL>[_nolearn].json)")
    p.add_argument("--journal", help="決策日誌(預設 journal/futures_<SYMBOL>_<MODEL>[_nolearn].jsonl)")
    p.add_argument("--reset", action="store_true", help="忽略既有狀態,重新開始")
    p.add_argument("--max-steps", type=int, default=0, help="問 AI 幾次後結束,0=持續執行")
    p.set_defaults(fn=cmd_futures)

    p = sub.add_parser("futures-replay", help="合約歷史回放(含歷史資金費、強平與停損)")
    _add_common(p)
    p.add_argument("--limit", type=int, default=500, help="K 線根數(預設 500)")
    p.add_argument("--every", type=int, default=4, help="每幾根 K 線問一次 AI(預設 4)")
    p.add_argument("--max-calls", type=int, default=100, help="Gemini 實際呼叫上限,快取命中不計(預設 100)")
    p.add_argument("--cache-dir", default="cache", help="模型回覆快取目錄(預設 cache)")
    p.add_argument("--journal", help="決策日誌(預設 journal/futures_replay_<SYMBOL>_<MODEL>[_nolearn].jsonl)")
    p.set_defaults(fn=cmd_futures_replay)
