"""本機儀表板:讀取 state/ 與 journal/ 的合約模擬盤資料,在瀏覽器顯示權益曲線、持倉、成交與 AI 決策。

只讀不寫、只綁 127.0.0.1(預設),不會碰到 API 金鑰;行情(標記價格、K 線)直接向 Binance 公開 API 取得並短暫快取。
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from urllib.parse import parse_qs
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional, Tuple

from .futures_agent import FuturesAgent, FuturesRiskConfig
from .futures import DUST
from . import market
from .market import fetch_premium
from .runtime import INTERVAL_SECS
from .state import load_json, try_lock

STYLES = ("conservative", "balanced", "aggressive")
STATE_RE = re.compile(r"^futures_([A-Z0-9]+)_([a-z]+)((?:_[a-z]+)*)\.json$")
PRICE_TTL, KLINE_TTL, LIVE_TTL = 8, 20, 2  # 行情快取秒數
REFRESH_SECS = 15  # 帳戶資料;即時價格另由 /api/live 每 3 秒更新
KLINE_BARS = {"1m": 180, "5m": 288, "15m": 192, "1h": 120, "4h": 120}  # 各週期顯示幾根
SYMBOL_OK = re.compile(r"^[A-Z0-9]{2,20}$")
DAY = 86400


class _Cache:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.items: Dict[str, Tuple[float, object]] = {}

    def get(self, key: str, ttl: float, load):
        with self.lock:
            hit = self.items.get(key)
            if hit and time.time() - hit[0] < ttl:
                return hit[1]
        value = load()
        with self.lock:
            self.items[key] = (time.time(), value)
        return value


def discover(state_dir: str) -> List[dict]:
    """找出合約模擬盤帳戶(不含回放):futures_<SYMBOL>_<MODEL>[_<STYLE>][_nolearn].json。"""
    out = []
    if not os.path.isdir(state_dir):
        return out
    for name in sorted(os.listdir(state_dir)):
        m = STATE_RE.match(name)
        if not m:
            continue
        symbol, model, suffix = m.group(1), m.group(2), [x for x in m.group(3).split("_") if x]
        style = next((x for x in suffix if x in STYLES), "auto")
        out.append({"id": name[:-5], "symbol": symbol, "model": model, "style": style, "learn": "nolearn" not in suffix,
                    "state": os.path.join(state_dir, name)})
    return out


def _running(state_path: str) -> Optional[bool]:
    """模擬盤是否執行中(狀態檔被鎖住);不支援檔案鎖的平台回 None。"""
    fh = try_lock(state_path)
    if fh is False:
        return True
    if fh is None:
        return None
    fh.close()
    return False


def _read_journal(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def _max_drawdown(points: List[Tuple[float, float]]) -> float:
    peak, dd = 0.0, 0.0
    for _, eq in points:
        peak = max(peak, eq)
        if peak > 0:
            dd = max(dd, (1 - eq / peak) * 100)
    return round(dd, 2)


def account_view(acc: dict, journal_dir: str, mark: Optional[float], now: float) -> dict:
    """一個帳戶給儀表板的資料(金額單位 USDT,前端再換算)。"""
    d = load_json(acc["state"])
    agent = FuturesAgent.from_dict(d["agent"], FuturesRiskConfig())  # 追蹤停損距離以預設值估算
    a = agent.acct
    recs = _read_journal(os.path.join(journal_dir, acc["id"] + ".jsonl"))
    running = _running(acc["state"])
    flat = abs(a.qty) < DUST
    # 權益曲線:每筆有 equity_after 的紀錄;執行中再加上現在的即時權益
    points = [(r["ts"], r["equity_after"]) for r in recs if "equity_after" in r]
    start_ts = recs[0]["ts"] if recs else d.get("saved_ts") or now
    points = [(start_ts, a.initial)] + points
    price = mark if (mark and running) else None
    equity = a.equity(price) if price else (points[-1][1] if points else a.wallet)
    if price:
        points.append((now, equity))
    pos = None
    if not flat:
        p = price or a.entry
        stop, trail, liq = a.stop_price(agent.pos_stop_pct), agent._trail_price(), a.liq_price()
        pos = {"side": "LONG" if a.qty > 0 else "SHORT", "qty": a.qty, "entry": a.entry,
               "leverage": round(abs(a.qty) * p / equity, 2) if equity > 0 else None,
               "upnl": a.upnl(p) if price else None, "stop": stop, "trail": trail, "liq": liq,
               "stop_distance_pct": round(abs(p - stop) / p * 100, 2) if (stop and price) else None}
    trades = [dict(r["trade"], forced=bool(r.get("forced")), why=r.get("action_taken"))
              for r in recs if r.get("kind") == "futures" and r.get("trade")]
    closes = [t for t in trades if abs(t["realized_pnl"]) > 1e-12]
    decisions = [{"ts": r["ts"], "price": r.get("price"), "forced": bool(r.get("forced")), "code": r.get("code"),
                  "action": (r.get("decision") or {}).get("action"), "leverage": (r.get("decision") or {}).get("leverage"),
                  "confidence": (r.get("decision") or {}).get("confidence"),
                  "reason": (r.get("decision") or {}).get("reason") or r.get("action_taken") or r.get("error"),
                  "taken": r.get("action_taken"), "alert": r.get("alert")}
                 for r in recs if r.get("kind") == "futures"][-30:]
    news = next((r for r in reversed(recs) if r.get("kind") == "futures_news" and not r.get("error")), None)
    usage: Dict[str, List[int]] = {"decide": [0, 0, 0], "reflect": [0, 0, 0], "news": [0, 0, 0]}
    for r in recs:
        if r["ts"] < now - DAY:
            continue
        m, k = r.get("meta") or {}, r.get("kind")
        key = {"futures": "decide", "futures_reflection": "reflect", "futures_news": "news"}.get(k)
        if not key or (k == "futures" and r.get("forced")) or (key != "news" and not m and not r.get("error")):
            continue
        u = usage[key]
        u[0] += m.get("calls", 1) if key == "news" else 1
        u[1] += m.get("tokens_in", 0)
        u[2] += m.get("tokens_out", 0)
    mem = agent.memory
    return {
        "id": acc["id"], "symbol": acc["symbol"], "style": acc["style"], "learn": acc["learn"], "running": running,
        "interval": d.get("interval"), "initial": a.initial, "equity": equity, "wallet": a.wallet,
        "return_pct": round((equity / a.initial - 1) * 100, 3), "max_drawdown_pct": _max_drawdown(points),
        "realized": a.realized_total, "fees": a.fees_paid, "funding": a.funding_paid, "trade_count": a.trade_count,
        "wins": sum(1 for t in closes if t["realized_pnl"] > 0), "closes": len(closes),
        "position": pos, "curve": points, "trades": trades[-200:], "decisions": decisions,
        "lessons": list(mem.lessons), "lessons_updated": mem.lessons_updated,
        "news": {k: news.get(k) for k in ("day", "summary", "sentiment", "events", "sources", "shared")} if news else None,
        "usage_24h": usage, "saved_ts": d.get("saved_ts"),
    }


class Dashboard:
    def __init__(self, state_dir: str = "state", journal_dir: str = "journal", currency: str = "USDT",
                 twd_rate: Optional[float] = None) -> None:
        self.state_dir, self.journal_dir = state_dir, journal_dir
        self.currency, self.fixed_rate = currency, twd_rate
        self.cache = _Cache()

    def _rate(self) -> Optional[float]:
        if self.currency != "TWD":
            return None
        if self.fixed_rate:
            return self.fixed_rate
        from .market import fetch_usdt_twd

        def load():
            try:
                return fetch_usdt_twd()
            except (OSError, ValueError, KeyError):
                return None
        return self.cache.get("usdt_twd", 300, load)

    def _mark(self, symbol: str) -> Optional[float]:
        def load():
            try:
                return fetch_premium(symbol)[0]
            except (OSError, ValueError, KeyError):
                return None
        return self.cache.get("mark:" + symbol, PRICE_TTL, load)

    def klines(self, symbol: str, interval: str, limit: Optional[int] = None, ttl: float = KLINE_TTL) -> List[list]:
        """含「正在形成」的最後一根(只供顯示;AI 決策另用 fetch_klines,只看已收盤的)。"""
        if not SYMBOL_OK.match(symbol) or interval not in KLINE_BARS:
            raise ValueError("symbol / interval 不合法")
        n = limit or KLINE_BARS[interval]

        def load():
            try:
                rows = market._get(f"{market.BINANCE_FUTURES}/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={n}")
                return [[k[0] / 1000] + [float(x) for x in k[1:6]] for k in rows]
            except (OSError, ValueError, KeyError, TypeError, IndexError):
                return []
        return self.cache.get(f"k:{symbol}:{interval}:{n}", ttl, load)

    def kline_data(self, symbol: str, interval: str) -> dict:
        rows = self.klines(symbol, interval)  # 先驗證(不合法丟 ValueError → 400)
        return {"symbol": symbol, "interval": interval, "secs": INTERVAL_SECS[interval], "klines": rows}

    def live(self, pairs: List[Tuple[str, str]]) -> dict:
        """即時:各交易對的標記價格與指定週期的最後兩根 K 線(網頁每 3 秒呼叫)。"""
        out = {}
        for sym, iv in pairs[:8]:
            def load(sym=sym):
                try:
                    return fetch_premium(sym)[0]
                except (OSError, ValueError, KeyError):
                    return None
            out[f"{sym}:{iv}"] = {"mark": self.cache.get("live:" + sym, LIVE_TTL, load),
                                  "klines": self.klines(sym, iv, 2, LIVE_TTL)}
        return {"now": time.time(), "items": out}

    def data(self) -> dict:
        now = time.time()
        accounts, errors = [], []
        for acc in discover(self.state_dir):
            try:
                accounts.append(account_view(acc, self.journal_dir, self._mark(acc["symbol"]), now))
            except (ValueError, OSError, KeyError) as e:
                errors.append(f"{acc['id']}:{e}")
        symbols = sorted({x["symbol"] for x in accounts})
        return {"now": now, "currency": self.currency, "rate": self._rate(), "accounts": accounts, "errors": errors,
                "prices": {s: {"mark": self._mark(s)} for s in symbols},
                "refresh_secs": REFRESH_SECS}


def _handler(dash: Dashboard):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            path, _, query = self.path.partition("?")
            q = {k: v[-1] for k, v in parse_qs(query).items()}
            js = "application/json; charset=utf-8"
            try:
                if path == "/api/data":
                    body, ctype = json.dumps(dash.data(), ensure_ascii=False).encode(), js
                elif path == "/api/klines":
                    body, ctype = json.dumps(dash.kline_data(q.get("symbol", ""), q.get("interval", ""))).encode(), js
                elif path == "/api/live":
                    pairs = [tuple(x.split(":", 1)) for x in q.get("q", "").split(",") if ":" in x]
                    for s, iv in pairs:
                        if not SYMBOL_OK.match(s) or iv not in KLINE_BARS:
                            raise ValueError("q 不合法")
                    body, ctype = json.dumps(dash.live(pairs)).encode(), js
                elif path in ("/", "/index.html"):
                    body, ctype = PAGE.encode(), "text/html; charset=utf-8"
                else:
                    self.send_error(404)
                    return
            except ValueError:
                self.send_error(400, "bad request")  # 狀態列只能用 latin-1
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):  # 不洗版
            pass
    return H


def serve(dash: Dashboard, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), _handler(dash))


def start_background(port: int, currency: str, twd_rate: Optional[float], tag: str = "futures") -> Optional[ThreadingHTTPServer]:
    """模擬盤順便開儀表板(背景執行緒)。port 已被占用(例如另一組模擬盤已經開了)時只提示,不影響交易。"""
    try:
        srv = serve(Dashboard(currency=currency, twd_rate=twd_rate), port=port)
    except OSError:
        print(f"[{tag}] 儀表板 port {port} 已被使用(可能另一組已開啟),直接打開 http://localhost:{port} 即可", flush=True)
        return None
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[{tag}] 儀表板:http://localhost:{port}", flush=True)
    return srv


PAGE = r"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI 合約模擬盤</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
<style>
:root{--bg:#0f1419;--panel:#171d24;--line:#26303b;--text:#e6edf3;--muted:#8b98a5;--up:#26a69a;--down:#ef5350;--accent:#58a6ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,BlinkMacSystemFont,"PingFang TC","Noto Sans TC",sans-serif}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:baseline;justify-content:space-between;padding:16px 20px;border-bottom:1px solid var(--line)}
h1{font-size:18px;margin:0}
h2{font-size:15px;margin:0 0 10px}
.muted{color:var(--muted)}
main{padding:16px 20px;display:grid;gap:16px;max-width:1500px;margin:0 auto}
.grid{display:grid;gap:12px;grid-template-columns:repeat(auto-fill,minmax(260px,1fr))}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}
.card .top{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
.big{font-size:22px;font-weight:600}
.up{color:var(--up)} .down{color:var(--down)}
.badge{font-size:12px;padding:1px 8px;border-radius:999px;border:1px solid var(--line);color:var(--muted)}
.badge.on{color:var(--up);border-color:var(--up)} .badge.off{color:var(--down);border-color:var(--down)}
.kv{display:grid;grid-template-columns:auto 1fr;gap:2px 10px;font-size:13px}
.kv span:nth-child(odd){color:var(--muted)}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:500}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.charts{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(420px,1fr))}
.chartbox{position:relative;height:300px}
.kbox{position:relative;height:380px}
.khead{display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:6px}
.khead h2{margin:0}
.livep{font-variant-numeric:tabular-nums;padding:0 4px;border-radius:4px}
.age{font-size:12px;font-weight:400}
@keyframes fu{from{background:rgba(38,166,154,.55)}to{background:transparent}}
@keyframes fd{from{background:rgba(239,83,80,.55)}to{background:transparent}}
.flash-up{animation:fu 1.2s ease-out;color:var(--up)} .flash-down{animation:fd 1.2s ease-out;color:var(--down)}
.legend{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12px;color:var(--muted);margin-top:6px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
.tabs button{background:none;border:1px solid var(--line);color:var(--muted);padding:3px 10px;border-radius:6px;cursor:pointer}
.tabs button.sel{color:var(--text);border-color:var(--accent)}
ul{margin:4px 0 0;padding-left:20px}
.err{color:var(--down)}
@media (max-width:600px){.charts{grid-template-columns:1fr}main,header{padding:12px}}
</style>
</head>
<body>
<header>
  <div><h1>AI 合約模擬盤</h1><div class="muted">只模擬,不下真單 · 帳戶每 15 秒、現價與 K 線每 3 秒更新</div></div>
  <div class="tabs"><span class="muted" id="updated"></span>
    <button data-unit="money" class="sel">金額</button><button data-unit="pct">%</button>
    <button id="stopped">含已停止</button></div>
</header>
<main>
  <section class="grid" id="totals"></section>
  <section class="grid" id="cards"></section>
  <section class="panel"><h2>權益曲線</h2><div class="chartbox"><canvas id="equity"></canvas></div></section>
  <section class="charts" id="prices"></section>
  <section class="panel"><h2>風格對照</h2><div id="compare"></div></section>
  <section class="panel"><h2>最近決策</h2><div id="decisions"></div></section>
  <section class="grid" id="notes"></section>
  <section class="panel"><h2>AI 用量(近 24 小時)</h2><div id="usage"></div></section>
  <div id="errors" class="err"></div>
</main>
<script>
const COLORS=["#58a6ff","#f0883e","#a371f7","#3fb950","#db61a2","#d29922","#79c0ff","#ff7b72"];
let RAW=null, D=null, unit="money", showStopped=false, charts={};
const el=(tag,attrs={},...kids)=>{const e=document.createElement(tag);for(const[k,v]of Object.entries(attrs)){if(k==="class")e.className=v;else e.setAttribute(k,v)}for(const c of kids)e.append(c instanceof Node?c:document.createTextNode(c??""));return e};
const money=v=>{if(v==null)return"-";const r=D.rate||1;const cur=D.rate?"元":"USDT";return Math.round(v*r).toLocaleString()+" "+cur};
const smoney=v=>(v>=0?"+":"")+money(v);
const pct=v=>v==null?"-":(v>=0?"+":"")+v.toFixed(2)+"%";
const cls=v=>v>0?"up":v<0?"down":"";
const t=ts=>new Date(ts*1000).toLocaleString("zh-TW",{month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit",hour12:false});
const label=a=>a.symbol.replace("USDT","")+" · "+a.style+(a.learn?"":" · 無學習");
const color=i=>COLORS[i%COLORS.length];
const fmtp=v=>v==null?"-":(v>=100?v.toFixed(2):v>=1?v.toFixed(3):v.toFixed(4));

// 合計:執行中的帳戶為主;另列「含已停止」(例如停掉的 ETH 的已實現 / 凍結損益)。
// 已停止且從未成交的帳戶(早期測試留下的)不算,否則本金會稀釋報酬率。
const counted=a=>!(a.running===false&&!a.trade_count);
function sumUp(list){const init=list.reduce((x,a)=>x+a.initial,0),eq=list.reduce((x,a)=>x+a.equity,0);return{init,pnl:eq-init,r:init?(eq/init-1)*100:0,n:list.length}}
function totals(){
  const box=document.getElementById("totals");box.replaceChildren();
  const styles=[...new Set(RAW.accounts.map(a=>a.style))];
  for(const s of styles){
    const all=RAW.accounts.filter(a=>a.style===s&&counted(a)),run=all.filter(a=>a.running!==false);
    const R=sumUp(run.length?run:all),A=sumUp(all),main=unit==="pct"?pct(R.r):smoney(R.pnl);
    const p=el("div",{class:"panel"},el("div",{class:"muted"},s+" 合計 · 執行中 "+R.n+" 個帳戶"),
      el("div",{class:"big "+cls(R.r)},main),el("div",{class:"muted"},(unit==="pct"?smoney(R.pnl):pct(R.r))+" · 本金 "+money(R.init)));
    if(A.n>R.n)p.append(el("div",{class:"muted",style:"margin-top:4px"},"含已停止("+A.n+" 個帳戶):",
      el("span",{class:cls(A.pnl)},smoney(A.pnl)+"("+pct(A.r)+")"),",本金 "+money(A.init)));
    box.append(p);
  }
}
function cards(){
  const box=document.getElementById("cards");box.replaceChildren();
  for(const a of D.accounts){
    const p=a.position,st=a.running===true?["on","執行中"]:a.running===false?["off","已停止"]:["","?"];
    const kv=el("div",{class:"kv"});
    const row=(k,v,c="")=>{kv.append(el("span",{},k),el("span",{class:c},v))};
    if(p){row("持倉",(p.side==="LONG"?"多 ":"空 ")+(p.leverage??"?")+" 倍 @ "+fmtp(p.entry),p.side==="LONG"?"up":"down");
      row("未實現",p.upnl==null?"-":smoney(p.upnl),cls(p.upnl));
      row("停損價",fmtp(p.stop)+(p.stop_distance_pct!=null?"(距 "+p.stop_distance_pct+"%)":""));
      if(p.trail)row("追蹤停利",fmtp(p.trail));}
    else row("持倉","空手");
    row("成交",a.trade_count+" 筆,平倉勝率 "+(a.closes?Math.round(a.wins/a.closes*100)+"%":"-"));
    row("手續費",money(a.fees));row("最大回撤",a.max_drawdown_pct+"%");
    box.append(el("div",{class:"panel card"},el("div",{class:"top"},el("b",{},label(a)),el("span",{class:"badge "+st[0]},st[1])),
      el("div",{class:"big "+cls(a.return_pct)},unit==="pct"?pct(a.return_pct):smoney(a.equity-a.initial)),
      el("div",{class:"muted"},(unit==="pct"?smoney(a.equity-a.initial):pct(a.return_pct))+" · 權益 "+money(a.equity)),kv));
  }
}
function lineChart(id,datasets,yfmt,xmin,xmax){
  const ctx=document.getElementById(id);if(charts[id])charts[id].destroy();
  charts[id]=new Chart(ctx,{type:"line",data:{datasets},options:{animation:false,maintainAspectRatio:false,parsing:false,
    interaction:{mode:"nearest",intersect:false},plugins:{legend:{labels:{color:"#8b98a5"}},
      tooltip:{callbacks:{title:i=>t(i[0].parsed.x),label:i=>i.dataset.label+":"+(i.raw.tip||yfmt(i.parsed.y))}}},
    scales:{x:{type:"linear",min:xmin,max:xmax,ticks:{color:"#8b98a5",callback:v=>t(v),maxTicksLimit:8},grid:{color:"#26303b"}},
            y:{ticks:{color:"#8b98a5",callback:yfmt},grid:{color:"#26303b"}}}}});
}
function equityChart(){
  const ds=D.accounts.map((a,i)=>({label:label(a),borderColor:color(i),backgroundColor:color(i),pointRadius:0,borderWidth:2,stepped:true,
    data:a.curve.map(([x,y])=>({x,y:unit==="pct"?(y/a.initial-1)*100:(y-a.initial)*(D.rate||1)}))}));
  const xs=ds.flatMap(d=>d.data.map(p=>p.x));  // 時間軸只到有資料的範圍(不自動往外延伸)
  lineChart("equity",ds,unit==="pct"?v=>v.toFixed(1)+"%":v=>Math.round(v).toLocaleString(),Math.min(...xs),Math.max(...xs,D.now));
}
// K 線(TradingView Lightweight Charts):可切換週期;成交點標在所屬 K 線上,持倉的進場 / 停損 / 追蹤停利畫成水平線
const tz=-new Date().getTimezoneOffset()*60;  // 圖表庫以 UTC 顯示:時間先加上本地時差
const TFS=["1m","5m","15m","1h"];
let kcharts={};
const savedTf=sym=>{try{return localStorage.getItem("tf_"+sym)}catch(e){return null}};
function ensureChart(sym){
  let k=kcharts[sym];if(k)return k;
  const holder=el("div",{class:"kbox"}),title=el("span",{}),price=el("span",{class:"livep"}),age=el("span",{class:"muted age"});
  const tfs=el("span",{class:"tabs"}),legend=el("div",{class:"legend"});
  const head=el("div",{class:"khead"},el("h2",{},title," ",price," ",age),tfs);
  document.getElementById("prices").append(el("div",{class:"panel","data-sym":sym},head,holder,legend));
  const chart=LightweightCharts.createChart(holder,{autoSize:true,layout:{background:{color:"#171d24"},textColor:"#8b98a5"},
    grid:{vertLines:{color:"#26303b"},horzLines:{color:"#26303b"}},rightPriceScale:{borderColor:"#26303b"},
    timeScale:{borderColor:"#26303b",timeVisible:true,secondsVisible:false},crosshair:{mode:0}});
  const candles=chart.addCandlestickSeries({upColor:"#26a69a",downColor:"#ef5350",borderVisible:false,wickUpColor:"#26a69a",wickDownColor:"#ef5350"});
  const vol=chart.addHistogramSeries({priceFormat:{type:"volume"},priceScaleId:"vol"});
  chart.priceScale("vol").applyOptions({scaleMargins:{top:0.82,bottom:0}});
  k=kcharts[sym]={sym,chart,candles,vol,title,price,age,legend,tfs,lines:[],tf:TFS.includes(savedTf(sym))?savedTf(sym):"15m",secs:900,lastTime:null,last:null};
  TFS.forEach(tf=>{const b=el("button",{},tf);b.onclick=()=>{k.tf=tf;try{localStorage.setItem("tf_"+sym,tf)}catch(e){}loadKlines(k,true)};tfs.append(b)});
  return k;
}
const bar=x=>({time:x[0]+tz,open:x[1],high:x[2],low:x[3],close:x[4]});
const vbar=x=>({time:x[0]+tz,value:x[5],color:x[4]>=x[1]?"rgba(38,166,154,.35)":"rgba(239,83,80,.35)"});
async function loadKlines(k,fit){  // 換週期或每分鐘整批重抓(新的 K 線會補上)
  try{
    const r=await (await fetch("/api/klines?symbol="+k.sym+"&interval="+k.tf)).json();
    k.secs=r.secs;k.candles.setData(r.klines.map(bar));k.vol.setData(r.klines.map(vbar));
    k.lastTime=r.klines.length?r.klines[r.klines.length-1][0]+tz:null;
    k.tfs.querySelectorAll("button").forEach(b=>b.classList.toggle("sel",b.textContent===k.tf));
    overlays(k);if(fit)k.chart.timeScale().fitContent();
  }catch(e){}
}
function overlays(k){  // 成交標記與持倉線(依目前週期把成交時間對齊到 K 線)
  if(!D)return;
  const prec=k.last>=100?2:k.last>=1?3:4;
  k.candles.applyOptions({priceFormat:{type:"price",precision:prec,minMove:Math.pow(10,-prec)}});
  k.lines.forEach(l=>k.candles.removePriceLine(l));k.lines=[];k.legend.replaceChildren();
  const marks=[];
  D.accounts.forEach((a,i)=>{if(a.symbol!==k.sym)return;
    k.legend.append(el("span",{},el("i",{style:"background:"+color(i)}),label(a)));
    a.trades.forEach(x=>{const buy=x.side==="buy";
      marks.push({time:Math.floor(x.ts/k.secs)*k.secs+tz,position:buy?"belowBar":"aboveBar",color:color(i),shape:buy?"arrowUp":"arrowDown",
        text:(x.forced?"強制":buy?"買":"賣")+(x.realized_pnl?" "+smoney(x.realized_pnl):"")})});
    const q=a.position;if(q&&a.running){
      const add=(price,title,style)=>{if(price)k.lines.push(k.candles.createPriceLine({price,color:color(i),lineWidth:1,lineStyle:style,axisLabelVisible:true,title}))};
      add(q.entry,a.style+" 進場",0);add(q.stop,a.style+" 停損",2);add(q.trail,a.style+" 追蹤",1);}
  });
  marks.sort((x,y)=>x.time-y.time);k.candles.setMarkers(marks);
}
function setPrice(k,p,ts){
  if(p==null)return;
  if(k.last!=null&&p!==k.last){k.price.classList.remove("flash-up","flash-down");void k.price.offsetWidth;k.price.classList.add(p>k.last?"flash-up":"flash-down")}
  k.last=p;k.price.textContent=fmtp(p);k.updated=ts;
}
function priceCharts(){
  const syms=Object.keys(D.prices);
  for(const s of Object.keys(kcharts))if(!syms.includes(s)){kcharts[s].chart.remove();delete kcharts[s];document.querySelector('[data-sym="'+s+'"]')?.remove()}
  syms.forEach(sym=>{const k=ensureChart(sym),fresh=k.lastTime==null;
    k.title.textContent=sym;if(k.last==null)setPrice(k,D.prices[sym].mark,D.now);
    if(fresh)loadKlines(k,true);else overlays(k);});
}
async function live(){  // 每 3 秒:現價與最後兩根 K 線
  const ks=Object.values(kcharts);if(!ks.length||document.hidden)return;
  try{
    const r=await (await fetch("/api/live?q="+ks.map(k=>k.sym+":"+k.tf).join(","))).json();
    for(const k of ks){const it=r.items[k.sym+":"+k.tf];if(!it)continue;
      setPrice(k,it.mark,r.now);
      if(k.lastTime!=null)for(const x of it.klines){const b=bar(x);if(b.time>=k.lastTime-k.secs){k.candles.update(b);k.vol.update(vbar(x));k.lastTime=Math.max(k.lastTime,b.time)}}}
  }catch(e){}
}
setInterval(()=>{const n=Date.now()/1000;for(const k of Object.values(kcharts))k.age.textContent=k.updated?Math.max(0,Math.round(n-k.updated))+" 秒前":""},1000);
setInterval(live,3000);
setInterval(()=>Object.values(kcharts).forEach(k=>loadKlines(k,false)),60000);
function compare(){
  const head=["帳戶","報酬","損益","最大回撤","成交","平倉勝率","手續費","資金費"];
  const tb=el("table",{},el("tr",{},...head.map((h,i)=>el("th",{class:i?"num":""},h))));
  for(const a of D.accounts)tb.append(el("tr",{},el("td",{},label(a)),el("td",{class:"num "+cls(a.return_pct)},pct(a.return_pct)),
    el("td",{class:"num "+cls(a.equity-a.initial)},smoney(a.equity-a.initial)),el("td",{class:"num"},a.max_drawdown_pct+"%"),
    el("td",{class:"num"},a.trade_count),el("td",{class:"num"},a.closes?Math.round(a.wins/a.closes*100)+"%("+a.wins+"/"+a.closes+")":"-"),
    el("td",{class:"num"},money(a.fees)),el("td",{class:"num"},smoney(-a.funding))));
  document.getElementById("compare").replaceChildren(tb);
}
function decisions(){
  const all=[];D.accounts.forEach(a=>a.decisions.forEach(d=>all.push([a,d])));all.sort((x,y)=>y[1].ts-x[1].ts);
  const tb=el("table",{},el("tr",{},...["時間","帳戶","動作","信心","理由 / 結果"].map(h=>el("th",{},h))));
  for(const[a,d]of all.slice(0,40)){
    const act=(d.alert?"⚡急跌 ":"")+(d.forced?"強制出場":(d.action||"-")+(d.leverage?" "+d.leverage+"x":""));
    tb.append(el("tr",{},el("td",{},t(d.ts)),el("td",{},label(a)),el("td",{class:d.action==="LONG"?"up":d.action==="SHORT"||d.forced?"down":""},act),
      el("td",{class:"num"},d.confidence??"-"),el("td",{},(d.reason||"")+(d.taken&&!d.forced?" → "+d.taken:""))));
  }
  document.getElementById("decisions").replaceChildren(tb);
}
function notes(){
  const box=document.getElementById("notes");box.replaceChildren();
  for(const a of D.accounts){
    const p=el("div",{class:"panel"},el("h2",{},label(a)+" 心得"+(a.lessons_updated?"("+a.lessons_updated+")":"")),
      a.lessons.length?el("ul",{},...a.lessons.map(x=>el("li",{},x))):el("div",{class:"muted"},"(尚無心得)"));
    if(a.news){const n=a.news,good=(n.sources||[]).filter(s=>s.reputable).length;
      p.append(el("h2",{style:"margin-top:12px"},"新聞 "+n.day+" · "+({BULLISH:"偏多",BEARISH:"偏空",NEUTRAL:"中性"}[n.sentiment]||n.sentiment)+(n.shared?" · 共用":"")),
        el("div",{},n.summary||""),el("div",{class:"muted"},"來源(主流 "+good+"/"+(n.sources||[]).length+"):"+(n.sources||[]).map(s=>s.title+(s.reputable?" ✓":"")).join("、")));}
    box.append(p);
  }
}
function usage(){
  const tb=el("table",{},el("tr",{},...["帳戶","決策 次數 / 輸入 / 輸出","檢討","新聞"].map((h,i)=>el("th",{class:i?"num":""},h))));
  const f=u=>u[0]+" 次 / "+u[1].toLocaleString()+" / "+u[2].toLocaleString();
  const sum={decide:[0,0,0],reflect:[0,0,0],news:[0,0,0]};
  for(const a of D.accounts){for(const k in sum)a.usage_24h[k].forEach((v,i)=>sum[k][i]+=v);
    tb.append(el("tr",{},el("td",{},label(a)),el("td",{class:"num"},f(a.usage_24h.decide)),el("td",{class:"num"},f(a.usage_24h.reflect)),el("td",{class:"num"},f(a.usage_24h.news))));}
  tb.append(el("tr",{},el("th",{},"合計"),el("th",{class:"num"},f(sum.decide)),el("th",{class:"num"},f(sum.reflect)),el("th",{class:"num"},f(sum.news))));
  document.getElementById("usage").replaceChildren(tb,el("div",{class:"muted"},"決策用 GEMINI_MODEL(Flash),檢討與新聞用 GEMINI_REFLECT_MODEL(Pro);新聞含沒附來源而重試的呼叫"));
}
function render(){
  if(!D)return;
  document.getElementById("updated").textContent="更新於 "+t(D.now)+(D.rate?" · 1 USDT ≈ "+D.rate.toFixed(3)+" 元":"")+"  ";
  totals();cards();compare();decisions();notes();usage();
  const miss=[];
  if(window.Chart)equityChart();else miss.push("權益曲線");
  if(window.LightweightCharts)priceCharts();else miss.push("K 線");
  document.getElementById("errors").textContent=miss.length?"圖表程式庫載入失敗(需要網路):"+miss.join("、")+";其他資料仍可查看\n":"";
  document.getElementById("errors").textContent+=(D.errors||[]).join("\n");
}
function filter(){  // 預設只看執行中的帳戶(停掉的幣種或沒開的風格自動隱藏)
  if(!RAW)return;D=Object.assign({},RAW);
  const run=RAW.accounts.filter(a=>a.running!==false);
  D.accounts=showStopped||!run.length?RAW.accounts:run;
  const syms=new Set(D.accounts.map(a=>a.symbol));D.prices=Object.fromEntries(Object.entries(RAW.prices).filter(([s])=>syms.has(s)));
  const hidden=RAW.accounts.length-D.accounts.length;
  document.getElementById("stopped").textContent=showStopped?"只看執行中":"含已停止"+(hidden?"("+hidden+")":"");
}
async function load(){try{const r=await fetch("/api/data");RAW=await r.json();filter();render()}catch(e){document.getElementById("errors").textContent="讀取資料失敗:"+e}}
document.getElementById("stopped").onclick=()=>{showStopped=!showStopped;filter();render()};
document.querySelectorAll(".tabs button[data-unit]").forEach(b=>b.onclick=()=>{unit=b.dataset.unit;document.querySelectorAll(".tabs button[data-unit]").forEach(x=>x.classList.toggle("sel",x===b));render()});
load();setInterval(load,15000);
</script>
</body>
</html>
"""
