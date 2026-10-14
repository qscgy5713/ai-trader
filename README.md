# ai-trader

用 **Google Gemini** 做交易決策的**模擬交易**系統:每根 K 線收盤後,把技術指標與目前持倉交給 Gemini,
由它建議「目標持倉比例」,再經過程式的**風控層**決定是否真的(模擬)成交。

- **只模擬,不下真單**,也不構成投資建議。
- AI 只提供建議;停損、單日虧損上限、持倉上限、交易次數等規則由程式強制執行。
- 每次決策(含 AI 的理由)都記錄到決策日誌,方便事後檢討。
- 兩種模式:
  - **目標持倉模式**(`once` / `paper` / `replay`):AI 每根 K 線決定要持有多少比例的幣,交易次數少。
  - **AI 調整網格模式**(`grid` / `grid-replay`):AI 每根 K 線決定網格區間與格數(或暫停),
    網格每幾秒撮合一次,負責頻繁的小額低買高賣。AI 費用和目標持倉模式一樣,成交次數多很多。

## 技術棧
- Python 3.9+,**僅用標準庫**(`urllib`、`json`、`unittest`),無第三方依賴
- AI:Gemini API `generateContent`(REST,JSON 結構化輸出)
- 行情:Binance 公開 API(免金鑰)

## 安裝與環境設定

1. 到 [Google AI Studio](https://aistudio.google.com/apikey) 申請 Gemini API 金鑰。
2. 在專案根目錄建立 `.env`(已被 `.gitignore` 排除,不會提交):
   ```bash
   cp .env.example .env
   # 編輯 .env,填入 GEMINI_API_KEY=你的金鑰
   ```
   | 變數 | 預設 | 說明 |
   |---|---|---|
   | `GEMINI_API_KEY` | 必填 | Gemini API 金鑰 |
   | `GEMINI_MODEL` | `gemini-3.8-flash` | 模型名稱;想省錢可改 `gemini-3.5-flash-lite` |
3. 跑測試:
   ```bash
   python3 -m unittest discover -s tests -t .
   ```

**不要把金鑰貼到對話、程式碼或 commit 裡。** 程式只從環境變數 / `.env` 讀取,金鑰放在 HTTP header,不會出現在網址、日誌或錯誤訊息中。

## 使用方式

```bash
# 1. 確認金鑰可用:用即時行情問一次 Gemini(只試算,不存檔)
python3 -m aitrader once

# 2. 模擬盤:每根 1h K 線收盤後問 Gemini,兩根之間每分鐘檢查停損
python3 -m aitrader paper

# 3. 歷史回放:最近 500 根 1h K 線,每 4 根問一次,Gemini 最多實際呼叫 100 次
python3 -m aitrader replay --limit 500 --every 4 --max-calls 100

# 4. 用不花錢的規則模型當對照組
python3 -m aitrader replay --model rule --limit 1000 --every 1

# 5. 檢視決策日誌
python3 -m aitrader report journal/paper_BTCUSDT_gemini.jsonl

# 6. AI 調整網格:模擬盤(每 5 秒撮合網格,每小時問 Gemini 是否調整)
python3 -m aitrader grid

# 7. AI 調整網格:歷史回放(先用規則模型看看,不花錢)
python3 -m aitrader grid-replay --model rule --limit 1000 --every 1

# 8. 檢視網格日誌(report 會自動辨識)
python3 -m aitrader report journal/grid_BTCUSDT_gemini.jsonl
```

### 子指令

| 指令 | 用途 |
|---|---|
| `once` | 用即時行情問一次模型,顯示決策與風控結果。不改任何狀態,適合確認金鑰與模型設定。 |
| `paper` | 模擬盤。狀態存在 `state/`,重跑相同指令自動接續;Ctrl+C 會在安全點存檔結束(再按一次強制結束)。 |
| `replay` | 歷史回放。決策後以**下一根 K 線開盤價**成交(避免偷看未來);每根 K 線都檢查停損。Gemini 回覆會快取在 `cache/`,重跑不重複付費。 |
| `report` | 統計決策日誌:各種決策次數、成交、已實現損益、手續費、被風控擋下的原因、token 用量、最近決策與理由。網格日誌會自動改用網格報表。 |
| `grid` | AI 調整網格的模擬盤,說明見下方「AI 調整網格模式」。 |
| `grid-replay` | AI 調整網格的歷史回放。 |

### 共用參數

| 參數 | 預設 | 說明 |
|---|---|---|
| `--symbol` | `BTCUSDT` | Binance 交易對 |
| `--interval` | `1h` | K 線週期(`paper` 支援 1m/5m/15m/30m/1h/4h/1d) |
| `--model` | `gemini` | `gemini` 或 `rule`(離線規則基準:均線 + RSI,不花錢) |
| `--cash` | 10000 | 模擬資金 |
| `--fee` | 0.001 | 單邊手續費率 |
| `--slippage-bps` | 5 | 滑價(基點,5 = 0.05%) |

### 風控參數

| 參數 | 預設 | 說明 |
|---|---|---|
| `--max-position` | 50 | 持倉上限(占總權益 %)。AI 要求更高會被截斷。 |
| `--min-confidence` | 0.6 | AI 信心低於此值不交易。 |
| `--min-trade` | 5 | 目標與目前持倉差距小於此 % 不交易,避免手續費磨損。 |
| `--stop-loss` | 5 | 持倉虧損達此 % 強制清倉(不問 AI)。 |
| `--take-profit` | 0 | 持倉獲利達此 % 強制清倉;0 = 不啟用。 |
| `--daily-loss-limit` | 3 | 權益較當日(UTC)開始下跌達此 %,清倉並停止交易到隔天。 |
| `--max-trades` | 6 | 每日最多成交筆數。 |

`paper` 接續時,風控參數**以這次啟動的參數為準**(可隨時調整),帳戶與當日統計沿用狀態檔。

### 其他參數
- `paper`:`--check-every`(兩根 K 線之間檢查停損的秒數,預設 60)、`--state`、`--journal`、`--reset`、`--max-steps`(問模型幾次後結束)。
- `replay`:`--limit`(K 線根數,預設 500)、`--every`(每幾根問一次,預設 4)、`--max-calls`(Gemini 實際呼叫上限,預設 100,快取命中不計)、`--cache-dir`、`--journal`。

## AI 調整網格模式

分工:**AI 決定「在哪個區間、切幾格」,網格負責「頻繁小額低買高賣」**。

- 每 `--tick` 秒(預設 5)取一次現價撮合網格:跌破買單價就買,漲過賣單價就賣(穿價才算成交)。
- 每根 `--interval` K 線(預設 1h)收盤後問 AI,AI 回覆以下三種之一:
  - `KEEP`:維持現狀(預設;重設網格要付手續費)。
  - `SET`:以現價為中心重設網格,下緣 `lower_pct`% / 上緣 `upper_pct`% / `grids` 格(可不對稱)。
  - `PAUSE`:撤掉網格、全部賣出(例如明顯跌破)。之後再 `SET` 才恢復。
- 重設網格時只買賣「新舊網格所需持幣的差額」,不會先全部賣掉再買回。
- 強制規則:跌破區間下緣再往下 `--stop-loss`% 停損;權益單日跌 `--daily-loss-limit`%,清倉。兩者觸發後**當日(UTC)停機、不問 AI**。
- **調整頻率**:`--interval` 決定多久問一次 AI(`1m`~`1d`,預設 `1h`)。例如 `grid --interval 4h` 每 4 小時一次,
  `--interval 15m` 每 15 分鐘一次(費用約 4 倍)。網格本身的撮合頻率由 `--tick` 決定,與此無關。

### 學習(預設開啟)

不是重新訓練模型,而是把 AI 自己的成績單放進提示詞:
1. **最近結果**:每次決策附上最近 8 次決策與「到下一次決策為止」的結果(價格、權益變化、網格成交、停損事件)。
2. **每日檢討**:UTC 換日後的第一次決策前,多呼叫一次 Gemini 檢討前一天的所有決策,整理成最多 5 條心得,
   之後每次決策都附上,下次檢討時更新(失敗會在之後的決策重試,最多 3 次)。

注意:樣本少時容易學錯,心得只是參考;停損、上限等規則仍由程式強制執行。費用約增加 30~50%
(每次決策多幾百 tokens,每天多一次檢討)。

用 `--no-learn` 關閉學習,預設狀態檔與日誌會加 `_nolearn` 後綴,可以和學習版**同時跑**來比較:
```bash
python3 -m aitrader grid               # 學習版
python3 -m aitrader grid --no-learn    # 對照組(另開一個終端機)
```
`report` 會顯示每日檢討次數與最新心得。

| 參數 | 預設 | 說明 |
|---|---|---|
| `--interval` | `1h` | 多久問一次 AI |
| `--tick` | 5 | 撮合網格的價格更新秒數(僅 `grid`) |
| `--min-confidence` | 0.6 | `SET` / `PAUSE` 的信心門檻,低於此值維持現狀 |
| `--min-spacing` | 0.5 | 最小格距 %。AI 要求的格數太多會自動減少(手續費 0.1% 時,一買一賣打平約需 0.2%) |
| `--max-range` | 15 | 區間上下緣各自最多離現價幾 %,超過會截斷 |
| `--stop-loss` | 5 | 區間下緣再往下幾 % 停損;0 = 不設 |
| `--daily-loss-limit` | 3 | 權益單日下跌達此 % 清倉,當日停機 |
| `--max-reconfigs` | 4 | 每日最多重設 / 暫停次數 |
| `--min-change` | 1 | 新舊區間上下緣都差不到此 %、且格數相同時不重設 |
| `--no-learn` | 關 | 關閉學習(見下方「學習」) |

另有 `--state`、`--journal`、`--reset`、`--max-steps`(`grid`)與 `--limit`、`--every`、`--max-calls`、`--cache-dir`(`grid-replay`),意義同上。
回放時 K 線內部走勢未知,以保守路徑(陽線先探低、陰線先拉高)撮合,成交次數會比實際少。

## 運作方式(目標持倉模式)

每個決策點:
1. **強制風控**:單日虧損上限、停損、停利。觸發就直接清倉,**不問 AI**。
2. **問 AI**:送出技術指標(報酬率、均線、MACD、RSI、ATR、波動度、區間位置、量能、最近 24 根收盤)與目前持倉;Gemini 以固定 JSON 格式回覆 `action`(BUY/SELL/HOLD)、`target_position_pct`、`confidence`、`reason`。
3. **驗證輸出**:格式不符、數值超出範圍、被安全機制擋下、API 錯誤 → 一律視為「不交易」並記錄。金鑰錯誤 / 模型不存在等設定錯誤會直接停止並提示。
4. **風控過濾**:信心門檻、持倉上限、最小調整幅度、每日交易次數。
5. **模擬成交**:市價成交,計入手續費與滑價;寫入決策日誌。

## 重要限制
- **AI 沒有被證明能穩定獲利。** 請先用 `replay --model rule` 當對照組,並讓 `paper` 跑一段時間再評估。
- **回放結果偏樂觀**:如果回放期間早於模型訓練資料的截止日,模型可能「記得」當時行情。模擬盤(即時)結果才可信。
- 不模擬委託簿深度與部分成交;停損只在檢查點觸發(回放每根 K 線、模擬盤每 `--check-every` 秒),實際可能成交在更差的價格。
- 費用:模擬盤每根 K 線呼叫一次(1h 週期約每天 24 次);回放用 `--every`、`--max-calls` 與快取控制呼叫次數。

## 目錄結構
```
aitrader/
  config.py     .env 與 Gemini 設定
  market.py     Binance 行情與技術指標
  llm.py        Gemini 客戶端、決策驗證、規則基準模型、快取
  prompt.py     系統提示詞與輸入格式
  portfolio.py  模擬帳戶(目標持倉再平衡、手續費、滑價)
  risk.py       風控(強制出場與決策過濾)
  agent.py      單步決策流程、歷史回放
  grid.py       網格引擎(等比網格、從既有持倉重建、停損)
  grid_agent.py AI 調整網格:風控、套用 AI 決策、歷史回放
  grid_cli.py   grid / grid-replay 指令與網格報表
  memory.py     學習紀錄:最近決策的結果、每日檢討的待檢資料與心得
  runtime.py    安全停止(訊號)與可中斷等待
  state.py      狀態原子寫入、檔案鎖、決策日誌
  cli.py        命令列入口(python3 -m aitrader)
tests/          單元測試(Gemini 以模擬 HTTP 回應測試,不需金鑰;test_grid.py 為網格模式)
state/ journal/ cache/   執行時產生(已 gitignore)
doc/            過程文件
```

## doc/ 文件索引
- [worklog.md](doc/worklog.md) 工作日誌
- [todo.md](doc/todo.md) 待辦清單
- [plan.md](doc/plan.md) 計畫與設計取捨
