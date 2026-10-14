# 工作日誌

## 2026-10-14
- 修正:paper 中模型暫時失敗會把該根 K 線標記為已處理,接續後要等一小時且畫面無提示(使用者回報看似卡住)。
  改為同一根 K 線失敗後每 `--check-every` 秒重試,最多 3 次;啟動與每次決策後顯示下一次問模型的時間;report 列出最近錯誤、無決策時明確說明。新增 2 個測試(共 29 個)。
- 使用者儲值後,`once` 實際呼叫 Gemini(gemini-3.8-flash)成功:回傳 HOLD、信心 0.65、繁中理由;延遲約 5.8 秒,tokens 輸入 638 / 輸出 506(輸出含思考 token)。結構化輸出、解析、風控流程端到端驗證完成。
- 帳務備忘:Gemini API 自 2026-03-23 起新帳號預設預付制,餘額歸零時同一帳務帳戶下所有金鑰回 402。
- 使用者提供 .env 後首次實際呼叫 Gemini:回 402「prepayment credits are depleted」(額度用完)。金鑰有效、請求有送達,但尚未驗證到實際決策。
  抓到 bug:402 原被當暫時性錯誤,`paper` 會每小時失敗卻不停止;`once` 失敗仍回 0。修正:402 歸為設定 / 帳務錯誤(直接停止並提示),`once` 模型出錯回 1。新增測試(共 27 個)。
- 建立專案:用 Gemini 做交易決策的模擬交易系統(預設 BTCUSDT 1h、只模擬)。
- 查證 Gemini API 現況:`generateContent` 仍有正式文件、無棄用公告;官方新範例改用 Interactions API。採 `generateContent` + `responseMimeType`/`responseSchema`;預設模型 `gemini-3.8-flash`。
- 實作:config(.env)、market(行情與指標)、llm(Gemini / 規則基準 / 快取、嚴格驗證)、prompt、portfolio、risk(含原因代碼)、agent(單步、回放)、state、cli(once / paper / replay / report)。
- 自我 code review 與修正:
  - report 把「已在目標持倉」誤算成「被風控擋下」→ 風控回傳原因代碼,報表分開統計。
  - replay 提前結束時的期末價格與根數計算錯誤 → 改以實際處理的 K 線計算。
  - report 時間用 UTC、模擬盤用本地時間,容易看錯 → 統一本地時間。
  - 測試:paper 接續的測試會真的等到下一根 K 線(最長一小時)→ 改用可控時鐘。
- 驗證:26 個單元測試通過;以規則模型實際跑 once / replay(1000 根)/ paper(真實行情)/ report。
- 未驗證:沒有 Gemini 金鑰,尚未實際呼叫 Gemini(請求格式以官方文件與模擬回應測試)。
