# 工作日誌

## 2026-10-14
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
