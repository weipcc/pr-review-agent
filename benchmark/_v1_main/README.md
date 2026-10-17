# PR Review Agent

一個用 LLM（目前串接 Google Gemini API）自動審查 GitHub PR 的小工具，
提供兩種使用方式：本機 CLI 版本、以及透過網頁呼叫的 API 版本。

## 架構

```
輸入（PR 網址 / 本機 repo）
   → 抓取 diff 與檔案內容
   → 呼叫 Gemini 進行結構化審查（bug / 效能 / 風格 / 安全性）
   → 彙整並依嚴重程度排序
   → 產生 Markdown 報告
```

## 檔案說明

### 核心邏輯（CLI、API 版本共用）
- `reviewer.py` — 呼叫 Gemini API，用 Pydantic schema 強制結構化輸出
- `aggregator.py` — 彙整所有檔案的問題，依嚴重程度排序
- `report.py` — 把彙整結果轉成 Markdown 報告

### CLI 版本（審查本機 git repo）
- `diff_reader.py` — 讀本機 `git diff`，解析成結構化資料
- `context_builder.py` — 讀本機檔案內容，組成 LLM 的上下文
- `main.py` — CLI 入口

### API 版本（審查遠端 GitHub PR）
- `github_diff_reader.py` — 透過 GitHub API 抓遠端 PR 的 diff
- `github_context_builder.py` — 透過 GitHub API 抓檔案內容
- `api.py` — 用 FastAPI 包裝，提供 `/review` 端點
- `review_page.html` — 獨立的小網頁，輸入 PR 網址、呼叫本機 API

### 已棄用的嘗試
- `bookmarklet.js` / `bookmarklet.min.txt` — 原本想用瀏覽器書籤在 GitHub
  頁面上直接呼叫 API，但會被 GitHub 的 CSP 安全機制擋掉，改用
  `review_page.html` 取代。保留作紀錄。

## 環境設定

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY=你的Gemini_key
```

## 使用方式

### CLI 版本（審查本機 git repo）

```bash
python main.py --repo-path /path/to/repo --base main
python main.py --repo-path . --output review.md
```

### API 版本（審查遠端 GitHub PR）

1. 啟動 API 服務：
   ```bash
   uvicorn api:app --reload --port 8000
   ```
2. 用 Chrome 打開 `review_page.html`（不要用 Safari，file:// 權限限制較嚴）
3. 貼上 PR 網址，點「開始審查」

目前僅支援**公開 repo**，未帶 GitHub token，會受匿名 API 速率限制
（每小時約 60 次請求）。

## 已知限制 / 待改進方向

- 匿名呼叫 GitHub API，速率限制較低，也無法讀取私有 repo
- 目前是手動觸發，尚未串接 webhook 自動觸發
- 目前只回傳報告，尚未自動貼回 GitHub PR 當 inline comment
- 目前只能在本機執行，尚未部署到雲端
- 審查邏輯目前是每個檔案獨立呼叫一次，看不到檔案之間的關聯
- 尚未建立 benchmark 量化審查品質（precision / recall）
