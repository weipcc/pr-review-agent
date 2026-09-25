# PR Review Agent

一個用 LLM（目前串接 Google Gemini API）自動審查 GitHub PR 的小工具，
提供兩種使用方式：本機 CLI 版本、以及透過網頁呼叫的 API 版本。

## 架構

```
輸入（PR 網址 / 本機 repo）
   → 抓取 diff、檔案內容，以及專案背景（PR 標題/描述、README 開頭一段）
   → 依總 token 數決定審查方式：
        小 PR → 批次模式：所有檔案一次送給 Gemini，能看到整個 PR 的全貌
        大 PR → 逐檔模式：一個檔案一次呼叫，但附上其他檔案的 diff 當參考
   → Gemini 依固定 schema 回傳結構化結果（分類、信心、引用的原始碼、建議…）
   → 彙整並排序
   → 產生 Markdown 報告（英文）
```

## 檔案說明

### 核心邏輯（CLI、API 版本共用）
- `reviewer.py` — 呼叫 Gemini API：批次 / 逐檔兩種模式、`review_pr_files()` 依 token 數自動分流；
  system prompt、few-shot 範例與 Pydantic schema 也都在這裡
- `aggregator.py` — 彙整所有檔案的結果，依類別與結論排序，並算出整份 PR 的總結論
- `report.py` — 把彙整結果轉成 Markdown 報告

### CLI 版本（審查本機 git repo）
- `diff_reader.py` — 讀本機 `git diff`，解析成結構化資料
- `context_builder.py` — 讀本機檔案內容與 README，組成 LLM 的上下文
- `main.py` — CLI 入口

### API 版本（審查遠端 GitHub PR）
- `github_diff_reader.py` — 透過 GitHub API 抓遠端 PR 的 diff 與基本資訊
- `github_context_builder.py` — 透過 GitHub API 抓檔案內容與 README
- `api.py` — 用 FastAPI 包裝，提供 `/review` 端點
- `review_page.html` — 獨立的小網頁，輸入 PR 網址、呼叫本機 API

### 已棄用的嘗試
- `bookmarklet.js` / `bookmarklet.min.txt` — 原本想用瀏覽器書籤在 GitHub
  頁面上直接呼叫 API，但會被 GitHub 的 CSP 安全機制擋掉，改用
  `review_page.html` 取代。保留作紀錄。

## 審查結果

每個問題（issue）包含：

| 欄位 | 說明 |
|---|---|
| `category` | 7 類：`bug` / `security` / `performance` / `style` / `refactor` / `nitpick` / `question` |
| `confidence` | `high` / `medium` / `low`，模型對這個問題有多確定 |
| `evidence` | 逐字引用觸發問題的原始程式碼（用來降低模型憑空捏造問題的機率） |
| `start_line` / `end_line` | 問題所在的行範圍 |
| `comment` / `suggestion` / `suggested_code` | 問題描述、修改建議、可直接套用的程式碼（可為空） |

每個檔案另有 `change_intent`（這個檔案的改動想做什麼）與 `recommendation`
（`looks_good` / `minor_comments` / `needs_changes`）；整份 PR 的結論取所有檔案中最嚴重的一個。

報告會先列出 PR 標題（僅 API 版本）、整體結論與檔案總覽表；有問題的檔案才展開細節，
沒問題的檔案收在最後、每個檔案一行。報告與終端輸出、網頁介面皆為英文。

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

不指定 `--base` 時，會審查尚未 commit 的變更。CLI 版本沒有 PR 標題/描述，
只會帶入該 repo 的 README 當專案背景。

### API 版本（審查遠端 GitHub PR）

1. 啟動 API 服務：
   ```bash
   uvicorn api:app --reload --port 8000
   ```
2. 用 Chrome 打開 `review_page.html`（不要用 Safari，file:// 權限限制較嚴）
3. 貼上 PR 網址，點 "Start review"

目前僅支援**公開 repo**，未帶 GitHub token，會受匿名 API 速率限制
（每小時約 60 次請求）。審查一個 PR 大約會用掉 3 + N 次請求
（PR 資訊、diff、README，加上 N 個檔案的內容），檔案多時較容易碰到上限。

## 已知限制 / 待改進方向

- 匿名呼叫 GitHub API，速率限制較低，也無法讀取私有 repo
- 目前是手動觸發，尚未串接 webhook 自動觸發
- 目前只回傳報告，尚未自動貼回 GitHub PR 當 inline comment
- 目前只能在本機執行，尚未部署到雲端
- 大 PR 走逐檔模式時，模型只看得到其他檔案的 diff，看不到它們的完整內容
- README 只取開頭 6000 字元、PR 描述只取前 3000 字元，尚未做完整的專案架構摘要
- 尚未建立 benchmark 量化審查品質（precision / recall），也還沒驗證
  few-shot 與專案背景資訊實際帶來多少幫助
- 逐檔模式每次呼叫都會帶上 few-shot 範例，大 PR 會增加一些 token 用量
- `refactor`、`question` 兩個類別目前沒有專門的 few-shot 範例
