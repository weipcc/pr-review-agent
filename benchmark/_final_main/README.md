# PR Review Agent

一個用 LLM（目前串接 Google Gemini API）自動審查 GitHub PR 的小工具，
透過 API 審查遠端 GitHub PR（需要 E2B_API_KEY 與 GEMINI_API_KEY；agent 會在 E2B sandbox 中探索 PR 程式碼）。

## 架構

```
輸入（PR 網址）
   → 抓取 diff，以及專案背景（PR 標題/描述、README 開頭一段）
   → 檢查所有 diff 的總 token 數：超過 BATCH_TOKEN_THRESHOLD 就不審查，
     並回覆「PR 太大」（API 回 413；@mention 時貼成 PR 留言）
   → 否則所有檔案一次送給 Gemini，agent 可在 E2B sandbox 中探索程式碼
   → Gemini 依固定 schema 回傳結構化結果（分類、信心、引用的原始碼、建議…）
   → 彙整並排序
   → 產生 Markdown 報告（英文）
```

## 檔案說明

### 核心邏輯
- `reviewer.py` — 呼叫 Gemini API：`check_pr_size()` 擋掉過大的 PR、`review_pr_files()` 一次審查所有檔案；
  system prompt、few-shot 範例與 Pydantic schema 也都在這裡
- `aggregator.py` — 彙整所有檔案的結果，依類別與結論排序，並算出整份 PR 的總結論
- `report.py` — 把彙整結果轉成 Markdown 報告

### Diff 解析
- `diff_reader.py` — 把 git diff 文字解析成結構化資料
- `context_builder.py` — 把 diff 組成 LLM 的輸入，並截斷 README / PR 描述
- `sandbox_tools.py` — E2B sandbox（checkout PR、關閉網路）與唯一的 `run_bash` 工具

### API 版本（審查遠端 GitHub PR）
- `github_diff_reader.py` — 透過 GitHub API 抓遠端 PR 的 diff 與基本資訊
- `github_context_builder.py` — 透過 GitHub API 抓 README
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
export E2B_API_KEY=你的E2B_key      # 必填：沒有 sandbox 就不會審查（.env 裡用 E2B_KEY 也可以）
export GITHUB_TOKEN=你的GitHub_token  # 選填：讀私有 repo、啟用 @mention 輪詢
```

## 使用方式

### API 版本（審查遠端 GitHub PR）

1. 啟動 API 服務：
   ```bash
   uvicorn api:app --reload --port 8000
   ```
2. 用 Chrome 打開 `review_page.html`（不要用 Safari，file:// 權限限制較嚴）
3. 貼上 PR 網址，點 "Start review"

讀取 PR 資訊、diff、README 是匿名呼叫 GitHub API（每小時約 60 次請求），
一次審查約用 3 次請求，所以目前只支援**公開 repo**。

審查流程：先檢查 diff 大小（太大就回 413 / 貼「PR 太大」留言），再建立 E2B sandbox，
shallow checkout PR 後立刻關閉網路；agent 只有一個 `run_bash` 工具，
最多 10 輪「LLM → 工具 → 結果」，用完後會被拿掉工具、強制輸出結果；單次工具輸出上限 5000 字元。

## 已知限制 / 待改進方向

- 匿名呼叫 GitHub API，速率限制較低，也無法讀取私有 repo（sandbox 的 checkout 會用 GITHUB_TOKEN，但 diff/README 不會）
- 目前是手動觸發，尚未串接 webhook 自動觸發
- 目前只回傳報告，尚未自動貼回 GitHub PR 當 inline comment
- 目前只能在本機執行，尚未部署到雲端
- 超過 token 門檻的 PR 不會被審查，只會提示拆小
- README 只取開頭 6000 字元、PR 描述只取前 3000 字元，尚未做完整的專案架構摘要
- 尚未建立 benchmark 量化審查品質（precision / recall），也還沒驗證
  few-shot 與專案背景資訊實際帶來多少幫助
- `refactor`、`question` 兩個類別目前沒有專門的 few-shot 範例
