"""
reviewer.py
負責呼叫 LLM（Google Gemini API）審查 PR，並支援兩種模式：

1. 批次模式（小 PR）：所有檔案的 diff + context 一次塞進同一個 prompt，
   一次呼叫 Gemini，模型可以看到整個 PR 的全貌，能發現跨檔案的問題。
2. 逐檔模式（大 PR，避免超出合理的 prompt 大小）：維持一個檔案一次呼叫，
   但額外附上「本次 PR 其他被改動檔案的 diff」作為補充資訊，
   讓模型至少知道「還有哪些檔案也被動過」，緩解看不到全貌的問題。

用「總 token 數」決定走哪個模式（用 Gemini 自己的 count_tokens 計算，
比單純算字數準確，因為中英文、程式碼的 token 換算比例不一樣）。
"""

import os
from typing import Literal

from google import genai
from pydantic import BaseModel, Field

MODEL_NAME = "gemini-3.6-flash"

# 判斷「小 PR / 大 PR」的門檻（token 數）。
# 這不是 Gemini 的技術上限（Gemini 的 context window 大很多），
# 純粹是我們自己訂的「值不值得一次性塞進同一個 prompt」的政策性門檻，
# 之後可依實測效果、成本考量調整。
BATCH_TOKEN_THRESHOLD = 6000

_client = None


def _get_client() -> genai.Client:
    """延遲初始化 Gemini client，避免模組載入時就要求 API key。"""
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    return _client


class Issue(BaseModel):
    line: int = Field(description="問題所在的行號（對應新版檔案的行號）")
    severity: Literal["bug", "performance", "style", "security"] = Field(
        description="問題類型/嚴重程度分類"
    )
    comment: str = Field(description="問題描述")
    suggestion: str = Field(description="具體修改建議")


class FileReview(BaseModel):
    summary: str = Field(description="對這個檔案改動的一句話整體評語")
    issues: list[Issue] = Field(description="發現的問題清單，如果沒有問題可以是空陣列")


class FileReviewItem(BaseModel):
    filename: str = Field(description="檔案名稱，必須對應到輸入裡給的檔名")
    summary: str = Field(description="對這個檔案改動的一句話整體評語")
    issues: list[Issue] = Field(description="發現的問題清單，如果沒有問題可以是空陣列")


class BatchReview(BaseModel):
    files: list[FileReviewItem] = Field(description="每個檔案各自的審查結果")


SYSTEM_PROMPT_SINGLE = """你是一位資深的 code reviewer。
你會收到某個檔案的 git diff 以及該檔案目前的完整內容（或相關片段）作為上下文，
另外可能會附上「本次 PR 其他被改動檔案的 diff」作為參考。
請針對這次改動（只針對 diff 中新增或修改的部分）進行審查，找出：
- bug / 邏輯錯誤（包含跨檔案的破壞性改動，例如其他檔案改了函式簽名但這個檔案沒同步更新）
- 效能疑慮
- 程式風格 / 慣例問題
- 安全性疑慮（如果有明顯問題）

只針對這次改動的程式碼提出意見，不要對沒有變動的既有程式碼吹毛求疵。
如果這次改動沒有問題，issues 可以回傳空陣列。"""

SYSTEM_PROMPT_BATCH = """你是一位資深的 code reviewer。
你會收到「同一個 PR」裡，所有被改動檔案的 diff 以及各自的完整內容（或相關片段）。
請針對每一個檔案分別進行審查，找出：
- bug / 邏輯錯誤（因為你能同時看到所有檔案，請特別留意跨檔案的破壞性改動，
  例如某個檔案改了函式簽名、變數名稱或回傳型別，但呼叫它的其他檔案沒有同步更新）
- 效能疑慮
- 程式風格 / 慣例問題
- 安全性疑慮（如果有明顯問題）

只針對這次改動的程式碼提出意見，不要對沒有變動的既有程式碼吹毛求疵。
如果某個檔案這次改動沒有問題，該檔案的 issues 可以回傳空陣列，
但仍要把這個檔案列在 files 裡（附上 summary）。
回傳的 files 清單，檔名必須跟輸入裡給的檔名完全一致。"""


def count_total_tokens(file_contexts: dict[str, dict]) -> int:
    """
    計算「所有檔案的 diff + context 合併後」的總 token 數，
    用來判斷要走批次模式還是逐檔模式。
    """
    combined_text = "\n".join(
        f"{data['diff']}\n{data['context']}" for data in file_contexts.values()
    )
    if not combined_text.strip():
        return 0
    response = _get_client().models.count_tokens(model=MODEL_NAME, contents=combined_text)
    return response.total_tokens


def review_file(
    filename: str, diff_text: str, context_text: str, other_files_diff: str = ""
) -> dict:
    """
    對單一檔案呼叫 Gemini 進行 review（逐檔模式用）。
    other_files_diff：本次 PR 其他被改動檔案的 diff（只給 diff，不給完整內容，
    避免又把 prompt 撐得太大），讓模型至少知道「還有哪些地方也被動過」。
    回傳格式: {"filename": ..., "summary": ..., "issues": [...]}
    """
    user_message_parts = [
        f"檔案名稱: {filename}",
        f"\n=== Diff（本次改動） ===\n{diff_text}",
        f"\n=== 檔案上下文（目前內容） ===\n{context_text}",
    ]
    if other_files_diff:
        user_message_parts.append(
            f"\n=== 本次 PR 其他被改動檔案的 diff（僅供參考關聯性，非本次審查重點）===\n{other_files_diff}"
        )
    user_message = "\n".join(user_message_parts)

    response = _get_client().models.generate_content(
        model=MODEL_NAME,
        contents=user_message,
        config={
            "system_instruction": SYSTEM_PROMPT_SINGLE,
            "response_mime_type": "application/json",
            "response_schema": FileReview,
        },
    )

    parsed: FileReview = response.parsed
    result = parsed.model_dump()
    result["filename"] = filename
    return result


def review_all_files_sequential(file_contexts: dict[str, dict]) -> list[dict]:
    """
    逐檔模式（大 PR 用）：一個檔案一次呼叫，
    但每次都附上「其他檔案的 diff」讓模型知道還有哪些地方被動過。
    """
    results = []
    filenames = list(file_contexts.keys())

    for filename in filenames:
        data = file_contexts[filename]
        other_diffs = "\n\n".join(
            file_contexts[other]["diff"] for other in filenames if other != filename
        )
        print(f"正在審查（逐檔模式）: {filename} ...")
        result = review_file(filename, data["diff"], data["context"], other_diffs)
        results.append(result)
    return results


def review_all_files_batched(file_contexts: dict[str, dict]) -> list[dict]:
    """
    批次模式（小 PR 用）：所有檔案一次塞進同一個 prompt，一次呼叫 Gemini。
    回傳格式跟逐檔模式一致，方便 aggregator.py 不用區分是哪種模式產生的。
    """
    sections = []
    for filename, data in file_contexts.items():
        sections.append(
            f"### 檔案: {filename}\n"
            f"--- Diff ---\n{data['diff']}\n"
            f"--- 檔案上下文 ---\n{data['context']}"
        )
    user_message = "\n\n".join(sections)

    print(f"正在審查（批次模式，共 {len(file_contexts)} 個檔案）...")
    response = _get_client().models.generate_content(
        model=MODEL_NAME,
        contents=user_message,
        config={
            "system_instruction": SYSTEM_PROMPT_BATCH,
            "response_mime_type": "application/json",
            "response_schema": BatchReview,
        },
    )

    parsed: BatchReview = response.parsed
    return [item.model_dump() for item in parsed.files]


def review_pr_files(file_contexts: dict[str, dict]) -> list[dict]:
    """
    主要對外接口：根據總 token 數自動決定要走批次模式還是逐檔模式。
    api.py / main.py 應該呼叫這個函式，而不是直接呼叫上面兩個模式各自的函式。
    """
    total_tokens = count_total_tokens(file_contexts)
    print(f"這次 PR 合併後估計 {total_tokens} tokens（門檻: {BATCH_TOKEN_THRESHOLD}）")

    if total_tokens < BATCH_TOKEN_THRESHOLD:
        return review_all_files_batched(file_contexts)
    return review_all_files_sequential(file_contexts)


# 保留舊名稱作為別名，避免既有程式碼（如果還有地方直接 import review_all_files）壞掉。
review_all_files = review_pr_files