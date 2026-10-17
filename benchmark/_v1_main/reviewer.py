"""
reviewer.py
負責呼叫 LLM（這裡改用 Google Gemini API）針對單一檔案的 diff + context
產生結構化的 review 結果。

使用 Pydantic schema 強制 Gemini 輸出符合格式的 JSON，避免格式跑掉。
"""

import os
from typing import Literal

from google import genai
from pydantic import BaseModel, Field

MODEL_NAME = "gemini-3.6-flash"


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


SYSTEM_PROMPT = """你是一位資深的 code reviewer。
你會收到某個檔案的 git diff 以及該檔案目前的完整內容（或相關片段）作為上下文。
請針對這次改動（只針對 diff 中新增或修改的部分）進行審查，找出：
- bug / 邏輯錯誤
- 效能疑慮
- 程式風格 / 慣例問題
- 安全性疑慮（如果有明顯問題）

只針對這次改動的程式碼提出意見，不要對沒有變動的既有程式碼吹毛求疵。
如果這次改動沒有問題，issues 可以回傳空陣列。"""


def review_file(filename: str, diff_text: str, context_text: str) -> dict:
    """
    對單一檔案呼叫 Gemini 進行 review。
    回傳格式: {"filename": ..., "summary": ..., "issues": [...]}
    """
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    user_message = f"""檔案名稱: {filename}

=== Diff（本次改動） ===
{diff_text}

=== 檔案上下文（目前內容） ===
{context_text}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=user_message,
        config={
            "system_instruction": SYSTEM_PROMPT,
            "response_mime_type": "application/json",
            "response_schema": FileReview,
        },
    )

    parsed: FileReview = response.parsed
    result = parsed.model_dump()
    result["filename"] = filename
    return result


def review_all_files(file_contexts: dict[str, dict]) -> list[dict]:
    """
    對 context_builder.build_context_for_files 產生的所有檔案逐一呼叫 review_file。
    回傳每個檔案的 review 結果清單。
    """
    results = []
    for filename, data in file_contexts.items():
        print(f"正在審查: {filename} ...")
        result = review_file(filename, data["diff"], data["context"])
        results.append(result)
    return results