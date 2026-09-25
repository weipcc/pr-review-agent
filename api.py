"""
api.py
用 FastAPI 包裝現有的 PR review pipeline，提供一個本機可呼叫的 API。

啟動方式:
    uvicorn api:app --reload --port 8000

呼叫方式:
    curl -X POST http://localhost:8000/review \
        -H "Content-Type: application/json" \
        -d '{"pr_url": "https://github.com/owner/repo/pull/1"}'
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from aggregator import aggregate_results
from context_builder import MAX_PR_DESCRIPTION_CHARS, truncate_text
from github_context_builder import build_context_for_files, get_readme
from github_diff_reader import get_parsed_pr_diff
from report import render_markdown
from reviewer import review_all_files

app = FastAPI(title="PR Review Agent API")

# 允許瀏覽器呼叫這支本機 API（僅供本機測試使用，不對外公開部署時無妨用 *）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)


class ReviewRequest(BaseModel):
    pr_url: str


class ReviewResponse(BaseModel):
    pr_url: str
    overall_summary: str
    overall_recommendation: str
    total_issues: int
    category_counts: dict
    files: list
    markdown_report: str


@app.get("/")
def health_check():
    """簡單的健康檢查，確認服務有啟動。"""
    return {"status": "ok", "message": "PR Review Agent API is running"}


@app.post("/review", response_model=ReviewResponse)
def review_pr(request: ReviewRequest):
    """
    輸入一個 GitHub PR 網址，回傳完整的 review 結果。
    僅支援公開 repo（未帶 GitHub token，會受匿名 API 速率限制）。
    """
    try:
        file_diffs, metadata = get_parsed_pr_diff(request.pr_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch PR data: {e}")

    if not file_diffs:
        raise HTTPException(status_code=404, detail="No changed files were detected in this PR")

    owner = metadata["_owner"]
    repo = metadata["_repo"]
    head_sha = metadata["_head_sha"]

    file_contexts = build_context_for_files(owner, repo, head_sha, file_diffs)

    # 專案背景：PR 標題/描述 + README 摘錄，讓模型知道這個 PR 想達成什麼、專案是做什麼的
    project_context = {
        "pr_title": metadata.get("title") or "",
        "pr_description": truncate_text(metadata.get("body") or "", MAX_PR_DESCRIPTION_CHARS),
        "readme": get_readme(owner, repo, head_sha),
    }

    try:
        review_results = review_all_files(file_contexts, project_context)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM review call failed: {e}")

    aggregated = aggregate_results(review_results)
    aggregated["pr_title"] = project_context["pr_title"]
    aggregated["pr_url"] = request.pr_url
    markdown_report = render_markdown(aggregated)

    return ReviewResponse(
        pr_url=request.pr_url,
        overall_summary=aggregated["overall_summary"],
        overall_recommendation=aggregated["overall_recommendation"],
        total_issues=aggregated["total_issues"],
        category_counts=aggregated["category_counts"],
        files=aggregated["files"],
        markdown_report=markdown_report,
    )