"""
api.py
Wraps the existing PR review pipeline with FastAPI, providing an API that can be called locally.

How to start:
    uvicorn api:app --reload --port 8000

How to call:
    curl -X POST http://localhost:8000/review \
        -H "Content-Type: application/json" \
        -d '{"pr_url": "https://github.com/owner/repo/pull/1"}'

Background poller:
    If the GITHUB_TOKEN environment variable is set, the server starts polling
    GitHub notifications after startup, automatically reviewing any PR that @mentions it and replying with a comment.
    The polling interval can be adjusted with POLL_INTERVAL_SECONDS (default 15 seconds).
"""

import asyncio
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from aggregator import aggregate_results
from context_builder import MAX_PR_DESCRIPTION_CHARS, truncate_text
from github_context_builder import build_context_for_files, get_readme
from github_diff_reader import get_parsed_pr_diff
from mention_poller import poll_once
from report import render_markdown
from reviewer import review_all_files

POLL_INTERVAL_SECONDS = int(os.environ.get("POLL_INTERVAL_SECONDS", 15))


async def _poll_loop() -> None:
    """Background task: run poll_once() on a fixed interval."""
    while True:
        print(f"[poller] Checking for @mentions ...")
        try:
            await asyncio.to_thread(poll_once)
        except RuntimeError as exc:
            print(f"[poller] Disabled: {exc}")
            return
        except Exception as exc:
            print(f"[poller] Error: {exc}")
        await asyncio.sleep(POLL_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the mention poller on startup if GITHUB_TOKEN is configured."""
    if os.environ.get("GITHUB_TOKEN", "").strip():
        print(f"[poller] Starting mention poller (interval={POLL_INTERVAL_SECONDS}s) ...")
        task = asyncio.create_task(_poll_loop())
    else:
        task = None
        print("[poller] GITHUB_TOKEN not set — mention poller is disabled.")
    yield
    if task:
        task.cancel()


app = FastAPI(title="PR Review Agent API", lifespan=lifespan)

# Allow browsers to call this local API (for local testing only; using * is fine as long as it is not publicly deployed)
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
    """Simple health check to confirm the service is up."""
    return {"status": "ok", "message": "PR Review Agent API is running"}


@app.post("/review", response_model=ReviewResponse)
def review_pr(request: ReviewRequest):
    """
    Takes a GitHub PR URL and returns the full review result.
    Only public repos are supported (no GitHub token is sent, so anonymous API rate limits apply).
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

    # Project context: PR title/description + README excerpt, so the model knows what the PR is trying to achieve and what the project does
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
