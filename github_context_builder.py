"""
github_context_builder.py
負責透過 GitHub API 讀取 PR head commit 版本的 README。
"""

import base64

import requests

from context_builder import MAX_README_CHARS, truncate_text

GITHUB_API_BASE = "https://api.github.com"


def get_readme(owner: str, repo: str, ref: str) -> str:
    """
    透過 GitHub README API 取得指定 commit（ref）版本的 README，並截斷到長度上限。
    README 只是補充背景，抓不到（沒有 README、網路或速率限制問題）就回傳空字串，
    不要讓整個審查因此失敗。
    """
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/readme"
    try:
        response = requests.get(url, params={"ref": ref}, timeout=30)
        if response.status_code == 404:
            return ""
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        print(f"Warning: could not fetch README ({e}); continuing without it.")
        return ""

    if data.get("encoding") == "base64":
        text = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    else:
        text = data.get("content", "")
    return truncate_text(text, MAX_README_CHARS)
