"""
github_diff_reader.py
負責透過 GitHub REST API 取得遠端 PR 的 diff，取代原本讀本機 git diff 的方式。

不需要 GitHub token 也能用（僅限公開 repo），但匿名請求有速率限制
（每小時約 60 次），足夠拿來測試小型 repo。
"""

import re

import requests

from diff_reader import FileDiff, parse_diff

GITHUB_API_BASE = "https://api.github.com"

PR_URL_RE = re.compile(
    r"github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<number>\d+)"
)


def parse_pr_url(pr_url: str) -> tuple[str, str, int]:
    """
    把 PR 網址解析成 (owner, repo, pr_number)。
    例如: https://github.com/octocat/Hello-World/pull/42
    """
    match = PR_URL_RE.search(pr_url)
    if not match:
        raise ValueError(f"Could not parse PR URL: {pr_url}")
    return match.group("owner"), match.group("repo"), int(match.group("number"))


def get_pr_metadata(owner: str, repo: str, pr_number: int) -> dict:
    """取得 PR 的基本資訊（包含 head commit sha，之後抓檔案內容會用到）。"""
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/pulls/{pr_number}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def get_pr_raw_diff(owner: str, repo: str, pr_number: int) -> str:
    """取得 PR 的原始 diff 文字（格式跟本機 git diff 幾乎相同）。"""
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/pulls/{pr_number}"
    headers = {"Accept": "application/vnd.github.v3.diff"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.text


def get_parsed_pr_diff(pr_url: str) -> tuple[list[FileDiff], dict]:
    """
    整合函式：輸入 PR 網址，回傳 (解析好的 FileDiff 清單, PR metadata)。
    PR metadata 裡的 head sha 會拿去給 context builder 抓對應版本的檔案內容。
    """
    owner, repo, pr_number = parse_pr_url(pr_url)
    metadata = get_pr_metadata(owner, repo, pr_number)
    raw_diff = get_pr_raw_diff(owner, repo, pr_number)
    file_diffs = parse_diff(raw_diff)

    metadata["_owner"] = owner
    metadata["_repo"] = repo
    metadata["_pr_number"] = pr_number
    metadata["_head_sha"] = metadata["head"]["sha"]

    return file_diffs, metadata


if __name__ == "__main__":
    # 簡單測試（換成任何公開 repo 的 PR 網址）
    test_url = "https://github.com/octocat/Hello-World/pull/1"
    diffs, meta = get_parsed_pr_diff(test_url)
    print(f"PR head sha: {meta['_head_sha']}")
    for f in diffs:
        print(f"File: {f.filename}, hunks: {len(f.hunks)}")
