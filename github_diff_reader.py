"""
github_diff_reader.py
Fetches a remote PR's diff through the GitHub REST API, replacing the original approach of reading a local git diff.

Works without a GitHub token (public repos only), but anonymous requests are rate limited
(about 60 per hour), which is enough for testing small repos.
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
    Parse a PR URL into (owner, repo, pr_number).
    Example: https://github.com/octocat/Hello-World/pull/42
    """
    match = PR_URL_RE.search(pr_url)
    if not match:
        raise ValueError(f"Could not parse PR URL: {pr_url}")
    return match.group("owner"), match.group("repo"), int(match.group("number"))


def get_pr_metadata(owner: str, repo: str, pr_number: int) -> dict:
    """Get the PR's basic info (including the head commit sha, which is used later to fetch file contents)."""
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/pulls/{pr_number}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def get_pr_raw_diff(owner: str, repo: str, pr_number: int) -> str:
    """Get the PR's raw diff text (nearly the same format as a local git diff)."""
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/pulls/{pr_number}"
    headers = {"Accept": "application/vnd.github.v3.diff"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.text


def get_parsed_pr_diff(pr_url: str) -> tuple[list[FileDiff], dict]:
    """
    Convenience function: takes a PR URL and returns (parsed list of FileDiff, PR metadata).
    The head sha in the PR metadata is passed to the context builder to fetch the matching version of each file.
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
    # Quick test (replace with a PR URL from any public repo)
    test_url = "https://github.com/octocat/Hello-World/pull/1"
    diffs, meta = get_parsed_pr_diff(test_url)
    print(f"PR head sha: {meta['_head_sha']}")
    for f in diffs:
        print(f"File: {f.filename}, hunks: {len(f.hunks)}")
