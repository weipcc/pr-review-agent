"""
github_context_builder.py
Reads the README at the PR head commit through the GitHub API.
"""

import base64

import requests

from context_builder import MAX_README_CHARS, truncate_text

GITHUB_API_BASE = "https://api.github.com"


def get_readme(owner: str, repo: str, ref: str) -> str:
    """
    Fetches the README at the given commit (ref) through the GitHub README API and truncates it to the length limit.
    The README is only supplementary background; if it cannot be fetched (no README, network or rate-limit problems), an empty string is returned
    so that the whole review does not fail because of it.
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
