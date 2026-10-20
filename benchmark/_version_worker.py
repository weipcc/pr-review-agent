"""
Worker: runs ONE historical version of the reviewer on ONE pull-request state and writes JSON.

Called by run_version_compare.py as a subprocess with cwd = the exported source tree of that
version, so `import reviewer` resolves to THAT version's real code (no copies, no re-implementation).

    python _version_worker.py <v1|v2> <state.json> <out.json> <model> <path-to-.env>
"""

import json
import os
import re
import sys
import time

version, state_path, out_path, model, env_path = sys.argv[1:6]
sys.path.insert(0, os.getcwd())

for line in open(env_path, encoding="utf-8"):
    m = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", line)
    if m and not line.lstrip().startswith("#"):
        os.environ.setdefault(m.group(1), m.group(2).strip().strip("\"'"))

state = json.load(open(state_path, encoding="utf-8"))

# The historical versions call the GitHub REST API (file contents, README) without authentication,
# so they hit the 60 requests/hour limit. Add GITHUB_TOKEN to those requests only; nothing else changes.
import requests  # noqa: E402

_gh_token = os.environ.get("GITHUB_TOKEN", "").strip()
_orig_request = requests.sessions.Session.request


def _with_token(self, method, url, *a, **k):
    if _gh_token and str(url).startswith("https://api.github.com/"):
        headers = dict(k.pop("headers", None) or {})
        headers.setdefault("Authorization", f"Bearer {_gh_token}")
        k["headers"] = headers
    return _orig_request(self, method, url, *a, **k)


requests.sessions.Session.request = _with_token

# Count tokens of every generate_content call (these versions do not report usage themselves).
from google.genai import models as _gm  # noqa: E402

usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "call_input_tokens": []}
_orig = _gm.Models.generate_content


def _counting(self, *a, **k):
    r = _orig(self, *a, **k)
    u = getattr(r, "usage_metadata", None)
    usage["calls"] += 1
    if u:
        usage["input_tokens"] += u.prompt_token_count or 0
        usage["call_input_tokens"].append(u.prompt_token_count or 0)
        usage["output_tokens"] += u.candidates_token_count or 0
    return r


_gm.Models.generate_content = _counting

import reviewer  # noqa: E402  (this version's code)
from diff_reader import parse_diff  # noqa: E402
from github_context_builder import build_context_for_files  # noqa: E402

reviewer.MODEL_NAME = model  # same model for every version: only the code/prompt differs

file_diffs = parse_diff(state["raw_diff"])
file_contexts = build_context_for_files(state["owner"], state["repo"], state["review_sha"], file_diffs)

# V2 routes by size (batch below BATCH_TOKEN_THRESHOLD, per-file otherwise): record which route it took.
route = {"mode": "per-file (always)" if version == "v1" else None, "routing_tokens": None}
if version != "v1" and hasattr(reviewer, "count_total_tokens"):
    _orig_count = reviewer.count_total_tokens

    def _count(fc):
        n = _orig_count(fc)
        route["routing_tokens"] = n
        route["mode"] = "batch" if n < reviewer.BATCH_TOKEN_THRESHOLD else "per-file"
        return n

    reviewer.count_total_tokens = _count

t0 = time.time()
if version == "v1":
    results = reviewer.review_all_files(file_contexts)
else:
    from context_builder import MAX_PR_DESCRIPTION_CHARS, truncate_text
    from github_context_builder import get_readme

    project_context = {
        "pr_title": state["title"],
        "pr_description": truncate_text(state["body"], MAX_PR_DESCRIPTION_CHARS),
        "readme": get_readme(state["owner"], state["repo"], state["review_sha"]),
    }
    results = reviewer.review_all_files(file_contexts, project_context)

json.dump(
    {"results": results, "n_files": len(file_diffs), "elapsed_s": round(time.time() - t0, 1), **usage, **route},
    open(out_path, "w", encoding="utf-8"),
    ensure_ascii=False,
    indent=1,
)
print(f"[{version}] done: {len(results)} file result(s), {usage['calls']} model call(s)")
