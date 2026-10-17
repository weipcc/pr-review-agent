#!/usr/bin/env python3
"""
End-to-end evaluation of the FINAL system (main after PR #9) on ONE real pull request,
reviewed in the state it had when a human reviewer commented.

Why this exists: the 50-line chunks of github-codereview cannot exercise the sandbox agent.
Here the whole PR (all files, one prompt) goes through the real pipeline, the agent reads the
repository in a real E2B sandbox, and its findings are compared with every human review
comment that was left on that commit.

Steps (each one is printed and saved to results/pr_eval/<owner>__<repo>__<pr>/):
  1. Pick a sample of the github-codereview benchmark (default test-522) and find the same
     comment on GitHub, which gives the commit it was written on (`original_commit_id`).
  2. Build the PR diff AT THAT COMMIT: GitHub compare API, base...commit.
  3. Ground truth = every top-level, non-bot inline comment whose original commit is that commit.
  4. Review: the pipeline code of main (pinned export in benchmark/_final_main): size check,
     project context, E2B sandbox checked out at that commit, Deep Agent with the run_bash tool.
  5. Judge: for each human comment, an LLM judge says whether some finding raises the same
     concern. Findings that match nothing are listed, not scored.

Run with the venv that has main's requirements:
    benchmark/_final_venv/bin/python benchmark/run_pr_eval.py --dry-run      # no keys needed
    benchmark/_final_venv/bin/python benchmark/run_pr_eval.py                # needs GEMINI_API_KEY, E2B_KEY
Keys are read from <repo>/.env and never printed.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FINAL = HERE / "_final_main"                 # pinned export of origin/main
sys.path.insert(0, str(FINAL))               # so `import reviewer` is the FINAL one
sys.path.insert(1, str(HERE))                # for run_benchmark (samples, judge prompt)

DEFAULT_MODEL = "gemini-3.8-flash"


# ----------------------------------------------------------------------------- utilities
def load_env() -> dict:
    """Read <repo>/.env into os.environ (values never printed). Returns {name: bool(set)}."""
    status = {}
    p = REPO / ".env"
    if p.is_file():
        for line in p.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", line)
            if m and not line.lstrip().startswith("#"):
                v = m.group(2).strip().strip("\"'")
                if v and m.group(1) not in os.environ:
                    os.environ[m.group(1)] = v
                status[m.group(1)] = bool(v)
    return status


def gh_headers(accept: str = "application/vnd.github+json") -> dict:
    h = {"Accept": accept}
    tok = os.environ.get("GITHUB_TOKEN", "").strip()
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def gh(path: str, accept: str = "application/vnd.github+json", **params):
    r = requests.get("https://api.github.com" + path, headers=gh_headers(accept), params=params, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"GitHub API {path} -> HTTP {r.status_code}: {r.text[:200]}")
    return r


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip()


# ----------------------------------------------------------------------------- 1-3: input + ground truth
def recover_review_state(sample: dict) -> dict:
    repo, n, path = sample["repo_name"], sample["pr_number"], sample["file_path"]
    owner, name = repo.split("/")
    pr = gh(f"/repos/{repo}/pulls/{n}").json()

    comments, page = [], 1
    while True:
        chunk = gh(f"/repos/{repo}/pulls/{n}/comments", per_page=100, page=page).json()
        comments += chunk
        if len(chunk) < 100:
            break
        page += 1

    target = norm(sample["reviewer_comment"])
    mine = next((c for c in comments if c["path"] == path and norm(c["body"]) == target), None)
    if mine is None:
        raise RuntimeError("Could not find the dataset comment among the PR's inline comments.")
    sha = mine["original_commit_id"]
    base = pr["base"]["sha"]

    ground_truth = [
        {"id": c["id"], "user": c["user"]["login"], "path": c["path"], "body": c["body"],
         "diff_hunk": c.get("diff_hunk", ""), "is_dataset_comment": c["id"] == mine["id"]}
        for c in comments
        if c["original_commit_id"] == sha and c["user"].get("type") != "Bot" and c.get("in_reply_to_id") is None
    ]
    raw_diff = gh(f"/repos/{repo}/compare/{base}...{sha}", accept="application/vnd.github.v3.diff").text
    return {"owner": owner, "repo": name, "full": repo, "pr_number": n, "title": pr["title"], "body": pr.get("body") or "",
            "base_sha": base, "review_sha": sha, "head_now": pr["head"]["sha"], "raw_diff": raw_diff,
            "ground_truth": ground_truth}


# ----------------------------------------------------------------------------- 4: review
def make_sandbox_class(sandbox_tools):
    """PRSandbox, but checked out at an arbitrary commit instead of the PR's current head."""

    class ReviewTimeSandbox(sandbox_tools.PRSandbox):
        def _checkout(self) -> None:
            token = os.environ.get("GITHUB_TOKEN", "").strip()
            auth = ""
            if token:
                basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
                auth = f"-c http.extraheader='Authorization: Basic {basic}' "
            url = f"https://github.com/{self.owner}/{self.repo}.git"
            rd = sandbox_tools.REPO_DIR
            head = ["mkdir -p " + rd, "cd " + rd, "git init -q", f"git remote add origin {url}"]
            by_sha = " && ".join(head + [
                f"git {auth}fetch -q --depth=1 origin {self.head_sha}",
                f"git {auth}fetch -q --depth=1 origin {self.base_sha}",
                f"git checkout -q --detach {self.head_sha}",
            ])
            by_ref = " && ".join(head + [   # fallback: the PR ref with some history, then check the commit out
                f"git {auth}fetch -q --depth=200 origin refs/pull/{self.pr_number}/head",
                f"git {auth}fetch -q --depth=1 origin {self.base_sha}",
                f"git checkout -q --detach {self.head_sha}",
            ])
            for how, script in (("fetch by SHA", by_sha), ("fetch PR ref", by_ref)):
                try:
                    res = self._sandbox.commands.run(script, timeout=240)
                    if res.exit_code == 0:
                        print(f"   sandbox checkout OK ({how})")
                        return
                except Exception as e:  # noqa: BLE001
                    print(f"   sandbox checkout via '{how}' failed: {str(e)[:160]}")
            raise RuntimeError("Sandbox checkout failed for both strategies.")

    return ReviewTimeSandbox


class TraceHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(self.format(record))


def run_review(state: dict, model_name: str) -> dict:
    import reviewer
    import sandbox_tools
    from context_builder import MAX_PR_DESCRIPTION_CHARS, build_diffs_for_files, truncate_text
    from diff_reader import parse_diff
    from github_context_builder import get_readme

    if model_name != reviewer.MODEL_NAME:
        reviewer.MODEL_NAME = model_name
        reviewer._model = None  # force re-creation with the chosen name

    file_diffs = parse_diff(state["raw_diff"])
    file_contexts = build_diffs_for_files(file_diffs)
    project_context = {
        "pr_title": state["title"],
        "pr_description": truncate_text(state["body"], MAX_PR_DESCRIPTION_CHARS),
        "readme": get_readme(state["owner"], state["repo"], state["review_sha"]),
    }
    n_tokens = reviewer.check_pr_size(file_contexts)  # raises PRTooLargeError, as in production

    if not sandbox_tools._api_key():
        raise RuntimeError("E2B_KEY / E2B_API_KEY is not set.")
    Box = make_sandbox_class(sandbox_tools)
    th = TraceHandler()
    th.setFormatter(logging.Formatter("%(message)s"))
    lg = logging.getLogger("pr_review")
    lg.setLevel(logging.DEBUG)
    lg.addHandler(th)

    t0 = time.time()
    box = Box(owner=state["owner"], repo=state["repo"], pr_number=state["pr_number"],
              head_sha=state["review_sha"], base_sha=state["base_sha"])
    try:
        # Same three lines as reviewer.review_pr_files, but we keep the message trace
        system_prompt = reviewer.build_system_prompt_batch()
        user_message = reviewer.build_batch_user_message(file_contexts, project_context)
        messages = reviewer.build_batch_few_shot_contents() + [{"role": "user", "content": user_message}]
        agent = reviewer._build_sandbox_agent(system_prompt, reviewer.BatchReview, box)
        result = agent.invoke({"messages": messages},
                              config={"recursion_limit": 2 * sandbox_tools.MAX_STEPS + 25})
    finally:
        box.close()
    elapsed = time.time() - t0

    parsed = result["structured_response"]
    results = [item.model_dump() for item in parsed.files]

    # Trace: tool calls, outputs, tokens (few-shot messages carry no usage_metadata and are skipped)
    steps, tin, tout = [], 0, 0
    outputs = {}
    for m in result["messages"]:
        if m.__class__.__name__ == "ToolMessage":
            outputs[getattr(m, "tool_call_id", None)] = str(m.content)
    for m in result["messages"]:
        if m.__class__.__name__ != "AIMessage":
            continue
        um = getattr(m, "usage_metadata", None)
        if um:
            tin += um.get("input_tokens", 0); tout += um.get("output_tokens", 0)
        for tc in getattr(m, "tool_calls", None) or []:
            steps.append({"command": (tc.get("args") or {}).get("command"),
                          "output_preview": outputs.get(tc.get("id"), "")[:400]})
    return {"results": results, "diff_tokens": n_tokens, "n_files": len(file_diffs),
            "agent_steps": len(steps), "tool_calls": steps, "input_tokens": tin, "output_tokens": tout,
            "elapsed_s": round(elapsed, 1), "log_lines": th.lines[-60:]}


# ----------------------------------------------------------------------------- 5: judge
def judge_all(state: dict, results: list[dict], model_name: str) -> list[dict]:
    from run_benchmark import JUDGE_SYSTEM, JudgeVerdict, call_model, ensure_api_key
    ensure_api_key()
    findings = []
    for f in results:
        for it in f.get("issues", []):
            findings.append({"file": f["filename"], "start": it["start_line"], "end": it["end_line"],
                             "category": it["category"], "confidence": it["confidence"], "comment": it["comment"]})
    listing = "\n".join(f"[{i}] {x['file']}:{x['start']}-{x['end']} · {x['category']} ({x['confidence']}): {x['comment']}"
                        for i, x in enumerate(findings)) or "(no findings)"
    verdicts = []
    for c in state["ground_truth"]:
        if not findings:
            verdicts.append({"comment_id": c["id"], "matched": False, "best_issue_index": None, "reason": "no findings"})
            continue
        prompt = (f"<human_comment file=\"{c['path']}\">\n{c['body']}\n</human_comment>\n\n"
                  f"<code_the_comment_was_attached_to>\n{c['diff_hunk']}\n</code_the_comment_was_attached_to>\n\n"
                  f"<automated_findings>\n{listing}\n</automated_findings>")
        v, _ = call_model(model_name, JUDGE_SYSTEM, prompt, JudgeVerdict, temperature=0.0)
        verdicts.append({"comment_id": c["id"], "matched": bool(v.matched), "best_issue_index": v.best_issue_index,
                         "reason": v.reason})
    return findings, verdicts


# ----------------------------------------------------------------------------- reporting
def write_outputs(out: Path, state: dict, rv: dict | None, findings, verdicts, model: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    gt = state["ground_truth"]
    lines = [f"# Evaluation of the final system on one PR\n",
             f"- PR: https://github.com/{state['full']}/pull/{state['pr_number']} — {state['title']}",
             f"- Reviewed state: commit `{state['review_sha'][:10]}` (PR head today: `{state['head_now'][:10]}`), base `{state['base_sha'][:10]}`",
             f"- Model: `{model}`; system code: main @ `{(FINAL / '.exported_from_sha').read_text().strip()[:7]}`\n"]
    lines.append(f"## Ground truth: {len(gt)} human inline comment(s) on that commit\n")
    for i, c in enumerate(gt):
        lines.append(f"{i + 1}. `{c['path']}` — @{c['user']}{' (the one in the dataset)' if c['is_dataset_comment'] else ''}: {norm(c['body'])[:200]}")
    if rv is None:
        lines.append("\n*(dry run: no review was executed)*")
    else:
        lines.append(f"\n## Findings of the system: {len(findings)}\n")
        for i, x in enumerate(findings):
            lines.append(f"- [{i}] `{x['file']}:{x['start']}-{x['end']}` · {x['category']} · {x['confidence']}: {norm(x['comment'])[:220]}")
        lines.append("\n## Matching (LLM judge)\n")
        lines.append("| # | Human comment | Covered? | By finding | Judge's reason |")
        lines.append("|---|---|---|---|---|")
        by = {v["comment_id"]: v for v in verdicts}
        n_ok = 0
        for i, c in enumerate(gt):
            v = by[c["id"]]; n_ok += v["matched"]
            bi = v["best_issue_index"]
            lines.append(f"| {i + 1} | {norm(c['body'])[:70]} | {'**yes**' if v['matched'] else 'no'} | "
                         f"{('[%d]' % bi) if bi is not None and v['matched'] else '–'} | {norm(v['reason'])[:140]} |")
        used = {v["best_issue_index"] for v in verdicts if v["matched"] and v["best_issue_index"] is not None}
        unmatched = [i for i in range(len(findings)) if i not in used]
        lines.append(f"\n**Recall: {n_ok}/{len(gt)} human comments covered.**  "
                     f"Findings not used to match any comment: {len(unmatched)}/{len(findings)} "
                     f"(noise or issues the humans did not raise: read them).")
        lines.append("\n## Agent\n")
        lines.append(f"- files in the PR: {rv['n_files']}; diff size ≈ {rv['diff_tokens']} tokens (limit 6000)")
        lines.append(f"- sandbox tool calls: {rv['agent_steps']} (budget 10); input tokens {rv['input_tokens']:,}, output tokens {rv['output_tokens']:,}; wall time {rv['elapsed_s']} s")
        for i, s in enumerate(rv["tool_calls"], 1):
            lines.append(f"  {i}. `{norm(s['command'] or '')[:140]}`")
    (out / "eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "result.json").write_text(json.dumps({
        "state": {k: v for k, v in state.items() if k != "raw_diff"}, "review": rv, "findings": findings,
        "verdicts": verdicts, "model": model}, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {out / 'eval.md'} and result.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", default="test-522", help="sample id from benchmark/data/samples_main.jsonl")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dry-run", action="store_true", help="only build the input and the ground truth")
    args = ap.parse_args()

    env = load_env()
    print("keys found in .env:", {k: ("set" if v else "EMPTY") for k, v in env.items()} or "(none)")

    from run_benchmark import paths, read_jsonl
    samples = {s["id"]: s for s in read_jsonl(paths("main")["samples"])}
    sample = samples[args.sample]
    print(f"\n[1-3] {sample['repo_name']}#{sample['pr_number']}  ({sample['comment_type']} comment in {sample['file_path']})")
    state = recover_review_state(sample)
    print(f"   review-time commit {state['review_sha'][:10]} | base {state['base_sha'][:10]} | "
          f"diff {len(state['raw_diff'])} chars (≈{len(state['raw_diff']) // 4} tokens) | ground truth: {len(state['ground_truth'])} comment(s)")

    out = REPO / "benchmark" / "results" / "pr_eval" / f"{state['owner']}__{state['repo']}__{state['pr_number']}"
    if args.dry_run:
        write_outputs(out, state, None, [], [], args.model)
        return
    missing = [k for k in ("GEMINI_API_KEY",) if not os.environ.get(k)]
    if not (os.environ.get("E2B_KEY") or os.environ.get("E2B_API_KEY")):
        missing.append("E2B_KEY")
    if missing:
        sys.exit(f"Missing in .env: {missing}. Nothing was run.")

    print("\n[4] reviewing with the real pipeline (E2B sandbox) ...")
    rv = run_review(state, args.model)
    print(f"   done: {rv['agent_steps']} tool call(s), {rv['input_tokens']:,} input tokens, {rv['elapsed_s']} s")
    print("\n[5] judging ...")
    findings, verdicts = judge_all(state, rv["results"], args.model)
    write_outputs(out, state, rv, findings, verdicts, args.model)

    # also keep the report exactly as the product would post it
    from aggregator import aggregate_results
    from report import render_markdown
    agg = aggregate_results(rv["results"])
    agg["pr_title"] = state["title"]
    agg["pr_url"] = f"https://github.com/{state['full']}/pull/{state['pr_number']}"
    (out / "report.md").write_text(render_markdown(agg), encoding="utf-8")


if __name__ == "__main__":
    main()
