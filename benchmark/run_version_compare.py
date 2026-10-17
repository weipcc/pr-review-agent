#!/usr/bin/env python3
"""
Compare THREE real versions of the system on the SAME pull request, with the SAME ground truth
and the SAME judge:

    V1  6ddb19c   the original reviewer (baseline): Chinese prompt, per file, no schema extras
    V2  2ed8205   after the prompt rework (#5): English prompt, few-shot, project context, no tools
    V3  cbe4576   the final system (main, #9): agent + E2B sandbox, whole PR in one prompt

What is being evaluated, exactly: given a PR as it was when a human reviewed it, which of the
concerns the human raised does each version also raise, and what else does it say?

    benchmark/_final_venv/bin/python benchmark/run_version_compare.py --sample test-2809
    benchmark/_final_venv/bin/python benchmark/run_version_compare.py --pr pallets/click#3817   # no human truth
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "_final_main"))
sys.path.insert(1, str(HERE))
import run_pr_eval as pe  # noqa: E402  (final-version code + judge helpers)

REPO = HERE.parent
PY = str(HERE / "_final_venv" / "bin" / "python")
MODEL = "gemini-3.8-flash"
VERSIONS = {
    "V1": {"dir": "_v1_main", "worker": "v1", "label": "V1 baseline (6ddb19c)"},
    "V2": {"dir": "_v2_main", "worker": "v2", "label": "V2 prompt rework (2ed8205)"},
    "V3": {"dir": "_final_main", "worker": None, "label": "V3 final, sandbox agent (cbe4576)"},
}
CJK = re.compile(r"[一-鿿]")


def state_for_pr(spec: str) -> dict:
    """A PR without dataset sample: review its current head; ground truth = its human inline comments."""
    repo, n = spec.split("#")
    owner, name = repo.split("/")
    pr = pe.gh(f"/repos/{repo}/pulls/{n}").json()
    sha, base = pr["head"]["sha"], pr["base"]["sha"]
    comments = pe.gh(f"/repos/{repo}/pulls/{n}/comments", per_page=100).json()
    gt = [{"id": c["id"], "user": c["user"]["login"], "path": c["path"], "body": c["body"], "diff_hunk": c.get("diff_hunk", ""),
           "is_dataset_comment": False}
          for c in comments if c["original_commit_id"] == sha and c["user"].get("type") != "Bot" and c.get("in_reply_to_id") is None]
    raw = pe.gh(f"/repos/{repo}/compare/{base}...{sha}", accept="application/vnd.github.v3.diff").text
    return {"owner": owner, "repo": name, "full": repo, "pr_number": int(n), "title": pr["title"], "body": pr.get("body") or "",
            "base_sha": base, "review_sha": sha, "head_now": sha, "raw_diff": raw, "ground_truth": gt}


def dataset_record(sample_id: str) -> dict | None:
    """The ORIGINAL dataset row, plus what the earlier chunk-level benchmark runs said about it."""
    idx = int(sample_id.split("-")[1])
    row = None
    for line in open(HERE / "data" / "test_split.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["row_idx"] == idx:
            row = r
            break
    if row is None:
        return None
    from run_benchmark import last_by_id, normalize, paths
    earlier = {}
    for cond, kind in (("A_ver1", "ver1"), ("B_prompt", "new")):
        rec = last_by_id(paths("main")["raw"] / f"{cond}.jsonl").get(sample_id)
        if rec and rec.get("ok"):
            earlier[cond] = normalize(kind, rec["output"])
    return {"row": row, "earlier": earlier}


def to_new_schema(results: list[dict], old: bool) -> list[dict]:
    """V1 output (line/severity) -> the shape the shared judge expects."""
    if not old:
        return results
    out = []
    for f in results:
        out.append({"filename": f["filename"], "summary": f.get("summary", ""), "recommendation": "n/a",
                    "issues": [{"start_line": i["line"], "end_line": i["line"], "category": i["severity"],
                                "confidence": "n/a", "comment": i["comment"]} for i in f.get("issues", [])]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample", help="dataset sample id (e.g. test-2809): reviewed at the commit its human comment was made on")
    g.add_argument("--pr", help="owner/repo#N: reviewed at its current head")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--only", help="comma list of versions to run, e.g. V1,V2")
    ap.add_argument("--rerender", action="store_true", help="rebuild compare.md from saved results; no model calls")
    args = ap.parse_args()

    pe.load_env()
    if args.sample:
        from run_benchmark import paths, read_jsonl
        sample = {s["id"]: s for s in read_jsonl(paths("main")["samples"])}[args.sample]
        state = pe.recover_review_state(sample)
    else:
        state = state_for_pr(args.pr)
    out = HERE / "results" / "version_compare" / f"{state['owner']}__{state['repo']}__{state['pr_number']}"
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "state.json"
    state_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    print(f"PR {state['full']}#{state['pr_number']} — {state['title']}")
    print(f"reviewed at {state['review_sha'][:10]} | diff {len(state['raw_diff'])} chars | human comments: {len(state['ground_truth'])}\n")

    dataset = dataset_record(args.sample) if args.sample else None
    if args.rerender:
        saved = json.loads((out / "compare.json").read_text(encoding="utf-8"))["runs"]
        write_compare(out, state, saved, args.model, dataset)
        return

    wanted = args.only.split(",") if args.only else list(VERSIONS)
    runs: dict[str, dict] = {}
    env = dict(os.environ)
    for v in wanted:
        spec = VERSIONS[v]
        print(f"=== {spec['label']} ===")
        t0 = time.time()
        if spec["worker"]:
            of = out / f"{v}_raw.json"
            r = subprocess.run([PY, str(HERE / "_version_worker.py"), spec["worker"], str(state_path), str(of), args.model, str(REPO / ".env")],
                               cwd=str(HERE / spec["dir"]), env=env, capture_output=True, text=True)
            if r.returncode != 0:
                print("   FAILED:", (r.stderr or r.stdout)[-600:])
                continue
            raw = json.loads(of.read_text(encoding="utf-8"))
            runs[v] = {"results": to_new_schema(raw["results"], old=(v == "V1")), "n_files": raw["n_files"],
                       "files_returned": len(raw["results"]),
                       "input_tokens": raw["input_tokens"], "output_tokens": raw["output_tokens"],
                       "calls": raw["calls"], "tool_calls": None, "elapsed_s": raw["elapsed_s"]}
        else:
            try:
                rv = pe.run_review(state, args.model)
            except Exception as e:  # noqa: BLE001
                print("   FAILED:", type(e).__name__, str(e)[:300])
                continue
            runs[v] = {"results": rv["results"], "n_files": rv["n_files"], "files_returned": len(rv["results"]),
                       "input_tokens": rv["input_tokens"],
                       "output_tokens": rv["output_tokens"], "calls": None, "tool_calls": rv["tool_calls"],
                       "elapsed_s": rv["elapsed_s"]}
        n_iss = sum(len(f["issues"]) for f in runs[v]["results"])
        print(f"   files returned {len(runs[v]['results'])}/{runs[v]['n_files']} | findings {n_iss} | {time.time()-t0:.0f}s")

    # ---- judge every version against every human comment (same judge, same prompt) ----
    print("\n=== judging ===")
    for v, rv in runs.items():
        findings, verdicts = pe.judge_all(state, rv["results"], args.model)
        rv["findings"], rv["verdicts"] = findings, verdicts
        print(f"   {v}: {sum(x['matched'] for x in verdicts)}/{len(verdicts)} human comment(s) covered")

    write_compare(out, state, runs, args.model, dataset)


def write_compare(out: Path, state: dict, runs: dict, model: str, dataset: dict | None = None) -> None:
    gt = state["ground_truth"]
    L = [f"# Version comparison — {state['full']}#{state['pr_number']}\n",
         f"**{state['title']}** — reviewed as of commit `{state['review_sha'][:10]}`; model `{model}` for every version; same judge.\n"]
    if dataset:
        r, ln = dataset["row"], dataset["row"]["comment_line"]
        chunk = r["before_code"].splitlines()
        n_files_pr = state["raw_diff"].count("\ndiff --git") + (1 if state["raw_diff"].startswith("diff --git") else 0)
        n_hunks_pr = sum(1 for x in state["raw_diff"].splitlines() if x.startswith("@@"))
        L.append("## 1. The original dataset record\n")
        L.append(f"From `ronantakizawa/github-codereview`, test split, row {r['row_idx']}. These are ALL the fields; "
                 "the model never sees the comment, it only gets code.\n")
        L.append("| field | value |")
        L.append("|---|---|")
        L.append(f"| repo / PR | `{r['repo_name']}` #{r['pr_number']} — {r['pr_title']} |")
        L.append(f"| file / language | `{r['file_path']}` / {r['language']} |")
        L.append(f"| reviewer / PR author | @{r['reviewer_username']} / @{r['author_username']} |")
        L.append(f"| `reviewer_comment` | {r['reviewer_comment']} |")
        L.append(f"| `comment_line` | {ln} (line {ln} of the {len(chunk)}-line chunk) |")
        L.append(f"| `comment_type` (dataset label) | {r['comment_type']} |")
        L.append(f"| `quality_score` | {r['quality_score']:.2f} |")
        L.append(f"| `diff_context` | {len(r['diff_context'])} chars, {len(r['diff_context'].splitlines())} lines |")
        L.append(f"| `before_code` | {len(r['before_code'])} chars, {len(chunk)} lines |\n")
        L.append("`diff_context`, the hunk the comment was attached to:\n")
        L.append("```diff\n" + r["diff_context"].rstrip() + "\n```\n")
        L.append(f"`before_code`, lines {max(ln-3,1)}–{min(ln+3,len(chunk))} of the chunk (the code as the reviewer saw it):\n")
        L.append("```")
        for i in range(max(ln - 4, 0), min(ln + 3, len(chunk))):
            L.append(f"{i+1:>3}: {chunk[i]}{'    <-- comment_line' if i + 1 == ln else ''}")
        L.append("```\n")
        L.append("### How this record differs from what the comparison below used\n")
        L.append("| | the dataset gives | this comparison used |")
        L.append("|---|---|---|")
        L.append(f"| code shown to the reviewer | the commented hunk ({len(r['diff_context'].splitlines())} lines) + a {len(chunk)}-line chunk | "
                 f"the **whole PR diff** at that commit: {n_files_pr} file(s), {n_hunks_pr} hunk(s), {len(state['raw_diff'])} chars |")
        L.append(f"| human comments | 1 | {len(gt)} on that commit (read from the GitHub API) |")
        L.append(f"| which commit | not given | `{state['review_sha'][:10]}` (the comment's `original_commit_id`) |")
        L.append("| label | `comment_type` = " + str(r["comment_type"]) + " | not used: the label is automatic and can be off |")
        L.append("| what the comment is about | not stated | see section 2 |\n")
        if dataset["earlier"]:
            L.append("### What the earlier chunk-level benchmark run said about this same sample\n")
            L.append("(only the 51-line chunk as input, no PR, no repository; this is the run used for the benchmark tables)\n")
            for cond, fs in dataset["earlier"].items():
                L.append(f"- `{cond}`: " + ("no findings" if not fs else "; ".join(
                    f"line {f['line']} [{f['category']}] {pe.norm(f['comment'])[:110]}" for f in fs)))
            L.append("")
    L.append(f"## {'2. ' if dataset else ''}What the human reviewer(s) said ({len(gt)} comment(s))\n")
    if not gt:
        L.append("*No human comments exist on this PR, so there is nothing to recall; only noise can be observed.*\n")
    for i, c in enumerate(gt, 1):
        L.append(f"{i}. `{c['path']}` — @{c['user']}: {pe.norm(c['body'])[:260]}")
    cols = list(runs)
    L.append(f"\n## {'3. ' if dataset else ''}Side by side\n")
    L.append("| | " + " | ".join(VERSIONS[v]["label"] for v in cols) + " |")
    L.append("|---|" + "---|" * len(cols))

    def row(name, fn):
        L.append(f"| {name} | " + " | ".join(fn(runs[v], v) for v in cols) + " |")

    if gt:
        row("Human concerns covered (judge)", lambda r, v: f"**{sum(x['matched'] for x in r['verdicts'])}/{len(gt)}**")
    row("Files returned / in PR", lambda r, v: f"{r.get('files_returned', len(r.get('results', [])))}/{r['n_files']}")
    row("Findings", lambda r, v: str(len(r["findings"])))
    row("…of which not matching any human comment", lambda r, v: str(len(r["findings"]) - len({x['best_issue_index'] for x in r['verdicts'] if x['matched'] and x['best_issue_index'] is not None})))
    row("Language of the comments", lambda r, v: "Chinese" if any(CJK.search(f["comment"]) for f in r["findings"]) else ("English" if r["findings"] else "–"))
    row("Has confidence + verbatim evidence", lambda r, v: "no" if v == "V1" else "yes")
    row("Reads the repository (tool calls)", lambda r, v: "no" if r["tool_calls"] is None else f"{len(r['tool_calls'])} / 10")
    row("Input tokens", lambda r, v: f"{r['input_tokens']:,}")
    row("Wall time (s)", lambda r, v: str(r["elapsed_s"]))

    for v in cols:
        r = runs[v]
        L.append(f"\n## {VERSIONS[v]['label']}\n")
        by = {x["comment_id"]: x for x in r["verdicts"]}
        for i, c in enumerate(gt, 1):
            x = by[c["id"]]
            L.append(f"- human comment {i}: **{'covered' if x['matched'] else 'NOT covered'}** — {pe.norm(x['reason'])[:200]}")
        L.append(f"- findings ({len(r['findings'])}):")
        for i, f in enumerate(r["findings"]):
            L.append(f"  - [{i}] `{f['file']}:{f['start']}` {f['category']}: {pe.norm(f['comment'])[:230]}")
        if r["tool_calls"]:
            L.append("- what the agent read: " + "; ".join(f"`{pe.norm(s['command'] or '')[:70]}`" for s in r["tool_calls"][:10]))
    (out / "compare.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / "compare.json").write_text(json.dumps({"state": {k: v for k, v in state.items() if k != "raw_diff"},
                                                  "runs": {v: r for v, r in runs.items()}},
                                                 ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n" + "\n".join(L))
    print(f"\nWrote {out / 'compare.md'}")


if __name__ == "__main__":
    main()
