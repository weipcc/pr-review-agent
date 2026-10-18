#!/usr/bin/env python3
"""
Evaluate V1 / V2 / V3 on the 10 shortlisted AACR-Bench pull requests, judged with the official AACR-Bench protocol (LLM judge).

    V1  6ddb19c   baseline: framework only, original prompt
    V2  2ed8205   prompt rework (workflow version, no tools)
    V3  cbe4576   final system: agent + E2B sandbox

Ground truth = every AACR comment the AACR experts verified (label = 1), whether a human or an
AI wrote it (`source` records which). Matching follows alibaba/aacr-bench evaluation/judge.py
(file-level semantic coverage): one judge call per finding against the same file's reference comments,
official prompt, temperature 0, greedy dedup for precision; results in judge_aacr_V?.json. Each PR is reviewed at `pr_target_commit` (the commit the comments were made on),
diffed against `pr_source_commit` (the PR base).

    # 1. build inputs + ground truth only (GitHub API, no model, no sandbox)
    benchmark/_final_venv/bin/python benchmark/run_aacr_eval.py build
    # 2. review (costs tokens); results are cached per PR and version, rerun only with --force
    benchmark/_final_venv/bin/python benchmark/run_aacr_eval.py run --pr valkey-io/valkey#1889 --only V1,V2,V3
    # 3. LLM judge (AACR-Bench protocol: one call per finding), then the tables
    benchmark/_final_venv/bin/python benchmark/run_aacr_eval.py judge
    benchmark/_final_venv/bin/python benchmark/run_aacr_eval.py summary
    # optional: spot-check sheet pre-filled with the judge's verdicts (fill `agree`, rerun summary)
    benchmark/_final_venv/bin/python benchmark/run_aacr_eval.py sheet
"""

from __future__ import annotations

import argparse
import csv
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
import run_pr_eval as pe  # noqa: E402
import run_version_compare as rvc  # noqa: E402

SEL = HERE / "data" / "selection" / "aacr"
OUT = HERE / "results" / "aacr"
MAX_DIFF_TOKENS = 6000  # V3 rejects larger PRs (reviewer.BATCH_TOKEN_THRESHOLD)


def pr_dir(url: str) -> Path:
    owner, repo, _, n = url.rstrip("/").split("/")[-4:]
    return OUT / f"{owner}__{repo}__{n}"


def shortlist() -> list[str]:
    return [x["url"] for x in json.loads((SEL / "shortlist_aacr.json").read_text(encoding="utf-8"))]


def hunk_for(raw_diff: str, path: str, line: int, slack: int = 5) -> str:
    """The hunk of `path` around `line` on the new side (for the judging sheet)."""
    m = re.search(rf"^diff --git a/{re.escape(path)} b/{re.escape(path)}\n(.*?)(?=^diff --git |\Z)", raw_diff, re.S | re.M)
    if not m:
        return ""
    best, gap = "", None  # the hunk containing the line, else the nearest one within `slack` lines
    for h in re.split(r"(?m)^(?=@@ )", m.group(1)):
        hm = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", h)
        if not hm:
            continue
        lo, hi = int(hm.group(1)), int(hm.group(1)) + int(hm.group(2) or 1) - 1
        g = max(lo - line, line - hi, 0)
        if g <= slack and (gap is None or g < gap):
            best, gap = h.rstrip(), g
    return best


# ----------------------------------------------------------------------------- build
def build_state(url: str, rows: list[dict]) -> dict:
    owner, repo, _, n = url.rstrip("/").split("/")[-4:]
    full = f"{owner}/{repo}"
    base, sha = rows[0]["pr_source_commit"], rows[0]["pr_target_commit"]
    pr = pe.gh(f"/repos/{full}/pulls/{n}").json()
    raw = pe.gh(f"/repos/{full}/compare/{base}...{sha}", accept="application/vnd.github.v3.diff").text
    gt = [{"id": i, "user": "AI (AACR, verified)" if r["is_ai_comment"] else "human (AACR, verified)",
           "source": "ai" if r["is_ai_comment"] else "human", "path": r["path"], "body": r["note"], "from_line": r["from_line"],
           "to_line": r["to_line"], "category": r["category"], "context": r["context"],
           "diff_hunk": hunk_for(raw, r["path"], r["to_line"]), "is_dataset_comment": True}
          for i, r in enumerate(x for x in rows if x["label"] == 1)]
    return {"owner": owner, "repo": repo, "full": full, "pr_number": int(n), "title": pr["title"],
            "body": pr.get("body") or "", "base_sha": base, "review_sha": sha, "head_now": pr["head"]["sha"],
            "raw_diff": raw, "ground_truth": gt, "language": rows[0]["project_main_language"]}


def check(state: dict) -> list[str]:
    raw = state["raw_diff"]
    files = re.findall(r"(?m)^diff --git a/(\S+) b/", raw)
    problems = []
    if not raw.strip():
        problems.append("empty diff")
    if len(raw) // 4 > MAX_DIFF_TOKENS:
        problems.append(f"diff ≈{len(raw) // 4} tokens > {MAX_DIFF_TOKENS}: V3 would reject it")
    for c in state["ground_truth"]:
        if c["path"] not in files:
            problems.append(f"comment {c['id']} is on {c['path']}, which is not in the diff (repo-level context)")
        elif not c["diff_hunk"]:
            problems.append(f"comment {c['id']} line {c['to_line']} is outside every hunk of {c['path']}")
    return problems


def cmd_build(urls: list[str], force: bool) -> None:
    data = json.loads((SEL / "aacr.json").read_text(encoding="utf-8"))
    rows_by = {}
    for r in data:
        rows_by.setdefault(r["pr_url"], []).append(r)
    print(f"{'PR':45} {'lang':10} files  diff≈tok  GT  problems")
    for url in urls:
        d = pr_dir(url)
        d.mkdir(parents=True, exist_ok=True)
        sp = d / "state.json"
        if sp.exists() and not force:
            state = json.loads(sp.read_text(encoding="utf-8"))
        else:
            try:
                state = build_state(url, rows_by[url])
            except Exception as e:  # noqa: BLE001
                print(f"{url:45} FAILED: {str(e)[:200]}")
                continue
            sp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
        files = re.findall(r"(?m)^diff --git a/(\S+) b/", state["raw_diff"])
        probs = check(state)
        name = f"{state['full']}#{state['pr_number']}"
        print(f"{name:45} {state['language']:10} {len(files):>5}  {len(state['raw_diff']) // 4:>7}  {len(state['ground_truth']):>2}  "
              + ("; ".join(probs) or "ok"))


# ----------------------------------------------------------------------------- run
def run_version(state: dict, sp: Path, v: str, model: str) -> dict | None:
    spec = rvc.VERSIONS[v]
    if spec["worker"]:
        of = sp.parent / f"{v}_raw.json"
        r = subprocess.run([rvc.PY, str(HERE / "_version_worker.py"), spec["worker"], str(sp), str(of), model, str(rvc.REPO / ".env")],
                           cwd=str(HERE / spec["dir"]), env=dict(os.environ), capture_output=True, text=True)
        if r.returncode != 0:
            print("   FAILED:", (r.stderr or r.stdout)[-600:])
            return None
        raw = json.loads(of.read_text(encoding="utf-8"))
        return {"results": rvc.to_new_schema(raw["results"], old=(v == "V1")), "n_files": raw["n_files"],
                "input_tokens": raw["input_tokens"], "output_tokens": raw["output_tokens"],
                "calls": raw["calls"], "tool_calls": None, "elapsed_s": raw["elapsed_s"]}
    try:
        rv = pe.run_review(state, model)
    except Exception as e:  # noqa: BLE001
        print("   FAILED:", type(e).__name__, str(e)[:300])
        return None
    return {"results": rv["results"], "n_files": rv["n_files"], "input_tokens": rv["input_tokens"],
            "output_tokens": rv["output_tokens"], "calls": None, "tool_calls": rv["tool_calls"], "elapsed_s": rv["elapsed_s"]}


def cmd_run(urls: list[str], versions: list[str], model: str, force: bool) -> None:
    pe.load_env()
    for url in urls:
        sp = pr_dir(url) / "state.json"
        if not sp.exists():
            sys.exit(f"{url}: run `build` first.")
        state = json.loads(sp.read_text(encoding="utf-8"))
        print(f"\n### {state['full']}#{state['pr_number']} — {state['title']}")
        for v in versions:
            rp = sp.parent / f"{v}.json"
            if rp.exists() and not force:
                print(f"   {v}: cached, skipped")
                continue
            print(f"=== {rvc.VERSIONS[v]['label']} ===")
            t0 = time.time()
            res = run_version(state, sp, v, model)
            if res is None:
                continue
            res["model"] = model
            rp.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
            n = sum(len(f["issues"]) for f in res["results"])
            print(f"   files returned {len(res['results'])}/{res['n_files']} | findings {n} | "
                  f"{res['input_tokens']:,} input tokens | {time.time() - t0:.0f}s")


# ----------------------------------------------------------------------------- judging (AACR-Bench official protocol)
# Mirrors alibaba/aacr-bench evaluation/judge.py (file-level semantic coverage, docs/metrics.md):
# one judge call per generated finding against all reference comments of the SAME file, the official
# prompt verbatim, temperature 0; findings on files without reference comments are unmatched without
# a call; precision counts a finding only if it covers a reference not covered by an earlier finding.
AACR_PROMPT = """
-Role-

You are an expert code reviewer comparing review comments for semantic equivalence.

-Task-

Given a Generated Comment and a numbered list of Reference Comments (all from the same file: {file_path}), determine which Reference Comments express the same concern or suggestion as the Generated Comment.

Focus solely on semantic equivalence of the underlying issue — ignore differences in wording, tone, formatting, or exact line numbers. If the Generated Comment addresses the same core problem as a Reference Comment, include that Reference Comment's number.

-Generated Comment-
[L{gen_lines}] {gen_note}

-Reference Comments-
{numbered_refs}

-Output-

For each Reference Comment, briefly state whether it matches and why (one line each).
Then return a JSON array of matched numbers on the LAST line. If none match, return [].

Example:
1. Match — both flag the uninitialized variable risk
2. No match — ref discusses naming, generated discusses null check
3. Match — same concern about missing error handling

[1, 3]

Your answer:
"""


def findings_of(res: dict) -> list[dict]:
    """All findings of one review, in generation order (file order, then issue order)."""
    return [{"file": f["filename"], "start": it["start_line"], "end": it["end_line"], "category": it["category"],
             "comment": pe.norm(it["comment"])} for f in res["results"] for it in f.get("issues", [])]


def norm_path(p: str) -> str:
    return (p or "").replace("\\/", "/").replace("\\", "/")


def call_judge(prompt: str, model: str, attempts: int = 6) -> str:
    from run_benchmark import FATAL_MARKERS, RETRY_MARKERS, FatalAPIError, ensure_api_key, get_client

    ensure_api_key()
    last = None
    for a in range(attempts):
        try:
            r = get_client().models.generate_content(
                model=model, contents=prompt, config={"temperature": 0, "top_p": 0.95, "max_output_tokens": 3000})
            return (r.text or "").strip()
        except Exception as e:  # noqa: BLE001
            last = e
            if any(m in str(e) for m in FATAL_MARKERS):
                raise FatalAPIError(str(e)[:300]) from e
            if not any(m in str(e) for m in RETRY_MARKERS) or a == attempts - 1:
                break
            time.sleep(min(2 ** a, 30))
    raise last


def judge_official(state: dict, fs: list[dict], model: str) -> list[dict]:
    refs_by_path: dict[str, list] = {}
    for c in state["ground_truth"]:
        refs_by_path.setdefault(norm_path(c["path"]), []).append(c)
    records = []
    for k, x in enumerate(fs):
        refs = refs_by_path.get(norm_path(x["file"]), [])
        rec = {"finding": k, "file": x["file"], "matched_gt": [], "reason": "no reference comments in this file (no judge call)"}
        if refs:
            prompt = AACR_PROMPT.format(
                file_path=norm_path(x["file"]), gen_lines=f"{x['start']}-{x['end']}", gen_note=x["comment"],
                numbered_refs="\n".join(f"{i + 1}. [L{c['from_line']}-{c['to_line']}] {c['body']}" for i, c in enumerate(refs)))
            text = call_judge(prompt, model)
            arrays = re.findall(r"\[[\d,\s]*\]", text)
            nums = json.loads(arrays[-1]) if arrays else []
            rec["matched_gt"] = sorted({refs[n - 1]["id"] for n in nums if isinstance(n, int) and 1 <= n <= len(refs)})
            rec["reason"] = text[:1500]
        records.append(rec)
    covered: set = set()
    for rec in records:  # greedy dedup in generation order
        new = set(rec["matched_gt"]) - covered
        rec["hit_new_gt"] = bool(new)
        covered |= set(rec["matched_gt"])
    return records


def cmd_judge(urls: list[str], versions: list[str], model: str, force: bool) -> None:
    pe.load_env()
    for url in urls:
        d = pr_dir(url)
        state = json.loads((d / "state.json").read_text(encoding="utf-8"))
        for v in versions:
            rp, jp = d / f"{v}.json", d / f"judge_aacr_{v}.json"
            if not rp.exists():
                continue
            fs = findings_of(json.loads(rp.read_text(encoding="utf-8")))
            key = {"protocol": "aacr-bench file_semantic", "model": model, "n_truth": len(state["ground_truth"]), "findings": fs}
            if jp.exists() and not force and json.loads(jp.read_text(encoding="utf-8"))["key"] == key:
                continue
            recs = judge_official(state, fs, model)
            jp.write_text(json.dumps({"key": key, "records": recs}, ensure_ascii=False, indent=1), encoding="utf-8")
            cov = {g for r in recs for g in r["matched_gt"]}
            print(f"{state['full']}#{state['pr_number']} {v}: covered {len(cov)}/{len(state['ground_truth'])} references, "
                  f"{sum(r['hit_new_gt'] for r in recs)}/{len(fs)} findings effective")


def load_judged(d: Path, v: str) -> tuple[dict, dict, dict] | None:
    rp, jp = d / f"{v}.json", d / f"judge_aacr_{v}.json"
    if not (rp.exists() and jp.exists()):
        return None
    return (json.loads((d / "state.json").read_text(encoding="utf-8")), json.loads(rp.read_text(encoding="utf-8")),
            json.loads(jp.read_text(encoding="utf-8")))


# ----------------------------------------------------------------------------- spot-check sheet + summary
def cmd_sheet() -> None:
    """spotcheck.csv: one row per (PR, version, finding judged against same-file references), pre-filled with the
    judge's verdict; a person fills `agree` (y/n) on a subset to measure judge agreement."""
    rows = []
    for d in sorted(OUT.glob("*__*__*")):
        for v in rvc.VERSIONS:
            got = load_judged(d, v)
            if not got:
                continue
            state, _, j = got
            refs = {c["id"]: c for c in state["ground_truth"]}
            for x, rec in zip(j["key"]["findings"], j["records"]):
                same_file = [c for c in state["ground_truth"] if norm_path(c["path"]) == norm_path(x["file"])]
                if not same_file:
                    continue
                rows.append({"pr": f"{state['full']}#{state['pr_number']}", "version": v, "finding_no": rec["finding"],
                             "finding": f"{x['file']}:{x['start']} {x['category']}: {x['comment']}",
                             "references": "\n".join(f"G{c['id']} L{c['from_line']}-{c['to_line']}: {pe.norm(c['body'])}" for c in same_file),
                             "judge_matched": " ".join(f"G{g}" for g in rec["matched_gt"]) or "none",
                             "judge_reason": rec["reason"], "agree": ""})
    p = OUT / "spotcheck.csv"
    if p.exists():
        sys.exit(f"{p} exists, not overwritten (delete it to regenerate)")
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {p} ({len(rows)} rows)")


def cmd_summary() -> None:
    agg = {v: {"prs": 0, "tok": 0, "s": 0.0, "nofind": 0, "missing": 0, "cap": 0, "find": 0, "eff": 0, "dup": 0,
               "ref": {}, "ctx": {}} for v in rvc.VERSIONS}
    per_pr = []
    for d in sorted(OUT.glob("*__*__*")):
        state = json.loads((d / "state.json").read_text(encoding="utf-8"))
        row = {"pr": f"{state['full'].split('/')[1]}#{state['pr_number']}", "lang": state["language"], "n": len(state["ground_truth"])}
        for v in rvc.VERSIONS:
            got = load_judged(d, v)
            if not got:
                continue
            _, r, j = got
            a = agg[v]
            recs = j["records"]
            covered = {g for x in recs for g in x["matched_gt"]}
            a["prs"] += 1; a["tok"] += r["input_tokens"]; a["s"] += r["elapsed_s"]
            a["nofind"] += not recs
            a["missing"] += len(r["results"]) < r["n_files"]
            a["cap"] += bool(r["tool_calls"] is not None and len(r["tool_calls"]) >= 10)
            a["find"] += len(recs)
            a["eff"] += sum(x["hit_new_gt"] for x in recs)
            a["dup"] += sum(bool(x["matched_gt"]) and not x["hit_new_gt"] for x in recs)
            for c in state["ground_truth"]:
                m = c["id"] in covered
                for k, key in (("ref", c["source"]), ("ref", "all"), ("ctx", c["context"])):
                    t = a[k].setdefault(key, [0, 0]); t[0] += m; t[1] += 1
            row[v] = f"{len(covered)} / {sum(x['hit_new_gt'] for x in recs)} / {len(recs)}"
        per_pr.append(row)
    vs = [v for v in rvc.VERSIONS if agg[v]["prs"]]
    pct = lambda h, n: f"{h}/{n} ({100 * h / n:.0f}%)" if n else "–"  # noqa: E731
    print("Protocol: AACR-Bench file-level semantic coverage (official judge prompt, temperature 0, greedy dedup).\n")
    print("## Table 1: overall\n")
    print("| version | PRs | recall (all) | recall (human-written) | recall (AI-written) | precision | balanced score | findings | duplicate hits | mean input tokens | mean time (s) | PRs with no findings | missing files in output | hit 10-step cap |")
    print("|---|" + "---|" * 13)
    for v in vs:
        a = agg[v]
        rh, rn = a["ref"].get("all", [0, 0])
        rec, prec = (rh / rn if rn else 0), (a["eff"] / a["find"] if a["find"] else 0)
        bal = 2 * rec * prec / (rec + prec) if rec + prec else 0
        print(f"| {v} | {a['prs']} | {pct(rh, rn)} | {pct(*a['ref'].get('human', [0, 0]))} | {pct(*a['ref'].get('ai', [0, 0]))} | "
              f"{pct(a['eff'], a['find'])} | {bal:.3f} | {a['find']} | {a['dup']} | {a['tok'] // a['prs']:,} | {a['s'] / a['prs']:.0f} | "
              f"{a['nofind']} | {a['missing']} | " + (str(a["cap"]) if v == "V3" else "–") + " |")
    print("\n## Table 2: per PR (covered references / effective findings / findings)\n")
    print("| PR | language | reference comments | " + " | ".join(vs) + " |")
    print("|---|---|---|" + "---|" * len(vs))
    for row in per_pr:
        print(f"| {row['pr']} | {row['lang']} | {row['n']} | " + " | ".join(row.get(v, "–") for v in vs) + " |")
    print("\n## Table 3: recall by context needed\n")
    ctxs = sorted({k for v in vs for k in agg[v]["ctx"]})
    print("| context | " + " | ".join(vs) + " |")
    print("|---|" + "---|" * len(vs))
    for k in ctxs:
        print(f"| {k} | " + " | ".join(pct(*agg[v]["ctx"].get(k, [0, 0])) for v in vs) + " |")
    p = OUT / "spotcheck.csv"
    if p.exists():
        rows = [x for x in csv.DictReader(p.open(encoding="utf-8")) if x["agree"].strip()]
        if rows:
            ag = sum(x["agree"].strip().lower() in ("y", "yes", "1") for x in rows)
            print(f"\nJudge agreement with hand check: {ag}/{len(rows)} ({100 * ag / len(rows):.0f}%)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "run", "judge", "sheet", "summary"])
    ap.add_argument("--pr", help="comma list of owner/repo#N (default: all 10 shortlisted)")
    ap.add_argument("--only", default="V1,V2,V3", help="versions to run")
    ap.add_argument("--model", default=rvc.MODEL)
    ap.add_argument("--force", action="store_true", help="rebuild / rerun even if cached")
    a = ap.parse_args()
    urls = shortlist()
    if a.pr:
        want = {f"https://github.com/{p.replace('#', '/pull/')}" for p in a.pr.split(",")}
        urls = [u for u in urls if u in want]
        if not urls:
            sys.exit("none of --pr is in the shortlist")
    if a.cmd == "build":
        cmd_build(urls, a.force)
    elif a.cmd == "run":
        cmd_run(urls, a.only.split(","), a.model, a.force)
    elif a.cmd == "judge":
        cmd_judge(urls, a.only.split(","), a.model, a.force)
    elif a.cmd == "sheet":
        cmd_sheet()
    else:
        cmd_summary()


if __name__ == "__main__":
    main()
