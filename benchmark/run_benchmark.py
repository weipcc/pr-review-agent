#!/usr/bin/env python3
"""
Benchmark for the PR-review prompt, on the `ronantakizawa/github-codereview` dataset.

What it measures
----------------
Each sample is one real human review comment (or a "no issues" negative) on a ~50-line code
chunk plus the diff hunk that was commented on. We feed that to the reviewer *exactly the way
the product feeds a file* (diff + numbered file context) and compare with the human comment.

Conditions (an ablation of the prompt work; same model, same inputs):
    A_ver1     original reviewer.py (Chinese prompt, 4 severities, no few-shot)      <- baseline
    B_prompt   current English prompt + new schema, no few-shot, no project context
    C_fewshot  B + the 3 few-shot calibration examples
    D_context  C + PR title as <pr_info> background

Metrics
-------
    recall         share of human comments covered by >=1 finding (LLM judge, semantic match)
    location@K     share of human comments with a finding within +-K lines of the human line
                   (reported next to a trivial "always answer line 26" baseline, because the
                   dataset puts the commented line at line 26 of its chunk: see README.md)
    FP rate        share of "no issues" negatives where the reviewer still reports something
    serious FP     ... where it reports a bug/security finding (confidence >= medium)
    tokens/latency mean input tokens and seconds per review call

What it can NOT measure: repo exploration (sandbox), cross-file breaking changes, README context.
The dataset only has single-file chunks. Use qualitative case studies for those.

Usage (run from anywhere; needs GEMINI_API_KEY in the environment or in <repo>/.env):
    python benchmark/run_benchmark.py fetch                 # cache the test split locally (once)
    python benchmark/run_benchmark.py sample --tag smoke --per-type 2 --n-neg 4
    python benchmark/run_benchmark.py run    --tag smoke    # review every condition
    python benchmark/run_benchmark.py judge  --tag smoke    # LLM-judge the positives
    python benchmark/run_benchmark.py report --tag smoke    # tables + results/<tag>/summary.md
    python benchmark/run_benchmark.py all    --tag main     # sample + run + judge + report
    python benchmark/run_benchmark.py inspect --tag main -n 30 --csv labels.csv   # validate the judge by hand
    python benchmark/run_benchmark.py agreement --csv labels.csv

Every model call is cached in results/<tag>/, so re-running resumes where it stopped and only
retries failures.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

DATASET = "ronantakizawa/github-codereview"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
SPLIT = "test"

# Where each condition's prompt comes from (git refs; the files are read with `git show`).
# Model for ALL conditions and for the judge. Same model everywhere, so differences between
# conditions come from the prompt, not the model. Override with --model / --judge-model.
DEFAULT_MODEL = "gemini-3.8-flash"
VER1_REF = "6ddb19c"  # "ver1": the original reviewer.py
# PR #5 squash merge. PR #7 (LangChain) only changed the transport layer, not the prompts.
DEFAULT_PROMPT_REF = "2ed8205"

CONDITIONS = {
    "A_ver1": {"kind": "ver1"},
    "B_prompt": {"kind": "new", "few_shot": False, "context": False},
    "C_fewshot": {"kind": "new", "few_shot": True, "context": False},
    "D_context": {"kind": "new", "few_shot": True, "context": True},
}

# Stratified quotas for positives. Security/performance are rare in the dataset (~1.3% each).
DEFAULT_QUOTAS = {
    "bug": 30, "security": 20, "performance": 20, "refactor": 15,
    "question": 20, "suggestion": 30, "nitpick": 10, "style": 10,
}
DEFAULT_N_NEG = 60
MAX_BEFORE_CHARS = 8000   # skip giant chunks (cost)
MAX_DIFF_CHARS = 6000
MAX_PER_PR = 2            # at most this many samples from one pull request
LOCATION_K = 3
# How model inputs are built from a dataset row (stored on every sample, part of the cache key):
#   1 = legacy: the hunk's new-side start is made chunk-relative but the OLD-side start stays an
#       absolute file line, so one header mixes two numberings and models sometimes report
#       absolute-looking lines outside the chunk (20% of one condition's findings).
#   2 = both sides of the aligned header are chunk-relative.
INPUT_VERSION = 2
# The dataset takes 25 lines of leading context before the commented line, so the commented
# line is line 26 of the chunk whenever the comment is >=25 lines into its file (83.6% of the
# test positives are exactly line 26). "Always answer line 26" is therefore the trivial baseline.
TRIVIAL_LINE = 26
SERIOUS = {"bug", "security"}

_lock = threading.Lock()
_abort = threading.Event()  # set when the API says retrying is pointless (e.g. credits depleted)


class FatalAPIError(Exception):
    """The API refused for a reason that retrying cannot fix (no credits, invalid key, ...)."""


FATAL_MARKERS = ("402", "credits are depleted", "prepayment", "billing", "API key not valid",
                 "API_KEY_INVALID", "PERMISSION_DENIED")


# --------------------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------------------
def ensure_api_key() -> None:
    if os.environ.get("GEMINI_API_KEY"):
        return
    for path in (REPO / ".env", ROOT / ".env"):
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                m = re.match(r"\s*(?:export\s+)?GEMINI_API_KEY\s*=\s*(.+?)\s*$", line)
                if m:
                    os.environ["GEMINI_API_KEY"] = m.group(1).strip().strip("\"'")
                    return
    sys.exit("GEMINI_API_KEY is not set (export it, or put it in <repo>/.env).")


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def last_by_id(path: Path) -> dict[str, dict]:
    """Latest record per id (later lines override earlier ones, so failures can be retried)."""
    return {r["id"]: r for r in read_jsonl(path)}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson score interval for a proportion; sensible for the small n used here."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def evidence_in_input(sample: dict, evidence: str) -> bool:
    """Is the quoted evidence really in what the model was shown? Whitespace-normalised substring
    match, so it is a lower bound (escaping or re-joined lines count as misses)."""
    norm = lambda t: re.sub(r"\s+", " ", t or "").strip()  # noqa: E731
    ev = norm(evidence)
    if not ev:
        return False
    stripped_ctx = re.sub(r"(?m)^\d+: ", "", sample["context_text"])  # drop the "N: " prefixes
    hay = norm(sample["diff_text"] + "\n" + sample["context_text"])
    hay2 = norm(stripped_ctx + "\n" + sample["diff_text"])
    return ev in hay or ev in hay2


def paired_bootstrap_diff(a: list[int], b: list[int], iters: int = 2000, seed: int = 0):
    """Mean(b) - mean(a) with a 95% percentile CI, resampling samples jointly (paired)."""
    n = len(a)
    if n == 0:
        return None
    rng = random.Random(seed)
    diffs = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(b[i] for i in idx) / n - sum(a[i] for i in idx) / n)
    diffs.sort()
    return sum(b) / n - sum(a) / n, diffs[int(0.025 * iters)], diffs[int(0.975 * iters) - 1]


# --------------------------------------------------------------------------------------
# dataset: fetch, sample, build model inputs
# --------------------------------------------------------------------------------------
def fetch_split(force: bool = False) -> Path:
    path = DATA_DIR / f"{SPLIT}_split.jsonl"
    if path.is_file() and not force:
        print(f"Using cached split: {path}")
        return path
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    offset, total, length = 0, None, 100
    with open(tmp, "w", encoding="utf-8") as f:
        while total is None or offset < total:
            params = {"dataset": DATASET, "config": "default", "split": SPLIT,
                      "offset": offset, "length": length}
            for attempt in range(7):
                r = requests.get(ROWS_URL, params=params, timeout=90)
                if r.status_code == 200:
                    break
                time.sleep(min(2 ** attempt, 30))
            else:
                raise RuntimeError(f"Could not fetch rows at offset {offset}: {r.status_code} {r.text[:200]}")
            d = r.json()
            total = d["num_rows_total"]
            for item in d["rows"]:
                row = item["row"]
                row.pop("after_code", None)  # not used; saves space
                row["row_idx"] = item["row_idx"]
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            offset += length
            if (offset // length) % 10 == 0 or offset >= total:
                print(f"  fetched {min(offset, total)}/{total}")
    tmp.replace(path)
    print(f"\nSaved {total} rows to {path}")
    return path


_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")


def _new_side(diff_lines: list[str]):
    """Lines of the hunk as they look in the *new* file (context + added)."""
    for ln in diff_lines:
        if ln.startswith("@@") or ln.startswith("-"):
            continue
        yield ln[1:] if ln[:1] in ("+", " ") else ln


def align_hunk(diff_lines: list[str], chunk_lines: list[str]) -> int | None:
    """
    1-based chunk line number of the hunk's first new-side line, or None if the hunk cannot be
    located inside the chunk. Lets us make the hunk header agree with the chunk's line numbers,
    as it does in production where context lines carry absolute file line numbers.
    """
    ns = [l.rstrip() for l in _new_side(diff_lines)]
    chunk = [l.rstrip() for l in chunk_lines]
    for i in range(min(len(ns), 12)):
        window = ns[i:i + 3]
        if len(window) < 2 or any(len(w.strip()) < 4 for w in window):
            continue
        for j in range(len(chunk) - len(window) + 1):
            if chunk[j:j + len(window)] == window:
                return j + 1 - i
    return None


def prepare_inputs(row: dict, version: int = INPUT_VERSION) -> dict:
    """
    Build the exact two strings the product gives the reviewer for a file:
      diff_text    "File: <path>" + hunk(s)
      context_text numbered file content ("N: code"); here N is relative to the 50-line chunk
    """
    chunk_lines = (row["before_code"] or "").splitlines()
    context_text = "\n".join(f"{i + 1}: {l}" for i, l in enumerate(chunk_lines))

    diff_lines = (row["diff_context"] or "").splitlines()
    start = align_hunk(diff_lines, chunk_lines)
    aligned = start is not None and start >= 1

    fixed, first = [], True
    for ln in diff_lines:
        m = _HUNK_RE.match(ln)
        if not m:
            fixed.append(ln)
            continue
        old_s, old_n, _new_s, new_n, tail = m.groups()
        if aligned and first:
            # v1 kept the absolute old-side start; v2 uses the same chunk-relative start on both sides
            old_part = f"-{old_s if version == 1 else start}" + (f",{old_n}" if old_n else "")
            new_part = f"+{start}" + (f",{new_n}" if new_n else "")
            fixed.append(f"@@ {old_part} {new_part} @@{tail}")
        else:  # numbers would disagree with the chunk numbering; drop them
            fixed.append(f"@@ ... @@{tail}")
        first = False
    diff_text = f"File: {row['file_path']}\n" + "\n".join(fixed)

    on_added = None
    if aligned and not row["is_negative"]:
        pos, on_added = start, False
        for ln in diff_lines:
            if ln.startswith("@@") or ln.startswith("-"):
                continue
            if ln.startswith("+") and pos == row["comment_line"]:
                on_added = True
            pos += 1

    return {
        "diff_text": diff_text,
        "context_text": context_text,
        "n_chunk_lines": len(chunk_lines),
        "input_version": version,
        "header_aligned": aligned,
        "on_added_line": on_added,
    }


def build_samples(rows: list[dict], quotas: dict[str, int], n_neg: int, seed: int,
                  input_version: int = INPUT_VERSION) -> list[dict]:
    rng = random.Random(seed)

    def usable(r: dict) -> bool:
        return (
            bool((r["diff_context"] or "").strip())
            and bool((r["before_code"] or "").strip())
            and len(r["before_code"]) <= MAX_BEFORE_CHARS
            and len(r["diff_context"]) <= MAX_DIFF_CHARS
        )

    pools: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        if usable(r):
            pools["__neg__" if r["is_negative"] else r["comment_type"]].append(r)

    chosen: list[dict] = []
    wanted = dict(quotas)
    wanted["__neg__"] = n_neg
    per_pr: collections.Counter = collections.Counter()  # global: cap applies across all types
    # Fill the rarest types first so the per-PR cap cannot starve them.
    for key in sorted(wanted, key=lambda t: (len(pools.get(t, [])), t)):
        k = wanted[key]
        pool = pools.get(key, [])
        rng.shuffle(pool)
        picked = []
        for r in pool:
            pr = (r["repo_name"], r["pr_number"])
            if per_pr[pr] >= MAX_PER_PR:
                continue
            per_pr[pr] += 1
            picked.append(r)
            if len(picked) == k:
                break
        if len(picked) < k:
            print(f"  warning: only {len(picked)} usable rows for '{key}' (wanted {k})")
        chosen += picked

    rng.shuffle(chosen)  # so --limit N gives a mixed subset
    samples = []
    for r in chosen:
        s = {
            "id": f"{SPLIT}-{r['row_idx']}",
            "is_negative": bool(r["is_negative"]),
            "comment_type": r["comment_type"],
            "comment_line": int(r["comment_line"]),
            "reviewer_comment": r["reviewer_comment"],
            "file_path": r["file_path"],
            "language": r["language"],
            "repo_name": r["repo_name"],
            "pr_number": r["pr_number"],
            "pr_title": r["pr_title"],
            "quality_score": r["quality_score"],
            "diff_context": r["diff_context"],
            "before_code": r["before_code"],
        }
        s.update(prepare_inputs(r, input_version))
        samples.append(s)
    return samples


# --------------------------------------------------------------------------------------
# calling the model
# --------------------------------------------------------------------------------------
# Without a timeout one stalled request blocks the whole run (the SDK default is to wait forever).
REQUEST_TIMEOUT_MS = 90_000

_client = None
_module_cache: dict[str, types.ModuleType] = {}


def get_client():
    global _client
    with _lock:
        if _client is None:
            from google import genai
            from google.genai import types as gtypes
            _client = genai.Client(
                api_key=os.environ.get("GEMINI_API_KEY"),
                http_options=gtypes.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
            )
        return _client


def load_reviewer_from_ref(ref: str) -> types.ModuleType:
    """Execute `reviewer.py` as it was at git ref `ref`, so each condition uses the real prompt code."""
    if ref in _module_cache:
        return _module_cache[ref]
    src = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{ref}:reviewer.py"],
        capture_output=True, text=True, check=True,
    ).stdout
    mod = types.ModuleType(f"reviewer_at_{ref}")
    sys.modules[mod.__name__] = mod
    try:
        exec(compile(src, f"{ref}:reviewer.py", "exec"), mod.__dict__)
    except ImportError as e:
        sys.exit(f"Could not import reviewer.py at {ref}: {e}\n"
                 f"(refs after PR #7 need `pip install -r requirements.txt`; the default ref does not.)")
    _module_cache[ref] = mod
    return mod


RETRY_MARKERS = ("429", "500", "502", "503", "504", "UNAVAILABLE", "RESOURCE_EXHAUSTED",
                 "DEADLINE", "timed out", "Timeout", "Connection", "overloaded")


def call_model(model: str, system_prompt: str, contents, schema, temperature, max_attempts: int = 7):
    config = {
        "system_instruction": system_prompt,
        "response_mime_type": "application/json",
        "response_schema": schema,
    }
    if temperature is not None:
        config["temperature"] = temperature
    last = None
    for attempt in range(max_attempts):
        try:
            t0 = time.time()
            resp = get_client().models.generate_content(model=model, contents=contents, config=config)
            if resp.parsed is None:
                raise ValueError("model returned no parsable JSON")
            usage = getattr(resp, "usage_metadata", None)
            return resp.parsed, {
                "latency_s": round(time.time() - t0, 2),
                "prompt_tokens": getattr(usage, "prompt_token_count", None),
                "output_tokens": getattr(usage, "candidates_token_count", None),
            }
        except Exception as e:  # noqa: BLE001
            last = e
            if any(m in str(e) for m in FATAL_MARKERS):
                raise FatalAPIError(str(e)[:300]) from e
            retryable = any(m in str(e) for m in RETRY_MARKERS) or "Timeout" in type(e).__name__
            if not retryable or attempt == max_attempts - 1:
                break
            time.sleep(min(2 ** attempt, 40) * (0.75 + random.random() / 2))
    raise last


def _to_gemini_contents(messages: list[dict]) -> list[dict]:
    """Accept both google-genai ({'parts': ...}) and LangChain ({'content': ...}) message shapes."""
    out = []
    for m in messages:
        if "parts" in m:
            out.append(m)
        else:
            role = "model" if m["role"] == "assistant" else m["role"]
            out.append({"role": role, "parts": [{"text": m["content"]}]})
    return out


def review_sample(cond: str, sample: dict, model: str, prompt_ref: str) -> tuple[dict, dict]:
    """Run one review for one condition. Returns (raw output dict, call metadata)."""
    spec = CONDITIONS[cond]
    model = model or DEFAULT_MODEL
    if spec["kind"] == "ver1":
        mod = load_reviewer_from_ref(VER1_REF)
        # Same text the original review_file() built inline.
        user_message = (
            f"檔案名稱: {sample['file_path']}\n\n"
            f"=== Diff（本次改動） ===\n{sample['diff_text']}\n\n"
            f"=== 檔案上下文（目前內容） ===\n{sample['context_text']}\n"
        )
        parsed, meta = call_model(model, mod.SYSTEM_PROMPT, user_message,
                                  mod.FileReview, temperature=None)
        return parsed.model_dump(), meta

    mod = load_reviewer_from_ref(prompt_ref)
    language = mod.detect_language(sample["file_path"])
    system_prompt = mod.build_system_prompt_single(language)
    project_context = {"pr_title": sample["pr_title"]} if spec["context"] else None
    user_message = mod.build_user_message(
        sample["file_path"], language, sample["diff_text"], sample["context_text"],
        project_context=project_context,
    )
    contents = []
    if spec["few_shot"]:
        contents += _to_gemini_contents(mod.build_few_shot_contents())
    contents.append({"role": "user", "parts": [{"text": user_message}]})
    parsed, meta = call_model(model, system_prompt, contents,
                              mod.FileReview, temperature=mod.TEMPERATURE)
    return parsed.model_dump(), meta


def normalize(kind: str, out: dict) -> list[dict]:
    """Map either schema to a common finding shape."""
    findings = []
    for it in out.get("issues", []):
        if kind == "ver1":
            findings.append({"line": it.get("line"), "category": it.get("severity"),
                             "confidence": None, "comment": it.get("comment", ""), "evidence": None})
        else:
            findings.append({"line": it.get("start_line"), "category": it.get("category"),
                             "confidence": it.get("confidence"), "comment": it.get("comment", ""),
                             "evidence": it.get("evidence")})
    return findings


# --------------------------------------------------------------------------------------
# LLM judge
# --------------------------------------------------------------------------------------
class JudgeVerdict(BaseModel):
    matched: bool = Field(description="True if some finding raises the same underlying concern")
    best_issue_index: int | None = Field(default=None, description="Index of the best matching finding")
    reason: str = Field(description="One short sentence")


JUDGE_SYSTEM = """You compare an automated code reviewer's findings with a comment written by a \
human reviewer on the same change, and decide whether the automated reviewer found what the human \
cared about.

matched = true only if at least one finding raises the same underlying concern as the human \
comment: the same problem, or a request for the same change, or the same doubt if the human asked a \
question. A finding that is broader or narrower still counts if a developer acting on it would \
address the human's concern.
matched = false if the findings are about other things, or merely touch the same line or topic \
without making the human's point, or only restate what the code does.
The human comment is the reference; do not judge whether the findings are good in their own right."""


def judge_prompt(sample: dict, findings: list[dict]) -> str:
    lines = []
    for i, f in enumerate(findings):
        lines.append(f"[{i}] line {f['line']} · {f['category']}: {f['comment']}")
    return (
        f"<human_comment line=\"{sample['comment_line']}\">\n{sample['reviewer_comment']}\n</human_comment>\n\n"
        f"<diff>\n{sample['diff_text']}\n</diff>\n\n"
        f"<file_context>\n{sample['context_text']}\n</file_context>\n\n"
        f"<automated_findings>\n" + "\n".join(lines) + "\n</automated_findings>"
    )


def findings_hash(findings: list[dict]) -> str:
    return hashlib.sha1(json.dumps(findings, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]


# --------------------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------------------
def paths(tag: str) -> dict[str, Path]:
    res = RESULTS_DIR / tag
    return {"samples": DATA_DIR / f"samples_{tag}.jsonl", "res": res, "raw": res / "raw",
            "judge": res / "judge", "summary_md": res / "summary.md", "summary_json": res / "summary.json"}


def load_samples(tag: str, limit: int | None = None) -> list[dict]:
    samples = read_jsonl(paths(tag)["samples"])
    if not samples:
        sys.exit(f"No samples for tag '{tag}'. Run `sample --tag {tag}` first.")
    return samples[:limit] if limit else samples


def stage_sample(args) -> None:
    rows = read_jsonl(fetch_split())
    quotas = ({k: args.per_type for k in DEFAULT_QUOTAS} if args.per_type else dict(DEFAULT_QUOTAS))
    for item in args.quota or []:
        k, v = item.split("=")
        quotas[k] = int(v)
    samples = build_samples(rows, quotas, args.n_neg, args.seed, args.input_version)
    p = paths(args.tag)["samples"]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in samples), encoding="utf-8")
    pos = [s for s in samples if not s["is_negative"]]
    aligned = [s for s in pos if s["header_aligned"]]
    print(f"Wrote {len(samples)} samples ({len(pos)} positive, {len(samples) - len(pos)} negative) to {p}")
    print("  by type:", dict(collections.Counter(s["comment_type"] for s in samples)))
    centre_hit = sum(abs(TRIVIAL_LINE - s["comment_line"]) <= LOCATION_K for s in pos)
    print(f"  human line within ±{LOCATION_K} of line {TRIVIAL_LINE}: {centre_hit}/{len(pos)} "
          f"({100 * centre_hit / max(len(pos), 1):.0f}%)  <- trivial location baseline")
    print(f"  diff hunk aligned to chunk numbering: {len(aligned)}/{len(pos)}")
    on_added = [s for s in aligned if s["on_added_line"]]
    print(f"  of those, human comment on an added line: {len(on_added)}/{len(aligned)} "
          f"({100 * len(on_added) / max(len(aligned), 1):.0f}%)  <- ceiling for 'only review added lines'")


def condition_signature(cond: str, prompt_ref: str, model: str | None, input_version: int = 1) -> str:
    """Identifies what produced a result: prompt source + model (+ input construction if not v1).
    Part of the cache key, so results from different inputs are never reused for each other."""
    ref = VER1_REF if CONDITIONS[cond]["kind"] == "ver1" else prompt_ref
    sig = f"{ref}|{model or DEFAULT_MODEL}"
    return sig if input_version == 1 else f"{sig}|in{input_version}"


def stage_run(args) -> None:
    ensure_api_key()
    samples = load_samples(args.tag, args.limit)
    versions = {s.get("input_version", 1) for s in samples}
    if len(versions) != 1:
        sys.exit(f"Samples mix input versions {sorted(versions)}; re-run `sample` to regenerate them.")
    input_version = versions.pop()
    print(f"Input construction: v{input_version}" + (" (legacy header numbering)" if input_version == 1 else ""))
    conds = args.conditions.split(",") if args.conditions else list(CONDITIONS)
    for cond in conds:
        if cond not in CONDITIONS:
            sys.exit(f"Unknown condition {cond}. Choose from {list(CONDITIONS)}")
    p = paths(args.tag)
    if any(CONDITIONS[c]["kind"] == "new" for c in conds):  # fail fast on an incompatible --prompt-ref
        mod = load_reviewer_from_ref(args.prompt_ref)
        need = ("detect_language", "build_system_prompt_single", "build_user_message",
                "build_few_shot_contents", "FileReview", "TEMPERATURE")
        missing = [n for n in need if not hasattr(mod, n)]
        if missing:
            sys.exit(f"reviewer.py at {args.prompt_ref} lacks {missing}; it is not a per-file prompt "
                     f"version this benchmark can drive (e.g. #9 needs its own condition).")
    for cond in conds:
        out_path = p["raw"] / f"{cond}.jsonl"
        sig = condition_signature(cond, args.prompt_ref, args.model, input_version)
        # A cached result is only reused if it came from the same prompt source AND model.
        done = {i for i, r in last_by_id(out_path).items() if r.get("ok") and r.get("sig") == sig}
        todo = [s for s in samples if s["id"] not in done]
        print(f"[{cond}] {len(done)} cached, {len(todo)} to run (workers={args.workers}, {sig})")
        if not todo:
            continue

        def work(s, cond=cond, out_path=out_path, sig=sig):
            if _abort.is_set():  # a fatal error already happened: do not call, do not record
                return False
            try:
                out, meta = review_sample(cond, s, args.model, args.prompt_ref)
                rec = {"id": s["id"], "condition": cond, "sig": sig, "ok": True, "output": out, **meta}
            except FatalAPIError as e:
                if not _abort.is_set():
                    print(f"\n\nFATAL: {e}\nStopping. Results so far are cached; fix the cause and re-run the same command to resume.")
                _abort.set()
                return False  # not recorded, so it is retried on resume
            except Exception as e:  # noqa: BLE001
                rec = {"id": s["id"], "condition": cond, "sig": sig, "ok": False, "error": str(e)[:300]}
            append_jsonl(out_path, rec)
            return rec["ok"]

        ok = err = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(work, s) for s in todo]
            for n, fut in enumerate(as_completed(futs), 1):
                ok += fut.result()
                err += not fut.result()
                print(f"  [{cond}] {n}/{len(todo)}  ok={ok} err={err}", end="\r")
        print()
        if _abort.is_set():
            sys.exit(2)


def stage_judge(args) -> None:
    ensure_api_key()
    samples = {s["id"]: s for s in load_samples(args.tag, args.limit)}
    conds = args.conditions.split(",") if args.conditions else list(CONDITIONS)
    p = paths(args.tag)
    model = args.judge_model or DEFAULT_MODEL
    for cond in conds:
        kind = CONDITIONS[cond]["kind"]
        raw = last_by_id(p["raw"] / f"{cond}.jsonl")
        judged = last_by_id(p["judge"] / f"{cond}.jsonl")
        todo = []
        for sid, s in samples.items():
            r = raw.get(sid)
            if s["is_negative"] or not r or not r.get("ok"):
                continue
            findings = normalize(kind, r["output"])
            if not findings:  # nothing reported: trivially not matched, no call needed
                rec = {"id": sid, "ok": True, "matched": False, "best_issue_index": None,
                       "reason": "no findings", "hash": findings_hash(findings), "judge_model": model}
                old = judged.get(sid, {})
                if old.get("hash") != rec["hash"] or old.get("judge_model") != model:
                    append_jsonl(p["judge"] / f"{cond}.jsonl", rec)
                continue
            h = findings_hash(findings)
            old = judged.get(sid, {})
            if old.get("ok") and old.get("hash") == h and old.get("judge_model") == model:
                continue
            todo.append((s, findings, h))
        print(f"[{cond}] judging {len(todo)} sample(s) with {model}")

        def work(item, cond=cond):
            s, findings, h = item
            if _abort.is_set():
                return
            try:
                v, _ = call_model(model, JUDGE_SYSTEM, judge_prompt(s, findings), JudgeVerdict, temperature=0.0)
                rec = {"id": s["id"], "ok": True, "matched": bool(v.matched),
                       "best_issue_index": v.best_issue_index, "reason": v.reason, "hash": h,
                       "judge_model": model}
            except FatalAPIError as e:
                if not _abort.is_set():
                    print(f"\n\nFATAL: {e}\nStopping. Re-run the same command to resume.")
                _abort.set()
                return
            except Exception as e:  # noqa: BLE001
                rec = {"id": s["id"], "ok": False, "error": str(e)[:300], "hash": h, "judge_model": model}
            append_jsonl(p["judge"] / f"{cond}.jsonl", rec)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(work, todo))
        if _abort.is_set():
            sys.exit(2)


def pct(k: int, n: int) -> str:
    if n == 0:
        return "n/a"
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.0f}% [{100 * lo:.0f}–{100 * hi:.0f}] ({k}/{n})"


def pct_tex(k: int, n: int) -> str:
    """LaTeX-safe 'xx\\% [lo--hi]' (95% Wilson)."""
    if n == 0:
        return "n/a"
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.0f}\\% [{100 * lo:.0f}--{100 * hi:.0f}]"


def build_latex_tables(tag, conds, per, samples, pos_ids, neg_ids, common, centre_hits) -> str:
    """Tables for report.tex (only the packages already in the course template are needed)."""
    short = {"A_ver1": "A", "B_prompt": "B", "C_fewshot": "C", "D_context": "D"}
    name = lambda c: short.get(c, c.replace("_", "\\_"))  # noqa: E731
    n_pos, n_neg = len(pos_ids), len(neg_ids)
    mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")  # noqa: E731
    head = "\\textbf{Metric} & " + " & ".join(f"\\textbf{{{name(c)}}}" for c in conds) + " \\\\\n\\hline\n"
    colspec = "p{6.0cm}" + "c" * len(conds)

    def row(label, cells):
        return label + " & " + " & ".join(cells) + " \\\\\n"

    t1 = ["\\begin{table}[H]\n\\centering\n\\footnotesize\n\\begin{tabular}{" + colspec + "}\n\\hline\n" + head]
    t1.append(row("Recall (judge) $\\uparrow$", [pct_tex(sum(per[c]["recall"]), n_pos) for c in conds]))
    t1.append(row("Location hit @$\\pm 3$ $\\uparrow$", [pct_tex(sum(per[c]["loc"]), n_pos) for c in conds]))
    t1.append(row("\\quad always-line-26 baseline", [pct_tex(centre_hits, n_pos)] * len(conds)))
    t1.append(row("False-positive rate on negatives $\\downarrow$", [pct_tex(sum(per[c]["fp"]), n_neg) for c in conds]))
    t1.append(row("Serious false positives $\\downarrow$", [pct_tex(sum(per[c]["fp_serious"]), n_neg) for c in conds]))
    t1.append(row("Evidence faithfulness, per finding $\\uparrow$", [
        "n/a" if per[c]["ev_total"] == 0 else pct_tex(per[c]["ev_ok"], per[c]["ev_total"]) for c in conds]))
    t1.append(row("Findings per sample (positives / negatives)", [
        f"{mean(per[c]['n_pos_findings']):.2f} / {mean(per[c]['n_neg_findings']):.2f}" for c in conds]))
    t1.append(row("Mean input tokens per call", [f"{mean(per[c]['tokens']):.0f}" for c in conds]))
    t1.append(row("Mean latency per call (s)", [f"{mean(per[c]['latency']):.1f}" for c in conds]))
    t1.append("\\hline\n\\end{tabular}\n\\caption{Overall results on " + str(len(common)) + " samples (" + str(n_pos)
              + " positives, " + str(n_neg) + " negatives); 95\\% Wilson intervals in brackets.}\n"
              "\\label{tab:overall}\n\\end{table}\n")

    types_ = sorted({samples[i]["comment_type"] for i in pos_ids},
                    key=lambda t: -sum(samples[i]["comment_type"] == t for i in pos_ids))
    t2 = ["\\begin{table}[H]\n\\centering\n\\footnotesize\n\\begin{tabular}{l" + "c" * len(conds) + "}\n\\hline\n"
          + "\\textbf{Human comment type (n)} & " + " & ".join(f"\\textbf{{{name(c)}}}" for c in conds) + " \\\\\n\\hline\n"]
    for ty in types_:
        idx = [k for k, i in enumerate(pos_ids) if samples[i]["comment_type"] == ty]
        t2.append(row(f"{ty} ({len(idx)})", [f"{100 * sum(per[c]['recall'][k] for k in idx) / len(idx):.0f}\\%" for c in conds]))
    t2.append("\\hline\n\\end{tabular}\n\\caption{Recall by type of human comment. Per-type n is small; "
              "read the point estimates with care.}\n\\label{tab:bytype}\n\\end{table}\n")

    out = ["% Generated by benchmark/run_benchmark.py report (tag: " + tag + "). Paste into report.tex.\n"] + t1 + ["\n"] + t2
    if len(conds) > 1:
        base = conds[0]
        t3 = ["\\begin{table}[H]\n\\centering\n\\footnotesize\n\\begin{tabular}{l" + "c" * (len(conds) - 1) + "}\n\\hline\n"
              + "\\textbf{Metric (percentage points)} & " + " & ".join(f"\\textbf{{{name(c)} vs {name(base)}}}" for c in conds[1:]) + " \\\\\n\\hline\n"]
        for key, label in (("recall", "Recall"), ("fp", "False-positive rate"), ("fp_serious", "Serious false positives")):
            cells = []
            for c in conds[1:]:
                r = paired_bootstrap_diff(per[base][key], per[c][key])
                cells.append("n/a" if r is None else
                             "${:+.0f}$ [${:+.0f}$, ${:+.0f}$]".format(100 * r[0], 100 * r[1], 100 * r[2]))
            t3.append(row(label, cells))
        t3.append("\\hline\n\\end{tabular}\n\\caption{Paired differences against the baseline with 95\\% bootstrap "
                  "intervals (same samples).}\n\\label{tab:paired}\n\\end{table}\n")
        out += ["\n"] + t3
    return "".join(out)


def stage_report(args) -> None:
    samples = {s["id"]: s for s in load_samples(args.tag, args.limit)}
    conds = args.conditions.split(",") if args.conditions else list(CONDITIONS)
    p = paths(args.tag)

    raw = {c: last_by_id(p["raw"] / f"{c}.jsonl") for c in conds}
    judged = {c: last_by_id(p["judge"] / f"{c}.jsonl") for c in conds}
    conds = [c for c in conds if raw[c]]
    if not conds:
        sys.exit("No results yet. Run `run` (and `judge`) first.")

    # Evaluate every condition on the same samples: those with a successful review in ALL
    # conditions (and, for positives, a successful judgement in all of them).
    common = []
    for sid, s in samples.items():
        if not all(raw[c].get(sid, {}).get("ok") for c in conds):
            continue
        if not s["is_negative"] and not all(judged[c].get(sid, {}).get("ok") for c in conds):
            continue
        common.append(sid)
    pos_ids = [i for i in common if not samples[i]["is_negative"]]
    neg_ids = [i for i in common if samples[i]["is_negative"]]
    n_err = {c: sum(1 for r in raw[c].values() if not r.get("ok")) for c in conds}

    per: dict[str, dict] = {}
    for c in conds:
        kind = CONDITIONS[c]["kind"]
        recs = {i: normalize(kind, raw[c][i]["output"]) for i in common}
        m = {
            "recall": [int(judged[c][i]["matched"]) for i in pos_ids],
            "loc": [int(any(f["line"] is not None and abs(f["line"] - samples[i]["comment_line"]) <= LOCATION_K
                            for f in recs[i])) for i in pos_ids],
            "fp": [int(len(recs[i]) > 0) for i in neg_ids],
            "fp_serious": [int(any(f["category"] in SERIOUS and f["confidence"] in (None, "medium", "high")
                                   for f in recs[i])) for i in neg_ids],
            "n_pos_findings": [len(recs[i]) for i in pos_ids],
            "n_neg_findings": [len(recs[i]) for i in neg_ids],
            "ev_total": sum(1 for i in common for f in recs[i] if f["evidence"]),
            "ev_ok": sum(1 for i in common for f in recs[i]
                         if f["evidence"] and evidence_in_input(samples[i], f["evidence"])),
            "tokens": [raw[c][i].get("prompt_tokens") for i in common if raw[c][i].get("prompt_tokens")],
            "latency": [raw[c][i].get("latency_s") for i in common if raw[c][i].get("latency_s")],
        }
        per[c] = m

    centre = [int(abs(TRIVIAL_LINE - samples[i]["comment_line"]) <= LOCATION_K) for i in pos_ids]
    mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")  # noqa: E731

    md = []
    md.append(f"# Benchmark summary — tag `{args.tag}`\n")
    md.append(f"Dataset: `{DATASET}` ({SPLIT} split). Evaluated on {len(common)} samples common to all "
              f"conditions: {len(pos_ids)} positive, {len(neg_ids)} negative. "
              f"Intervals are 95% Wilson.\n")
    sigs = {c: sorted({r.get("sig", "unknown") for r in raw[c].values() if r.get("ok")}) for c in conds}
    judge_models = sorted({r.get("judge_model", "unknown") for c in conds for r in judged[c].values() if r.get("ok")})
    md.append("Prompt source | model per condition: " + "; ".join(f"`{c}` = {' + '.join(sigs[c])}" for c in conds)
              + f". Judge model: {', '.join(judge_models)}.\n")
    mixed = [c for c in conds if len(sigs[c]) > 1]
    if mixed:
        md.append(f"> **Warning:** {', '.join(mixed)} contain results from more than one prompt source/model "
                  f"(latest per sample is used). Re-run with a fresh `--tag` for a clean comparison.\n")
    if len({sig.split('|')[-1] for c in conds for sig in sigs[c]}) > 1:
        md.append("> **Note:** conditions were run with different reviewer models; differences mix prompt and "
                  "model effects.\n")
    md.append("## Overall\n")
    md.append("| Metric | " + " | ".join(conds) + " |")
    md.append("|---|" + "---|" * len(conds))
    md.append("| Recall — human concern covered (judge) ↑ | " + " | ".join(pct(sum(per[c]["recall"]), len(pos_ids)) for c in conds) + " |")
    md.append(f"| Location hit @±{LOCATION_K} ↑ | " + " | ".join(pct(sum(per[c]["loc"]), len(pos_ids)) for c in conds) + " |")
    md.append(f"| *trivial baseline: always answer line {TRIVIAL_LINE}* | " + " | ".join([pct(sum(centre), len(pos_ids))] * len(conds)) + " |")
    md.append("| False-positive rate on negatives ↓ | " + " | ".join(pct(sum(per[c]["fp"]), len(neg_ids)) for c in conds) + " |")
    md.append("| Serious FP (bug/security, conf ≥ medium) ↓ | " + " | ".join(pct(sum(per[c]["fp_serious"]), len(neg_ids)) for c in conds) + " |")
    md.append("| Evidence found verbatim in the input, per finding ↑ | " + " | ".join(
        "n/a" if per[c]["ev_total"] == 0 else pct(per[c]["ev_ok"], per[c]["ev_total"]) for c in conds) + " |")
    md.append("| Findings per positive sample | " + " | ".join(f"{mean(per[c]['n_pos_findings']):.2f}" for c in conds) + " |")
    md.append("| Findings per negative sample | " + " | ".join(f"{mean(per[c]['n_neg_findings']):.2f}" for c in conds) + " |")
    md.append("| Mean input tokens / call | " + " | ".join(f"{mean(per[c]['tokens']):.0f}" for c in conds) + " |")
    md.append("| Mean latency (s) / call | " + " | ".join(f"{mean(per[c]['latency']):.1f}" for c in conds) + " |")
    md.append("| Failed review calls (excluded) | " + " | ".join(str(n_err[c]) for c in conds) + " |")
    md.append("")

    # recall by human comment type
    types_ = sorted({samples[i]["comment_type"] for i in pos_ids},
                    key=lambda t: -sum(samples[i]["comment_type"] == t for i in pos_ids))
    md.append("## Recall by type of human comment\n")
    md.append("| Human comment type (n) | " + " | ".join(conds) + " |")
    md.append("|---|" + "---|" * len(conds))
    for t in types_:
        idx = [k for k, i in enumerate(pos_ids) if samples[i]["comment_type"] == t]
        md.append(f"| {t} ({len(idx)}) | " + " | ".join(
            f"{100 * sum(per[c]['recall'][k] for k in idx) / len(idx):.0f}%" for c in conds) + " |")
    md.append("")

    # recall split by whether the human line is on an added line (our prompt only reviews those)
    md.append("## Recall when the human comment is on an added vs. an unchanged line\n")
    md.append("(only samples whose diff hunk could be aligned to the chunk)\n")
    md.append("| Subset (n) | " + " | ".join(conds) + " |")
    md.append("|---|" + "---|" * len(conds))
    for label, want in (("on an added line", True), ("on an unchanged line", False)):
        idx = [k for k, i in enumerate(pos_ids) if samples[i]["on_added_line"] is want]
        if idx:
            md.append(f"| {label} ({len(idx)}) | " + " | ".join(
                f"{100 * sum(per[c]['recall'][k] for k in idx) / len(idx):.0f}%" for c in conds) + " |")
    md.append("")

    # paired differences vs the first condition (baseline)
    base = conds[0]
    if len(conds) > 1:
        md.append(f"## Paired difference vs `{base}` (95% bootstrap CI)\n")
        md.append("| Metric | " + " | ".join(conds[1:]) + " |")
        md.append("|---|" + "---|" * (len(conds) - 1))
        for key, label in (("recall", "Recall (pp)"), ("fp", "FP rate (pp)"), ("fp_serious", "Serious FP (pp)")):
            cells = []
            for c in conds[1:]:
                r = paired_bootstrap_diff(per[base][key], per[c][key])
                cells.append("n/a" if r is None else f"{100 * r[0]:+.0f} [{100 * r[1]:+.0f}, {100 * r[2]:+.0f}]")
            md.append(f"| {label} | " + " | ".join(cells) + " |")
        md.append("")

    md.append("## Read this before quoting numbers\n")
    md.append(
        "- Human comments are not exhaustive ground truth: a finding the human did not write is "
        "counted as noise, so precision-like numbers are pessimistic.\n"
        "- \"Negative\" means *changed but no comment was left*; it does not prove the code is fine.\n"
        "- Each chunk starts 25 lines before the commented line, so the human line is almost always "
        "line 26 and line-number metrics have a high trivial floor (see the baseline row). Prefer "
        "the judge-based recall.\n"
        "- The judge is the same model family as the reviewer (self-preference is possible); "
        "spot-check a sample of verdicts by hand before reporting.\n"
        "- ~60% of human comments are `suggestion`-type, which is a label our schema lacks.\n"
        "- Small per-type n: look at the intervals, not the point estimates.\n"
        "- Evidence faithfulness is a whitespace-normalised substring check, so it is a lower bound; "
        "its unit is the finding, not the sample.\n"
        "- Not covered here: repo exploration (sandbox), cross-file changes, README context."
    )
    text = "\n".join(md)
    p["summary_md"].parent.mkdir(parents=True, exist_ok=True)
    (p["res"] / "tables.tex").write_text(
        build_latex_tables(args.tag, conds, per, samples, pos_ids, neg_ids, common, sum(centre)), encoding="utf-8")
    p["summary_md"].write_text(text, encoding="utf-8")
    p["summary_json"].write_text(json.dumps({
        "tag": args.tag, "n_common": len(common), "n_pos": len(pos_ids), "n_neg": len(neg_ids),
        "conditions": conds,
        "metrics": {c: {k: v for k, v in per[c].items()
                        if k in ("recall", "loc", "fp", "fp_serious", "ev_total", "ev_ok")} for c in conds},
        "ids": {"pos": pos_ids, "neg": neg_ids},
    }, indent=1), encoding="utf-8")
    print(text)
    print(f"\nWrote {p['summary_md']} and {p['res'] / 'tables.tex'}")


def stage_inspect(args) -> None:
    """Show human comment vs findings vs judge verdict; optionally write a CSV for manual labelling."""
    import csv
    samples = {s["id"]: s for s in load_samples(args.tag)}
    p = paths(args.tag)
    cond = args.condition
    kind = CONDITIONS[cond]["kind"]
    raw = last_by_id(p["raw"] / f"{cond}.jsonl")
    jud = last_by_id(p["judge"] / f"{cond}.jsonl")
    ids = [i for i, s in samples.items()
           if not s["is_negative"] and raw.get(i, {}).get("ok") and jud.get(i, {}).get("ok")]
    random.Random(args.seed).shuffle(ids)
    ids = ids[: args.n]
    rows = []
    for k, i in enumerate(ids, 1):
        s, j = samples[i], jud[i]
        fs = normalize(kind, raw[i]["output"])
        findings = " | ".join(f"L{f['line']} {f['category']}: {f['comment']}" for f in fs) or "(no findings)"
        rows.append({"id": i, "comment_type": s["comment_type"], "human_comment": s["reviewer_comment"],
                     "findings": findings, "judge_matched": int(j["matched"]),
                     "judge_reason": j["reason"], "human_label": ""})
        print(f"\n[{k}] {i} · {s['comment_type']} · judge={'MATCH' if j['matched'] else 'no match'}")
        print("  HUMAN :", s["reviewer_comment"][:300].replace("\n", " "))
        print("  MODEL :", findings[:400])
        print("  JUDGE :", j["reason"][:200])
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {args.csv}. Fill `human_label` with y/n "
              f"(did the findings cover the human comment?), then run `agreement --csv {args.csv}`.")


def stage_agreement(args) -> None:
    """Agreement between your manual labels and the judge (accuracy and Cohen's kappa)."""
    import csv
    truthy, falsy = {"y", "yes", "1", "true", "t"}, {"n", "no", "0", "false", "f"}
    pairs = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lab = row["human_label"].strip().lower()
            if lab in truthy | falsy:
                pairs.append((lab in truthy, bool(int(row["judge_matched"]))))
    n = len(pairs)
    if n == 0:
        sys.exit("No labelled rows (fill the human_label column with y/n).")
    agree = sum(h == j for h, j in pairs)
    po = agree / n
    ph, pj = sum(h for h, _ in pairs) / n, sum(j for _, j in pairs) / n
    pe = ph * pj + (1 - ph) * (1 - pj)
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    lo, hi = wilson(agree, n)
    print(f"Labelled: {n} | agreement: {agree}/{n} = {100 * po:.0f}% [95% CI {100 * lo:.0f}–{100 * hi:.0f}] "
          f"| Cohen's kappa: {kappa:.2f}")
    print(f"Human says covered: {sum(h for h, _ in pairs)} | judge says covered: {sum(j for _, j in pairs)}")
    fn = sum(h and not j for h, j in pairs)
    fp = sum(j and not h for h, j in pairs)
    print(f"Judge too strict (human y / judge n): {fn} | judge too lenient (human n / judge y): {fp}")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, run_opts=False):
        p.add_argument("--tag", default="main", help="name of this experiment (separate samples/results)")
        p.add_argument("--limit", type=int, help="use only the first N samples (smoke tests)")
        p.add_argument("--conditions", help=f"comma list, default all: {','.join(CONDITIONS)}")
        if run_opts:
            p.add_argument("--workers", type=int, default=4)
            p.add_argument("--model", default=DEFAULT_MODEL,
                           help=f"reviewer model for every condition (default: {DEFAULT_MODEL})")
            p.add_argument("--prompt-ref", default=DEFAULT_PROMPT_REF,
                           help="git ref whose reviewer.py provides the B/C/D prompts")

    sub.add_parser("fetch", help="cache the dataset test split locally").add_argument("--force", action="store_true")
    ps = sub.add_parser("sample", help="draw a stratified sample")
    ps.add_argument("--tag", default="main")
    ps.add_argument("--seed", type=int, default=0)
    ps.add_argument("--n-neg", type=int, default=DEFAULT_N_NEG)
    ps.add_argument("--per-type", type=int, help="same quota for every comment type (smoke tests)")
    ps.add_argument("--input-version", type=int, default=INPUT_VERSION, choices=(1, 2),
                    help="1 = legacy header numbering, 2 = fully chunk-relative (default)")
    ps.add_argument("--quota", action="append", help="override one type, e.g. --quota bug=40")
    pi = sub.add_parser("inspect", help="human comment vs findings vs judge verdict")
    pi.add_argument("--tag", default="main")
    pi.add_argument("--condition", default="B_prompt", choices=list(CONDITIONS))
    pi.add_argument("-n", type=int, default=30)
    pi.add_argument("--seed", type=int, default=0)
    pi.add_argument("--csv", help="also write a CSV to label by hand")
    pa = sub.add_parser("agreement", help="agreement of manual labels with the judge")
    pa.add_argument("--csv", required=True)
    common_p = {}
    for name in ("run", "judge", "report", "all"):
        common_p[name] = sub.add_parser(name)
        common(common_p[name], run_opts=name in ("run", "judge", "all"))
    common_p["judge"].add_argument("--judge-model", help=f"default: {DEFAULT_MODEL}")
    common_p["all"].add_argument("--judge-model")
    common_p["all"].add_argument("--seed", type=int, default=0)
    common_p["all"].add_argument("--n-neg", type=int, default=DEFAULT_N_NEG)
    common_p["all"].add_argument("--per-type", type=int)
    common_p["all"].add_argument("--input-version", type=int, default=INPUT_VERSION, choices=(1, 2))
    common_p["all"].add_argument("--resample", action="store_true", help="redraw samples even if they exist")
    common_p["all"].add_argument("--quota", action="append")

    args = ap.parse_args()
    if args.cmd == "fetch":
        fetch_split(force=args.force)
    elif args.cmd == "sample":
        stage_sample(args)
    elif args.cmd == "run":
        stage_run(args)
    elif args.cmd == "judge":
        stage_judge(args)
    elif args.cmd == "report":
        stage_report(args)
    elif args.cmd == "inspect":
        stage_inspect(args)
    elif args.cmd == "agreement":
        stage_agreement(args)
    elif args.cmd == "all":
        if paths(args.tag)["samples"].is_file() and not args.resample:
            print(f"Keeping existing samples for tag '{args.tag}' (use --resample to redraw).")
        else:
            stage_sample(args)
        stage_run(args)
        stage_judge(args)
        stage_report(args)


if __name__ == "__main__":
    main()
