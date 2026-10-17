# Benchmark

## AACR-Bench evaluation (used in the report)

Compares three versions of the reviewer on 10 real pull requests from
[AACR-Bench](https://huggingface.co/datasets/Alibaba-Aone/aacr-bench), each reviewed at the commit the
reference comments were made on:

| | commit | what it is | code |
|---|---|---|---|
| V1 | 6ddb19c | original framework and prompt (baseline) | `_v1_main/` |
| V2 | 2ed8205 | reworked prompt, few-shot, project context, no tools | `_v2_main/` |
| V3 | cbe4576 | final system: agent + E2B sandbox (main after PR #9) | `_final_main/` |

The `_v*_main/` folders are `git archive` exports of those commits, so each version runs its own real code.
Ground truth = every comment in AACR-Bench with `label = 1` (17 human-written + 17 AI-written = 34).
Matching is done by an LLM judge (`gemini-3.8-flash`); it has not been checked against human labels.

```bash
# needs GEMINI_API_KEY and E2B_KEY in <repo>/.env (GITHUB_TOKEN recommended: anonymous API = 60 requests/hour)
# a venv with the final version's requirements (+ requests, pydantic, google-genai) -> benchmark/_final_venv
benchmark/_final_venv/bin/python benchmark/run_aacr_eval.py build     # inputs + ground truth, no model calls
benchmark/_final_venv/bin/python benchmark/run_aacr_eval.py run       # V1/V2/V3 on all 10 PRs (cached; --force reruns)
benchmark/_final_venv/bin/python benchmark/run_aacr_eval.py judge     # LLM judge
benchmark/_final_venv/bin/python benchmark/run_aacr_eval.py summary   # tables -> results/aacr/summary.md
python benchmark/plot_aacr.py figure.pdf                              # needs matplotlib
```

- `data/selection/aacr/`: AACR-Bench rows (`aacr.json`), the 38 PRs with <= 2 files and a valid human
  comment (`aacr_ok.json`), the 10 chosen (`shortlist_aacr.json`; ClickHouse#85266 was replaced by
  bitcoin#31886 because its review commit no longer exists on GitHub) and their diffs.
- `results/aacr/<owner>__<repo>__<n>/`: `state.json` (input + ground truth), `V?.json` (each version's
  review, tokens, time, agent tool calls), `judge_V?.json` (verdicts). `summary.md` has the tables.

Headline (one run per version): recall V1 10/34, V2 9/34, V3 3/34; precision (lower bound) 10/22, 9/16, 3/5;
mean input tokens 4.5k / 9.9k / 87.8k. V3 used all 10 tool steps on 8 of 10 PRs and 7 of those ended with no
findings (3 missing files in the output), which the product would show as "Looks Good".

## Earlier snippet-level benchmark (github-codereview)

The rest of this file describes the first benchmark. Its data and results are not committed
(`run_benchmark.py fetch` downloads the data again).

Measures how well the reviewer's **prompt** matches real human code review, on the
[`ronantakizawa/github-codereview`](https://huggingface.co/datasets/ronantakizawa/github-codereview)
dataset (permissively licensed open-source repos; the *test* split is used, 11,013 rows).

Everything is in `run_benchmark.py`. It needs only `requests`, `pydantic` and `google-genai`
(already in the venv) and a `GEMINI_API_KEY`.

## Quick start

```bash
source venv/bin/activate
export GEMINI_API_KEY=...                      # or put it in <repo>/.env

python benchmark/run_benchmark.py fetch                                   # once, ~2 min
python benchmark/run_benchmark.py sample --tag smoke --per-type 2 --n-neg 4
python benchmark/run_benchmark.py run    --tag smoke                      # 4 conditions
python benchmark/run_benchmark.py judge  --tag smoke
python benchmark/run_benchmark.py report --tag smoke                      # -> results/smoke/summary.md

python benchmark/run_benchmark.py all --tag main --workers 6              # the real run
```

Every model call is cached under `results/<tag>/`; re-running resumes and only retries failures.
Sampling is seeded (`--seed`), so the same tag gives the same samples.

## Design

**Sample.** One human review comment (a *positive*) or a "changed but no comment" file
(a *negative*) on a ~50-line chunk plus the diff hunk that was commented on. We build
exactly what the product sends for a file: `File: <path>` + hunk, and the chunk as `N: code`.
Positives are stratified by comment type (default 155) and 60 negatives are added; at most two
samples per pull request; oversized chunks are skipped.

**Conditions** (same model, same inputs; only the prompt differs). The model is
`gemini-3.8-flash` for every condition and for the judge (`DEFAULT_MODEL`; override with `--model`):

| | Prompt | Source |
|---|---|---|
| `A_ver1` | original: Chinese, 4 severities, no few-shot | `reviewer.py` @ `6ddb19c` |
| `B_prompt` | English prompt + 7-category schema | `reviewer.py` @ `2ed8205` |
| `C_fewshot` | B + 3 few-shot examples | same |
| `D_context` | C + PR title as `<pr_info>` | same |

The code at each ref is read with `git show` and executed, so each condition uses the real prompt
code rather than a copy. PR #7 only swapped the LLM client (prompts unchanged), hence `2ed8205`.
Use `--prompt-ref` to benchmark another commit.

**Metrics**

- **Recall (judge)** — does at least one finding raise the same concern as the human comment?
  Decided by an LLM judge (`JUDGE_SYSTEM` in the script). *Main metric.*
- **Location hit @±3** — a finding within 3 lines of the human line. Shown next to a trivial
  baseline, see below.
- **False-positive rate** — on negatives, share with any finding; and with a bug/security finding
  of confidence ≥ medium.
- Findings per sample, mean input tokens and latency per call.

Intervals are 95% Wilson; differences between conditions use a paired bootstrap on the same samples.

## Things you must know before quoting numbers

1. **Line numbers are a weak signal.** The dataset cuts each chunk with 25 lines of context *before*
   the commented line, so 83.6% of the human lines are exactly line 26 (85.3% within ±3; measured on
   the whole test split). A reviewer that always says "line 26" scores ~85% on location. The report
   prints this baseline; rely on judge recall.
2. **Absolute recall is low by construction.** Many human comments need information a diff-only
   reviewer cannot have (team conventions, history, a linter that ran in CI). Compare conditions
   with each other; do not read the absolute number as "quality".
3. **Human comments are not exhaustive ground truth**, so extra findings count as noise, and
   "negative" only means *no comment was left*.
4. **~61% of human comments are `suggestion`-type**, a category our schema does not have.
5. **About 1 in 5 human comments sit on unchanged lines**, which the prompt tells the model not to
   review (only ~50% of hunks can be aligned to the chunk, so this is measured on that subset).
6. **The judge is the same model family as the reviewer.** Validate it by hand:
   `inspect -n 30 --csv labels.csv`, fill `human_label` with y/n, then `agreement --csv labels.csv`.
7. **The dataset cannot test** repo exploration (sandbox), cross-file breaking changes, or README
   context: chunks are single files with no repository. Use case studies on real PRs for those.
8. Small strata (security/performance, 20 each) give wide intervals. Look at the intervals.

## Not implemented yet

- A condition for the sandbox agent (PR #9). Its prompt lives in a different code path
  (`build_system_prompt_batch`, `RULES`) and it needs a checked-out repo, so it cannot be fed a
  50-line chunk. If it should be compared on the prompt alone, add a new condition `kind`.
- README context (`D_context` only adds the PR title; the dataset has no README).
