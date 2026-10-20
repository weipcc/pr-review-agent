#!/usr/bin/env python3
"""
Select the AACR-Bench pull requests for the V1/V2 evaluation (writes data/selection/aacr/selection_100.json).

Rule (fixed before any results were seen):
  1. Eligible: the review commit is still available on GitHub, at least one valid reference comment
     (label = 1), and at most 20 changed files in the diff between pr_source_commit and pr_target_commit.
  2. Groups by diff size (characters / 4 of that diff): small <= 6,000 tokens, large > 6,000 tokens
     (6,000 is V2's routing threshold and V3's size limit).
  3. Small group: the 10 pull requests of the pilot plus 70 more; large group: 20.
  4. Within each group: random, stratified by language (allocation proportional to the eligible pool,
     at least one per language), seeded shuffle over URL-sorted pools.
  5. Each language keeps the rest of its shuffled list as reserves: if a selected pull request cannot
     be evaluated, the next reserve of the same group and language replaces it.

    python benchmark/select_aacr.py <dir with <owner>__<repo>__pull__<n>.diff files and _status.json>

The diffs are the GitHub web compare diffs (https://github.com/<repo>/compare/<base>...<target>.diff).
"""

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEL = HERE / "data" / "selection" / "aacr"
SEED = 20261018
N_SMALL, N_LARGE, MAX_FILES, LIMIT = 80, 20, 20, 6000


def allocate(pool_counts: dict[str, int], total: int, fixed: dict[str, int]) -> dict[str, int]:
    """Per-language targets proportional to the pool, >= 1 each, summing to `total`, never below `fixed`."""
    n = sum(pool_counts.values())
    raw = {k: total * v / n for k, v in pool_counts.items()}
    tgt = {k: max(1, int(raw[k]), fixed.get(k, 0)) for k in pool_counts}
    tgt = {k: min(tgt[k], pool_counts[k]) for k in tgt}
    while sum(tgt.values()) < total:  # largest remainder among languages with room left
        k = max((k for k in tgt if tgt[k] < pool_counts[k]), key=lambda k: raw[k] - tgt[k])
        tgt[k] += 1
    while sum(tgt.values()) > total:
        k = max((k for k in tgt if tgt[k] > max(1, fixed.get(k, 0))), key=lambda k: tgt[k] - raw[k])
        tgt[k] -= 1
    return tgt


def main() -> None:
    diffs = Path(sys.argv[1])
    status = json.loads((diffs / "_status.json").read_text())
    rows = json.loads((SEL / "aacr.json").read_text(encoding="utf-8"))
    pilot = [x["url"] for x in json.loads((SEL / "shortlist_aacr.json").read_text(encoding="utf-8"))]
    prs = {}
    for r in rows:
        p = prs.setdefault(r["pr_url"], {"url": r["pr_url"], "lang": r["project_main_language"],
                                         "lines": r["pr_change_line_count"], "gt": 0, "cats": Counter()})
        if r["label"] == 1:
            p["gt"] += 1
            p["cats"][r["category"]] += 1
    excluded = Counter()
    for u, p in prs.items():
        f = diffs / (u.split("github.com/")[1].replace("/", "__") + ".diff")
        if status.get(u) != 200 or not f.exists():
            p["ok"], why = False, "review commit unavailable"
        else:
            d = f.read_text(errors="ignore")
            p["files"], p["diff_tok"] = len(re.findall(r"(?m)^diff --git ", d)), len(d) // 4
            why = "no valid reference comment" if p["gt"] == 0 else "more than 20 files" if p["files"] > MAX_FILES else (
                "empty diff" if p["files"] == 0 else None)
            p["ok"] = why is None
        if why:
            excluded[why] += 1
        p["group"] = ("small" if p.get("diff_tok", 0) <= LIMIT else "large") if p["ok"] else None
    bad_pilot = [u for u in pilot if not prs[u]["ok"] or prs[u]["group"] != "small"]
    assert not bad_pilot, f"pilot PRs not eligible/small: {bad_pilot}"

    rng = random.Random(SEED)
    selected, reserves = [], {}
    for group, total, fixed_urls in (("small", N_SMALL, pilot), ("large", N_LARGE, [])):
        pool = sorted((p for p in prs.values() if p["group"] == group), key=lambda p: p["url"])
        counts = Counter(p["lang"] for p in pool)
        fixed = Counter(prs[u]["lang"] for u in fixed_urls)
        tgt = allocate(dict(counts), total, dict(fixed))
        reserves[group] = {}
        for lang in sorted(counts):
            cand = [p for p in pool if p["lang"] == lang and p["url"] not in fixed_urls]
            rng.shuffle(cand)
            k = tgt[lang] - fixed[lang]
            selected += [dict(p, pilot=True) for p in pool if p["url"] in fixed_urls and p["lang"] == lang]
            selected += [dict(p, pilot=False) for p in cand[:k]]
            reserves[group][lang] = [p["url"] for p in cand[k:]]
    # Batch 1 (run first, to measure cost): the pilot + 30 small + 10 large, taken round-robin over languages
    # from each group's selected pull requests in a seeded random order, so the batch keeps the language spread.
    batch1 = set(pilot)
    rr_rank: dict[str, int] = {}
    for group, k in (("small", 30), ("large", 10)):
        by_lang: dict[str, list] = {}
        for p in sorted((p for p in selected if p["group"] == group and not p["pilot"]), key=lambda p: p["url"]):
            by_lang.setdefault(p["lang"], []).append(p["url"])
        for lst in by_lang.values():
            rng.shuffle(lst)
        langs = sorted(by_lang, key=lambda L: (-len(by_lang[L]), L))
        picked = []
        while any(by_lang.values()):  # full round-robin order; the first k form batch 1
            for L in langs:
                if by_lang[L]:
                    picked.append(by_lang[L].pop(0))
        batch1 |= set(picked[:k])
        rr_rank.update({u: i for i, u in enumerate(picked)})
    out = {
        "batch1_rule": "pilot (10) + 30 small + 10 large, round-robin over languages in a seeded random order",
        "small50_rule": "final set: only the small group is evaluated: the pilot (10) + the first 40 small PRs of the same round-robin order (rr_rank < 40)",
        "rule": __doc__.split("Rule (fixed before any results were seen):")[1].split("python benchmark")[0].strip(),
        "seed": SEED,
        "excluded_from_pool": dict(excluded),
        "selected": [{"url": p["url"], "group": p["group"], "lang": p["lang"], "files": p["files"], "diff_tok_est": p["diff_tok"],
                      "lines": p["lines"], "gt": p["gt"], "categories": dict(p["cats"]), "pilot": p["pilot"],
                      "batch": 1 if p["url"] in batch1 else 2, "rr_rank": rr_rank.get(p["url"]),
                      # final evaluation set (2026-10-19): only small PRs are run; the pilot + the first 40 of the
                      # small round-robin order (= batch 1's 30 + the next 10)
                      "in_small50": p["group"] == "small" and (p["pilot"] or rr_rank[p["url"]] < 40)} for p in selected],
        "reserve": reserves,
    }
    (SEL / "selection_100.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for g in ("small", "large"):
        s = [p for p in selected if p["group"] == g]
        pool_n = sum(1 for p in prs.values() if p["group"] == g)
        print(f"{g}: {len(s)} selected from {pool_n} eligible | GT {sum(p['gt'] for p in s)} | "
              f"files median {sorted(p['files'] for p in s)[len(s) // 2]} max {max(p['files'] for p in s)} | "
              f"diff tokens median {sorted(p['diff_tok'] for p in s)[len(s) // 2]} max {max(p['diff_tok'] for p in s)}")
        print("   languages:", dict(sorted(Counter(p["lang"] for p in s).items())))
        print("   categories:", dict(sum((p["cats"] for p in s), Counter())))
    print("excluded from pool:", dict(excluded))
    b1 = [p for p in selected if p["url"] in batch1]
    print(f"batch 1: {len(b1)} PRs ({sum(p['group'] == 'small' for p in b1)} small incl. pilot, {sum(p['group'] == 'large' for p in b1)} large), "
          f"GT {sum(p['gt'] for p in b1)}, languages {dict(sorted(Counter(p['lang'] for p in b1).items()))}")


if __name__ == "__main__":
    main()
