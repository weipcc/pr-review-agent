#!/usr/bin/env python3
"""
Figure for the report: recall / precision and mean input tokens of V1-V3 on the 10 AACR PRs.
Reads benchmark/results/aacr/*/{V?.json,judge_aacr_V?.json} (written by run_aacr_eval.py run + judge).

    python benchmark/plot_aacr.py <out.pdf>        # needs matplotlib
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "aacr"
VERSIONS = ["V1", "V2", "V3"]
LABELS = ["V1\nbaseline", "V2\nworkflow", "V3\nsandbox"]

stats = {v: {"ref": 0, "ref_hit": 0, "find": 0, "find_hit": 0, "tok": [], "prs": 0} for v in VERSIONS}
for d in sorted(OUT.glob("*__*__*")):
    for v in VERSIONS:
        rp, jp = d / f"{v}.json", d / f"judge_aacr_{v}.json"
        if not (rp.exists() and jp.exists()):
            continue
        r, j = json.loads(rp.read_text(encoding="utf-8")), json.loads(jp.read_text(encoding="utf-8"))
        s = stats[v]
        s["prs"] += 1
        s["tok"].append(r["input_tokens"])
        s["ref"] += j["key"]["n_truth"]
        s["ref_hit"] += len({g for x in j["records"] for g in x["matched_gt"]})
        s["find"] += len(j["records"])
        s["find_hit"] += sum(x["hit_new_gt"] for x in j["records"])  # AACR-Bench precision: greedy dedup

fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.5, 1]})
x = range(len(VERSIONS))
w = 0.36
rec = [stats[v]["ref_hit"] / stats[v]["ref"] for v in VERSIONS]
pre = [stats[v]["find_hit"] / stats[v]["find"] if stats[v]["find"] else 0 for v in VERSIONS]
b1 = a.bar([i - w / 2 for i in x], rec, w, color="#2a78d6", label="recall")
b2 = a.bar([i + w / 2 for i in x], pre, w, color="#eb6834", hatch="//", edgecolor="white", label="precision (lower bound)")
for bar, v in zip(b1, VERSIONS):
    a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{stats[v]['ref_hit']}/{stats[v]['ref']}", ha="center", fontsize=8)
for bar, v in zip(b2, VERSIONS):
    a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{stats[v]['find_hit']}/{stats[v]['find']}", ha="center", fontsize=8)
a.set_xticks(list(x), LABELS, fontsize=8)
a.set_ylim(0, 0.8)
a.set_ylabel("share", fontsize=8)
a.set_title("(a) Recall and precision", fontsize=9)
a.legend(fontsize=7, frameon=False, loc="upper left")

mean_tok = [sum(stats[v]["tok"]) / len(stats[v]["tok"]) for v in VERSIONS]
bars = b.bar(list(x), mean_tok, 0.55, color=["#2a78d6", "#2a78d6", "#1baf7a"])
for bar, t in zip(bars, mean_tok):
    b.text(bar.get_x() + bar.get_width() / 2, t * 1.15, f"{t / 1000:.1f}k", ha="center", fontsize=8)
b.set_yscale("log")
b.set_ylim(1000, 300000)
b.set_xticks(list(x), LABELS, fontsize=8)
b.set_ylabel("input tokens per PR (log)", fontsize=8)
b.set_title("(b) Mean input tokens", fontsize=9)

for ax in (a, b):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig(sys.argv[1])
print({v: {"recall": f"{stats[v]['ref_hit']}/{stats[v]['ref']}", "precision": f"{stats[v]['find_hit']}/{stats[v]['find']}",
           "mean_tok": round(sum(stats[v]["tok"]) / len(stats[v]["tok"])), "total_tok": sum(stats[v]["tok"])} for v in VERSIONS})
