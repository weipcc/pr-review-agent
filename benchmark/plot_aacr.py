#!/usr/bin/env python3
"""
Figure for the report: V1 vs V2 on the 50 small AACR-Bench PRs (selection_100.json, in_small50).
(a) recall and precision (AACR-Bench protocol, judge_aacr_V?.json); (b) where the findings point:
share of findings whose start line is an added line of the diff, and (V2 only, V1 has no evidence
field) share whose quoted evidence appears verbatim in the diff.

    python benchmark/plot_aacr.py <out.pdf|png>        # needs matplotlib
"""

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "aacr"
VERSIONS = ["V1", "V2"]
LABELS = ["V1\nbaseline", "V2\nreworked prompt"]
norm = lambda s: re.sub(r"\s+", " ", s or "").strip()  # noqa: E731


def added_lines(raw: str) -> dict[str, set[int]]:
    out, cur, nl = {}, None, 0
    for ln in raw.splitlines():
        if ln.startswith("diff --git "):
            cur = ln.split(" b/", 1)[1]
            out[cur] = set()
            continue
        h = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)", ln)
        if h:
            nl = int(h.group(1))
            continue
        if cur is None or ln.startswith(("---", "+++")):
            continue
        if ln.startswith("+"):
            out[cur].add(nl)
            nl += 1
        elif not ln.startswith("-"):
            nl += 1
    return out


sel = [x for x in json.loads((HERE / "data/selection/aacr/selection_100.json").read_text())["selected"] if x.get("in_small50")]
st = {v: {"ref": 0, "cov": 0, "find": 0, "eff": 0, "on_added": 0, "ev_ok": 0} for v in VERSIONS}
for x in sel:
    o, r, _, n = x["url"].split("/")[-4:]
    d = OUT / f"{o}__{r}__{n}"
    state = json.loads((d / "state.json").read_text())
    add = added_lines(state["raw_diff"])
    diff_text = norm(state["raw_diff"].replace("\n+", "\n").replace("\n-", "\n"))
    for v in VERSIONS:
        res = json.loads((d / f"{v}.json").read_text())["results"]
        recs = json.loads((d / f"judge_aacr_{v}.json").read_text())["records"]
        s = st[v]
        s["ref"] += len(state["ground_truth"])
        s["cov"] += len({g for z in recs for g in z["matched_gt"]})
        s["eff"] += sum(z["hit_new_gt"] for z in recs)
        for f in res:
            for it in f.get("issues", []):
                s["find"] += 1
                s["on_added"] += it.get("start_line", it.get("line")) in add.get(f["filename"], set())
                ev = norm(it.get("evidence", ""))
                s["ev_ok"] += bool(ev) and ev[:60] in diff_text

fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.2, 1]})
x = range(len(VERSIONS))
w = 0.36
rec = [st[v]["cov"] / st[v]["ref"] for v in VERSIONS]
pre = [st[v]["eff"] / st[v]["find"] for v in VERSIONS]
b1 = a.bar([i - w / 2 for i in x], rec, w, color="#2a78d6", label="recall")
b2 = a.bar([i + w / 2 for i in x], pre, w, color="#eb6834", hatch="//", edgecolor="white", label="precision (lower bound)")
for bars, key, den in ((b1, "cov", "ref"), (b2, "eff", "find")):
    for bar, v in zip(bars, VERSIONS):
        a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, f"{st[v][key]}/{st[v][den]}", ha="center", fontsize=8)
a.set_xticks(list(x), LABELS, fontsize=8)
a.set_ylim(0, 0.65)
a.set_ylabel("share", fontsize=8)
a.set_title("(a) Recall and precision", fontsize=9)
a.legend(fontsize=7, frameon=False, loc="upper left")

on = [st[v]["on_added"] / st[v]["find"] for v in VERSIONS]
bars = b.bar(list(x), on, 0.5, color="#1baf7a")
for bar, val in zip(bars, on):
    b.text(bar.get_x() + bar.get_width() / 2, val + 0.015, f"{100 * val:.0f}%", ha="center", fontsize=8)
b.set_xticks(list(x), LABELS, fontsize=8)
b.set_ylim(0, 1.1)
b.set_ylabel("share of findings", fontsize=8)
b.set_title("(b) Findings on an added line", fontsize=9)

for ax in (a, b):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig(sys.argv[1])
print({v: {"recall": f"{st[v]['cov']}/{st[v]['ref']}", "precision": f"{st[v]['eff']}/{st[v]['find']}",
           "on_added": f"{st[v]['on_added']}/{st[v]['find']}", "evidence_in_diff": f"{st[v]['ev_ok']}/{st[v]['find']}"} for v in VERSIONS})
