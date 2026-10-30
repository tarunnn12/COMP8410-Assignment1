"""Pair 7 extra: HateCheck results per target group, from the saved decisions only; no model is run.

The target group (target_ident) comes from the public HateCheck test file (Paul/hatecheck on Hugging Face, test.csv, case_id = uid without the hc_ prefix).
Hateful cases: recall per group. Non-hateful cases (counter speech, slur reclaimed, negation, profanity, etc.) that name a group: false-block rate per group.
Pooled over both runs (a case drawn in both runs is counted once). 95% Wilson intervals.
Output: hatecheck_by_group.csv (same folder).   Run:  python hatecheck_by_group.py
"""
import json, pathlib, glob
import numpy as np, pandas as pd
R = pathlib.Path(__file__).resolve().parents[1]
import os
HC = [p for p in [os.environ.get("P7_HATECHECK", ""), "C:/data/p7/hatecheck/test.csv"] + glob.glob(str(pathlib.Path.home() / ".cache/huggingface/hub/datasets--Paul--hatecheck/snapshots/*/test.csv")) if p and pathlib.Path(p).exists()]
assert HC, "HateCheck test.csv not found: run python common/download_assets.py --only p7 (or set P7_HATECHECK)"
hc = pd.read_csv(HC[0])
hc["uid"] = "hc_" + hc.case_id.astype(str)
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); h = h / d; return round(max(0, c - h), 3), round(min(1, c + h), 3)
P = pd.concat([pd.DataFrame([json.loads(l) for l in open(p, encoding="utf-8")]) for p in (R / "predictions.jsonl", R / "replicate_seed1" / "predictions.jsonl")])
P = P[P.set == "hatecheck"].drop_duplicates(subset=["uid", "model"], keep="first").merge(hc[["uid", "target_ident"]], on="uid", how="left")
print("cases with a group:", P.target_ident.notna().groupby(P.model).sum().to_dict(), "| without:", P.target_ident.isna().groupby(P.model).sum().to_dict())
rows = []
for m, g in P.groupby("model"):
    for grp, gg in g.groupby("target_ident"):
        for lab, kind in ((1, "recall on hateful"), (0, "false-block rate on non-hateful")):
            x = gg[gg.label == lab]; k = int(x.decision.sum()); n = len(x)
            if n: lo, hi = wilson(k, n); rows.append({"model": m, "group": grp, "measure": kind, "n": n, "flagged": k, "rate": round(k / n, 3), "ci_low": lo, "ci_high": hi})
out = pd.DataFrame(rows); out.to_csv(pathlib.Path(__file__).with_name("hatecheck_by_group.csv"), index=False)
print(out.pivot_table(index="group", columns=["measure", "model"], values="rate").round(2).to_string())
print(out.pivot_table(index="group", columns=["measure", "model"], values="n").to_string())
