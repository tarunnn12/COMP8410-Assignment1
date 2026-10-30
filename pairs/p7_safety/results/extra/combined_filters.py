"""Offline check of combining the two safety filters (pair 7), from the saved decisions only; no model is run.

Rules: A = Llama Guard 3-1B flags, B = ShieldGemma-2B flags, "either" = flag when A or B flags, "both" = flag when both flag.
Sets: ToxicChat, HateCheck and the retail set, pooled over both runs (a prompt drawn in both runs is counted once, first run kept).
Output: combined_filters.csv (same folder) with recall on unsafe prompts and false-block rate on safe prompts, 95% Wilson intervals.
Run:  python combined_filters.py
"""
import json, pathlib
import numpy as np, pandas as pd

R = pathlib.Path(__file__).resolve().parents[1]

def load(p):
    return pd.DataFrame([json.loads(l) for l in open(p, encoding="utf-8")])

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)

P = pd.concat([load(R / "predictions.jsonl"), load(R / "replicate_seed1" / "predictions.jsonl")]).drop_duplicates(subset=["set", "uid", "model"], keep="first")
rows = []
for s in ("toxicchat", "hatecheck", "retail"):
    g = P[P.set == s]
    D = g.pivot_table(index="uid", columns="model", values="decision", aggfunc="first")
    L = g.groupby("uid").label.first().reindex(D.index)
    A, B = D["llama_guard_3_1b"], D["shieldgemma_2b"]
    for rule, d in (("A alone", A), ("B alone", B), ("either flags", ((A == 1) | (B == 1)).astype(int)), ("both flag", ((A == 1) & (B == 1)).astype(int))):
        npos, nneg = int((L == 1).sum()), int((L == 0).sum()); kr, kf = int(d[L == 1].sum()), int(d[L == 0].sum())
        rl, rh = wilson(kr, npos); fl, fh = wilson(kf, nneg)
        rows.append({"set": s, "rule": rule, "unsafe_n": npos, "safe_n": nneg, "recall": round(kr / npos, 3), "recall_ci_low": round(rl, 3), "recall_ci_high": round(rh, 3),
                     "false_block_rate": round(kf / nneg, 3), "fbr_ci_low": round(fl, 3), "fbr_ci_high": round(fh, 3)})
out = pd.DataFrame(rows); out.to_csv(pathlib.Path(__file__).with_name("combined_filters.csv"), index=False)
print(out.to_string(index=False))
