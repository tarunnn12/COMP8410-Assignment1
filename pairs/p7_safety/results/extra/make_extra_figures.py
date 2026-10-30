"""Pair 7 extra figures from saved results (no model is run): per-group HateCheck results and the ShieldGemma ROC curves.
The Llama Guard 3-1B score is not used for any curve: its saved score does not follow its decision (see the report), so only ShieldGemma has a valid AUC.
Output: ../figures/hatecheck_by_group.png and ../figures/roc_shieldgemma.png.   Run: python make_extra_figures.py
"""
import json, pathlib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
R = pathlib.Path(__file__).resolve().parents[1]; F = R / "figures"
C = {"llama_guard_3_1b": "#1f77b4", "shieldgemma_2b": "#ff7f0e"}; NAME = {"llama_guard_3_1b": "A Llama Guard 3-1B", "shieldgemma_2b": "B ShieldGemma-2B"}

g = pd.read_csv(pathlib.Path(__file__).with_name("hatecheck_by_group.csv"))
order = g[(g.measure == "recall on hateful") & (g.model == "llama_guard_3_1b")].sort_values("rate").group.tolist()
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for a, meas, ttl in zip(ax, ("recall on hateful", "false-block rate on non-hateful"), ("Hateful cases caught (higher is better)", "Non-hateful cases blocked (lower is better)")):
    for k, m in enumerate(("llama_guard_3_1b", "shieldgemma_2b")):
        d = g[(g.measure == meas) & (g.model == m)].set_index("group").loc[order]; y = np.arange(len(order)) + (k - 0.5) * 0.28
        a.errorbar(d.rate, y, xerr=[d.rate - d.ci_low, d.ci_high - d.rate], fmt="o", color=C[m], label=NAME[m], capsize=2, ms=5)
    a.set_title(ttl, fontsize=10); a.set_xlim(0, 1.02); a.set_xlabel("share of cases flagged (95% Wilson interval)"); a.grid(axis="x", alpha=.3)
ax[0].set_yticks(range(len(order))); ax[0].set_yticklabels(order); ax[0].legend(loc="lower left", fontsize=8)
plt.tight_layout(); plt.savefig(F / "hatecheck_by_group.png", dpi=170); plt.close()

P = pd.concat([pd.DataFrame([json.loads(l) for l in open(p, encoding="utf-8")]) for p in (R / "predictions.jsonl", R / "replicate_seed1" / "predictions.jsonl")])
P = P[P.model == "shieldgemma_2b"].drop_duplicates(subset=["set", "uid"], keep="first")
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.5), sharey=True); rows = []
for a, s in zip(ax, ("toxicchat", "hatecheck", "retail")):
    d = P[P.set == s]; fpr, tpr, _ = roc_curve(d.label, d.score); auc = roc_auc_score(d.label, d.score)
    a.plot(fpr, tpr, color=C["shieldgemma_2b"], label=f"ShieldGemma-2B (AUC {auc:.2f})"); a.plot([0, 1], [0, 1], "k:", lw=.8)
    dd = d.decision == 1; a.plot((dd & (d.label == 0)).sum() / (d.label == 0).sum(), (dd & (d.label == 1)).sum() / (d.label == 1).sum(), "o", color=C["shieldgemma_2b"], mec="k", label="default decision")
    a.set_title(s, fontsize=10); a.set_xlabel("false-block rate"); a.legend(fontsize=7, loc="lower right"); a.grid(alpha=.3); rows.append((s, len(d), round(auc, 3)))
ax[0].set_ylabel("recall"); plt.tight_layout(); plt.savefig(F / "roc_shieldgemma.png", dpi=170); plt.close()
print(rows)
