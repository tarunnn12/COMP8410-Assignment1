"""Pair 6 extra figures 2 from saved results (no model is run): which questions each model hallucinated on, and speed and energy.
Inputs: ../failures_hallucination.csv and ../replicate_seed1/failures_hallucination.csv (pooled, a question counted once), latency_median_p95.csv, ../business.csv.
Output: ../figures/hallucination_overlap.png and ../figures/speed_energy.png.   Run: python make_extra_figures2.py
"""
import itertools, pathlib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = pathlib.Path(__file__).resolve().parents[1]; F = R / "figures"; E = pathlib.Path(__file__).resolve().parent
ORDER = ["qwen3-vl-8b-instruct (ollama q4)", "internvl3-8b (bnb nf4)", "qwen2.5-vl-7b (ollama q4)"]
SHORT = {ORDER[0]: "A", ORDER[1]: "B", ORDER[2]: "Ref"}; COL = {"A": "#1f77b4", "B": "#ff7f0e", "Ref": "#7f7f7f"}

u = pd.concat([pd.read_csv(R / "failures_hallucination.csv"), pd.read_csv(R / "replicate_seed1" / "failures_hallucination.csv")]).drop_duplicates(["model", "id"])
sets = {SHORT[m]: set(u[u.model == m].id) for m in ORDER}
combos = []
for r in (1, 2, 3):
    for c in itertools.combinations(["A", "B", "Ref"], r):
        only = set.intersection(*[sets[k] for k in c]) - set.union(*[sets[k] for k in ("A", "B", "Ref") if k not in c], set())
        combos.append(("+".join(c) if r > 1 else c[0] + " only", len(only)))
combos = sorted(combos, key=lambda x: -x[1])
fig, ax = plt.subplots(figsize=(6.6, 3.3))
ax.barh([c[0] for c in combos][::-1], [c[1] for c in combos][::-1], color="#5b7fa6")
for i, c in enumerate(combos[::-1]): ax.text(c[1] + 0.3, i, str(c[1]), va="center", fontsize=8)
tot = len(set.union(*sets.values()))
ax.set_xlabel(f"questions answered yes although the object is absent ({tot} different questions, both samples pooled)", fontsize=8); ax.grid(axis="x", alpha=.3)
plt.tight_layout(); plt.savefig(F / "hallucination_overlap.png", dpi=170); plt.close()
print(combos, tot)

lat = pd.read_csv(E / "latency_median_p95.csv").set_index("model").loc[ORDER]; bus = pd.read_csv(R / "business.csv").set_index("model").loc[ORDER]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4)); x = np.arange(3); w = 0.38
ax[0].bar(x - w / 2, lat.median_s, w, color="#9ecae1", label="median"); ax[0].bar(x + w / 2, lat.p95_s, w, color="#3182bd", label="95th percentile")
for i in range(3): ax[0].text(i - w / 2, lat.median_s.iloc[i] + 0.08, f"{lat.median_s.iloc[i]:.2f}", ha="center", fontsize=8); ax[0].text(i + w / 2, lat.p95_s.iloc[i] + 0.08, f"{lat.p95_s.iloc[i]:.2f}", ha="center", fontsize=8)
ax[0].set_xticks(x); ax[0].set_xticklabels(["A", "B", "Ref"]); ax[0].set_ylabel("seconds per answer (run 1, 700 answers)"); ax[0].legend(fontsize=8); ax[0].grid(axis="y", alpha=.3)
ax[1].bar(x, bus.gpu_wh_per_1000_answers, color=[COL["A"], COL["B"], COL["Ref"]])
for i in range(3): ax[1].text(i, bus.gpu_wh_per_1000_answers.iloc[i] + 0.8, f"{bus.gpu_wh_per_1000_answers.iloc[i]:.0f} Wh\n{bus.aud_per_1000_answers.iloc[i]:.3f} AUD", ha="center", fontsize=8)
ax[1].set_xticks(x); ax[1].set_xticklabels(["A", "B", "Ref"]); ax[1].set_ylabel("GPU energy per 1,000 answers (Wh)"); ax[1].set_ylim(0, 62); ax[1].grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig(F / "speed_energy.png", dpi=170); plt.close()
