"""Pair 7 extra figures 2 from saved results (no model is run): the evasion probe and the review workload.
Inputs: ../evasion_recall.csv (run 1 sample, 150 unsafe and 150 safe prompts per set), ../workload_sensitivity.csv (run 1 rates).
Output: ../figures/evasion_probe.png and ../figures/review_workload.png.   Run: python make_extra_figures2.py
"""
import pathlib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = pathlib.Path(__file__).resolve().parents[1]; F = R / "figures"
CS = {"A Llama Guard 3-1B": "#1f77b4", "B ShieldGemma-2B": "#ff7f0e"}
e = pd.read_csv(R / "evasion_recall.csv")
fig, ax = plt.subplots(2, 2, figsize=(9, 5.6), sharey=True)
for i, src in enumerate(("toxicchat", "hatecheck")):
    for j, meas in enumerate(("recall (unsafe prompts)", "false-block rate (safe prompts)")):
        a = ax[i, j]
        for k, (sysn, c) in enumerate(CS.items()):
            r = e[(e.source == src) & (e.measure == meas) & (e.system == sysn)].iloc[0]
            vals = [r.original, r.lower, r.leet]; lo = [float(r[c_].split("-")[0]) for c_ in ("original_ci95", "lower_ci95", "leet_ci95")]; hi = [float(r[c_].split("-")[1]) for c_ in ("original_ci95", "lower_ci95", "leet_ci95")]
            x = np.arange(3) + (k - 0.5) * 0.36
            a.bar(x, vals, 0.34, color=c, label=sysn if (i, j) == (0, 0) else None); a.errorbar(x, vals, yerr=[np.array(vals) - lo, np.array(hi) - vals], fmt="none", color="k", capsize=2)
            for xx, v, h in zip(x, vals, hi): a.text(xx, min(h + 0.02, 1.0), f"{v:.0%}", ha="center", fontsize=7)
        a.set_xticks(range(3)); a.set_xticklabels(["as written", "lower case", "leetspeak"]); a.set_ylim(0, 1.08); a.grid(axis="y", alpha=.3)
        a.set_title(f"{src}: {meas}", fontsize=9)
h, l = ax[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=2, fontsize=8); plt.tight_layout(rect=(0, 0.05, 1, 1)); plt.savefig(F / "evasion_probe.png", dpi=170); plt.close()

w = pd.read_csv(R / "workload_sensitivity.csv")
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=True)
for a, src, ttl in zip(ax, ("toxicchat", "retail", "hatecheck"), ("rates from ToxicChat", "rates from the retail set", "rates from HateCheck (stress case)")):
    for sysn, c in CS.items():
        d = w[(w.rates_from == src) & (w.system == sysn)].sort_values("unsafe_share"); a.plot(d.unsafe_share * 100, d.review_hours_per_day, "o-", color=c, label=sysn)
    a.set_title(ttl, fontsize=9); a.set_xlabel("share of messages that are unsafe (%)"); a.grid(alpha=.3)
ax[0].set_ylabel("review hours per day"); ax[0].legend(fontsize=8); plt.tight_layout(); plt.savefig(F / "review_workload.png", dpi=170); plt.close()
