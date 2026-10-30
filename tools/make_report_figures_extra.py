"""Redraw three figures of the report from files that are in the repository (no model is run, CPU only, a few seconds).

    Figure 2   my chair video: three test views, real photo and render, nerfacto above splatfacto
               <- pairs/p2_3d_reconstruction/results/renders/chair_video_{nerfacto,splatfacto}/view{0,4,8}_img.png
               -> pairs/p2_3d_reconstruction/results/figures/chair_video_views.png
    Figure 9   success rate per task of the random, text and vision agents (run 1, 10 seeds per task)
               <- pairs/p4_agents/results/per_task_success.csv
               -> pairs/p4_agents/results/figures/success_per_task_report.png
    Figure 27  hardware telemetry written every 5 seconds during the evaluation runs
               <- docs/diagnostics/telemetry.csv (time of day only; the date is counted up at each midnight)
               -> docs/diagnostics/telemetry.png

Usage:  python tools/make_report_figures_extra.py
"""
import pathlib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
P2 = ROOT / "pairs" / "p2_3d_reconstruction" / "results"; P4 = ROOT / "pairs" / "p4_agents" / "results"

# ---------------------------------------------------------------- figure 2: three test views of the chair video, real (left half) and render (right half)
rows = []
for method in ("nerfacto", "splatfacto"):
    ims = [Image.open(P2 / "renders" / f"chair_video_{method}" / f"view{v}_img.png").convert("RGB") for v in (0, 4, 8)]
    h = min(i.height for i in ims); ims = [i.resize((int(i.width * h / i.height), h)) for i in ims]
    gap = 6; w = sum(i.width for i in ims) + gap * (len(ims) - 1); row = Image.new("RGB", (w, h), "white"); x = 0
    for i in ims: row.paste(i, (x, 0)); x += i.width + gap
    rows.append(row)
W = max(r.width for r in rows); H = sum(r.height for r in rows) + 6
fig2 = Image.new("RGB", (W, H), "white"); y = 0
for r in rows: fig2.paste(r, (0, y)); y += r.height + 6
fig2.save(P2 / "figures" / "chair_video_views.png"); print("figure 2:", fig2.size)

# ---------------------------------------------------------------- figure 9: success rate per task, run 1
d = pd.read_csv(P4 / "per_task_success.csv").set_index("task")
fig, ax = plt.subplots(figsize=(7.3, 2.9), dpi=200); w = 0.27; x = np.arange(len(d))
for k, (col, lab, c) in enumerate((("random", "random", "#bcbcbc"), ("text_qwen2.5-7b", "A: text agent", "#7b1f1f"), ("vlm_qwen2.5vl-7b", "B: vision agent", "#1f4e79"))):
    ax.bar(x + (k - 1) * w, d[col], w, color=c, label=lab)
ax.set_xticks(x); ax.set_xticklabels(d.index, rotation=30, ha="right", fontsize=7); ax.set_ylabel("success rate (10 seeds)", fontsize=8); ax.set_ylim(0, 1.17)
ax.grid(axis="y", alpha=.25); ax.set_axisbelow(True); ax.legend(ncol=3, loc="upper center", frameon=False, fontsize=7)
plt.tight_layout(); plt.savefig(P4 / "figures" / "success_per_task_report.png"); plt.close(); print("figure 9: drawn")

# ---------------------------------------------------------------- figure 27: telemetry; a gap of more than 5 minutes breaks the line
t = pd.read_csv(ROOT / "docs" / "diagnostics" / "telemetry.csv"); n_bad = int(t.time.isna().sum()); t = t.dropna(subset=["time"]).reset_index(drop=True)   # a row that was only partly written when the laptop crashed has no time
sec = pd.to_timedelta(t.time).dt.total_seconds().values; day = 0; ab = [sec[0]]
for a, b in zip(sec, sec[1:]):
    if b < a - 1: day += 1
    ab.append(b + 86400 * day)
t["when"] = pd.Timestamp("2026-10-21") + pd.to_timedelta(ab, unit="s"); seg = (t.when.diff().dt.total_seconds() > 300).cumsum()
fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.65, 3.2), dpi=200, sharex=True)
for _, g in t.groupby(seg):
    a1.plot(g.when, g.gpu_power_w, color="#7b1f1f", lw=.35); a2.plot(g.when, g.gpu_temp_c, color="#1f4e79", lw=.35)
a1.set_ylabel("GPU power (W)", fontsize=7); a2.set_ylabel("GPU temperature (C)", fontsize=7)
a1.set_title("Hardware telemetry written every 5 seconds during the evaluation runs (gaps: machine off, crash or logger stopped)", fontsize=6.5)
for a in (a1, a2): a.grid(alpha=.25); a.tick_params(labelsize=7)
a2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H")); plt.setp(a2.get_xticklabels(), rotation=20, ha="right")
plt.tight_layout(); plt.savefig(ROOT / "docs" / "diagnostics" / "telemetry.png"); plt.close()
print("figure 27: drawn;", t.time.iloc[0], "to", t.time.iloc[-1], f"({len(t)} rows, {seg.nunique()} continuous stretches; {n_bad} partly written row dropped)")
