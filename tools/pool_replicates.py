"""Pool the original run and the replicate run (`results/replicate_seed1`) of each pair.

Usage:  python tools/pool_replicates.py            (all pairs)
        python tools/pool_replicates.py p4 p6      (selected pairs: p2, p4, p6, p7)

Writes `pairs/<pair>/results/pooled/`. Nothing in the original results is changed.

What "pooled" means for each pair (the unit of analysis is stated in the output files):
  p2  two training seeds per method and scene: values per seed, mean and range only (n = 2 is too small for an interval).
  p4  episodes of environment seeds 0 to 9 and 10 to 19 (20 seeds per task), then the extra analysis notebook on the pooled episodes.
  p6  unique questions of the two random samples (a question drawn in both samples counts once), then the extra analysis notebook.
  p7  unique prompts of the two random samples (a prompt drawn in both samples counts once); retail messages are the same 60 in both runs.
"""
import json, pathlib, shutil, subprocess, sys, tempfile
import numpy as np, pandas as pd
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parent.parent / "pairs"
rng = np.random.default_rng(0)

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0, c - h), 3), round(min(1, c + h), 3))

def mcnemar_exact(a, b):
    n = a + b
    return 1.0 if n == 0 else float(stats.binomtest(a, n, 0.5).pvalue)

def boot_mean_ci(x, n=2000):
    x = np.asarray(x, float); m = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)

def jl(path): return pd.DataFrame([json.loads(l) for l in pathlib.Path(path).read_text(encoding="utf-8").splitlines()])

def dirs(pair):
    r = ROOT / pair / "results"; rep = r / "replicate_seed1"
    if not rep.exists(): sys.exit(f"{rep} is missing: run the replicate first")
    out = r / "pooled"; out.mkdir(exist_ok=True); return r, rep, out

def run_extra(pair, notebook, inputs):
    """Execute an analysis notebook on pooled inputs: it reads ../results and ../data relative to its working folder."""
    src_nb = ROOT / pair / "src" / notebook
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp); (tmp / "src").mkdir(); (tmp / "results").mkdir(); (tmp / "data").mkdir()
        for name, path in inputs.items():
            dst = tmp / name; dst.parent.mkdir(exist_ok=True); shutil.copy2(path, dst)
        shutil.copy2(src_nb, tmp / "src" / notebook)
        kernel = json.loads(src_nb.read_text(encoding="utf-8"))["metadata"]["kernelspec"]["name"]
        subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", notebook, f"--ExecutePreprocessor.kernel_name={kernel}",
                        "--output", notebook.replace(".ipynb", "_pooled_executed.ipynb")], cwd=tmp / "src", check=True)
        out = ROOT / pair / "results" / "pooled"; (out / "extra").mkdir(parents=True, exist_ok=True)
        for f in (tmp / "results" / "extra").glob("*"): shutil.copy2(f, out / "extra" / f.name)
        shutil.copy2(tmp / "src" / notebook.replace(".ipynb", "_pooled_executed.ipynb"), out / "extra" / notebook.replace(".ipynb", "_pooled_executed.ipynb"))

# ---------------------------------------------------------------- pair 2
def p2():
    r, rep, out = dirs("p2_3d_reconstruction")
    S = pd.concat([pd.read_csv(r / "summary.csv").assign(seed="run 1 (default seed 42)"), pd.read_csv(rep / "summary.csv").assign(seed="run 2 (seed 1)")])
    cols = ["psnr", "ssim", "lpips", "render_fps", "train_wall_s"]
    S[["scene", "method", "iterations", "seed"] + cols].to_csv(out / "p2_values_per_seed.csv", index=False)
    g = S.groupby(["scene", "method"])[cols].agg(["mean", "min", "max"]).round(3); g.columns = ["_".join(c) for c in g.columns]; g.to_csv(out / "p2_mean_and_range.csv"); print(g.to_string())

# ---------------------------------------------------------------- pair 4
def p4():
    r, rep, out = dirs("p4_agents")
    E = pd.concat([jl(r / "episodes.jsonl"), jl(rep / "episodes.jsonl")], ignore_index=True)
    assert not E.duplicated(["agent", "phase", "task", "seed"]).any(), "episodes overlap between the two runs"
    main = E[E.phase == "main"]; rows = []
    for ag, g in main.groupby("agent"):
        k, n = int(g.success.sum()), len(g); lo, hi = wilson(k, n); rows.append({"agent": ag, "episodes": n, "successes": k, "success_rate": round(k / n, 3), "ci95_low": lo, "ci95_high": hi, "seeds_per_task": g.seed.nunique()})
    pd.DataFrame(rows).to_csv(out / "p4_success_by_agent.csv", index=False); print(pd.DataFrame(rows).to_string())
    with tempfile.TemporaryDirectory() as tmp:
        pe = pathlib.Path(tmp) / "episodes.jsonl"; pe.write_text("\n".join(json.dumps(x, default=str) for x in E.to_dict("records")) + "\n", encoding="utf-8")
        run_extra("p4_agents", "p4_agents_extra_analysis.ipynb", {"results/episodes.jsonl": pe})

# ---------------------------------------------------------------- pair 6
def p6():
    r, rep, out = dirs("p6_lmm_vqa")
    P = pd.concat([jl(r / "predictions.jsonl"), jl(rep / "predictions.jsonl")], ignore_index=True)
    P1 = P.drop_duplicates(["model", "id", "variant"], keep="first")
    dup = P[P.duplicated(["model", "id", "variant"], keep=False)]; ndiff = int((dup.groupby(["model", "id", "variant"]).prediction.nunique() > 1).sum())
    print(f"repeated (model, id, variant) rows: {len(dup) // 2}; of these {ndiff} have a different prediction in the two runs (the first run's answer is kept)")
    M1 = json.loads((r.parent / "data" / "eval_manifest.json").read_text(encoding="utf-8")); M2 = json.loads((r.parent / "data_seed1" / "eval_manifest.json").read_text(encoding="utf-8"))
    D = pd.DataFrame(M1 + M2); cols = [c for c in D.columns if c not in ("id", "image")]    # the image path differs between the two data folders
    bad = int((D[D.duplicated("id", keep=False)].groupby("id")[cols].agg(lambda x: x.astype(str).nunique()) > 1).any(axis=1).sum()) if D.id.duplicated().any() else 0
    print(f"manifest items: {len(D)}, unique ids: {D.id.nunique()}, repeated ids whose question, truth or answers differ between the runs: {bad}")
    D = D.drop_duplicates("id", keep="first")
    pope = P1[(P1.set == "pope") & (P1.variant == "original")].merge(D[["id", "truth"]], on="id"); pope["yes"] = pope.prediction.str.strip().str.lower().str.startswith("yes"); pope["pos"] = pope.truth == "yes"
    rows = []
    for m, g in pope.groupby("model"):
        tp, fp, fn, tn = int((g.yes & g.pos).sum()), int((g.yes & ~g.pos).sum()), int((~g.yes & g.pos).sum()), int((~g.yes & ~g.pos).sum())
        rows.append({"model": m, "unique_pope_questions": len(g), "accuracy": round((tp + tn) / len(g), 3), "precision": round(tp / max(tp + fp, 1), 3), "recall": round(tp / max(tp + fn, 1), 3),
                     "hallucination_rate": round(fp / max(fp + tn, 1), 3), "yes_ratio": round((tp + fp) / len(g), 3)})
    pd.DataFrame(rows).to_csv(out / "p6_pooled_pope_summary.csv", index=False); print(pd.DataFrame(rows).to_string())
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp); (tmp / "predictions.jsonl").write_text("\n".join(json.dumps(x, default=str) for x in P1.to_dict("records")) + "\n", encoding="utf-8")
        (tmp / "eval_manifest.json").write_text(json.dumps(D.to_dict("records")), encoding="utf-8")
        run_extra("p6_lmm_vqa", "p6_lmm_vqa_extra_analysis.ipynb", {"results/predictions.jsonl": tmp / "predictions.jsonl", "data/eval_manifest.json": tmp / "eval_manifest.json"})

# ---------------------------------------------------------------- pair 7
def p7():
    r, rep, out = dirs("p7_safety")
    P = pd.concat([jl(r / "predictions.jsonl"), jl(rep / "predictions.jsonl")], ignore_index=True).drop_duplicates(["model", "uid"])
    E = pd.concat([pd.read_csv(r / "eval_set.csv"), pd.read_csv(rep / "eval_set.csv")]).drop_duplicates("uid")[["uid", "set", "label"]]
    P = P.drop(columns=[c for c in ("set", "label") if c in P]).merge(E, on="uid")
    names = sorted(P.model.unique()); a, b = names[0], names[1]; rows, tests = [], []
    for st in ("toxicchat", "hatecheck", "retail"):
        S = P[P.set == st]; W = S.pivot_table(index="uid", columns="model", values="decision", aggfunc="first").join(E.set_index("uid")[["label"]])
        for m in names:
            for kind, lab in (("recall", 1), ("false_block_rate", 0)):
                g = W[W.label == lab][m].astype(int); k, n = int(g.sum()), len(g); lo, hi = wilson(k, n); rows.append({"set": st, "system": m, "measure": kind, "k": k, "n": n, "rate": round(k / n, 3), "ci95_low": lo, "ci95_high": hi})
        for kind, lab in (("recall", 1), ("false_block_rate", 0)):
            g = W[W.label == lab]; x, y = g[a].astype(int), g[b].astype(int); oa, ob = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
            tests.append({"set": st, "measure": kind, "first": a, "second": b, "items": len(g), "only_first_blocks": oa, "only_second_blocks": ob, "exact_mcnemar_p": mcnemar_exact(oa, ob)})
    pd.DataFrame(rows).to_csv(out / "p7_pooled_rates.csv", index=False); pd.DataFrame(tests).to_csv(out / "p7_pooled_paired_tests.csv", index=False)
    print(pd.DataFrame(rows).to_string()); print(pd.DataFrame(tests).to_string())

if __name__ == "__main__":
    todo = sys.argv[1:] or ["p2", "p4", "p6", "p7"]
    for t in todo:
        print("\n==========", t); globals()[t]()
