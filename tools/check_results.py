"""Completeness check for the pair result folders.

Usage:  python tools/check_results.py [pair_folder_name ...]
Without arguments every pair is checked. Exit code 1 if anything is missing or empty.
"""
import json
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent / "pairs"

# file -> minimum number of data rows (None = file only has to be non-empty)
EXPECTED = {
    "p2_3d_reconstruction": {"environment.json": None, "summary.csv": 4, "business.csv": 2, "train_log.jsonl": 2, "quality_vs_effort.csv": 6, "chair_video_preparation.json": None,
                             "replicate_seed1/summary.csv": 4, "pooled/p2_mean_and_range.csv": 4, "figures/architecture_p2.png": None},
    "p4_agents": {"environment.json": None, "episodes.jsonl": 300, "summary.csv": 3, "per_task_success.csv": 3,
                  "business.csv": 2, "injection_summary.csv": 2, "figures/architecture_p4.png": None, "figures/make_report_figures.py": None, "failures.csv": 1, "vlm_calibration.csv": 2, "energy.jsonl": 1, "extra/per_task_paired.csv": 10, "extra/injection_with_intervals.csv": 2,
                  "replicate_seed1/episodes.jsonl": 300, "pooled/p4_success_by_agent.csv": 3, "pooled/extra/per_task_paired.csv": 10},
    "p6_lmm_vqa": {"environment.json": None, "predictions.jsonl": 2100, "pope_summary.csv": 3, "vqa_summary.csv": 2,
                   "robustness_flip_rate.csv": 3, "business.csv": 3, "energy.jsonl": 1, "extra/pope_paired_tests.csv": 9, "extra/pope_rates_with_ci.csv": 3, "extra/latency_median_p95.csv": 3,
                   "replicate_seed1/predictions.jsonl": 2100, "pooled/p6_pooled_pope_summary.csv": 3, "pooled/extra/pope_paired_tests.csv": 9,
                   # added analyses (scripts, saved results and the report figures drawn from them)
                   "extra/image_text_injection.py": None, "extra/injection_predictions.jsonl": 600, "extra/injection_summary.csv": 10,
                   "extra/make_extra_figures.py": None, "extra/make_extra_figures2.py": None, "figures/architecture_p6.png": None, "figures/error_types_flips.png": None,
                   "figures/hallucination_overlap.png": None, "figures/speed_energy.png": None, "figures/image_text_injection.png": None},
    "p7_safety": {"environment.json": None, "eval_set.csv": 500, "predictions.jsonl": 2000, "summary.csv": 6,
                  "business.csv": 2, "evasion_recall.csv": 4, "hatecheck_by_functionality.csv": 20, "retail_by_type.csv": 5,
                  "energy.jsonl": 2, "illustrative_workload.csv": 2, "workload_sensitivity.csv": 20,
                  "replicate_seed1/summary.csv": 6, "pooled/p7_pooled_rates.csv": 12, "pooled/p7_pooled_paired_tests.csv": 6,
                  # added analyses (scripts, saved results and the report figures drawn from them)
                  "extra/combined_filters.py": None, "extra/combined_filters.csv": 12, "extra/hatecheck_by_group.py": None, "extra/hatecheck_by_group.csv": 28,
                  "extra/make_extra_figures.py": None, "extra/make_extra_figures2.py": None, "figures/architecture_p7.png": None, "figures/recall_false_blocks.png": None,
                  "figures/hatecheck_by_group.png": None, "figures/roc_shieldgemma.png": None, "figures/evasion_probe.png": None, "figures/review_workload.png": None},
}
COMMON = ["feasibility", "figures"]          # folders every finished pair should hold
SRC_NOTEBOOK = "{pair}_eval.ipynb"           # checked loosely: one .ipynb in src/


def rows(path):
    if path.suffix == ".csv":
        return len(pd.read_csv(path))
    if path.suffix == ".jsonl":
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if path.suffix == ".json":
        json.loads(path.read_text(encoding="utf-8"))
        return 1
    return 1


def has_nan_only_column(path):
    if path.suffix != ".csv":
        return []
    df = pd.read_csv(path)
    return [c for c in df.columns if len(df) and df[c].isna().all()]


def check(pair):
    res = ROOT / pair / "results"
    problems, notes = [], []
    nbs = list((ROOT / pair / "src").glob("*.ipynb"))
    if not nbs:
        problems.append("no notebook in src/")
    executed = list(res.glob("*executed*.ipynb")) if res.exists() else []
    if not executed:
        problems.append("no executed notebook in results/")
    for name, minrows in EXPECTED[pair].items():
        f = res / name
        if not f.exists():
            problems.append(f"missing {name}")
            continue
        try:
            n = rows(f)
        except Exception as e:
            problems.append(f"unreadable {name}: {e}")
            continue
        if f.stat().st_size == 0 or (minrows is not None and n < minrows):
            problems.append(f"{name} has {n} rows, expected at least {minrows}")
        empty = has_nan_only_column(f)
        if empty:
            problems.append(f"{name} has all-empty columns: {empty}")
    for folder in [f for f in COMMON if not (pair == "p7_safety" and f == "feasibility")]:
        d = res / folder
        if not d.exists() or not any(d.iterdir()):
            problems.append(f"folder results/{folder} missing or empty")
    return problems, notes


def main():
    pairs = sys.argv[1:] or sorted(EXPECTED)
    bad = 0
    for p in pairs:
        if p not in EXPECTED:
            print(f"unknown pair {p}")
            bad += 1
            continue
        problems, _ = check(p)
        print(f"[{'OK ' if not problems else 'GAP'}] {p}")
        for pr in problems:
            print("      -", pr)
        bad += bool(problems)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
