# Pair 4: web agents, text agent (Qwen2.5-7B) against vision agent (Qwen2.5-VL-7B) on MiniWoB++

## What is compared

Both agents operate small web tasks in a browser through the MiniWoB++ environment: System A reads the page as a list of DOM elements, System B sees only a screenshot. A random agent is included as a lower bound.

## Evaluation design

10 tasks x 10 seeds per agent in each of two runs (seeds 0 to 9 in `results/`, seeds 10 to 19 in `results/replicate_seed1/`, pooled in `results/pooled/` by `python tools/pool_replicates.py`), step limit 6, temperature 0; the in-page timer is raised from 10 s to 300 s in a local copy of the task pages. A calibration of the screenshot scale (red square at known positions) is part of the notebook. A prompt-injection probe adds an instruction to the page (30 episodes per agent and run: 3 seeds per task). `p4_agents_extra_analysis.ipynb` adds per-task paired tests, a task-level bootstrap and intervals for the injection probe, from the saved episodes (CPU only).

## Implementation statement

**Written by me:** the evaluation notebook `src/p4_agents_eval.ipynb` (the two observation builders, namely the DOM element list for the text agent and the screenshot for the vision agent; the shared prompt and the JSON action parsing; the random, text and vision agents; the screenshot-scale calibration; the episode runner with GPU power sampling; the prompt-injection probe; the scoring and statistics), `src/p4_agents_extra_analysis.ipynb` (paired tests, task-level bootstrap and intervals), `results/figures/make_report_figures.py` and `tools/pool_replicates.py`.

**Third-party, used as installed:** MiniWoB++ 1.1.0 (the tasks and their reward; the 10 s page timer is raised to 300 s in a private copy of the pages), gymnasium, selenium with Chrome, and Ollama serving `qwen2.5:7b` and `qwen2.5vl:7b`.

**Test data:** none is stored. MiniWoB++ generates every page from its environment seed (0 to 19, two runs); the injected sentence is mine.

## What has to be available

Virtual environment `C:/envs/p4_agents` (MiniWoB++ 1.1.0, selenium, Chrome), Ollama with `qwen2.5:7b` and `qwen2.5vl:7b` (digests in `environment.json`). The notebook builds `data/miniwob_html_300s` from the installed MiniWoB++ pages.

Hardware used for all reported numbers: Windows 11, RTX 5070 Laptop GPU (8 GB), 32 GB RAM. Versions are in `../../env/ENVIRONMENT.md` and in `results/environment.json`.

## How to run

```
cd pairs/p4_agents/src
python -m nbconvert --to notebook --execute p4_agents_eval.ipynb --ExecutePreprocessor.kernel_name=comp8410-p4_agents --ExecutePreprocessor.timeout=-1 --output-dir ../results --output p4_agents_eval_executed.ipynb
```

- Environment variables: P4_QUICK=1 (smoke test), P4_RESULTS, P4_DATA, P4_SEED_START (first environment seed; the replicate run used 10, that is seeds 10 to 19 and injection seeds 10 to 12).
- A quick mode (the first variable above) runs a very small version in a few minutes to check the setup before a full run. Use a different `P?_RESULTS` folder for it so that the full results are not mixed.
- Runs are resumable: finished items are written to disk and skipped when the notebook is run again.
- Time on the test laptop: about 35 minutes of model time for the 360 episodes (the vision agent needs about 12 s per episode; the text agent 1.6 s) plus setup and calibration.
- Jupyter kernel `comp8410-p4_agents` is registered from the environment named in `../../env/ENVIRONMENT.md` (`python -m ipykernel install --user --name comp8410-p4_agents`).

## Outputs (`results/`)

`episodes.jsonl` (every episode with its step log), `summary.csv`, `per_task_success.csv`, `injection_summary.csv`, `business.csv`, `failures.csv`, `vlm_calibration.csv`, `extra/` (additional analysis), `pooled/` and `replicate_seed1/` (second run and pooled analysis), `figures/` (the charts of the notebook and the figures of the report: `make_report_figures.py` redraws them from the episode logs and re-renders the pages with the same seeds; no model is run), `environment.json`, `energy.jsonl`.

## Notes and limits

`data/miniwob_html_300s` is a copy of the installed MiniWoB++ task pages with the page timer raised from 10 s to 300 s. The notebook builds it on the first run, and it is third-party code, so it is not part of the submission. A second run with another seed is stored in `results/replicate_seed1/` and the two runs are pooled by `python tools/pool_replicates.py` into `results/pooled/` (20 environment seeds per task, then the additional analysis notebook on the pooled episodes). The replicate must be run with the `p4_agents` environment and kernel.

Energy and latency depend on the machine; electricity cost uses an assumed 0.30 AUD per kWh. Quality results (accuracy, success, scores) should be reproducible up to the nondeterminism of GPU kernels and of the Ollama runtime.
