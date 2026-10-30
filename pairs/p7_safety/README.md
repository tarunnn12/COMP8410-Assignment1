# Pair 7: safety classifiers as chatbot input filters, Llama Guard 3-1B against ShieldGemma-2B

## What is compared

Every customer message is screened before the chatbot answers. System A: Llama Guard 3-1B (13 hazard categories, label by generation). System B: ShieldGemma-2B (four policies, probability of violation; flagged when the highest is at least 0.5).

## Evaluation design

ToxicChat test sample (150 toxic including 30 jailbreaks, 450 harmless), HateCheck (30 cases from each of 29 functional tests), a 60-message retail set written for this project, and an evasion probe (150 unsafe and 150 safe prompts per source in lower case and leetspeak). Recall and false-block rate with Wilson intervals, AUC for ShieldGemma only (the saved score of Llama Guard does not follow its decision, so it is not used for AUC or thresholds), exact McNemar tests, latency, memory, energy, and an illustrative review-workload table with a sensitivity grid (all volumes and review times are assumptions).

## Implementation statement

**Written by me:** the evaluation notebook `src/p7_safety_eval.ipynb` (the seeded samples of ToxicChat and HateCheck, the 60 retail messages, the lower-case and leetspeak versions for the evasion probe, the scoring wrappers with the chat template of Llama Guard and the policy prompts and yes/no probabilities of ShieldGemma, the metrics, the paired tests and the review-workload model), `results/extra/combined_filters.py` (combining the two decisions), `results/extra/hatecheck_by_group.py` (results per target group) and the figure scripts in `results/extra/`.

**Third-party, used as installed:** transformers with `Llama-Guard-3-1B` and `shieldgemma-2b`, and the ToxicChat and HateCheck datasets.

**Test data:** `results/eval_set.csv` lists every prompt of the main evaluation (set, subtype, text and label); the retail messages were written for this project.

## What has to be available

`C:/models/Llama-Guard-3-1B` and `C:/models/shieldgemma-2b` (both gated), ToxicChat and HateCheck in `C:/data/p7` (`python common/download_assets.py --only p7`).

Hardware used for all reported numbers: Windows 11, RTX 5070 Laptop GPU (8 GB), 32 GB RAM. Versions are in `../../env/ENVIRONMENT.md` and in `results/environment.json`.

## How to run

```
cd pairs/p7_safety/src
python -m nbconvert --to notebook --execute p7_safety_eval.ipynb --ExecutePreprocessor.kernel_name=comp8410-gen --ExecutePreprocessor.timeout=-1 --output-dir ../results --output p7_safety_eval_executed.ipynb
```

- Environment variables: P7_QUICK=1 (smoke test), P7_RESULTS, P7_SEED (selects the random samples of prompts; the replicate run used 1).
- A quick mode (the first variable above) runs a very small version in a few minutes to check the setup before a full run. Use a different `P?_RESULTS` folder for it so that the full results are not mixed.
- Runs are resumable: finished items are written to disk and skipped when the notebook is run again.
- Time on the test laptop: about 20 minutes (2,730 prompts per model).
- Jupyter kernel `comp8410-gen` is registered from the environment named in `../../env/ENVIRONMENT.md` (`python -m ipykernel install --user --name comp8410-gen`).

## Outputs (`results/`)

`eval_set.csv` (every prompt used), `predictions.jsonl`, `summary.csv`, `evasion_recall.csv`, `hatecheck_by_functionality.csv`, `retail_by_type.csv`, `retail_missed_unsafe.csv`, `retail_false_blocks.csv`, `workload_sensitivity.csv`, `illustrative_workload.csv`, `toxicchat_by_subtype.csv`, `llama_guard_category_counts.csv`, `business.csv`, `extra/combined_filters.py` and `combined_filters.csv` (offline check of flagging when either or both filters flag, from the saved decisions; `python results/extra/combined_filters.py`), `extra/hatecheck_by_group.py` and `.csv` (HateCheck results per target group from the saved decisions; needs the public `Paul/hatecheck` test.csv in the Hugging Face cache), `extra/make_extra_figures.py` and `extra/make_extra_figures2.py` (draw `figures/hatecheck_by_group.png`, `roc_shieldgemma.png`, `evasion_probe.png` and `review_workload.png` from the saved results, without running a model), `pooled/` and `replicate_seed1/` (second sample and pooled analysis), `figures/` (`roc.png` is kept for the record but its Llama Guard curve is not valid, see above; the report uses `roc_shieldgemma.png`), `environment.json`, `energy.jsonl`.

## Notes and limits

ToxicChat is licensed CC BY-NC 4.0 and is used for research evaluation only. The retail messages are requests and statements only, no harmful content is produced. A second run with another seed is stored in `results/replicate_seed1/` and the two runs are pooled by `python tools/pool_replicates.py` into `results/pooled/` (the unique prompts of the two samples; the 60 retail messages are the same in both runs).

Energy and latency depend on the machine; electricity cost uses an assumed 0.30 AUD per kWh. Quality results (accuracy, success, scores) should be reproducible up to the nondeterminism of GPU kernels and of the Ollama runtime.
