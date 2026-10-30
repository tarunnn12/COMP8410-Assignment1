# Pair 6: vision-language models, Qwen3-VL-8B against InternVL3-8B (reference: Qwen2.5-VL-7B)

## What is compared

The models answer questions about photos. System A: Qwen3-VL-8B-Instruct (4-bit GGUF, Ollama, `qwen3-vl:8b-instruct`; the default `qwen3-vl:8b` tag is a reasoning variant and was replaced, see the report). System B: InternVL3-8B (4-bit NF4 with bitsandbytes, transformers). Qwen2.5-VL-7B (4-bit GGUF, Ollama) is kept as a reference line: it is also System B of pair 4 and is not counted as a system of this pair.

## Evaluation design

POPE adversarial (300 yes/no object-presence questions), VQAv2 validation (300 home and furniture questions), 100 re-worded POPE questions for robustness, latency, memory and energy. InternVL tiling is capped at 6 tiles and its output layer is quantised so that the model fits the 8 GB GPU. `p6_lmm_vqa_extra_analysis.ipynb` adds the split by true answer with intervals and tests, an image-clustered bootstrap, a home-object slice and median and 95th percentile latency (CPU only).

## Implementation statement

**Written by me:** the evaluation notebook `src/p6_lmm_vqa_eval.ipynb` (the seeded sampling of the POPE and VQAv2 questions and the manifest of images, the prompts, the wrappers around the Ollama models and around InternVL3 with its 4-bit setting and tile cap, the re-worded question set, the answer normalisation and scoring, GPU sampling and the tests), `src/p6_lmm_vqa_extra_analysis.ipynb`, `results/extra/image_text_injection.py` (the banner test of the report) and the figure scripts in `results/extra/`.

**Third-party, used as installed:** Ollama (`qwen3-vl:8b-instruct`, `qwen2.5vl:7b`), transformers and bitsandbytes with `InternVL3-8B-hf`, and the POPE and VQAv2 datasets (COCO images).

**Test data:** the evaluation questions are a seeded sample. `data/eval_manifest.json` (run 1) and `data_seed1/eval_manifest.json` (run 2) list every question with its true answer; the images are in the Drive folder named in the report (`SLIM_PACKAGE_NOTE.txt` in the submitted Evaluation folder says which files were left out).

## What has to be available

`C:/models/InternVL3-8B-hf`, POPE and VQAv2 parquet files in `C:/data/p6` (`python common/download_assets.py --only p6`); Ollama with `qwen3-vl:8b-instruct` and `qwen2.5vl:7b` (`ollama pull qwen3-vl:8b-instruct`; the default `qwen3-vl:8b` tag is a reasoning variant and was not used for the final run).

Hardware used for all reported numbers: Windows 11, RTX 5070 Laptop GPU (8 GB), 32 GB RAM. Versions are in `../../env/ENVIRONMENT.md` and in `results/environment.json`.

## How to run

```
cd pairs/p6_lmm_vqa/src
python -m nbconvert --to notebook --execute p6_lmm_vqa_eval.ipynb --ExecutePreprocessor.kernel_name=comp8410-gen --ExecutePreprocessor.timeout=-1 --output-dir ../results --output p6_lmm_vqa_eval_executed.ipynb
```

- Environment variables: P6_QUICK=1 (smoke test), P6_RESULTS, P6_DATA, P6_SRC, P6_INTERNVL, P6_SEED (selects the random question sample; the replicate run used 1 with its own `P6_RESULTS` and `P6_DATA` folders).
- A quick mode (the first variable above) runs a very small version in a few minutes to check the setup before a full run. Use a different `P?_RESULTS` folder for it so that the full results are not mixed.
- Runs are resumable: finished items are written to disk and skipped when the notebook is run again.
- Time on the test laptop: about 40 minutes for the first two models (700 answers each) and about 20 more for the third.
- Jupyter kernel `comp8410-gen` is registered from the environment named in `../../env/ENVIRONMENT.md` (`python -m ipykernel install --user --name comp8410-gen`).

## Outputs (`results/`)

`predictions.jsonl`, `pope_summary.csv`, `vqa_summary.csv`, `vqa_summary_lenient.csv`, `answer_length.csv`, `robustness_flip_rate.csv`, `business.csv`, `failures_hallucination.csv`, `failures_vqa.csv`, `extra/` (including `image_text_injection.py`, the banner test with `injection_predictions.jsonl` and `injection_summary.csv`, run with the `gen` environment; `make_extra_figures.py` and `make_extra_figures2.py`, which redraw the report figures from the saved results without running a model), `pooled/` and `replicate_seed1/` (second sample and pooled analysis), `feasibility/` (the first Qwen3-VL attempt), `figures/`, `environment.json`, `energy.jsonl`.

## Notes and limits

Latency drifted within the runs (see `extra/latency_drift.png`); medians are the fairer figure. `data/eval_manifest.json` stores absolute paths of the images created by the notebook; the notebook rebuilds it. A second run with another seed is stored in `results/replicate_seed1/` and the two runs are pooled by `python tools/pool_replicates.py` into `results/pooled/` (the unique questions of the two samples, then the additional analysis notebook). Of the 300 POPE questions of the second sample, 38 were also in the first sample (counted once in the pooled analysis, which leaves 562 unique questions); of its 300 VQA questions, 194 were also in the first sample, so VQA results are reported per run.

The image-text injection test (`results/extra/image_text_injection.py`) is resumable: `results/extra/injection_log.txt` is its first launch, which finished the Qwen3-VL answers and then stopped at the InternVL3 step because the wrong Python environment (an older transformers) was used; the run was continued in the `gen` environment (`injection_log2.txt`). Energy and latency depend on the machine; electricity cost uses an assumed 0.30 AUD per kWh. Quality results (accuracy, success, scores) should be reproducible up to the nondeterminism of GPU kernels and of the Ollama runtime.
