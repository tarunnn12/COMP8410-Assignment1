# COMP8410 Assignment 1: Evaluation of four pairs of AI models

Evaluation of two AI systems (four pairs, eight distinct models) as the Chief AI Officer of a home-decor online retailer. Each pair compares two recent models on the same task, inputs and seeds, with one shared framework (`docs/EVALUATION_FRAMEWORK.md`).

The assignment brief was revised during the work (due 8 November 2026, four pairs instead of seven). The four pairs with the strongest evidence and the most different categories were kept and keep their original numbers 2, 4, 6 and 7, so that file names, the work log and cross-references stay valid (see `docs/DECISIONS.md`).

| Pair | Business need | System A | System B | Folder |
|---|---|---|---|---|
| 2 | 3D view from a capture | nerfacto (NeRF) | splatfacto (3D Gaussian Splatting) | `pairs/p2_3d_reconstruction` |
| 4 | Back-office web agent | Qwen2.5-7B text agent | Qwen2.5-VL-7B vision agent | `pairs/p4_agents` |
| 6 | Questions about photos | Qwen3-VL-8B-Instruct (Qwen2.5-VL-7B as reference) | InternVL3-8B | `pairs/p6_lmm_vqa` |
| 7 | Chat message screening | Llama Guard 3-1B | ShieldGemma-2B | `pairs/p7_safety` |

## How to reproduce

1. Set up the machine and environments as described in `env/ENVIRONMENT.md` (Windows, CUDA 12.8, Python 3.11 virtual environments, Ollama). The installed package versions of the three environments are in `env/requirements_gen.txt`, `env/requirements_p4_agents.txt` and `env/requirements_p2_nerf.txt` (torch 2.11.0 with CUDA 12.8 comes from the PyTorch index). The third-party patches that Windows needed are listed in `env/ENVIRONMENT.md`.
2. Download the models and datasets: `python common/download_assets.py --list`, then `python common/download_assets.py --only p2 p6 p7` (pair 4 needs only the Ollama models and the MiniWoB++ package). Two model families are gated on Hugging Face (Llama Guard, ShieldGemma) and need an accepted licence.
3. Run one pair at a time, following `pairs/pN_*/README.md` (each notebook has a quick mode for a first check and is resumable). `tools/run_quick.ps1 -Pair pN` runs the quick mode.
The executed copies of the main runs in `results/` were saved before small changes were made to the shipped notebooks: environment-variable defaults for the seed (`P4_SEED_START`, `P6_SEED`, `P7_SEED`), the seed-generalised bootstrap of pair 4, the 15,000-iteration default of pair 2 and a portable lookup of the build tools. With the defaults the shipped notebooks run the settings that produced the reported results.
4. Check that each result folder is complete: `python tools/check_results.py`.
5. Each pair was run twice with a different seed or sample (`results/` and `results/replicate_seed1/`); `python tools/pool_replicates.py` produces the pooled tables in `results/pooled/`.
6. Additional analyses and the report figures that are drawn from saved results (CPU only, a few seconds each): in `pairs/p7_safety/results/extra/` run `combined_filters.py`, `hatecheck_by_group.py` (needs `python common/download_assets.py --only p7`), `make_extra_figures.py` and `make_extra_figures2.py`; in `pairs/p6_lmm_vqa/results/extra/` run `make_extra_figures.py` and `make_extra_figures2.py`; `python tools/make_architecture_figures.py`; `pairs/p4_agents/results/figures/make_report_figures.py` (needs Chrome and the `p4_agents` environment) and `python tools/make_report_figures_extra.py` (the chair-video views, the success-per-task chart and the telemetry chart). The one added analysis that runs a model is the image-text injection test of pair 6: with Ollama running, `python pairs/p6_lmm_vqa/results/extra/image_text_injection.py` in the `gen` environment (roughly half an hour on the RTX 5070 Laptop GPU, resumable; it writes `injection_predictions.jsonl` and `injection_summary.csv`).
7. Check the report against the raw files: `python tools/verify_report_numbers.py --report "49030000 Report.pdf"` recomputes 29 headline numbers from the prediction and episode files, compares them with the printed values and checks that each is in the report (CPU, about 10 seconds).
8. The other deliverables are submitted next to this folder: `49030000 Report.pdf`, `49030000 Presentation.pptx` and `49030000 Video` (they are submitted on iLearn with the zip and are not part of this repository). The submitted Evaluation folder is a slim copy built by `python tools/make_submission.py`; it leaves out the large image and video files, listed in `SLIM_PACKAGE_NOTE.txt`, which are in the shared Drive folder named in the report.

## Layout

- `pairs/pN_*/src/` evaluation notebooks (the code); `results/` raw outputs, tables, figures and executed notebooks; `data/` test sets built by the notebooks
- `common/` asset download script; `tools/` completeness check, quick-run script, replicate pooling and submission builder
- `docs/` evaluation framework, decisions, dated work log, diagnostics from the long runs
- `env/` hardware and software versions, with the package lists of the three environments

## User feedback

Following the revised brief, no external participants were recruited. The feedback in the report is my own structured account of each pair (set-up effort, failures met, what the outputs looked like in use, disagreements between a first impression and a metric), linked to files in `results/` and `docs/WORK_LOG.md`.

## Work Proof

The dated work log is `docs/WORK_LOG.md`; the Git history of this repository is the version record; every result folder carries `environment.json` and, where applicable, `energy.jsonl` with measured run times.
