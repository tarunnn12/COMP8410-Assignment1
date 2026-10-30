# Pair 2: 3D reconstruction from images, nerfacto (NeRF) against splatfacto (3D Gaussian Splatting)

## What is compared

Both methods are trained with nerfstudio on the same capture and scored on held-out views. System A is nerfacto (hash-grid NeRF, Mildenhall et al., 2020; Muller et al., 2022), run in its pure-PyTorch mode. System B is splatfacto (3D Gaussian Splatting, Kerbl et al., 2023, with the gsplat library, Ye et al., 2025).

## Evaluation design

Two scenes. `poster`: the nerfstudio sample data (a handheld capture of a folding chair, 100 frames at quarter resolution, COLMAP poses). `chair_video`: my own phone video of an office chair (`data/chair_orbit_low.mp4` for training, `data/chair_orbit_high.mp4` for testing; frames, COLMAP poses and the nerfstudio file are produced by `src/prepare_chair_video.py`). Main comparison: nerfacto 2,500 iterations against splatfacto 15,000. A second table (`quality_vs_effort.csv`) repeats the comparison for more nerfacto iterations (5,000 and 10,000) and fewer splatfacto iterations (1,000, 3,000, 7,000) so that the effect of the training budget can be read. Metrics: PSNR, SSIM, LPIPS on held-out views (10 for the poster scene, 20 for the chair video), render speed, training time, GPU energy, model size, number of Gaussians. Memory limits: each nerfstudio job runs inside a 16 GB memory cap and a 92% GPU memory cap.

## Implementation statement

**Written by me:** the evaluation notebook `src/p2_3d_reconstruction_eval.ipynb` (the build wrapper, the training runs as `ns-train` child processes with timing and GPU power sampling, an evaluation worker that renders the held-out views and scores them, the quality-against-effort curve, the paired tests, the summaries and the figures), `src/prepare_chair_video.py` (frame extraction, blur filter, COLMAP poses and `transforms.json`), the viewer-check and opacity-check scripts in `results/feasibility/`, and `tools/pool_replicates.py` for the two-seed pooling.

**Third-party, used as installed:** nerfstudio 1.1.5 (nerfacto, splatfacto and its metric code), gsplat, nerfacc, PyTorch, COLMAP and ffmpeg. The small Windows patches are listed in `env/ENVIRONMENT.md`. No model weights are shipped; both methods are trained from scratch on each scene.

**Test data:** the scene `poster` is the nerfstudio sample capture (downloaded by `common/download_assets.py`); the scene `chair_video` is my own phone video (`data/chair_orbit_low.mp4` and `data/chair_orbit_high.mp4`; in the shared Drive folder named in the report, and in the full working repository).

## What has to be available

Virtual environment `C:/envs/p2_nerf` with nerfstudio 1.1.5, gsplat 1.5.3 and nerfacc 0.5.2 (see `env/ENVIRONMENT.md` for the small patches that Windows with torch 2.11 needed), CUDA 12.8, Visual Studio 2022 Build Tools and Ninja (gsplat compiles its CUDA kernels on first use, about 80 seconds). Data: `python common/download_assets.py --only p2` (also writes `transforms_subset.json`, because the online copy holds only 100 of the 226 listed frames).

Hardware used for all reported numbers: Windows 11, RTX 5070 Laptop GPU (8 GB), 32 GB RAM. Versions are in `../../env/ENVIRONMENT.md` and in `results/environment.json`.

## How to run

```
# 1. only for the chair video: frames, COLMAP poses and transforms.json (needs ffmpeg and COLMAP; the two videos are in data/)
cd pairs/p2_3d_reconstruction/src
python prepare_chair_video.py

# 2. the evaluation (the notebook defaults are the reported settings: 2,500 nerfacto and 15,000 splatfacto iterations)
python -m nbconvert --to notebook --execute p2_3d_reconstruction_eval.ipynb --ExecutePreprocessor.kernel_name=comp8410-gen --ExecutePreprocessor.timeout=-1 --output-dir ../results --output p2_3d_reconstruction_eval_executed.ipynb

# 3. the replicate run (nerfstudio seed 1; skips the extra curve runs), then pooling
$env:P2_SEED = "1"; $env:P2_RESULTS = "<path to>/results/replicate_seed1"
python -m nbconvert --to notebook --execute p2_3d_reconstruction_eval.ipynb --ExecutePreprocessor.kernel_name=comp8410-gen --ExecutePreprocessor.timeout=-1 --output-dir ../results/replicate_seed1 --output p2_3d_reconstruction_eval_executed.ipynb
cd ../../..
python tools/pool_replicates.py
```

- Environment variables: P2_QUICK=1 (smoke test), P2_RESULTS, P2_SCENES (default `poster,chair_video`; adding `chair` brings in the NeRF-synthetic chair, which did not fit nerfacto on an 8 GB GPU), P2_CHAIR_VIDEO (path of the chair scene `transforms.json`), P2_NERF_ITERS (default 2500), P2_SPLAT_ITERS (default 15000, as reported; the executed notebooks in `results/` were run with this value set by the environment variable, before it became the default), P2_VCVARS and P2_CUDA_HOME (the Visual Studio `vcvars64.bat` and the CUDA 12.8 folder, if they are not in their default install folders; ninja is found on PATH or in the WinGet folder), P2_SEED (nerfstudio machine seed; the replicate run used 1 and skips the extra curve runs).
- A quick mode (the first variable above) runs a very small version in a few minutes to check the setup before a full run. Use a different `P?_RESULTS` folder for it so that the full results are not mixed.
- Runs are resumable: finished items are written to disk and skipped when the notebook is run again.
- Time on the test laptop: about 2 hours including the quality-against-effort runs (nerfacto 10,000 iterations alone takes about 34 minutes at 0.21 s per iteration; 2,500 iterations took about 10 minutes at 0.23 s per iteration).
- Jupyter kernel `comp8410-gen` is registered from the environment named in `../../env/ENVIRONMENT.md` (`python -m ipykernel install --user --name comp8410-gen`).

## Outputs (`results/`)

`summary.csv`, `quality_vs_effort.csv`, `business.csv`, `chair_video_preparation.json`, `train_log.jsonl`, `feasibility/viewer_check/` (the models opened in the nerfstudio viewer: launcher script, screenshots and notes), `feasibility/opacity_check/` (script and output of the opacity check of the saved splatfacto checkpoints), `eval_*.json` (per-view metrics), `renders/` (ground truth and rendered views), `figures/`, `ns_runs/` (nerfstudio runs; the checkpoints in `nerfstudio_models/` take about 2.8 GB with the logs and are left out of the submission zip by `tools/make_submission.py`), `pooled/` and `replicate_seed1/` (second run and pooled values), `environment.json`.

## Notes and limits

Two scenes. On the poster scene the 10 held-out views are taken from the same handheld sweep as the training views, so PSNR measures interpolation. On my own chair video the 20 held-out views come from a higher circle that was not used for training, so they test new viewpoints; both circles come from one capture, one chair and one polished floor. A second run with another seed is stored in `results/replicate_seed1/` and the two runs are pooled by `python tools/pool_replicates.py` into `results/pooled/` (values per seed, mean and range; two seeds are too few for an interval).

Energy and latency depend on the machine; electricity cost uses an assumed 0.30 AUD per kWh. Quality results (accuracy, success, scores) should be reproducible up to the nondeterminism of GPU kernels and of the Ollama runtime.
