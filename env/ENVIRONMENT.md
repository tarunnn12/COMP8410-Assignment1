# Environment

Recorded 2026-10-22 from the machine that produced the results. Everything below was read from the installed packages and tools, not copied from documentation.

## Machine

- OS: Windows 11 Home 10.0.26200 (the `environment.json` files of pair 2 record 10.0.26300, because its notebook was executed again on 27 October)
- GPU: NVIDIA GeForce RTX 5070 Laptop, 8 GB VRAM (about 7.8 GB usable), compute capability 12.0, driver 610.88
- RAM: 31.4 GB, fixed 32 GB pagefile (a smaller pagefile led to a crash during testing, see `docs/WORK_LOG.md`)
- Python 3.11.9 in four virtual environments under `C:\envs` (all use the system site packages, which hold torch 2.11.0+cu128)
- Energy and cost numbers are specific to this machine and to the assumed electricity price of 0.30 AUD per kWh (an assumption, not a measurement)

## Virtual environments

| Environment | Used by | Main packages |
|---|---|---|
| `C:\envs\gen` (kernel `comp8410-gen`) | pairs 6 and 7 and every notebook driver | torch 2.11.0+cu128, torchvision 0.26.0, transformers 5.20.2, diffusers 0.41.0, accelerate 1.13.0, bitsandbytes 0.49.2, scikit-learn 1.8.0, scipy 1.17.1, pandas 3.0.2, numpy 2.4.3, matplotlib 3.10.8, opencv-python 4.13.0.92, imageio 2.38.1, ollama (python) 0.6.1, lpips 0.1.4, nbconvert 7.17.2 |
| `C:\envs\p4_agents` (kernel `comp8410-p4_agents`) | pair 4 | miniwob 1.1.0, gymnasium 1.2.3, selenium 4.43.0, transformers 4.49.0, ollama (python) 0.6.1 |
| `C:\envs\p2_nerf` | pair 2 (child processes started by the notebook) | nerfstudio 1.1.5, gsplat 1.5.3, nerfacc 0.5.2, pytorch-msssim 1.0.0, torchmetrics 1.9.0, numpy 1.26.4 |
| `C:\envs\p2_3d` | gsplat smoke test only | gsplat 1.5.3 |

## Tools

- CUDA Toolkit 12.8 (nvcc 12.8.93), Visual Studio 2022 Build Tools C++ (cl 14.44), CMake 4.4.4, Ninja 1.13.2
- ffmpeg 9.0.2, COLMAP 4.2.1 (CUDA build)
- Ollama server 0.32.9 with models `qwen2.5vl:7b` (pair 4 vision agent and pair 6 reference model), `qwen3-vl:8b-instruct` (pair 6 system A) and `qwen2.5:7b` (pair 4 text agent), all pulled with the default Q4 quantisation

## Patches to third-party code (all small, all listed here and in the work log)

- `gsplat/cuda/_backend.py` line 177: the GCC-only flag `-Wno-attributes` is skipped on Windows (original kept as `_backend.py.orig`).
- A stand-in module `fpsample.py` (NumPy farthest-point sampling) in the `p2_nerf` environment, because the package source does not build on Windows and the nerfstudio command line imports it at start-up.
- nerfstudio is installed without its pinned gsplat 1.4.0 and nerfacc kernels (they do not compile against torch 2.11 with MSVC); nerfacto therefore runs in its pure-PyTorch mode.
- Build environment for CUDA extensions: `vcvars64.bat`, `CUDA_HOME` set to CUDA 12.8, `TORCH_CUDA_ARCH_LIST=12.0`, `DISTUTILS_USE_SDK=1`, `NVCC_APPEND_FLAGS=-DWIN32_LEAN_AND_MEAN -DNOMINMAX`.

## Local model and data folders (not part of the submission; downloaded as listed in each pair README)

`C:\models` (model weights), `C:\data` (datasets), `C:\data\p6` (POPE and VQAv2 files), `C:\data\p7` (ToxicChat and HateCheck). Hugging Face access had to be accepted for Gemma (ShieldGemma) and Llama (Llama Guard 3) before download.

## Safety settings used for long runs

A Windows job object limits the memory of the nerfstudio jobs (16 GB) and the nerfstudio jobs cap GPU memory at 92 percent of the card, so that an overflow fails with an error instead of spilling into shared memory (`docs/diagnostics/`). The pair 6 and pair 7 notebooks run one model at a time.

## How the environments were created

Each environment is a Python 3.11 virtual environment that uses the system site packages (which hold torch 2.11.0+cu128), for example `python -m venv --system-site-packages C:\envs\gen`. The package versions of the three environments that run notebooks are in `requirements_gen.txt`, `requirements_p4_agents.txt` and `requirements_p2_nerf.txt` (written with `pip list --format=freeze`; they include the system packages). After installing, register the notebook kernels (`python -m ipykernel install --user --name comp8410-gen` and `--name comp8410-p4_agents`) and apply the patches above. Pair 2 needs the CUDA build tools listed under Tools; its capture step (`pairs/p2_3d_reconstruction/src/prepare_chair_video.py`) needs ffmpeg and COLMAP on the path.
