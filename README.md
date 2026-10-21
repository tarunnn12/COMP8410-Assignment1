# COMP8410 Assignment 1: Evaluation Presentation

Evaluation of two AI systems (seven model pairs, fourteen models) for a home-decor online retailer. Each pair compares two models on the same task under a shared evaluation framework.

Status: setup phase (started 2026-10-21). Due 2026-11-06.

## Planned pairs

| Pair | Topic | System A | System B |
|---|---|---|---|
| p1 | Image editing | InstructPix2Pix | SD inpainting + ControlNet |
| p2 | 3D reconstruction | 3D Gaussian Splatting | NeRF (nerfacto) |
| p3 | Video generation | Wan2.1-1.3B | CogVideoX-2B |
| p4 | Agents | LLM text agent | VLM agent (MiniWoB++) |
| p5 | Flow vs diffusion | SD3.5-Medium | SDXL |
| p6 | Multimodal LMM / VQA | Qwen2.5-VL-7B | InternVL |
| p7 | Safety / governance | Llama Guard 3-1B | ShieldGemma-2B |

Pair choices may change after feasibility testing. Final choices are recorded in `docs/DECISIONS.md`.

## Layout

- `common/` shared metrics and helper code
- `pairs/pN_*/` per-pair `src/` (code), `data/` (test sets), `results/` (raw outputs and tables)
- `docs/` evaluation framework, decisions, dated work log
- `env/` hardware, library versions, setup notes
- `report/`, `slides/`, `video/` deliverables

## Reproducing results

Per-pair instructions are added to each `pairs/pN_*/README.md` as the pair is completed. Environment details are in `env/ENVIRONMENT.md`.

## Work Proof

The dated work log is `docs/WORK_LOG.md`. Git history is the version record.
