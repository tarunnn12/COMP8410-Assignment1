# Decisions

Each decision is recorded with its date, the options considered and the reason.

## 2026-10-21

- Organisation setting: online home-decor retailer. Reason: image, 3D, video, agent, multimodal and safety tasks all have a natural use there, and it stays clear of earlier course material.
- Two systems are two competing stacks (System A and System B). Each pair compares two models on the same task.
- Evaluation, not training, is the focus. Pretrained models are run for inference. Per-scene optimisation (3DGS, NeRF) is the only training expected.
- Toolchain: CUDA 12.8 chosen to match the installed torch cu128 build, instead of the latest CUDA 13.x.
- Pair 7 (safety) depends on Llama Guard approval. Fallback is a topic from weeks 11 or 12, or video editing.
