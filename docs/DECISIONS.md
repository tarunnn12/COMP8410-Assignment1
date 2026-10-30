# Decisions

Each decision is recorded with its date, the options considered and the reason.

## 2026-10-21

- Organisation setting: online home-decor retailer. Reason: image, 3D, video, agent, multimodal and safety tasks all have a natural use there, and it stays clear of earlier course material.
- Two systems are two competing stacks (System A and System B). Each pair compares two models on the same task.
- Evaluation, not training, is the focus. Pretrained models are run for inference. Per-scene optimisation (3DGS, NeRF) is the only training expected.
- Toolchain: CUDA 12.8 chosen to match the installed torch cu128 build, instead of the latest CUDA 13.x.
- Pair 7 (safety) depends on Llama Guard approval. A different topic was kept in reserve in case the Llama Guard licence was not granted.

## 2026-10-27

- The assignment was revised: the submission is due on Sunday 8 November 2026, four pairs of models are required instead of seven, and recruiting external participants for user feedback is not required (feedback may come from the student's own evaluations and observations).
- Options for the four pairs: keep the four with the strongest evidence, or the four most different categories. Both gave the same set: pair 2 (3D reconstruction), pair 4 (web agents), pair 6 (vision-language models) and pair 7 (safety filters). They cover four different categories of Section 2.1 (AI generation in 3D, agentic AI, multimodal learning, governance and safety), use recent models (Qwen3-VL, InternVL3, Qwen2.5-VL, Llama Guard 3 and ShieldGemma are from 2024 and 2025; nerfacto and splatfacto are methods from 2022 and 2023), and have the largest evaluation sets with replicate runs.
- Dropped: pair 1 (both models are from 2023, which weakens the "more recent" selection criterion; the same applies to the methods of pair 2, which was kept because it is the only pair of the 3D category and has the largest effect sizes and a second scene of my own capture), pair 3 (small 1.3B and 2B video models, poor clips, 12 prompts, one prompt blank on every seed) and pair 5 (the automatic judge disagreed with the one human check in the direction of the main result). Their code and results were archived outside the submission.
- The kept pairs keep their original numbers (2, 4, 6, 7) so that every file name, log entry and cross-reference stays valid.
- The rater test prepared for pair 2 was removed with the other rating material; no external-rater claim is made anywhere.

