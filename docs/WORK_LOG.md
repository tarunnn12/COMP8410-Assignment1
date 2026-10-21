# Work log

Dated record of what was done and why. Entries are factual and added as work happens.

## 2026-10-21

- Read the assignment brief and the lecture material (weeks 1 to 10; weeks 11 and 12 not released yet).
- Updated brief received (`COMP8410_Assignments_1-updatedd.pdf`). The convenor's forum reply on AI use is saved separately by the student.
- Selected a candidate set of seven model pairs aligned with lecture topics and the Section 2.1 categories (see `README.md`). To be confirmed after feasibility tests.
- Hardware and software feasibility checks run: GPU/PyTorch (RTX 5070 Laptop, 8GB, torch 2.11+cu128), disk, RAM, installed libraries, Ollama models, network access to Hugging Face.
- Findings: no CUDA toolkit, C++ build tools, cmake, ninja, ffmpeg or colmap installed at the time, so 3DGS/NeRF could not be built yet.
- Hugging Face licences accepted: SD3.5-Medium, Gemma family (ShieldGemma). Llama family request pending.
- First toolchain install attempt failed silently (script argument bug, nothing installed). Second attempt started.
- Tooling used in these sessions: Claude Code desktop app (assistant for planning, checks and drafting), PowerShell, Python 3.11, winget.
- Repository created with folder structure for seven pairs.
