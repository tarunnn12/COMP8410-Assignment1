"""Architecture and workflow diagrams for the four model pairs (Section X.1 of the report).

Each diagram has one lane per system (A above, B below). Every box is a step of the pipeline that was run in this evaluation; the sources of the
architecture facts are printed under each diagram (papers, model cards, nerfstudio documentation) and the settings come from the notebooks.
No model is run. Output: pairs/pN_*/results/figures/architecture_pN.png
Run:  python tools/make_architecture_figures.py
"""
import pathlib, textwrap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = pathlib.Path(__file__).resolve().parent.parent
COL = {"A": ("#fdecea", "#7a1f1f"), "B": ("#e8f0f8", "#1f4e79")}

PAIRS = {
 "p2_3d_reconstruction": ("Pair 2: how the two methods turn posed photos into new views", [
   ("A", "System A: nerfacto (implicit radiance field)", [
     "Posed photos\n(camera poses\nfrom COLMAP)", "One ray per\ntraining pixel", "Proposal sampler:\n2 small density fields\nplace samples\nnear surfaces",
     "Main field:\nhash grid + small MLP\n(+ appearance\nembedding) gives\ndensity and colour", "Volume rendering:\ncomposite the samples\nalong the ray\ninto one pixel", "Loss against the\nphoto; update the\nfields and the\ncamera poses", "New view:\nrendered ray by ray\n(0.36 frames per\nsecond here)"]),
   ("B", "System B: splatfacto (explicit 3D Gaussians)", [
     "Posed photos +\nsparse points\nfrom COLMAP", "Initialise 3D\nGaussians (position,\nscale, rotation,\nopacity, colour)", "Project the Gaussians\nand sort them by\ndepth in each\nscreen tile",
     "Alpha-blend front\nto back to get\nthe pixel", "Loss against the\nphoto; update\nthe Gaussians", "Density control:\nsplit, clone and\nprune Gaussians", "New view:\nrasterised\n(270 to 300 frames\nper second here)"])],
   "Sources: A: nerfstudio documentation (nerfacto); Mildenhall et al. 2020; Müller et al. 2022; Tancik et al. 2023. B: Kerbl et al. 2023; Ye et al. 2025 (gsplat). Settings: notebooks of pair 2."),
 "p4_agents": ("Pair 4: the observe, decide, act loop of the two agents (at most 6 steps per episode)", [
   ("A", "System A: text agent (Qwen2.5-7B, DOM elements)", [
     "Web page\n(MiniWoB++ task,\nfixed seed)", "Observation:\nup to 50 DOM elements\n(ref, text, centre,\nsize)", "Prompt: rules +\ninstruction + last\nactions + element\nlist", "Qwen2.5-7B (4 bit,\nOllama) writes JSON:\nreason + action", "Check: valid action\ntype and reference?\n(invalid: step lost)", "Run the action in\nthe browser: click\nelement, type, key", "Reward, or next\nstep"]),
   ("B", "System B: vision agent (Qwen2.5-VL-7B, screenshot)", [
     "Web page\n(MiniWoB++ task,\nfixed seed)", "Observation:\nscreenshot 8 times\nupscaled\n(1280 x 1680 px)", "Prompt: rules +\ninstruction + last\nactions + image size", "Qwen2.5-VL (4 bit):\nViT, merger, LLM\nwrites JSON with\npixel x and y", "Check: valid action\ntype and coordinates?\n(invalid: step lost)", "Divide x and y by 8\nand click in the\nbrowser", "Reward, or next\nstep"])],
   "Sources: Week 8 lecture (agent as a policy, slides 11 to 17); the notebook of pair 4 (prompts, parser, settings); Bai et al. 2025 (Qwen2.5-VL architecture); Qwen Team 2024."),
 "p6_lmm_vqa": ("Pair 6: how the two vision-language models turn a photo and a question into an answer", [
   ("A", "System A: Qwen3-VL-8B-Instruct (4-bit GGUF, Ollama)", [
     "Photo and\nquestion", "Vision transformer\nencodes the photo\n(dynamic resolution,\nabout 1,100 prompt\ntokens)", "DeepStack: features\nfrom several vision\nlayers are fused", "Qwen3 8B decoder\nreads image tokens\nand question", "Answer, one word\n(16 tokens at most)", "Normalise: yes/no\nor VQA matching", "Score against the\ntrue answer"]),
   ("B", "System B: InternVL3-8B (4-bit NF4, transformers)", [
     "Photo and\nquestion", "Tile the photo:\n448 px tiles, at most\n6 plus a thumbnail", "InternViT-300M\nencodes each tile", "Pixel unshuffle\n(4 times fewer tokens)\nand MLP projector", "Qwen2.5-7B decoder\nreads image tokens\nand question (V2PE)", "Answer, one word\n(about 1,740 tokens\nin the prompt)", "Normalise and score\nagainst the truth"])],
   "Sources: Qwen3-VL-8B-Instruct model card (DeepStack, Interleaved-MRoPE); Zhu et al. 2025 and the InternVL3-8B model card; notebook of pair 6 (tiling cap, answer budget, scoring)."),
 "p7_safety": ("Pair 7: how the two filters decide whether a customer message is unsafe", [
   ("A", "System A: Llama Guard 3-1B (decision is a generated label)", [
     "Customer\nmessage", "Chat template with\nthe 13 hazard\ncategories S1 to S13", "Llama Guard 3-1B\n(Llama 3.2 1B pruned\nto 12 layers,\ndistilled)", "Generates the word\nsafe or unsafe plus\ncategory codes", "Decision = the label", "Unsafe: block or\nsend to human\nreview"]),
   ("B", "System B: ShieldGemma-2B (decision is a probability rule)", [
     "Customer\nmessage", "Four prompts: the\nmessage plus one\npolicy text each", "ShieldGemma-2B\n(Gemma 2 2B,\nfine-tuned)", "Yes and No logits\nbecome a probability\nper policy", "Decision rule of this\nevaluation: highest\nof the four >= 0.5", "Flagged: block or\nsend to human\nreview"])],
   "Sources: Llama Guard 3-1B model card (categories, pruning and distillation, output format); ShieldGemma-2B model card and Zeng et al. 2024 (prompt, Yes/No scoring); notebook of pair 7 (decision rules)."),
}

def lane(ax, y, kind, title, steps, width=100.0, h=19.0):
    face, edge = COL[kind]
    n = len(steps); gap = 3.2; bw = (width - gap * (n - 1)) / n
    ax.text(0, y + h + 2.2, title, fontsize=13.5, fontweight="bold", color=edge, va="bottom")
    for i, s in enumerate(steps):
        x = i * (bw + gap)
        ax.add_patch(FancyBboxPatch((x, y), bw, h, boxstyle="round,pad=0.25,rounding_size=1.4", fc=face, ec=edge, lw=1.6))
        ax.text(x + bw / 2, y + h / 2, s, ha="center", va="center", fontsize=8.5, color="#222222", linespacing=1.22)
        ax.text(x + 1.2, y + h - 1.0, str(i + 1), ha="left", va="top", fontsize=9.5, fontweight="bold", color=edge)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x + bw + 0.5, y + h / 2), (x + bw + gap - 0.5, y + h / 2), arrowstyle="-|>", mutation_scale=15, color=edge, lw=1.8))

for folder, (title, lanes, src) in PAIRS.items():
    fig, ax = plt.subplots(figsize=(16.5, 6.6)); ax.set_xlim(-1, 101); ax.set_ylim(-9, 62); ax.axis("off")
    ax.text(0, 61, title, fontsize=15.5, fontweight="bold", va="top")
    lane(ax, 32.0, *lanes[0]); lane(ax, 4.0, *lanes[1])
    ax.text(0, -1.0, "\n".join(textwrap.wrap(src, 175)), fontsize=8.8, color="#444444", va="top")
    out = ROOT / "pairs" / folder / "results" / "figures" / f"architecture_{folder.split('_')[0]}.png"
    out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig); print("wrote", out.relative_to(ROOT))
