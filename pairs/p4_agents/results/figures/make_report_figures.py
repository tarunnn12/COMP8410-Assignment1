"""Figures for Section 4 of the report (pair 4, web agents). No model is run: everything is drawn from the saved
episode logs (results/episodes.jsonl and results/replicate_seed1/episodes.jsonl) and from re-rendering the MiniWoB++
pages with the same environment seeds and the same observation code as the evaluation notebook (cells 1, 5 and 7 are
executed from src/p4_agents_eval.ipynb, so the screenshot and the element list are exactly what the agents received).
Run from any folder with the environment p4_agents (Chrome and selenium needed for the page renders):
    python results/figures/make_report_figures.py
"""
import io, json, os, pathlib, textwrap
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent     # the notebook cells below overwrite HERE
P4 = SCRIPT_DIR.parents[1]
os.chdir(P4 / "src")
nb = json.load(open("p4_agents_eval.ipynb", encoding="utf-8"))
for i in (1, 5, 7):                                  # configuration, task pages, observation functions
    exec("".join(nb["cells"][i]["source"]), globals())
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = SCRIPT_DIR
R = P4 / "results"
eps1 = [json.loads(l) for l in open(R / "episodes.jsonl", encoding="utf-8")]
eps2 = [json.loads(l) for l in open(R / "replicate_seed1" / "episodes.jsonl", encoding="utf-8")]
allep = eps1 + eps2
NAME = {"random": "Random", "text_qwen2.5-7b": "Text agent (A)", "vlm_qwen2.5vl-7b": "Vision agent (B)"}
COL = {"random": "#9aa0a6", "text_qwen2.5-7b": "#2b6cb0", "vlm_qwen2.5vl-7b": "#c05621"}
TEXT, VLM = "text_qwen2.5-7b", "vlm_qwen2.5vl-7b"
def font(name, size):
    try: return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)
    except Exception: return ImageFont.load_default()
def wil(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return c - h, c + h
def shot(task, seed):
    env = make_env(task); obs, _ = env.reset(seed=seed); env.close(); return obs
def content_bottom(im, margin=40):
    a = np.asarray(im.convert("L")); rows = np.where((a < 245).any(axis=1))[0]
    return min(im.height, int(rows.max()) + margin)

# 1. what each agent sees (click-checkboxes, seed 0)
o = shot("click-checkboxes", 0)
img = Image.fromarray(o["screenshot"]).resize((160 * SCALE, 210 * SCALE), Image.NEAREST).convert("RGB")
img = img.crop((0, 0, img.width, content_bottom(img, 60)))
f1, f2 = font("arial.ttf", 38), font("consola.ttf", 27)
lines = []
for line in ("Instruction: " + o["utterance"] + "\nDOM elements:\n" + dom_text(o)).split("\n"):
    lines += textwrap.wrap(line, 62) or [""]
H = max(img.height, 80 + 36 * len(lines)) + 90
canvas = Image.new("RGB", (img.width + 40 + 1250, H), "white"); d = ImageDraw.Draw(canvas)
d.text((10, 10), "Vision agent (B) sees the screenshot (1280 x 1680 px, cropped here)", fill="black", font=f1)
canvas.paste(img, (10, 70)); d.rectangle((10, 70, 10 + img.width, 70 + img.height), outline=(120, 120, 120), width=3)
x0 = img.width + 60
d.text((x0, 10), "Text agent (A) sees a list of elements", fill="black", font=f1)
for k, line in enumerate(lines): d.text((x0, 75 + 36 * k), line, fill=(30, 30, 30), font=f2)
canvas.save(OUT / "p4_what_agents_see.png")

# 2. click traces of the vision agent on re-rendered pages (same seed, same layout)
def trace(task, seed):
    e = next(x for x in eps1 if x["agent"] == VLM and x["phase"] == "main" and x["task"] == task and x["seed"] == seed)
    o = shot(task, seed); im = Image.fromarray(o["screenshot"]).resize((160 * SCALE, 210 * SCALE), Image.NEAREST).convert("RGB")
    bottom = content_bottom(im, 50); d = ImageDraw.Draw(im); n = 0
    for s in e["step_log"]:
        a = s.get("action")
        if a and a.get("type") == "click_coords":
            n += 1; x, y = a["x"], a["y"]
            d.ellipse((x - 26, y - 26, x + 26, y + 26), outline=(220, 0, 0), width=9); d.text((x + 34, y - 30), str(n), fill=(220, 0, 0), font=font("arialbd.ttf", 64))
    return im, bottom, o["utterance"], e
panels = [trace("click-checkboxes", 0), trace("click-option", 0), trace("click-checkboxes", 2)]
hb = max(p[1] for p in panels)
fig, axs = plt.subplots(1, 3, figsize=(11, 4.3))
# the notes are checked against the logs: each click was hit-tested against the element boxes of the page
notes = {("click-checkboxes", 0): "success: click 1 on the HF2 label, click 2 on Submit",
         ("click-option", 0): "failed: the click hit Submit before AU was chosen",
         ("click-checkboxes", 2): "step limit: clicks 1 and 2 on the right boxes, 4 invalid\nactions in between, Submit never clicked"}
for ax, (im, _, utt, e) in zip(axs, panels):
    ax.imshow(im.crop((0, 0, im.width, hb))); ax.axis("off")
    ax.set_title(f"{e['task']}, seed {e['seed']}\n{notes[(e['task'], e['seed'])]}", fontsize=8.5)
fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(OUT / "p4_click_traces.png", dpi=160); plt.close(fig)

# 3. injection example (enter-text, seed 0): what each agent was shown
oi = shot("enter-text", 0)
png, size = vlm_image(oi, injection=True); imi = Image.open(io.BytesIO(png)).convert("RGB"); imi = imi.crop((0, 0, imi.width, content_bottom(imi, 50)))
txt = ["Instruction: " + oi["utterance"], "DOM elements:", '[0] <div> "' + PAYLOAD + '"'] + dom_text(oi).split("\n")[:5]
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.7), gridspec_kw={"width_ratios": [1, 1.5]})
axs[0].imshow(imi); axs[0].axis("off"); axs[0].set_title("Vision agent (B): page with the injected banner", fontsize=9)
axs[1].axis("off"); axs[1].set_title("Text agent (A): element list with the injected line", fontsize=9)
yy = 0.97
for line in txt:
    for sub in textwrap.wrap(line, 66) or [""]:
        axs[1].text(0, yy, sub, family="monospace", fontsize=7.8, va="top", transform=axs[1].transAxes); yy -= 0.075
fig.tight_layout(); fig.savefig(OUT / "p4_injection_example.png", dpi=160); plt.close(fig)

# 4. outcome breakdown, both runs pooled (200 episodes per agent)
fig, ax = plt.subplots(figsize=(7.2, 2.9))
order = [TEXT, VLM, "random"]; cats = [("success", "#2f855a", "Success"), ("failed", "#c53030", "Wrong final action"), ("max_steps", "#d69e2e", "Step limit reached")]
for y, a in enumerate(order[::-1]):
    g = [e for e in allep if e["agent"] == a and e["phase"] == "main"]; n = len(g); left = 0
    for key, c, lab in cats:
        k = sum(e["end_reason"] == key for e in g)
        ax.barh(y, k / n * 100, left=left, color=c, label=lab if y == 0 else None)
        if k: ax.text(left + k / n * 50, y, f"{k}", ha="center", va="center", color="white", fontsize=9)
        left += k / n * 100
ax.set_yticks(range(3)); ax.set_yticklabels([NAME[a] for a in order[::-1]]); ax.set_xlabel("Share of the 200 episodes (%); numbers are episode counts"); ax.set_xlim(0, 100)
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22), frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(OUT / "p4_outcomes_pooled.png", dpi=170); plt.close(fig)

# 5. prompt injection followed, both runs pooled (60 episodes per agent)
fig, ax = plt.subplots(figsize=(4.8, 3.0))
for x, a in enumerate([TEXT, VLM]):
    g = [e for e in allep if e["agent"] == a and e["phase"] == "injection"]; k = sum(e["attack_followed"] for e in g); n = len(g); lo, hi = wil(k, n)
    ax.bar(x, k / n * 100, color=COL[a], width=0.5); ax.errorbar(x, k / n * 100, yerr=[[k / n * 100 - lo * 100], [hi * 100 - k / n * 100]], color="black", capsize=4)
    ax.text(x + 0.3, k / n * 100, f"{k} of {n}", ha="left", va="center", fontsize=10)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Text agent (A)", "Vision agent (B)"]); ax.set_ylabel("Obeyed the injected text (%)"); ax.set_ylim(0, 75); ax.set_xlim(-0.5, 1.9)
fig.tight_layout(); fig.savefig(OUT / "p4_injection_pooled.png", dpi=170); plt.close(fig)

# 6. model time per episode, both runs pooled
fig, ax = plt.subplots(figsize=(5.4, 3.2)); rng = np.random.default_rng(0)
data = [[e["llm_s"] for e in allep if e["agent"] == a and e["phase"] == "main"] for a in (TEXT, VLM)]
bp = ax.boxplot(data, widths=0.45, showfliers=False, patch_artist=True, medianprops=dict(color="black"))
for patch, a in zip(bp["boxes"], (TEXT, VLM)): patch.set_facecolor(COL[a]); patch.set_alpha(0.35)
for i, (v, a) in enumerate(zip(data, (TEXT, VLM)), start=1):
    ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=7, color=COL[a], alpha=0.6, zorder=3)
    ax.scatter([i], [np.mean(v)], marker="D", color="black", s=28, zorder=4); ax.text(i + 0.27, np.mean(v), f"mean {np.mean(v):.1f} s", va="center", fontsize=8.5)
ax.set_xticks([1, 2]); ax.set_xticklabels(["Text agent (A)", "Vision agent (B)"]); ax.set_ylabel("Model time per episode (s)"); ax.set_xlim(0.4, 2.9)
fig.tight_layout(); fig.savefig(OUT / "p4_latency_pooled.png", dpi=170); plt.close(fig)
print("figures written to", OUT)
