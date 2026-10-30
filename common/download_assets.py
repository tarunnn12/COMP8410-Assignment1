"""Download the model weights and datasets that the evaluation notebooks expect, to the folders named in the pair READMEs.

Usage (Windows, from the repository root):
    python common/download_assets.py --list                 show what would be downloaded
    python common/download_assets.py --only p6 p7           download assets for selected pairs
    python common/download_assets.py                        download everything (about 100 GB of models)

Access: Stable Diffusion 3.5 Medium, Gemma (ShieldGemma) and Llama (Llama Guard 3) are gated on Hugging Face. Accept their terms on the
model pages with the account whose token you use (`huggingface-cli login`), otherwise those downloads fail with a 401 or 403 error.
Local folders can be moved with the environment variables listed in each pair README.
"""
import argparse, json, pathlib, sys

MODELS = pathlib.Path("C:/models"); DATA = pathlib.Path("C:/data")

# (pair, repo id, repo type, local folder, allow_patterns or None, note)
ASSETS = [
    ("p6", "OpenGVLab/InternVL3-8B-hf", "model", MODELS / "InternVL3-8B-hf", None, "System B vision-language model"),
    ("p7", "meta-llama/Llama-Guard-3-1B", "model", MODELS / "Llama-Guard-3-1B", ["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "README.md"], "GATED: accept the Llama 3.2 licence first"),
    ("p7", "google/shieldgemma-2b", "model", MODELS / "shieldgemma-2b", ["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "README.md"], "GATED: accept the Gemma terms first"),
    ("p6", "lmms-lab/POPE", "dataset", DATA / "p6" / "POPE", ["Full/adversarial-*.parquet"], "POPE adversarial split (COCO images)"),
    ("p6", "lmms-lab/VQAv2", "dataset", DATA / "p6" / "VQAv2", ["data/validation-*.parquet"], "VQAv2 validation split (large)"),
    ("p7", "lmsys/toxic-chat", "dataset", DATA / "p7" / "toxic-chat", None, "ToxicChat (CC BY-NC 4.0, research use only)"),
    ("p7", "Paul/hatecheck", "dataset", DATA / "p7" / "hatecheck", None, "HateCheck (CC BY 4.0)"),
    ("p2", "pablovela5620/nerf-synthetic-mirror", "dataset", DATA / "nerf_synth", ["chair/*"], "NeRF-synthetic chair scene (optional second scene)"),
    ("p2", "nerfstudioteam/datasets", "dataset", DATA / "ns_sample", ["poster/*"], "nerfstudio sample capture (shows a folding chair)"),
]

def make_poster_subset():
    """The online copy of the poster capture holds 100 of the 226 frames listed in transforms.json; list only the existing ones."""
    root = DATA / "ns_sample" / "poster"; src = root / "transforms.json"
    if not src.exists(): return
    t = json.loads(src.read_text()); have = {p.name for p in (root / "images_4").glob("*")}
    t["frames"] = [f for f in t["frames"] if pathlib.Path(f["file_path"]).name in have]
    (root / "transforms_subset.json").write_text(json.dumps(t))
    print("transforms_subset.json written with", len(t["frames"]), "frames")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--list", action="store_true"); ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    todo = [x for x in ASSETS if not a.only or any(o in x[0].split() for o in a.only)]
    for pairs, repo, kind, folder, patterns, note in todo:
        print(f"{pairs:8s} {repo:48s} -> {folder}   {note}")
    if a.list: return
    from huggingface_hub import snapshot_download
    for pairs, repo, kind, folder, patterns, note in todo:
        print("downloading", repo, flush=True)
        try: snapshot_download(repo, repo_type=kind, local_dir=str(folder), allow_patterns=patterns)
        except Exception as e: print("  FAILED:", type(e).__name__, str(e)[:160]); continue
    if not a.only or "p2" in a.only: make_poster_subset()

if __name__ == "__main__":
    main()
