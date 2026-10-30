"""Assemble the submission folder `COMP8410 A1 <StudentID>` from the repository, the report, the presentation and the video.

Usage:  python tools/make_submission.py --id 49030000 --out "C:/path/to/output"
        [--report path.pdf] [--slides path.pptx] [--video path_or_link.txt] [--git-log] [--full] [--dry-run] [--zip]

Layout produced (names follow Section 4.1 of the brief):
    COMP8410 A1 <id>/
        <id> Report.pdf
        <id> Presentation.pptx
        <id> Video.<ext>            (or <id> Video LINK.txt when a link is given)
        <id> Evaluation/            README.md, pairs/, common/, tools/, docs/, env/, SLIM_PACKAGE_NOTE.txt

Two sizes:
  slim (default)  the code, notebooks, READMEs, executed notebooks, CSV/JSON/JSONL results, figures and logs. Left out because they are large
                  and can be rebuilt or downloaded: the COCO and VQAv2 photos of pair 6 (`pairs/p6_lmm_vqa/data*/images`, rebuilt by the notebook
                  from the files that `common/download_assets.py` fetches), the two chair videos of pair 2 (`pairs/p2_3d_reconstruction/data/*.mp4`),
                  the rendered test views (`renders/` folders), and the TensorBoard logs of the nerfstudio runs (`events.out.tfevents*`).
  --full          the same as slim plus those folders and files.
Always left out: Git data, notebook checkpoints, Python caches, model checkpoints of pair 2 (`ns_runs/**/nerfstudio_models`, about 1.6 GB), the copy of the
MiniWoB++ pages of pair 4 (`data/miniwob_html_300s`, third-party code that the notebook rebuilds from the installed package), `demo_runs`, and saved prompt embeddings (`_embeds_*.pt`).
--git-log   writes `docs/git_log.txt` (the commit history, read with `git log`) into the Evaluation folder, because the Git data itself is not copied.
--dry-run   prints what would be copied and its size, and writes nothing.
"""
import argparse, pathlib, shutil, subprocess, sys, zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
SKIP_DIR = {".git", ".ipynb_checkpoints", "__pycache__", "nerfstudio_models", "edge_profile", "miniwob_html_300s", "demo_runs"}
SKIP_SUFFIX = {".pt", ".pyc"}
SLIM_DIR = {"renders"}                                  # rendered test views of pair 2 (figures/ keeps the side-by-side images)
SLIM_FILE_PREFIX = ("events.out.tfevents",)             # TensorBoard logs of the nerfstudio runs
SLIM_VIDEO_SUFFIX = {".mp4", ".mov", ".mkv", ".avi", ".webm"}

def is_p6_images(p):
    parts = p.parts
    return p.name == "images" and "p6_lmm_vqa" in parts and any(x.startswith("data") for x in parts)

def make_ignore(slim):
    def ignore(d, names):
        out = []
        for n in names:
            p = pathlib.Path(d) / n
            if n in SKIP_DIR or (p.is_file() and p.suffix in SKIP_SUFFIX): out.append(n); continue
            if slim:
                if p.is_dir() and (n in SLIM_DIR or is_p6_images(p)): out.append(n)
                elif p.is_file() and (n.startswith(SLIM_FILE_PREFIX) or (p.suffix.lower() in SLIM_VIDEO_SUFFIX and "p2_3d_reconstruction" in p.parts and "data" in p.parts)): out.append(n)
        return out
    return ignore

def walk_size(slim):
    """Size and file count that the copy would have (no files are written)."""
    ig = make_ignore(slim); total = n = 0
    def rec(d):
        nonlocal total, n
        names = [x.name for x in d.iterdir()]; skip = set(ig(str(d), names))
        for x in d.iterdir():
            if x.name in skip: continue
            if x.is_dir(): rec(x)
            else: total += x.stat().st_size; n += 1
    for part in ("pairs", "common", "tools", "docs", "env"): rec(ROOT / part)
    for f in ("README.md",): total += (ROOT / f).stat().st_size; n += 1
    return total / 2**20, n

def size_mb(path):
    return sum(f.stat().st_size for f in pathlib.Path(path).rglob("*") if f.is_file()) / 2**20

NOTE = """Slim package (what is not in this folder)

This folder holds the code, notebooks, READMEs, executed notebooks, result tables (CSV, JSON, JSONL), figures and logs of the four pairs.
To keep the upload small, these items were left out. They are in the shared Drive folder named in the report, and the notebooks can rebuild the data:
- pairs/p6_lmm_vqa/data/images and data_seed1/images: the COCO and VQAv2 photos. python common/download_assets.py --only p6 downloads the source files; the notebook rebuilds the images and the manifest.
- pairs/p2_3d_reconstruction/data/chair_orbit_low.mp4 and chair_orbit_high.mp4: my own phone videos of the chair (the input of src/prepare_chair_video.py).
- renders/ folders of pair 2 (rendered test views); the side-by-side figures are kept in figures/.
- TensorBoard logs (events.out.tfevents*) of the nerfstudio runs; the training logs (train_*.log) and the run configurations (config.yml) are kept.
- Model checkpoints of pair 2 (nerfstudio_models, about 1.6 GB) and the third-party MiniWoB++ page copy of pair 4, which the notebook rebuilds.
"""

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--id", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--report"); ap.add_argument("--slides"); ap.add_argument("--video"); ap.add_argument("--zip", action="store_true")
    ap.add_argument("--full", action="store_true", help="also copy the photos, chair videos, renders and TensorBoard logs")
    ap.add_argument("--git-log", action="store_true", help="write docs/git_log.txt from `git log`")
    ap.add_argument("--dry-run", action="store_true", help="only print the size of the copy")
    a = ap.parse_args(); slim = not a.full
    if a.dry_run:
        mb, n = walk_size(slim); print(f"{'slim' if slim else 'full'} Evaluation folder: {n} files, {mb:.1f} MB (before compression)"); return
    top = pathlib.Path(a.out) / f"COMP8410 A1 {a.id}"
    if top.exists(): sys.exit(f"{top} exists already; remove it or choose another --out")
    ev = top / f"{a.id} Evaluation"; ev.mkdir(parents=True)
    for part in ("pairs", "common", "tools", "docs", "env"):
        shutil.copytree(ROOT / part, ev / part, ignore=make_ignore(slim))
    shutil.copy2(ROOT / "README.md", ev / "README.md")
    if slim: (ev / "SLIM_PACKAGE_NOTE.txt").write_text(NOTE, encoding="utf-8")
    if a.git_log:
        log = subprocess.run(["git", "-C", str(ROOT), "log", "--date=short", "--pretty=format:%h | %an | %ad | %s"], capture_output=True, text=True)
        if log.returncode == 0: (ev / "docs" / "git_log.txt").write_text(log.stdout + "\n", encoding="utf-8")
        else: print("git log failed:", log.stderr.strip())
    if a.report: shutil.copy2(a.report, top / f"{a.id} Report.pdf")
    if a.slides: shutil.copy2(a.slides, top / f"{a.id} Presentation{pathlib.Path(a.slides).suffix}")
    if a.video:
        v = pathlib.Path(a.video)
        if v.suffix.lower() in SLIM_VIDEO_SUFFIX: shutil.copy2(v, top / f"{a.id} Video{v.suffix}")
        else: shutil.copy2(v, top / f"{a.id} Video LINK.txt")
    print(f"built {top} ({size_mb(top):.0f} MB)")
    for p in sorted(top.iterdir()): print(f"  {p.name:45s} {size_mb(p) if p.is_dir() else p.stat().st_size / 2**20:8.1f} MB")
    if a.zip:
        z = pathlib.Path(a.out) / f"COMP8410 A1 {a.id}.zip"
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in top.rglob("*"):
                if f.is_file(): zf.write(f, f.relative_to(top.parent))
        print(f"zip written: {z} ({z.stat().st_size / 2**20:.0f} MB)")

if __name__ == "__main__":
    main()
