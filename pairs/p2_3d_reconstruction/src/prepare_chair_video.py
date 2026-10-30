"""Prepare the author's own chair capture for pair 2: frames, camera poses (COLMAP) and a nerfstudio transforms.json.

Input: two phone videos of the same chair, a low circle (training views) and a higher circle looking down (test views):
    data/chair_orbit_low.mp4   -> frames named train_0001.jpg ...
    data/chair_orbit_high.mp4  -> frames named eval_0001.jpg ...
Both videos are registered in one COLMAP model, so the test views come from a different camera height and path than the training views.

Usage (Windows, from pairs/p2_3d_reconstruction/src; needs ffmpeg and colmap on PATH, and the p2_nerf environment for the conversion step):
    python prepare_chair_video.py [--out C:/data/chair_video] [--fps 2.5]

Steps and settings (as used for the reported run):
  1. ffmpeg: 2.5 frames per second, 960 x 540, JPEG quality 2.
  2. Frames with a variance of the Laplacian below 45 are dropped (motion blur; 6 of 188 frames).
  3. COLMAP 4.2.1 on the CPU: feature extraction (single OPENCV camera), exhaustive matching, incremental mapping.
  4. nerfstudio's colmap_to_json writes transforms.json (image names are kept, which the filename split needs); quarter and half size copies of the images are written.
Result of the reported run: 182 of 182 frames registered in one model, 16,914 points, mean track length 11.0, mean reprojection error 0.58 px.
"""
import argparse, pathlib, subprocess, shutil, json, sys
import cv2, pandas as pd
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parent / "data"

def run(cmd):
    print(">", " ".join(map(str, cmd)), flush=True); subprocess.run(list(map(str, cmd)), check=True)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="C:/data/chair_video"); ap.add_argument("--fps", default="2.5"); ap.add_argument("--blur", type=float, default=45.0)
    a = ap.parse_args(); out = pathlib.Path(a.out); allf = out / "frames_all"; fr = out / "frames"; ns = out / "ns"
    for d in (allf, fr, ns / "images", ns / "colmap" / "sparse"): d.mkdir(parents=True, exist_ok=True)
    for name, prefix in (("chair_orbit_low.mp4", "train"), ("chair_orbit_high.mp4", "eval")):
        run(["ffmpeg", "-v", "error", "-y", "-i", DATA / name, "-vf", f"fps={a.fps},scale=960:540", "-qscale:v", 2, allf / f"{prefix}_%04d.jpg"])
    rows = [(p.name, cv2.Laplacian(cv2.imread(str(p), 0), cv2.CV_64F).var()) for p in sorted(allf.glob("*.jpg"))]
    d = pd.DataFrame(rows, columns=["frame", "laplacian_var"]); d.to_csv(out / "frame_sharpness.csv", index=False)
    keep = d[d.laplacian_var >= a.blur]; print("frames kept:", len(keep), "of", len(d))
    for f in keep.frame:
        shutil.copy(allf / f, fr / f); shutil.copy(allf / f, ns / "images" / f)
    db = ns / "colmap" / "database.db"; sparse = ns / "colmap" / "sparse"
    run(["colmap", "feature_extractor", "--database_path", db, "--image_path", ns / "images", "--ImageReader.single_camera", 1, "--ImageReader.camera_model", "OPENCV", "--FeatureExtraction.use_gpu", 0])
    run(["colmap", "exhaustive_matcher", "--database_path", db, "--FeatureMatching.use_gpu", 0])
    run(["colmap", "mapper", "--database_path", db, "--image_path", ns / "images", "--output_path", sparse])
    # the mapper may write several models; the one with the most registered images is used
    best = max((m for m in sparse.iterdir() if m.is_dir()), key=lambda m: (m / "images.bin").stat().st_size)
    print("model used:", best.name)
    from nerfstudio.process_data import colmap_utils          # needs the p2_nerf environment
    n = colmap_utils.colmap_to_json(recon_dir=best, output_dir=ns); print("frames in transforms.json:", n)
    for k in (2, 4):
        o = ns / f"images_{k}"; o.mkdir(exist_ok=True)
        for p in sorted((ns / "images").glob("*.jpg")):
            im = Image.open(p); im.resize((im.width // k, im.height // k), Image.LANCZOS).save(o / p.name, quality=95)
    t = json.loads((ns / "transforms.json").read_text())
    print("train frames:", sum("train" in f["file_path"] for f in t["frames"]), "| eval frames:", sum("eval" in f["file_path"] for f in t["frames"]))

if __name__ == "__main__":
    main()
