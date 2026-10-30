"""Opacity check of the saved splatfacto checkpoints (27 October 2026).

Question: was the 3,000-iteration model stopped right after an opacity reset?
A reset caps every opacity at prune_opa * 2 (nerfstudio cull_alpha_thresh 0.1, so 0.2). If the checkpoint
was taken right after a reset, its opacities would all be at or below 0.2.
Run: python opacity_check.py   (environment p2_nerf; reads ../../ns_runs/*)
"""
import glob, pathlib, torch, importlib.util

root = pathlib.Path(__file__).resolve().parents[2] / "ns_runs"
for run in ["chair_video_splatfacto_1000", "chair_video_splatfacto_3000", "chair_video_splatfacto_7000", "chair_video_splatfacto",
            "poster_splatfacto_1000", "poster_splatfacto_3000", "poster_splatfacto_7000", "poster_splatfacto"]:
    f = glob.glob(str(root / run / "splatfacto" / "run" / "nerfstudio_models" / "*.ckpt"))
    if not f:
        print(run, "no checkpoint"); continue
    c = torch.load(f[0], map_location="cpu", weights_only=False)
    key = [k for k in c["pipeline"] if k.endswith("gauss_params.opacities")][0]
    o = torch.sigmoid(c["pipeline"][key].float()).flatten()
    print(f"{run}: step {c['step']}, gaussians {o.numel()}, mean opacity {o.mean():.3f}, median {o.median():.3f}, "
          f"share above 0.2: {(o > 0.2).float().mean():.3f}")

# the reset condition of the installed gsplat (strategy/default.py): "if step % self.reset_every == 0 & step > 0:"
# Python reads "a == 0 & step > 0" as the chained comparison (a == (0 & step)) and ((0 & step) > 0), which is always false.
reset_every = 3000
print("reset condition true for any step in 0..30000:", any((s % reset_every == 0 & s > 0) for s in range(0, 30001)))
spec = importlib.util.find_spec("gsplat")
print("gsplat file:", spec.origin if spec else "not found")
