
import sys, json, time, pathlib, torch
torch.cuda.set_per_process_memory_fraction(0.92)
_load = torch.load      # nerfstudio checkpoints hold config objects; torch >= 2.6 defaults to weights_only=True and refuses them (own checkpoints, trusted)
torch.load = lambda *a, **k: _load(*a, **{**k, 'weights_only': k.get('weights_only', False)})
from nerfstudio.utils.eval_utils import eval_setup
cfg_path, mode, n_views, out_json, img_dir = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), pathlib.Path(sys.argv[4]), pathlib.Path(sys.argv[5])
img_dir.mkdir(parents=True, exist_ok=True)
def use_plain_datamanager(config):
    # the parallel data manager spawns worker processes and pickles this whole script (including the loaded model) on Windows, which hangs;
    # evaluation only reads images, so the plain data manager gives the same views
    dm = config.pipeline.datamanager
    if type(dm).__name__ == "ParallelDataManagerConfig":
        from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
        config.pipeline.datamanager = VanillaDataManagerConfig(data=dm.data, dataparser=dm.dataparser, train_num_rays_per_batch=dm.train_num_rays_per_batch,
                                                                eval_num_rays_per_batch=dm.eval_num_rays_per_batch, camera_optimizer=dm.camera_optimizer)
    elif type(dm).__name__ == "VanillaDataManagerConfig":
        from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
        dm._target = VanillaDataManager
    return config
config, pipeline, ckpt, step = eval_setup(cfg_path, eval_num_rays_per_chunk=4096, test_mode=mode, update_config_callback=use_plain_datamanager)
pipeline.eval(); model = pipeline.model
dl = pipeline.datamanager.fixed_indices_eval_dataloader
has_loader = hasattr(dl, 'input_dataset')            # ray-based methods give a dataloader, splatfacto gives a list of (camera, batch)
total = len(dl.input_dataset) if has_loader else len(dl)
idx = sorted(set(round(i * (total - 1) / max(n_views - 1, 1)) for i in range(n_views))) if n_views > 1 else [0]
if has_loader: dl.image_indices = tuple(idx)
loader = dl if has_loader else [dl[i] for i in idx]
torch.cuda.reset_peak_memory_stats()
rows = []
with torch.no_grad():
    for k, (camera, batch) in enumerate(loader):
        torch.cuda.synchronize(); t = time.perf_counter()
        outputs = model.get_outputs_for_camera(camera=camera)
        torch.cuda.synchronize(); dt = time.perf_counter() - t
        metrics, images = model.get_image_metrics_and_images(outputs, batch)
        row = {"view": int(idx[k]), "render_s": dt, "fps": 1.0 / dt, "height": int(camera.height), "width": int(camera.width)}
        row.update({kk: float(v) for kk, v in metrics.items() if kk in ("psnr", "ssim", "lpips")})
        rows.append(row)
        if k < 3:
            import torchvision
            for name, im in images.items():
                torchvision.utils.save_image(im.permute(2, 0, 1).cpu(), str(img_dir / f"view{idx[k]}_{name}.png"))
info = {"checkpoint": str(ckpt), "step": int(step), "ckpt_mb": round(ckpt.stat().st_size / 2**20, 1) if ckpt.is_file() else None,
        "num_gaussians": int(model.means.shape[0]) if hasattr(model, "means") else None,
        "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "rows": rows}
out_json.write_text(json.dumps(info))
print("EVAL DONE", len(rows), "views")
