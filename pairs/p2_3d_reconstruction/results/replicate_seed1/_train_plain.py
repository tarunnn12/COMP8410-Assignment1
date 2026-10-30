
# nerfacto training with the plain data manager: the default parallel one stalled on this Windows laptop after a few hundred iterations
import os, sys, copy, pathlib, torch
torch.cuda.set_per_process_memory_fraction(0.92)
from nerfstudio.configs.method_configs import all_methods
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.scripts.train import main
if __name__ == "__main__":
    data, kind, iters, outdir, exp, ts = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
    ds, em = (int(sys.argv[7]), sys.argv[8]) if len(sys.argv) > 8 else (4, "fraction")
    cfg = copy.deepcopy(all_methods["nerfacto"]); old = cfg.pipeline.datamanager
    parser = BlenderDataParserConfig() if kind == "blender" else NerfstudioDataParserConfig(downscale_factor=ds, eval_mode=em)
    cfg.pipeline.datamanager = VanillaDataManagerConfig(dataparser=parser, train_num_rays_per_batch=old.train_num_rays_per_batch,
                                                        eval_num_rays_per_batch=old.eval_num_rays_per_batch, camera_optimizer=old.camera_optimizer)
    cfg.data = pathlib.Path(data)
    cfg.pipeline.model.implementation = "torch"
    cfg.pipeline.model.eval_num_rays_per_chunk = 4096       # the default (32768 rays per chunk) needs more than the 8 GB card holds when it renders the periodic eval image
    cfg.steps_per_eval_image = 10**9; cfg.steps_per_eval_batch = 10**9   # no in-training evaluation, the models are scored afterwards on fixed views
    cfg.machine.seed = int(os.environ["P2_SEED"]) if os.environ.get("P2_SEED") else cfg.machine.seed; cfg.max_num_iterations = iters; cfg.steps_per_save = iters; cfg.vis = "tensorboard"; cfg.viewer.quit_on_train_completion = True
    cfg.output_dir = pathlib.Path(outdir); cfg.experiment_name = exp; cfg.timestamp = ts
    main(cfg)
