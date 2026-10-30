"""Open a trained nerfstudio model in the interactive viewer (browser), with the same fixes that the evaluation worker uses on this laptop.
Usage: python viewer_run.py --load-config <config.yml> --viewer.websocket-port <port>"""
import sys, torch
torch.cuda.set_per_process_memory_fraction(0.92)                       # an overflow stops with an error instead of spilling into shared memory
_load = torch.load
torch.load = lambda *a, **k: _load(*a, **{**k, "weights_only": k.get("weights_only", False)})   # own checkpoints, trusted
import nerfstudio.scripts.viewer.run_viewer as rv
_setup = rv.eval_setup
def use_plain_datamanager(config):
    dm = config.pipeline.datamanager
    if type(dm).__name__ == "ParallelDataManagerConfig":
        from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
        config.pipeline.datamanager = VanillaDataManagerConfig(data=dm.data, dataparser=dm.dataparser, train_num_rays_per_batch=dm.train_num_rays_per_batch,
                                                                eval_num_rays_per_batch=dm.eval_num_rays_per_batch, camera_optimizer=dm.camera_optimizer)
    elif type(dm).__name__ == "VanillaDataManagerConfig":
        from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
        dm._target = VanillaDataManager
    return config
rv.eval_setup = lambda path, eval_num_rays_per_chunk=None, test_mode="inference", update_config_callback=None: _setup(path, eval_num_rays_per_chunk=4096, test_mode=test_mode, update_config_callback=use_plain_datamanager)
if __name__ == "__main__":
    rv.entrypoint()
