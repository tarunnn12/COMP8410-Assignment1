import time, torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0))
import gsplat
print("gsplat", gsplat.__version__)
from gsplat import rasterization

dev = "cuda"
torch.manual_seed(0)
N = 2000
means = torch.randn(N, 3, device=dev) * 0.5
quats = torch.nn.functional.normalize(torch.randn(N, 4, device=dev), dim=-1)
scales = torch.rand(N, 3, device=dev) * 0.05 + 0.01
opacities = torch.rand(N, device=dev)
colors = torch.rand(N, 3, device=dev)
viewmats = torch.eye(4, device=dev)[None]
viewmats[0, 2, 3] = 3.0
W, H = 256, 256
K = torch.tensor([[200.0, 0, W / 2], [0, 200.0, H / 2], [0, 0, 1]], device=dev)[None]

t = time.time()
img, alpha, meta = rasterization(means, quats, scales, opacities, colors, viewmats, K, W, H)
torch.cuda.synchronize()
print("first call (includes JIT compile) s:", round(time.time() - t, 1))
print("render shape", tuple(img.shape), "mean", float(img.mean()))

means.requires_grad_(True)
img, _, _ = rasterization(means, quats, scales, opacities, colors, viewmats, K, W, H)
img.sum().backward()
torch.cuda.synchronize()
print("backward ok, grad norm", float(means.grad.norm()))
print("peak VRAM MB", round(torch.cuda.max_memory_allocated() / 2**20, 1))
print("GSPLAT SMOKE TEST PASSED")
