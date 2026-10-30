import io, json, ollama, numpy as np
from PIL import Image, ImageDraw
OPT = {"temperature": 0, "seed": 0, "num_ctx": 4096, "num_predict": 80}
pts = [(0.25, 0.25), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.75, 0.75), (0.5, 0.85)]
sysmsg = 'You locate objects in images. Reply with one JSON object only: {"x":<int>,"y":<int>} giving the pixel coordinates (origin top-left, in the image shown) of the center of the red square.'
for scale in (5, 6, 8):
    W, H = 160 * scale, 210 * scale
    xs, ys, gx, gy = [], [], [], []
    for fx, fy in pts:
        im = Image.new("RGB", (W, H), (255, 255, 255)); d = ImageDraw.Draw(im)
        cx, cy, r = fx * W, fy * H, 10 * scale
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=(220, 0, 0))
        buf = io.BytesIO(); im.save(buf, format="PNG")
        out = ollama.chat(model="qwen2.5vl:7b", format="json", options=OPT, keep_alive="10m",
                          messages=[{"role": "system", "content": sysmsg}, {"role": "user", "content": f"Image size: {W}x{H} pixels. Where is the red square?", "images": [buf.getvalue()]}])
        try:
            o = json.loads(out["message"]["content"]); px, py = float(o["x"]), float(o["y"])
        except Exception:
            px = py = float("nan")
        xs.append(px); ys.append(py); gx.append(cx); gy.append(cy)
    xs, ys, gx, gy = map(np.array, (xs, ys, gx, gy))
    ok = ~np.isnan(xs)
    kx = np.polyfit(gx[ok], xs[ok], 1)[0] if ok.sum() > 2 else float("nan")
    ky = np.polyfit(gy[ok], ys[ok], 1)[0] if ok.sum() > 2 else float("nan")
    err = np.nanmean(np.hypot((xs - gx) / scale, (ys - gy) / scale))
    print(f"scale x{scale} ({W}x{H}): slope x={kx:.2f} y={ky:.2f} | mean error in page pixels = {err:.1f}")
    print("   truth :", [(round(a), round(b)) for a, b in zip(gx, gy)])
    print("   answer:", [(None if np.isnan(a) else round(a), None if np.isnan(b) else round(b)) for a, b in zip(xs, ys)])
