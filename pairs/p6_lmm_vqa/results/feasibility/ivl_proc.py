import warnings, glob; warnings.filterwarnings("ignore")
from PIL import Image
from transformers import AutoProcessor
proc = AutoProcessor.from_pretrained("C:/models/InternVL3-8B-hf"); ip = proc.image_processor
print(type(ip).__name__, "defaults:", {k: getattr(ip, k, None) for k in ("crop_to_patches", "min_patches", "max_patches", "size")})
for size in ((640, 480), (480, 640), (500, 375), (640, 427)):
    img = Image.new("RGB", size, (120, 100, 90)); row = []
    for n in (1, 2, 4, 6, 12):
        out = ip(images=img, crop_to_patches=True, max_patches=n, return_tensors="pt")
        row.append(f"max_patches={n}: {out['pixel_values'].shape[0]} tiles")
    out = ip(images=img, crop_to_patches=False, return_tensors="pt"); row.append(f"no crop: {out['pixel_values'].shape[0]} tile")
    print(size, " | ".join(row))
# how does the chat template path pick tiles for a plain call?
msgs = [{"role": "user", "content": [{"type": "image", "image": Image.new("RGB", (640, 480))}, {"type": "text", "text": "hi"}]}]
for kw in ({}, {"images_kwargs": {"max_patches": 4}}, {"images_kwargs": {"crop_to_patches": False}}):
    try:
        x = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", **kw)
        print("apply_chat_template", kw, "->", x["pixel_values"].shape[0], "tiles,", x["input_ids"].shape[1], "tokens")
    except Exception as e:
        print("apply_chat_template", kw, "failed:", str(e)[:120])
