import time, torch, io, warnings, pathlib, glob
warnings.filterwarnings("ignore")
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
P = "C:/models/InternVL3-8B-hf"
proc = AutoProcessor.from_pretrained(P)
ip = proc.image_processor
print("image processor:", type(ip).__name__, "| max_patches:", getattr(ip, "max_patches", None), "| min_patches:", getattr(ip, "min_patches", None), "| crop_to_patches:", getattr(ip, "crop_to_patches", None), flush=True)
q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=["vision_tower", "multi_modal_projector", "lm_head"])
t = time.time()
model = AutoModelForImageTextToText.from_pretrained(P, quantization_config=q, dtype=torch.bfloat16, device_map="cuda").eval()
print(f"load {time.time()-t:.1f}s | allocated after load {torch.cuda.memory_allocated()/2**30:.2f} GB", flush=True)
img_path = sorted(glob.glob("<scratch>/p6_quick/data/images/pope_*.jpg"))[0]
img = Image.open(img_path).convert("RGB"); print("test image", img.size, flush=True)
msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "Is there a person in the image? Answer with Yes or No."}]}]
for tag, kw in (("default", {}), ("max_patches=6", {"max_patches": 6}), ("max_patches=4", {"max_patches": 4}), ("max_patches=2", {"max_patches": 2}), ("crop_to_patches=False", {"crop_to_patches": False})):
    if tag != "default":
        for k, v in kw.items(): setattr(ip, k, v)
    inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t = time.time()
    with torch.no_grad(): out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    torch.cuda.synchronize(); dt = time.time() - t
    ans = proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    print(f"{tag:24s} tiles={inputs['pixel_values'].shape[0]:2d} prompt_tokens={inputs['input_ids'].shape[1]:5d} | {dt:6.1f}s | peak alloc {torch.cuda.max_memory_allocated()/2**30:.2f} GB | answer '{ans}'", flush=True)
