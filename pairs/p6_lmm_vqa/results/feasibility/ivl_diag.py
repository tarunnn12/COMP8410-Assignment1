import time, torch, glob, threading, subprocess, warnings, sys
warnings.filterwarnings("ignore")
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
P = "C:/models/InternVL3-8B-hf"
proc = AutoProcessor.from_pretrained(P)
q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=["vision_tower", "multi_modal_projector", "lm_head"])
model = AutoModelForImageTextToText.from_pretrained(P, quantization_config=q, dtype=torch.bfloat16, device_map="cuda").eval()
print("loaded; allocated GB:", round(torch.cuda.memory_allocated()/2**30, 2), flush=True)
imgs = sorted(glob.glob("<scratch>/p6_quick4/data/images/pope_*.jpg"))[:6]
def ask(path, tiles=6):
    img = Image.open(path).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "Is there a chair in the image? Answer with Yes or No."}]}]
    x = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", images_kwargs={"max_patches": tiles}).to(model.device, dtype=torch.bfloat16)
    torch.cuda.synchronize(); t = time.time()
    with torch.no_grad(): out = model.generate(**x, max_new_tokens=16, do_sample=False)
    torch.cuda.synchronize(); dt = time.time() - t
    return dt, x["input_ids"].shape[1], x["pixel_values"].shape[0], proc.decode(out[0][x["input_ids"].shape[1]:], skip_special_tokens=True).strip()
def smi_mem(): return subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip()
print("--- A: plain calls (no background sampler)", flush=True)
for p in imgs[:3]:
    dt, n, tiles, a = ask(p); print(f"{dt:6.1f}s tokens={n} tiles={tiles} ans={a} | peak alloc {torch.cuda.max_memory_allocated()/2**30:.2f} GB | smi used {smi_mem()} MiB", flush=True)
print("--- B: calls with nvidia-smi polled every second in a thread (what the notebook does)", flush=True)
run = True
def poll():
    while run:
        try: subprocess.check_output(["nvidia-smi", "--query-gpu=power.draw,utilization.gpu,memory.used", "--format=csv,noheader,nounits"], text=True)
        except Exception: pass
        time.sleep(1.0)
th = threading.Thread(target=poll, daemon=True); th.start()
for p in imgs[3:6]:
    dt, n, tiles, a = ask(p); print(f"{dt:6.1f}s tokens={n} tiles={tiles} ans={a}", flush=True)
run = False
