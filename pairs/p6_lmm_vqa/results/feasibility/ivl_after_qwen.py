import time, torch, glob, subprocess, warnings, ollama
warnings.filterwarnings("ignore")
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
imgs = sorted(glob.glob("<scratch>/p6_quick4/data/images/pope_*.jpg"))[:4]
def smi(q): return subprocess.check_output(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"], text=True).strip()
print("start: GPU mem", smi("memory.used"), "MiB", flush=True)
# 1. Qwen through Ollama (as in the notebook)
for p in imgs[:2]:
    t = time.time(); r = ollama.chat(model="qwen2.5vl:7b", messages=[{"role": "user", "content": "Is there a chair in the image? Answer with Yes or No.", "images": [open(p, "rb").read()]}], options={"temperature": 0, "num_predict": 16}, keep_alive="30m")
    print(f"qwen {time.time()-t:.1f}s -> {r['message']['content']!r} | GPU mem {smi('memory.used')} MiB", flush=True)
# 2. unload exactly like the notebook
ollama.generate(model="qwen2.5vl:7b", prompt="", keep_alive=0)
for i in range(90):
    if not ollama.ps().models: break
    time.sleep(1)
print(f"ollama unloaded after {i}s | GPU mem {smi('memory.used')} MiB", flush=True)
for i in range(120):
    if float(smi("memory.used").split("\n")[0]) < 2000: break
    time.sleep(1)
print(f"GPU below 2000 MiB after {i}s | mem {smi('memory.used')} MiB", flush=True)
print("GPU processes:", subprocess.run(["nvidia-smi", "pmon", "-c", "1"], capture_output=True, text=True).stdout.strip().splitlines()[-4:], flush=True)
# 3. InternVL
P = "C:/models/InternVL3-8B-hf"; proc = AutoProcessor.from_pretrained(P)
q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=["vision_tower", "multi_modal_projector", "lm_head"])
t = time.time(); model = AutoModelForImageTextToText.from_pretrained(P, quantization_config=q, dtype=torch.bfloat16, device_map="cuda").eval()
print(f"internvl load {time.time()-t:.1f}s | GPU mem {smi('memory.used')} MiB", flush=True)
for p in imgs:
    img = Image.open(p).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "Is there a chair in the image? Answer with Yes or No."}]}]
    x = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", images_kwargs={"max_patches": 6}).to(model.device, dtype=torch.bfloat16)
    torch.cuda.synchronize(); t = time.time()
    with torch.no_grad(): out = model.generate(**x, max_new_tokens=16, do_sample=False)
    torch.cuda.synchronize(); print(f"internvl {time.time()-t:.1f}s tokens={x['input_ids'].shape[1]} | GPU mem {smi('memory.used')} MiB | util {smi('utilization.gpu')}%", flush=True)
