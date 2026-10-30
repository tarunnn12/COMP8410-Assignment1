import time, torch, glob, subprocess, warnings
warnings.filterwarnings("ignore")
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
imgs = sorted(glob.glob("<scratch>/p6_quick4/data/images/pope_*.jpg"))[:3]
def smi(q): return subprocess.check_output(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"], text=True).strip()
P = "C:/models/InternVL3-8B-hf"; proc = AutoProcessor.from_pretrained(P)
q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=["vision_tower", "multi_modal_projector", "lm_head"])
model = AutoModelForImageTextToText.from_pretrained(P, quantization_config=q, dtype=torch.bfloat16, device_map="cuda").eval()
def run(tag):
    ts = []
    for p in imgs:
        img = Image.open(p).convert("RGB"); msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "Is there a chair in the image? Answer with Yes or No."}]}]
        x = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", images_kwargs={"max_patches": 6}).to(model.device, dtype=torch.bfloat16)
        torch.cuda.synchronize(); t = time.time()
        with torch.no_grad(): model.generate(**x, max_new_tokens=16, do_sample=False)
        torch.cuda.synchronize(); ts.append(time.time() - t)
    print(f"{tag:22s} latencies {[round(v,1) for v in ts]} | smi used {smi('memory.used')} MiB", flush=True)
run("no ballast")
ballast = []
for mb in (200, 400, 600):
    ballast.append(torch.empty(200 * 2**20, dtype=torch.uint8, device="cuda")); ballast[-1].fill_(1)
    run(f"ballast +{mb} MiB")
