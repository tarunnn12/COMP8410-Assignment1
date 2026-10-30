import time, sys
import numpy as np
import gymnasium as gym
import miniwob
from miniwob.action import ActionTypes

env = gym.make("miniwob/click-test-2-v1")
obs, info = env.reset(seed=0)
print("utterance:", obs["utterance"])
doms = obs["dom_elements"]
print("n dom elements:", len(doms), "| keys:", list(doms[0].keys()))
print("sample elements:", [(d["ref"], d["tag"], d["text"]) for d in doms[:8]])
shot = obs["screenshot"]
print("screenshot shape:", shot.shape)

# rule-based baseline from the lecture: find element whose text matches the target word and click it
word = obs["utterance"].replace("Click button", "").strip(" .").upper()
ref = next((d["ref"] for d in doms if d["text"].strip().upper() == word), None)
print("target word:", word, "| ref:", ref)
action = env.unwrapped.create_action(ActionTypes.CLICK_ELEMENT, ref=ref)
obs, reward, terminated, truncated, info = env.step(action)
print("rule-based step: reward", reward, "terminated", terminated)

obs, info = env.reset(seed=1)
shot = obs["screenshot"]
try:
    from PIL import Image
    Image.fromarray(shot).save(sys.argv[1])
except Exception as e:
    print("save shot failed", e)
env.close()

import ollama
for model in ("qwen2.5:7b", "qwen2.5vl:7b"):
    t = time.time()
    msg = {"role": "user", "content": "Reply with the single word OK."}
    if "vl" in model:
        msg = {"role": "user", "content": "Describe this image in one short sentence.", "images": [sys.argv[1]]}
    r = ollama.chat(model=model, messages=[msg], options={"temperature": 0, "seed": 0})
    print(model, "->", r["message"]["content"][:120].replace("\n", " "), "| s:", round(time.time() - t, 1))
print("P4 SMOKE TEST DONE")
