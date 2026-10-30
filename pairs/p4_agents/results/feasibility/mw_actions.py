import gymnasium as gym, miniwob, numpy as np, warnings, re
warnings.filterwarnings("ignore")
from miniwob.action import ActionTypes
base = "file:///C:/data/miniwob_html_300s/miniwob/"
env = gym.make("miniwob/enter-text-v1", base_url=base)
cfg = env.unwrapped.action_space_config
print("action types:", [a.value for a in cfg.action_types])
print("allowed keys (first 8):", list(cfg.allowed_keys)[:8], "... n =", len(cfg.allowed_keys))
obs, _ = env.reset(seed=0)
print("utterance:", obs["utterance"])
for e in obs["dom_elements"]:
    if e["width"] > 0: print(e["ref"], e["tag"], repr(e["text"][:20]), repr(e["value"]), e["classes"][:20], e["flags"], (e["left"], e["top"], e["width"], e["height"]))
word = re.search(r'"([^"]+)"', obs["utterance"]).group(1)
inp = next(e["ref"] for e in obs["dom_elements"] if e["tag"] == "input_text")
sub = next(e["ref"] for e in obs["dom_elements"] if e["text"].strip().lower() == "submit")
a1 = env.unwrapped.create_action(ActionTypes.FOCUS_ELEMENT_AND_TYPE_TEXT, ref=inp, text=word)
obs, r, t, tr, _ = env.step(a1); print("after type: reward", r, "term", t)
a2 = env.unwrapped.create_action(ActionTypes.CLICK_ELEMENT, ref=sub)
obs, r, t, tr, _ = env.step(a2); print("after submit: reward", r, "term", t)
# PRESS_KEY test
obs, _ = env.reset(seed=1)
word = re.search(r'"([^"]+)"', obs["utterance"]).group(1)
inp = next(e["ref"] for e in obs["dom_elements"] if e["tag"] == "input_text")
env.step(env.unwrapped.create_action(ActionTypes.FOCUS_ELEMENT_AND_TYPE_TEXT, ref=inp, text=word))
keys = list(cfg.allowed_keys)
print("Enter in keys:", "<Enter>" in keys)
a3 = env.unwrapped.create_action(ActionTypes.PRESS_KEY, key=keys.index("<Enter>") if "<Enter>" in keys else 0)
obs, r, t, tr, _ = env.step(a3); print("after Enter: reward", r, "term", t)
env.close()
