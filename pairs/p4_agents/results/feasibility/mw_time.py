import time, gymnasium as gym, numpy as np, miniwob
from miniwob.action import ActionTypes
env = gym.make("miniwob/click-test-2-v1")
for wait in (0, 12):
    obs, info = env.reset(seed=3)
    word = obs["utterance"].replace("Click button", "").strip(" .").upper()
    ref = next(d["ref"] for d in obs["dom_elements"] if d["text"].strip().upper() == word)
    time.sleep(wait)
    obs, r, term, trunc, info = env.step(env.unwrapped.create_action(ActionTypes.CLICK_ELEMENT, ref=ref))
    print(f"agent delay {wait:>2}s -> reward {r:.4f} terminated {term} truncated {trunc}")
env.close()
