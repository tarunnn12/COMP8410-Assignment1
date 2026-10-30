import shutil, re, time, pathlib, gymnasium as gym, miniwob
from miniwob.action import ActionTypes
src = pathlib.Path(miniwob.__file__).parent / "html"
dst = pathlib.Path("C:/data/miniwob_html_300s")
if dst.exists(): shutil.rmtree(dst)
shutil.copytree(src, dst)
core = dst / "core" / "core.js"
txt = core.read_text(encoding="utf-8")
new = txt.replace("core.EPISODE_MAX_TIME = 10000;", "core.EPISODE_MAX_TIME = 300000;")
print("patched:", txt != new)
core.write_text(new, encoding="utf-8")
base = (dst / "miniwob").as_uri() + "/"
print("base_url:", base)
env = gym.make("miniwob/click-test-2-v1", base_url=base)
for wait in (0, 12):
    obs, info = env.reset(seed=3)
    word = obs["utterance"].replace("Click button", "").strip(" .").upper()
    ref = next(d["ref"] for d in obs["dom_elements"] if d["text"].strip().upper() == word)
    time.sleep(wait)
    obs, r, term, trunc, info = env.step(env.unwrapped.create_action(ActionTypes.CLICK_ELEMENT, ref=ref))
    print(f"agent delay {wait:>2}s -> reward {r:.4f} terminated {term}")
env.close()
