import gymnasium as gym, miniwob, warnings
warnings.filterwarnings("ignore")
ids = sorted(k for k in gym.registry if k.startswith("miniwob/"))
print("total registered:", len(ids))
want = ["click-button","click-link","click-option","click-tab","click-checkboxes","click-dialog","enter-text","focus-text","choose-list","login-user","click-widget","enter-password","click-collapsible","search-engine"]
have = [w for w in want if f"miniwob/{w}-v1" in ids]
print("available of wanted:", have)
print("missing:", [w for w in want if w not in have])
base = "file:///C:/data/miniwob_html_300s/miniwob/"
for w in have[:12]:
    env = gym.make(f"miniwob/{w}-v1", base_url=base)
    obs, _ = env.reset(seed=0)
    d = obs["dom_elements"]
    vis = [(e["ref"], e["tag"], (e["text"] or "")[:18], e["classes"][:14]) for e in d if e["width"] > 0 and e["height"] > 0][:6]
    print(f"- {w}: '{obs['utterance'][:70]}' | {len(d)} dom | e.g. {vis[:4]}")
    env.close()
