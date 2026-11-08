# -*- coding: utf-8 -*-
"""Workflow V12 (rev2): Pika Flex official recipe via GENERIC interfaces.

Vendor-specific values live in the WORKFLOW, not the pack:
- guider sampler=lcm (native FLOW_AV route), scheduler stays native_flow
- a ManualSigmas node carries the official Flex sigma string and feeds
  SamplerCustomAdvanced's sigmas input (replacing the guider contract wire) -
  the segmented loop uses it as-is per segment
- refine sampler=lcm, denoise 0.30 simple tail (refine's new external sigmas
  socket left unconnected - internal recipe is already official-adjacent)

Model chain: UNET -> Sage -> pika@1.0 -> guider; refine chain: pika ->
cinematic -> antiburn -> refine. Old acceleration LoRAs / sol_attn /
attention-backend bypassed (mode 4). User live edits kept: 5s shots, range 1-5,
refine base=9 coarse=6 anchor context_noise=1.
"""
import io
import json

SRC = r"E:\rock\Downloads\长视频 h3_V8.json"
DST = r"F:\长视频 h3_V12_pika.json"

FLEX_SIGMA_STRING = "1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.0"

data = json.load(io.open(SRC, encoding="utf-8"))
nodes = {n["id"]: n for n in data["nodes"]}
links = data["links"]

REMOVE_LINKS = {3, 11, 12, 13, 14, 17, 18, 99, 118, 119, 122, 144, 153, 162, 163, 164}
NEW_LINKS = [
    [178, 105, 5, 47, 2, "LK_H3_RUNTIME"],  # composer 解码设置 -> guider
    [179, 105, 5, 115, 3, "LK_H3_RUNTIME"], # composer 解码设置 -> refine
    [182, 4, 0, 96, 0, "MODEL"],    # sage -> pika
    [183, 96, 0, 19, 0, "MODEL"],   # pika -> cinematic (refine style chain)
    [184, 118, 0, 71, 3, "SIGMAS"], # ManualSigmas -> SCA.sigmas
]

# --- ManualSigmas node (from the official template) ---
nodes[118] = {
    "id": 118,
    "type": "ManualSigmas",
    "pos": [4430, 3210],
    "size": [370, 100],
    "flags": {},
    "order": 31,
    "mode": 0,
    "inputs": [],
    "outputs": [
        {"name": "SIGMAS", "type": "SIGMAS", "links": [184]},
    ],
    "title": "Pika Flex 官方 6 步 Sigma",
    "properties": {"Node name for S&R": "ManualSigmas"},
    "widgets_values": [FLEX_SIGMA_STRING],
    "widgets_values_named": {"sigmas": FLEX_SIGMA_STRING},
}

# --- composer outputs rebuilt to the 6-slot bundle-era layout ---
nodes[105]["outputs"] = [
    {"name": "segments", "type": "LANXI_H3_SEGMENTS", "links": [114, 160]},
    {"name": "空白AV模板", "type": "LATENT", "links": [116]},
    {"name": "CLIP", "type": "CLIP", "links": []},
    {"name": "视频 VAE", "type": "VAE", "links": [169, 174]},
    {"name": "音频 VAE", "type": "VAE", "links": [170, 175]},
    {"name": "解码设置", "type": "LK_H3_RUNTIME", "links": [178, 179]},
]

# --- guider/refine input arrays rebuilt per the new schemas ---
def inputs_for(spec):
    entries = []
    for name, ltype, shape7, link in spec:
        e = {"name": name, "type": ltype, "link": link}
        if shape7:
            e["shape"] = 7
        entries.append(e)
    return entries


nodes[47]["inputs"] = inputs_for([
    ("model", "MODEL", False, 97),
    ("segments", "LANXI_H3_SEGMENTS", False, 114),
    ("runtime", "LK_H3_RUNTIME", True, 178),
])
nodes[115]["inputs"] = inputs_for([
    ("av_latent", "LATENT", False, 159),
    ("segments", "LANXI_H3_SEGMENTS", False, 160),
    ("model", "MODEL", False, 161),
    ("runtime", "LK_H3_RUNTIME", True, 179),
    ("sigmas", "SIGMAS", True, None),
    ("noise", "NOISE", True, 165),
])

# --- widget resets ---
nodes[107]["widgets_values"] = [5, "fixed"]
nodes[107]["widgets_values_named"] = {"value": 5, "control_after_generate": "fixed"}

n105 = nodes[105]
w = list(n105["widgets_values"])
w[-2], w[-1] = 1, 5
n105["widgets_values"] = w
n105.setdefault("widgets_values_named", {})["range_start"] = 1
n105["widgets_values_named"]["range_end"] = 5

n47 = nodes[47]
n47["widgets_values"] = [6, 12, 3, "lcm", "native_flow", "fixed", 0.15, True, True]
n47["widgets_values_named"] = {
    "steps": 6, "shift_video": 12, "shift_audio": 3,
    "sampler_name": "lcm", "scheduler": "native_flow",
    "seed_policy": "fixed", "context_noise": 0.15,
    "release_clip": True, "free_between_segments": True,
}

n115 = nodes[115]
n115["widgets_values"] = [12, 3, 9, 6, "lcm", "keep", True, True, "anchor", 1.0, 0.30]
n115["widgets_values_named"] = {
    "shift_video": 12, "shift_audio": 3, "base_steps": 9, "coarse_steps": 6,
    "sampler_name": "lcm", "audio_policy": "keep",
    "release_clip": True, "free_between_segments": True,
    "carry_source": "anchor", "context_noise": 1.0, "refine_entry_sigma": 0.30,
}

# --- model chain surgery ---
for nid in (5, 20, 16, 15, 17, 18):
    nodes[nid]["mode"] = 4
DANGLING = {5: 3, 16: 18, 15: 11, 17: 12, 18: 13, 20: 17}
for nid, old in DANGLING.items():
    for inp in nodes[nid].get("inputs", []):
        if inp.get("link") == old:
            inp["link"] = None
nodes[4]["outputs"][0]["links"] = [182]
nodes[96]["outputs"][0]["links"] = [97, 183]
nodes[19]["outputs"][0]["links"] = [155]
for nid in (5, 15, 16, 17, 18, 20):
    if nodes[nid].get("outputs"):
        nodes[nid]["outputs"][0]["links"] = []
for inp in nodes[96].get("inputs", []):
    if inp.get("link") == 99:
        inp["link"] = 182
for inp in nodes[19].get("inputs", []):
    if inp.get("link") == 14:
        inp["link"] = 183
# guider's contract sigmas output no longer feeds SCA
nodes[47]["outputs"][2]["links"] = []
# SCA sigmas input now from ManualSigmas
for inp in nodes[71].get("inputs", []):
    if inp.get("link") == 144:
        inp["link"] = 184

data["links"] = [l for l in links if l[0] not in REMOVE_LINKS] + NEW_LINKS
for l in data["links"]:
    if l[0] == 165:
        l[4] = 5  # refine noise wire -> its slot in [av, segments, model, runtime, sigmas, noise]
data["nodes"] = list(nodes.values())
data["last_node_id"] = 118
data["last_link_id"] = 184

json.dump(data, io.open(DST, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

# ---- verify ---------------------------------------------------------------------
chk = json.load(io.open(DST, encoding="utf-8"))
cnodes = {n["id"]: n for n in chk["nodes"]}
problems = []
for lid, origin, oslot, target, tslot, ltype in chk["links"]:
    o, t = cnodes.get(origin), cnodes.get(target)
    if o is None or t is None:
        problems.append(f"link {lid}: missing node {origin}->{target}")
        continue
    if o.get("mode") == 4:
        problems.append(f"link {lid}: ACTIVE link from BYPASSED node {origin}")
    if t.get("mode") == 4:
        problems.append(f"link {lid}: ACTIVE link to BYPASSED node {target}")
    if oslot >= len(o.get("outputs", [])) or lid not in (o["outputs"][oslot].get("links") or []):
        problems.append(f"link {lid}: origin {origin}.{oslot} output list mismatch")
    tin = t["inputs"][tslot] if tslot < len(t.get("inputs", [])) else None
    if tin is None or tin.get("link") != lid:
        problems.append(f"link {lid}: target {target}.{tslot} mismatch")
adj = {}
for lid, origin, oslot, target, tslot, ltype in chk["links"]:
    adj.setdefault(target, []).append(origin)
state = {}


def dfs(nid, stack):
    if state.get(nid) == 2:
        return True
    if state.get(nid) == 1:
        problems.append(f"CYCLE: {stack[stack.index(nid):] + [nid]}")
        return False
    state[nid] = 1
    ok = True
    for up in adj.get(nid, []):
        ok = dfs(up, stack + [nid]) and ok
    state[nid] = 2
    return ok


for sink in (110, 14, 116, 117):
    dfs(sink, [])
print("ManualSigmas:", cnodes[118]["widgets_values"])
print("SCA sigmas wire:", [l for l in chk["links"] if l[5] == "SIGMAS"])
print("PROBLEMS:" if problems else "NO PROBLEMS")
for p in problems:
    print(" -", p)
print("written:", DST)
