# -*- coding: utf-8 -*-
"""Workflow V11 (rev2): runtime-bundle consolidation, interop-safe.

Composer gains a FIFTH+ 解码设置 (LK_H3_RUNTIME) output APPENDED AFTER the
original five slots, so pre-bundle canvases keep every bare wire on its
original slot.  Bare CLIP/VAE outputs stay for third-party nodes (the user's
canvas decodes via T8 MiniMaxH3AVDecodeT8 nodes that need raw VAEs).

  node 105 (Composer): outputs -> 6 entries; runtime(slot 5) -> guider/refine
  node 47  (guider):   inputs  -> [model(97), segments(114), runtime(178)]
  node 115 (refine):   inputs  -> [av_latent(159), segments(160), model(161), runtime(179), noise(165)]
  nodes 116/117 (T8 decode): untouched - keep their raw VAE wires 169/170/174/175

Base: the user's current canvas E:\\rock\\Downloads\\长视频 h3_V8.json.
"""
import io
import json

SRC = r"E:\rock\Downloads\长视频 h3_V8.json"
DST = r"F:\长视频 h3_V11.json"

data = json.load(io.open(SRC, encoding="utf-8"))
nodes = {n["id"]: n for n in data["nodes"]}
links = data["links"]
# only the guider/refine clip/VAE wires are replaced by the bundle
OLD = {118, 119, 122, 162, 163, 164}
NEW = [
    [178, 105, 5, 47, 2, "LK_H3_RUNTIME"],
    [179, 105, 5, 115, 3, "LK_H3_RUNTIME"],
]

n105 = nodes[105]
n105["outputs"] = [
    {"name": "segments", "type": "LANXI_H3_SEGMENTS", "links": [114, 160]},
    {"name": "空白AV模板", "type": "LATENT", "links": [116]},
    {"name": "CLIP", "type": "CLIP", "links": []},
    {"name": "视频 VAE", "type": "VAE", "links": [169, 174]},
    {"name": "音频 VAE", "type": "VAE", "links": [170, 175]},
    {"name": "解码设置", "type": "LK_H3_RUNTIME", "links": [178, 179]},
]


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
    ("noise", "NOISE", True, 165),
])
# nodes 116/117 (T8 decodes) keep their original raw-VAE inputs - untouched.

data["links"] = [l for l in links if l[0] not in OLD] + NEW
# noise wire into the refine moves from input slot 6 to 4 (clip/vaes -> runtime)
for l in data["links"]:
    if l[0] == 165:
        l[4] = 4
data["last_link_id"] = 179

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
    if oslot >= len(o.get("outputs", [])) or lid not in (o["outputs"][oslot].get("links") or []):
        problems.append(f"link {lid}: origin {origin}.{oslot} output list mismatch")
    tin = t["inputs"][tslot] if tslot < len(t.get("inputs", [])) else None
    if tin is None or tin.get("link") != lid:
        problems.append(f"link {lid}: target {target}.{tslot} mismatch")
    elif tin.get("type") != ltype:
        problems.append(f"link {lid}: type mismatch {tin.get('type')} != {ltype}")
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
print("composer outputs:", [(o["name"], o["links"]) for o in cnodes[105]["outputs"]])
print("runtime wires   :", [l for l in chk["links"] if l[5] == "LK_H3_RUNTIME"])
print("PROBLEMS:" if problems else "NO PROBLEMS")
for p in problems:
    print(" -", p)
print("written:", DST)
