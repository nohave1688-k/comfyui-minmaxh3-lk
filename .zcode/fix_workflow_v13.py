# -*- coding: utf-8 -*-
"""Workflow V13: V12 (pika official recipe) + FULL de-T8.

- The T8 learned upscaler (44) is REMOVED: the refine node upscales internally
  (upscale_scale=1.5, model from the official latent_upscale_models folder) —
  spatial-only upscale keeps lk_layout spans exact, so the drift class dies.
  The refine's av_latent now takes the PASS-1 output directly (SCA output).
- The two T8 decodes (116/117) are replaced by comfy-core VAEDecode +
  VAEDecodeAudio pairs (the official Pika template's decode idiom — they read
  the packed AV latent directly).
- Everything else from V12: pika-only model chain (UNET->Sage->pika->guider),
  ManualSigmas (official Flex 6-step sigmas) -> SCA.sigmas, guider lcm,
  runtime bundle wires, refine denoise 0.30 / anchor / audio keep.
- User live edits kept: 5s shots, range 1-5.
"""
import io
import json

SRC = r"E:\rock\Downloads\长视频 h3_V8.json"
DST = r"F:\长视频 h3_V13_pika_not8.json"

FLEX_SIGMA_STRING = "1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.0"
UPSCALE_MODEL = "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"

data = json.load(io.open(SRC, encoding="utf-8"))
nodes = {n["id"]: n for n in data["nodes"]}
links = data["links"]

REMOVE_LINKS = {3, 11, 12, 13, 14, 17, 18, 40, 99, 118, 119, 122, 144, 153, 159, 162, 163, 164, 168}
NEW_LINKS = [
    [159, 71, 0, 115, 0, "LATENT"],         # SCA output -> refine av_latent (replaces the upscaler hop; id reused)
    [178, 105, 5, 47, 2, "LK_H3_RUNTIME"],  # composer 解码设置 -> guider
    [179, 105, 5, 115, 3, "LK_H3_RUNTIME"], # composer 解码设置 -> refine
    [182, 4, 0, 96, 0, "MODEL"],            # sage -> pika
    [183, 96, 0, 19, 0, "MODEL"],           # pika -> cinematic (refine style chain)
    [184, 118, 0, 71, 3, "SIGMAS"],         # ManualSigmas -> SCA.sigmas
    [185, 115, 0, 120, 0, "LATENT"],        # refine output -> VAEDecodeAudio samples
    [187, 71, 0, 122, 0, "LATENT"],         # SCA output -> VAEDecodeAudio samples (direct branch)
]
# retargets of surviving links (T8 decode replacement)
RETARGET = {
    169: (105, 3, 119, 1),  # video VAE -> VAEDecode.vae (refined branch)
    170: (105, 4, 120, 1),  # audio VAE -> VAEDecodeAudio.vae (refined branch)
    171: (119, 0, 110, 0),  # VAEDecode.IMAGE -> WASSave.images
    172: (120, 0, 110, 2),  # VAEDecodeAudio.AUDIO -> WASSave.audio
    173: (71, 0, 121, 0),   # SCA output -> VAEDecode.samples (direct branch)
    174: (105, 3, 121, 1),  # video VAE -> VAEDecode.vae (direct branch)
    175: (105, 4, 122, 1),  # audio VAE -> VAEDecodeAudio.vae (direct branch)
    176: (121, 0, 14, 0),   # VAEDecode.IMAGE -> WASSave.images
    177: (122, 0, 14, 2),   # VAEDecodeAudio.AUDIO -> WASSave.audio
}

# --- new nodes: official decoders ---
def _decode_node(nid, stype, pos, title, samples_link, vae_link, out_links):
    return {
        "id": nid, "type": stype,
        "pos": pos, "size": [270, 60], "flags": {}, "order": 32, "mode": 0,
        "inputs": [
            {"name": "samples", "type": "LATENT", "link": samples_link},
            {"name": "vae", "type": "VAE", "link": vae_link},
        ],
        "outputs": [
            {"name": "IMAGE" if stype == "VAEDecode" else "AUDIO",
             "type": "IMAGE" if stype == "VAEDecode" else "AUDIO", "links": out_links},
        ],
        "title": title,
        "properties": {"Node name for S&R": stype, "cnr_id": "comfy-core"},
        "widgets_values": [], "widgets_values_named": {},
    }


nodes[119] = _decode_node(119, "VAEDecode", [5960, 3170], "Decode HIGH video", 168, 169, [171])
nodes[120] = _decode_node(120, "VAEDecodeAudio", [5960, 3320], "Decode HIGH audio", 185, 170, [172])
nodes[121] = _decode_node(121, "VAEDecode", [6280, 2450], "Decode LOW video (pass-1 direct)", 173, 174, [176])
nodes[122] = _decode_node(122, "VAEDecodeAudio", [6280, 2600], "Decode LOW audio (pass-1 direct)", 187, 175, [177])

# --- ManualSigmas (official Flex schedule lives in the workflow, not the pack) ---
nodes[118] = {
    "id": 118, "type": "ManualSigmas",
    "pos": [4430, 3210], "size": [370, 100], "flags": {}, "order": 31, "mode": 0,
    "inputs": [],
    "outputs": [{"name": "SIGMAS", "type": "SIGMAS", "links": [184]}],
    "title": "Pika Flex 官方 6 步 Sigma",
    "properties": {"Node name for S&R": "ManualSigmas"},
    "widgets_values": [FLEX_SIGMA_STRING],
    "widgets_values_named": {"sigmas": FLEX_SIGMA_STRING},
}

# --- remove T8 nodes ---
for dead in (44, 116, 117):
    nodes.pop(dead, None)

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
n115["widgets_values"] = [12, 3, 9, 6, "lcm", "keep", True, True, "anchor", 1.0, 0.30, UPSCALE_MODEL, 1.5]
n115["widgets_values_named"] = {
    "shift_video": 12, "shift_audio": 3, "base_steps": 9, "coarse_steps": 6,
    "sampler_name": "lcm", "audio_policy": "keep",
    "release_clip": True, "free_between_segments": True,
    "carry_source": "anchor", "context_noise": 1.0, "refine_entry_sigma": 0.30,
    "upscale_model": UPSCALE_MODEL, "upscale_scale": 1.5,
}

# --- composer outputs rebuilt (6-slot bundle-era layout) ---
n105["outputs"] = [
    {"name": "segments", "type": "LANXI_H3_SEGMENTS", "links": [114, 160]},
    {"name": "空白AV模板", "type": "LATENT", "links": [116]},
    {"name": "CLIP", "type": "CLIP", "links": []},
    {"name": "视频 VAE", "type": "VAE", "links": [169, 174]},
    {"name": "音频 VAE", "type": "VAE", "links": [170, 175]},
    {"name": "解码设置", "type": "LK_H3_RUNTIME", "links": [178, 179]},
]

# --- guider/refine input arrays per current schemas ---
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
# guider contract sigmas no longer feed SCA (ManualSigmas does)
nodes[47]["outputs"][2]["links"] = []
for inp in nodes[71].get("inputs", []):
    if inp.get("link") == 144:
        inp["link"] = 184
# SCA output now feeds: refine (159), VAEDecode (173), VAEDecodeAudio (187)
nodes[71]["outputs"][0]["links"] = [159, 173, 187]

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

# --- links: removals, retargets, additions, noise slot fix ---
data["links"] = [l for l in links if l[0] not in REMOVE_LINKS]
for l in data["links"]:
    if l[0] == 165:
        l[4] = 4  # refine noise wire slot (runtime input moved ahead of it)
    if l[0] in RETARGET:
        o, oslot, t, tslot = RETARGET[l[0]]
        l[1], l[2], l[3], l[4] = o, oslot, t, tslot
data["links"].extend(NEW_LINKS)
# refine output slot 0 now feeds both decoders
for out in nodes[115]["outputs"]:
    if out.get("name") in ("AV 潜空间", "av_latent"):
        out["links"] = [168, 185]
data["nodes"] = list(nodes.values())
data["last_node_id"] = 122
data["last_link_id"] = 187

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
types = {119: ("IMAGE", 171), 120: ("AUDIO", 172), 121: ("IMAGE", 176), 122: ("AUDIO", 177)}
for nid, (want, lid) in types.items():
    got = cnodes[nid]["outputs"][0]["type"]
    if got != want:
        problems.append(f"node {nid}: output type {got} != {want}")
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


for sink in (110, 14, 119, 120, 121, 122):
    dfs(sink, [])
t8_types = {n["type"] for n in chk["nodes"] if str(n.get("cnr_id", "")).find("T8") >= 0 or "T8" in n["type"]}
print("T8 node types remaining:", sorted(t8_types) or "NONE")
print("PROBLEMS:" if problems else "NO PROBLEMS")
for p in problems:
    print(" -", p)
print("written:", DST)
