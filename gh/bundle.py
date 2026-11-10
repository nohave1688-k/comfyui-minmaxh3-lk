"""Per-segment latent cache for the LK long-video pipeline.

A bundle is ONE auto-generated cache folder per film — disposable; the next
run regenerates whatever it needs:

    {project_dir}/
    ├─ manifest.json        cache ledger: film identity + per-segment keys
    ├─ segNNN_first.latent  per-segment raw latents (safetensors)
    └─ segNNN_second.latent refined-pass outputs (trimmed)

The folder is created automatically under output/LK_H3_projects/ when the
composer's 项目目录 is left empty.  Cache keys chain the upstream carry hash
plus content hashes of every consumed reference, so any change invalidates
exactly the right segments - a hit is never served for changed content.
sweep_latent_caches prunes whole stale bundles by size/age.
"""
from __future__ import annotations

import datetime
import hashlib
import io as _io
import json
import logging
import os
import shutil
import time

import folder_paths
import torch

MANIFEST_NAME = "manifest.json"
PASS_FIRST = "first"
PASS_SECOND = "second"


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _user_cache_dir() -> str:
    """Registry home for {film_key: {path, updated}} - the sweep's book of
    record.  Losing it is harmless: bundles are self-describing and the
    next run re-registers itself."""
    base = getattr(folder_paths, "get_user_directory", None)
    root = base() if callable(base) else folder_paths.get_temp_directory()
    return os.path.join(root, "LK_cache")


# ---------------------------------------------------------------- hashing


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    """Content hash of a tensor; streams the raw buffer without copying."""
    data = tensor.detach().to("cpu").contiguous()
    digest = hashlib.sha256()
    view = data.numpy()
    digest.update(memoryview(view).cast("B"))
    return digest.hexdigest()


def canonical_hash(payload) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pair_hash(video, audio) -> str:
    """Hash of a latent pair (the carry-chain link between segments)."""
    return f"{sha256_tensor(video)[:16]}:{sha256_tensor(audio)[:16]}"


# ------------------------------------------------------------ bundle paths


def manifest_path(bundle_dir: str) -> str:
    return os.path.join(bundle_dir, MANIFEST_NAME)


def segment_path(bundle_dir: str, pass_name: str, index: int) -> str:
    return os.path.join(bundle_dir, f"seg{index + 1:03d}_{pass_name}.safetensors")


def load_manifest(bundle_dir: str):
    try:
        with open(manifest_path(bundle_dir), "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def save_manifest(bundle_dir: str, manifest: dict) -> None:
    manifest["updated"] = _now()
    tmp = manifest_path(bundle_dir) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=1)
    os.replace(tmp, manifest_path(bundle_dir))


def is_bundle(path: str) -> bool:
    return bool(path) and os.path.isfile(manifest_path(path))


# --------------------------------------------------------- bundle creation


def film_identity(*, clip_name, video_vae_name, audio_vae_name, width, height, segment_count) -> dict:
    """Stable film identity: deliberately EXCLUDES per-shot content so that
    editing one shot's prompt keeps the same bundle.  Structural changes
    (model, canvas, shot count) start a new bundle by design."""
    return {
        "clip_name": clip_name or "",
        "video_vae_name": video_vae_name or "",
        "audio_vae_name": audio_vae_name or "",
        "width": int(width),
        "height": int(height),
        "segment_count": int(segment_count),
    }


def resolve_bundle(project_dir: str, identity: dict) -> dict | None:
    """Create/open the bundle folder for this film (the composer passes the
    auto-generated per-film subfolder, or the user's explicit path).  An
    empty path returns None - cache fully off.  Every resolve also runs one
    whole-bundle sweep pass over the other indexed bundles."""
    project_dir = (project_dir or "").strip()
    if not project_dir:
        return None
    bundle_dir = os.path.abspath(project_dir)
    os.makedirs(bundle_dir, exist_ok=True)
    film_key = canonical_hash(identity)[:8]
    manifest = load_manifest(bundle_dir)
    if not isinstance(manifest, dict) or manifest.get("film_key") != film_key:
        # A foreign or stale folder: start THIS film's manifest.  The old
        # file is kept on disk as manifest.bak for forensics.
        if manifest is not None:
            try:
                shutil.copyfile(manifest_path(bundle_dir), manifest_path(bundle_dir) + ".bak")
            except OSError:
                pass
        manifest = {
            "version": 1,
            "film_key": film_key,
            "identity": identity,
            "created": _now(),
            "updated": _now(),
            PASS_FIRST: {},
            PASS_SECOND: {},
        }
    save_manifest(bundle_dir, manifest)
    update_index(film_key, bundle_dir)
    # 束解析即巡检：其它束按阈值整束清扫（活动束豁免）。
    sweep_latent_caches(active_dir=bundle_dir)
    return {"dir": bundle_dir, "film_key": film_key, "manifest": manifest}


def update_index(film_key: str, bundle_dir: str) -> None:
    if not film_key:
        return
    index = read_index()
    index[film_key] = {"path": bundle_dir, "updated": _now()}
    write_index(index)


def read_index():
    try:
        index_path = os.path.join(_user_cache_dir(), "index.json")
        with open(index_path, "r", encoding="utf-8") as handle:
            index = json.load(handle)
        return index if isinstance(index, dict) else {}
    except (OSError, ValueError):
        return {}


def write_index(index: dict) -> None:
    try:
        root = _user_cache_dir()
        os.makedirs(root, exist_ok=True)
        with open(os.path.join(root, "index.json"), "w", encoding="utf-8") as handle:
            json.dump(index, handle, ensure_ascii=False, indent=1)
    except OSError as exc:
        logging.info("[LKH3] 项目索引写入失败（不影响束）：%s", exc)


# ------------------------------------------------------- whole-bundle sweep


def sweep_latent_caches(active_dir: str | None = None) -> None:
    """束缓存自动清理：束内容（manifest + latent）全部可再生，所以到龄/
    超限直接**整束删除**。只删"确认是束"的文件夹（manifest.json 存在），
    防止索引指向被挪用的普通目录时误删；活动束豁免；删掉的束连同索引
    条目一并剔除，失效条目也顺带清理。

    阈值（环境变量，每次队列解析束时执行一次）：
    - LK_BUNDLE_CACHE_GB   全部束的磁盘占用上限（GB），最旧优先整束删。
                           默认 16，设 0 关闭清理。
    - LK_BUNDLE_CACHE_DAYS 束最后活动时间距今超过 N 天整束删除。默认 0
                           = 不按龄。
    """
    try:
        cap_gb = float(os.environ.get("LK_BUNDLE_CACHE_GB", "16") or 0)
    except (TypeError, ValueError):
        cap_gb = 16.0
    try:
        max_age_days = float(os.environ.get("LK_BUNDLE_CACHE_DAYS", "0") or 0)
    except (TypeError, ValueError):
        max_age_days = 0.0
    if cap_gb <= 0 and max_age_days <= 0:
        return
    active = os.path.abspath(active_dir) if active_dir else None
    index = read_index()
    stale_keys = []
    bundles = []  # (mtime, film_key, path)
    for film_key, item in index.items():
        path = str((item or {}).get("path") or "")
        if not path or (active and os.path.abspath(path) == active):
            continue
        if not is_bundle(path):
            stale_keys.append(film_key)  # 文件夹已删/被挪用：索引条目失效
            continue
        try:
            bundles.append((os.stat(path).st_mtime, film_key, path))
        except OSError:
            stale_keys.append(film_key)
    if not bundles and not stale_keys:
        return

    changed = bool(stale_keys)
    for film_key in stale_keys:
        index.pop(film_key, None)
        logging.info("[LKH3] 缓存清理：索引条目失效，剔除 %s", film_key)

    def _bundle_size(path: str) -> int:
        total = 0
        for base, _dirs, files in os.walk(path):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(base, name))
                except OSError:
                    pass
        return total

    freed = 0

    def _remove_bundle(film_key: str, path: str, size: int, why: str) -> None:
        nonlocal changed, freed
        try:
            shutil.rmtree(path)
            index.pop(film_key, None)
            changed = True
            freed += size
            logging.info("[LKH3] 缓存清理：整束删除 %s（%.1f MB，%s）",
                         path, size / (1 << 20), why)
        except OSError as exc:
            logging.info("[LKH3] 缓存清理跳过 %s：%s", path, exc)

    if max_age_days > 0:
        deadline = time.time() - max_age_days * 86400
        for mtime, film_key, path in list(bundles):
            if mtime < deadline:
                _remove_bundle(film_key, path, _bundle_size(path),
                               f"超过 {max_age_days:g} 天未活动")
    if cap_gb > 0:
        limit = int(cap_gb * (1 << 30))
        live = [(mtime, film_key, path) for mtime, film_key, path in bundles
                if os.path.isdir(path)]
        total = sum(_bundle_size(path) for _, _, path in live)
        if total > limit:
            live.sort(key=lambda entry: entry[0])  # 最旧优先
            for mtime, film_key, path in live:
                if total <= limit:
                    break
                size = _bundle_size(path)
                _remove_bundle(film_key, path, size, "总量超限，最旧优先")
                total -= size
    if changed:
        write_index(index)
    if freed:
        logging.info("[LKH3] 缓存清理完成：本次释放 %.2f GB", freed / (1 << 30))


# ------------------------------------------------------- media digests
# 参考媒体不再入束：只算内容 hash 喂缓存键；媒体本体留在用户图上。
# hash 值与旧版 refs 写盘时代逐位一致——已有缓存键继续有效。


def _image_png_bytes(tensor: torch.Tensor) -> bytes:
    from PIL import Image  # local import: only needed for the content hash

    frame = tensor.detach().to("cpu")[0]
    array = frame.clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
    buffer = _io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def _audio_wav_bytes(audio: dict) -> bytes:
    """Encode an AUDIO dict to WAV bytes (torchaudio → soundfile → stdlib)."""
    waveform, sample_rate = audio["waveform"], int(audio["sample_rate"])
    wave = waveform.detach().to("cpu")
    if wave.ndim == 3:
        wave = wave[0]
    try:
        import torchaudio
        buffer = _io.BytesIO()
        torchaudio.save(buffer, wave, sample_rate, format="wav")
        return buffer.getvalue()
    except Exception:
        pass
    try:
        import soundfile as sf
        buffer = _io.BytesIO()
        sf.write(buffer, wave.transpose(0, 1).numpy(), sample_rate, format="wav")
        return buffer.getvalue()
    except Exception:
        pass
    import wave as wave_mod
    import numpy as np
    samples = wave.transpose(0, 1).numpy()
    samples = (samples.clip(-1, 1) * 32767).astype("<i2")
    buffer = _io.BytesIO()
    with wave_mod.open(buffer, "wb") as handle:
        handle.setnchannels(samples.shape[1])
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())
    return buffer.getvalue()


def _video_source_path(video):
    """Best-effort source path of a V3 VIDEO input (files only; in-memory
    components cannot be referenced by name).  Mirrors
    long_video._video_source_path's get_stream_source contract — kept local
    to avoid an import cycle."""
    source = getattr(video, "get_stream_source", None)
    if callable(source):
        try:
            source = source()
        except Exception:
            return None
    return source if isinstance(source, str) and os.path.isfile(source) else None


def shot_media_digests(shots) -> dict:
    """Compute the per-shot media content digests that feed the segment cache
    keys (content-based, NOT file names - swapping a reference image always
    misses the segment).  Pure hashing: nothing is written to disk; reference
    media stays wherever the user's graph keeps it.

    Returns {shot_number: {slot: content_hash}}."""
    digests = {}
    for shot_index, shot in enumerate(shots, start=1):
        if not isinstance(shot, dict):
            continue
        shot_str = str(shot_index)
        pairs = [
            ("first_frame", shot.get("first_frame"), "image"),
            ("last_frame", shot.get("last_frame"), "image"),
        ]
        for slot, tensor in enumerate(shot.get("images") or [], start=1):
            pairs.append((f"img{slot}", tensor, "image"))
        for slot, video in enumerate(shot.get("videos") or [], start=1):
            pairs.append((f"vid{slot}", video, "video"))
        for slot, audio in enumerate(shot.get("audios") or [], start=1):
            pairs.append((f"aud{slot}", audio, "audio"))
        for slot, value, kind in pairs:
            if value is None:
                continue
            name = f"shot{shot_index}_{slot}"
            try:
                if kind == "image" and isinstance(value, torch.Tensor):
                    digest = hashlib.sha256(_image_png_bytes(value)).hexdigest()
                elif kind == "audio":
                    audio = value if isinstance(value, dict) else None
                    digest = (hashlib.sha256(_audio_wav_bytes(audio)).hexdigest()
                              if audio is not None and audio.get("waveform") is not None else None)
                elif kind == "video":
                    digest = None
                    source = _video_source_path(value)
                    if source:
                        digest = sha256_file(source)
                    else:
                        save_to = getattr(value, "save_to", None)
                        if callable(save_to):
                            tmp = os.path.join(folder_paths.get_temp_directory(),
                                               f"_lkhash_{name}.mp4")
                            save_to(tmp, format="mp4", codec="h264")
                            try:
                                digest = sha256_file(tmp)
                            finally:
                                if os.path.exists(tmp):
                                    os.remove(tmp)
                else:
                    digest = None
                if digest:
                    digests.setdefault(shot_str, {})[slot] = digest
            except Exception as exc:
                logging.info("[LKH3] 参考媒体 hash 失败（跳过 %s）：%s", name, exc)
    return digests


# ------------------------------------------------------- segment cache


def spec_hash(plan, index: int, seg_seed: int, sampling: dict, pass_name: str) -> str:
    """Content hash of ONE segment's full conditioning contract: the segment
    spec, film-level params, this segment's absolute seed, and (second pass)
    the refine recipe.  Changing any of these must miss this segment only."""
    segment = plan["segments"][index]
    spec = {
        "pass": pass_name,
        "segment": {
            key: segment.get(key)
            for key in ("duration", "prompt", "length", "loras",
                        "audio_mode",
                        "first_frame", "last_frame", "images", "videos", "audios")
        },
        # 媒体内容 hash（Composer 在 shot_media_digests 时计算）：段规格里的媒体
        # 字段只是文件名串，内容盲——换参考图必须通过这里才能 miss。
        "media": (plan.get("media_hashes") or {}).get(str(index + 1)),
        "seed": int(seg_seed),
        "film": {
            key: plan.get(key)
            for key in ("film_prompt",) if plan.get(key) is not None
        } | {
            "width": plan.get("width"), "height": plan.get("height"),
            "latent_continue": bool(plan.get("latent_continue", True)),
            "audio_mode": plan.get("audio_mode"),
        },
        "sampling": sampling,
    }
    return canonical_hash(spec)


def key_of(spec_hash: str, carry_hash: str) -> str:
    return hashlib.sha256((spec_hash + "|" + carry_hash).encode("utf-8")).hexdigest()


def save_segment(bundle_dir: str, pass_name: str, index: int, video, audio) -> bool:
    """Persist a segment's latent pair.  First pass stores the UNTRIMMED
    output (the next segment's carry source); second pass stores the trimmed
    film span.  Atomic-ish: temp file + os.replace.  Uses safetensors
    directly (comfy.utils.load_torch_file dispatches by file extension, so
    it cannot round-trip arbitrary cache file names)."""
    path = segment_path(bundle_dir, pass_name, index)
    tmp = path + ".tmp"
    try:
        from safetensors.torch import save_file
        save_file(
            {"video": video.detach().to("cpu").contiguous(),
             "audio": audio.detach().to("cpu").contiguous()},
            tmp,
        )
        os.replace(tmp, path)
        return True
    except Exception as exc:
        logging.info("[LKH3] 分段缓存写入失败（镜头%d %s）：%s", index + 1, pass_name, exc)
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False


def load_segment(bundle_dir: str, pass_name: str, index: int):
    path = segment_path(bundle_dir, pass_name, index)
    if not os.path.isfile(path):
        return None
    try:
        from safetensors.torch import load_file
        data = load_file(path, device="cpu")
        video, audio = data.get("video"), data.get("audio")
        if not isinstance(video, torch.Tensor) or not isinstance(audio, torch.Tensor):
            return None
        return {"video": video, "audio": audio}
    except Exception as exc:
        logging.info("[LKH3] 分段缓存读取失败（镜头%d %s）：%s", index + 1, pass_name, exc)
        return None


def has_segment(bundle_dir: str, pass_name: str, index: int) -> bool:
    return os.path.isfile(segment_path(bundle_dir, pass_name, index))


def miss_reason(manifest, pass_name: str, index: int, key: str, spec_hash: str, carry_hash: str) -> str:
    entry = (manifest or {}).get(pass_name, {}).get(str(index))
    if not entry:
        return "新段"
    if entry.get("spec_hash") != spec_hash:
        return "本镜规格或参数变更"
    if entry.get("carry_hash") != carry_hash:
        return "上游续接内容变更"
    return "键不匹配"


def note_segment(manifest: dict, pass_name: str, index: int, *, key, spec_hash, carry_hash,
                 seed: int, audio_mode: str | None = None) -> None:
    entries = manifest.setdefault(pass_name, {})
    entries[str(index)] = {
        "key": key, "spec_hash": spec_hash, "carry_hash": carry_hash,
        "seed": int(seed), "updated": _now(),
        **({"audio_mode": audio_mode} if audio_mode else {}),
    }


def wrap_samples(data: dict, audio_mode: str | None = None) -> dict:
    """Rebuild a first-pass `samples` dict from a cached latent pair, shaped
    exactly like `_sample_bundle`'s return for the trim/collect path."""
    import comfy.nested_tensor
    samples = {"samples": comfy.nested_tensor.NestedTensor((data["video"], data["audio"]))}
    if audio_mode:
        samples["gh_h3_audio_mode"] = audio_mode
    return samples
