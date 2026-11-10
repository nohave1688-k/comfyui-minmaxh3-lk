from __future__ import annotations

import gc
import json
import logging
import os
import queue
import shutil
import subprocess
import threading
import time
from collections import deque

import torch
import torch.nn.functional as torch_functional
import torchaudio

import comfy.utils
import folder_paths
from comfy import model_management
from comfy_api.latest import VideoFromFile, io, ui
from comfy_extras.nodes_audio import vae_decode_audio

from .core import (
    FPS,
    RUNTIME_TYPE,
    encode_audio_once,
    nested_av_parts,
    replace_audio_latent,
    runtime_parts,
    validate_audio,
)


def clean_generated_audio_start(audio: dict) -> dict:
    """Remove the characteristic truncated-word burst at the start of H3 audio."""
    waveform, sample_rate = validate_audio(audio, "decoded_audio")
    analysis_samples = min(waveform.shape[-1], round(sample_rate * 0.9))
    window_samples = max(1, round(sample_rate * 0.01))
    window_count = analysis_samples // window_samples
    if window_count < 20:
        return audio

    analysis = waveform[..., : window_count * window_samples]
    mono = analysis.float().pow(2).mean(dim=(0, 1)).reshape(window_count, window_samples)
    rms = mono.mean(dim=1).sqrt()
    peak = float(rms.max().item())
    if peak < 0.01:
        return audio

    # Generated H3 audio can have a relatively quiet valid onset. Use softer
    # thresholds than the old detector, then require a strong energy contrast
    # and a genuine pause so ordinary soft speech is not removed.
    active_threshold = max(0.004, peak * 0.08)
    silence_threshold = max(0.0015, peak * 0.07)
    active = rms >= active_threshold
    quiet = rms <= silence_threshold

    # The artifact starts at (or essentially at) t=0 and is normally the tail
    # of a word lasting roughly 80-280 ms. Do not classify a later onset as a
    # start burst.
    initial_limit = min(window_count, 3)
    initial_indices = torch.nonzero(active[:initial_limit], as_tuple=False).flatten()
    if initial_indices.numel() == 0:
        return audio
    burst_start = int(initial_indices[0].item())

    quiet_run = 3
    burst_end = None
    burst_end_limit = min(window_count - quiet_run, 32)
    for index in range(max(burst_start + 6, 8), burst_end_limit + 1):
        if bool(quiet[index : index + quiet_run].all().item()):
            burst_end = index
            break
    if burst_end is None:
        return audio

    burst = rms[burst_start:burst_end]
    burst_active_count = int((burst >= active_threshold).sum().item())
    if burst_active_count < 5 or burst_active_count > 30:
        return audio

    # Require an abrupt drop after the initial fragment. This contrast check
    # is what distinguishes the artifact from a normal quiet lead-in.
    pause_end = min(window_count, burst_end + 12)
    pause = rms[burst_end:pause_end]
    if pause.numel() < 6:
        return audio
    burst_level = float(burst.mean().item())
    pause_level = float(pause.mean().item())
    if burst_level <= 0.0 or pause_level > burst_level * 0.42:
        return audio

    # Find the real program onset after a meaningful gap. Allow a gradual
    # recovery, but require most windows in a 60-100 ms region to be active.
    sustained_windows = 7
    main_onset = None
    search_start = burst_end + 8
    search_end = min(window_count - sustained_windows, 80)
    for index in range(search_start, search_end + 1):
        if int(active[index : index + sustained_windows].sum().item()) >= 5:
            main_onset = index
            break
    if main_onset is None:
        return audio

    gap = quiet[burst_end:main_onset]
    if gap.numel() < 10 or float(gap.float().mean().item()) < 0.55:
        return audio

    silence_end_sample = min(waveform.shape[-1], burst_end * window_samples)
    fade_end_sample = min(waveform.shape[-1], main_onset * window_samples)
    if fade_end_sample <= silence_end_sample:
        return audio

    cleaned = waveform.clone()
    cleaned[..., :silence_end_sample] = 0
    fade_length = fade_end_sample - silence_end_sample
    phase = torch.linspace(
        0.0, torch.pi / 2, fade_length,
        device=cleaned.device, dtype=cleaned.dtype,
    )
    fade = torch.sin(phase).square()
    cleaned[..., silence_end_sample:fade_end_sample] *= fade
    return {**audio, "waveform": cleaned}


def inject_audio_latent(av_latent: dict, source_audio: dict, audio_vae, mode: str, strength: float):
    mode = mode.lower()
    if mode not in {"lock", "remix"}:
        raise ValueError("Audio latent control mode must be lock or remix")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")
    encoded = encode_audio_once(audio_vae, source_audio)
    denoise = 0.0 if mode == "lock" else strength
    return replace_audio_latent(av_latent, encoded, denoise), source_audio


def decode_av_latent(av_latent: dict, video_vae, audio_vae):
    try:
        video, audio = nested_av_parts(av_latent)
        images = video_vae.decode(video)
        if images.ndim == 5:
            images = images.reshape(-1, *images.shape[-3:])
        decoded_audio = vae_decode_audio(audio_vae, {"samples": audio})
        if av_latent.get("gh_h3_audio_mode") in {"native", "reference_only"}:
            decoded_audio = clean_generated_audio_start(decoded_audio)
        video_latent = {key: value for key, value in av_latent.items() if key not in {"samples", "noise_mask"}}
        audio_latent = video_latent.copy()
        video_latent["samples"] = video
        audio_latent["samples"] = audio
        masks = av_latent.get("noise_mask")
        if getattr(masks, "is_nested", False):
            video_mask, audio_mask = masks.unbind()
            video_latent["noise_mask"] = video_mask
            audio_latent["noise_mask"] = audio_mask
        return images, decoded_audio, video_latent, audio_latent
    finally:
        # Release temporary decode allocations without unloading VAE models or
        # invalidating tensors that are returned to downstream nodes.
        gc.collect()
        model_management.soft_empty_cache()


def _resample(waveform: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    if source_rate == target_rate:
        return waveform
    return torchaudio.functional.resample(waveform, source_rate, target_rate)


def _match_channels(waveform: torch.Tensor, target_channels: int) -> torch.Tensor:
    channels = waveform.shape[1]
    if channels == target_channels:
        return waveform
    if channels == 1:
        return waveform.expand(-1, target_channels, -1)
    if target_channels == 1:
        return waveform.mean(dim=1, keepdim=True)
    # For uncommon multichannel inputs, make a deterministic stereo-compatible downmix.
    mono = waveform.mean(dim=1, keepdim=True)
    return mono.expand(-1, target_channels, -1)


def mix_audio(
    source_audio: dict,
    generated_audio: dict,
    source_gain_db: float = 0.0,
    generated_gain_db: float = -6.0,
    duck_generated: float = 0.5,
    output_sample_rate: str = "source",
    peak_limit_dbfs: float = -1.0,
):
    source, source_rate = validate_audio(source_audio, "source_audio")
    generated, generated_rate = validate_audio(generated_audio, "generated_audio")
    if output_sample_rate == "source":
        target_rate = source_rate
    elif output_sample_rate == "generated":
        target_rate = generated_rate
    else:
        target_rate = int(output_sample_rate)

    source = _resample(source, source_rate, target_rate)
    generated = _resample(generated, generated_rate, target_rate)
    target_channels = max(source.shape[1], generated.shape[1])
    source = _match_channels(source, target_channels)
    generated = _match_channels(generated, target_channels)
    target_samples = max(source.shape[-1], generated.shape[-1])
    source = torch_functional.pad(source, (0, target_samples - source.shape[-1]))
    generated = torch_functional.pad(generated, (0, target_samples - generated.shape[-1]))

    source = source * (10.0 ** (source_gain_db / 20.0))
    generated = generated * (10.0 ** (generated_gain_db / 20.0))
    if duck_generated > 0:
        envelope = source.abs().mean(dim=1, keepdim=True)
        window = max(1, round(target_rate * 0.02))
        envelope = torch_functional.avg_pool1d(envelope, window, stride=1, padding=window // 2)
        envelope = envelope[..., :target_samples]
        activity = (envelope / 0.05).clamp(0.0, 1.0)
        generated = generated * (1.0 - float(duck_generated) * activity)

    mixed = source + generated
    limit = 10.0 ** (peak_limit_dbfs / 20.0)
    peak = mixed.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    scale = torch.minimum(torch.ones_like(peak), torch.full_like(peak, limit) / peak)
    mixed = mixed * scale
    return {"waveform": mixed, "sample_rate": target_rate}


def trim_av_output(frames: torch.Tensor, start_seconds: float, duration_seconds: float, audio=None, fps: float = 24.0):
    if frames.ndim != 4:
        raise ValueError(f"frames must be IMAGE [N,H,W,C], got {tuple(frames.shape)}")
    if start_seconds < 0 or duration_seconds <= 0 or fps <= 0:
        raise ValueError("trim start must be nonnegative; duration and fps must be positive")
    start_frame = round(start_seconds * fps)
    frame_count = max(1, round(duration_seconds * fps))
    end_frame = start_frame + frame_count
    if end_frame > frames.shape[0]:
        raise ValueError(
            f"Requested output frames [{start_frame}:{end_frame}] exceed decoded frame count {frames.shape[0]}"
        )
    trimmed_frames = frames[start_frame:end_frame]

    trimmed_audio = None
    if audio is not None:
        waveform, sample_rate = validate_audio(audio)
        start_sample = round(start_seconds * sample_rate)
        sample_count = round(duration_seconds * sample_rate)
        sliced = waveform[..., start_sample : start_sample + sample_count]
        if sliced.shape[-1] < sample_count:
            sliced = torch_functional.pad(sliced, (0, sample_count - sliced.shape[-1]))
        trimmed_audio = {"waveform": sliced, "sample_rate": sample_rate}

    report = json.dumps(
        {
            "start_seconds": start_seconds,
            "requested_duration_seconds": duration_seconds,
            "fps": fps,
            "start_frame": start_frame,
            "frame_count": frame_count,
            "actual_video_duration_seconds": frame_count / fps,
        },
        ensure_ascii=False,
        indent=2,
    )
    return trimmed_frames, trimmed_audio, report


class MiniMaxH3AVDecodeGH(io.ComfyNode):
    """Decodes the packed AV latent back to frames + audio.

    The downstream half of the sampler's output contract: the long-video
    sampler no longer decodes internally, so chain this node wherever frames /
    audio are actually needed (one decode per consumption point).
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LKH3AVDecode",
            display_name="MiniMax-H3 AV解码（LK）",
            category="LK/MiniMax H3 Integration",
            description="Decodes the packed AV latent into frames and audio.",
            inputs=[
                io.Latent.Input(
                    "av_latent",
                    tooltip="长视频分段采样（或下游采样链）输出的嵌套 AV 潜空间。",
                ),
                io.Custom(RUNTIME_TYPE).Input(
                    "runtime", display_name="解码设置",
                    tooltip="CLIP + 视频/音频 VAE 捆绑 — 从 MiniMaxH3 长视频编排 的 解码设置 连一根线。",
                ),
            ],
            outputs=[
                io.Image.Output(
                    "frames", display_name="图像",
                    tooltip="解码后的全部帧（1 帧 = 1 张图）。",
                ),
                io.Audio.Output(
                    "audio", display_name="音频",
                    tooltip="解码后的整段音频。",
                ),
            ],
        )

    @classmethod
    def execute(cls, av_latent, runtime):
        _clip, video_vae, audio_vae = runtime_parts(runtime)
        if video_vae is None or audio_vae is None:
            raise ValueError(
                "AV解码 缺少 解码设置 —— 请把 MiniMaxH3 长视频编排 的 解码设置 输出连到本节点。"
            )
        frames, audio, _video_latent, _audio_latent = decode_av_latent(
            av_latent, video_vae, audio_vae
        )
        frames = frames.detach().to("cpu")
        if audio is not None and audio.get("waveform") is not None:
            audio = dict(audio)
            audio["waveform"] = audio["waveform"].detach().to("cpu")
        return io.NodeOutput(frames, audio)


def _resolve_ffmpeg() -> str:
    """Same resolution chain as the audio-compat fallback in gh/nodes.py."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as exc:
            raise RuntimeError(
                "AV落盘解码 找不到可用的 ffmpeg：请安装系统 ffmpeg，"
                f"或在本环境 pip install imageio-ffmpeg。（{exc}）"
            ) from exc
    return ffmpeg


_SUBPROCESS_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _concat_entry(path: str) -> str:
    """One concat-demuxer list line; forward slashes + apostrophe escaping."""
    return "file '" + path.replace(os.sep, "/").replace("'", "'\\''") + "'\n"


def _run_ffmpeg(command, feed=None):
    """Run an ffmpeg command to completion; ``feed`` is an iterable of bytes.

    stderr is drained from a daemon thread so a chatty ffmpeg can never fill
    its stderr pipe and block while we are still feeding stdin (deadlock).
    A broken pipe while feeding means ffmpeg died early; the nonzero
    returncode branch surfaces its stderr.
    """
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE if feed is not None else subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        creationflags=_SUBPROCESS_FLAGS,
    )
    error_chunks = []

    def _drain_stderr():
        try:
            error_chunks.append(process.stderr.read() or b"")
        except Exception:
            pass
        finally:
            try:
                process.stderr.close()
            except Exception:
                pass

    reader = threading.Thread(target=_drain_stderr, daemon=True)
    reader.start()
    try:
        if feed is not None:
            try:
                for block in feed:
                    process.stdin.write(block)
            except BrokenPipeError:
                pass
    finally:
        # Close stdin on EVERY exit path: a feed generator that raises
        # mid-stream must not leave ffmpeg waiting for EOF (wait() would hang
        # forever).  The original exception then propagates after reaping.
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        try:
            code = process.wait()
        except BaseException:
            # Cancel/interrupt while waiting: never orphan the encoder.
            process.kill()
            raise
        reader.join(timeout=5)
    if code != 0:
        details = b"".join(error_chunks).decode("utf-8", errors="replace").strip()[-1200:]
        raise RuntimeError(f"ffmpeg 失败（code {code}）：{details or '无错误输出'}")


def _pcm_bytes(wave, chunk_samples=1 << 20):
    """Yield interleaved f32le PCM bytes from a validated [1, C, S] waveform."""
    interleaved = wave[0].transpose(0, 1).contiguous()
    for start in range(0, int(interleaved.shape[0]), chunk_samples):
        block = interleaved[start:start + chunk_samples]
        yield block.numpy().astype("<f4", copy=False).tobytes()


_FEED_DONE = object()


class _BackgroundEncode:
    """One ffmpeg encode running beside the main thread.

    The main thread stays on the GPU — decoding the next chunk — while this
    process's x264 works.  Frame bytes reach stdin through a bounded queue,
    so a slow encode throttles the producer instead of ballooning RAM, and
    the tail of segment k keeps encoding while segment k+1 is already
    decoding.

    Failure semantics (a video writer dies realistically often — disk full,
    OOM, killed process): ``put`` polls the process while the queue is full
    and raises with the stderr digest instead of blocking forever on a dead
    encoder; ``abort`` kills FIRST and then drains the queue, so cleanup can
    never wedge behind a wedged encoder; ``finish`` reaps and surfaces
    nonzero exits, recorded feeder errors, and the stderr digest.
    """

    def __init__(self, command, queue_depth=4):
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=_SUBPROCESS_FLAGS,
        )
        self._queue = queue.Queue(maxsize=queue_depth)
        self._error = bytearray()
        self._feed_error = None
        self._closed = False
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._feed_thread = threading.Thread(target=self._feed_stdin, daemon=True)
        self._stderr_thread.start()
        self._feed_thread.start()

    def _drain_stderr(self):
        try:
            self._error.extend(self._process.stderr.read() or b"")
        except Exception:
            pass
        finally:
            try:
                self._process.stderr.close()
            except Exception:
                pass

    def _feed_stdin(self):
        try:
            while True:
                block = self._queue.get()
                if block is _FEED_DONE:
                    break
                self._process.stdin.write(block)
        except Exception as exc:  # recorded; finish() decides whether it matters
            self._feed_error = exc
        finally:
            try:
                self._process.stdin.close()
            except OSError:
                pass

    def _failure(self, label):
        details = bytes(self._error).decode("utf-8", errors="replace").strip()[-1200:]
        code = self._process.poll()
        return RuntimeError(
            f"ffmpeg 失败（code {code if code is not None else '?'}，{label}）：{details or '无错误输出'}"
        )

    def put(self, block):
        while True:
            if self._process.poll() is not None:
                # The encoder died mid-segment; feeding further would block
                # forever on a full queue that nobody drains.
                raise self._failure(f"编码进程中断，剩余帧未写入")
            try:
                self._queue.put(block, timeout=0.5)
                return
            except queue.Full:
                continue

    def close(self):
        if not self._closed:
            self._closed = True
            self._queue.put(_FEED_DONE)

    def is_dead(self):
        return self._process.poll() is not None

    def finish(self, label):
        self.close()
        # The feeder stays alive while ffmpeg is still ingesting bytes, which
        # is normal for a slow encode — but a genuinely wedged encoder stops
        # consuming entirely.  Bound the wait so that can never hang the node.
        ingest_deadline = time.time() + 300
        while self._feed_thread.is_alive():
            self._feed_thread.join(timeout=1.0)
            if self._process.poll() is not None:
                # Dead encoder: the feeder exits via the broken pipe.
                break
            if time.time() > ingest_deadline:
                self._process.kill()
                raise RuntimeError(
                    f"ffmpeg 编码器停滞超过 300 秒（{label}），已终止："
                    f"{bytes(self._error).decode('utf-8', errors='replace').strip()[-400:] or '无错误输出'}"
                )
        code = self._process.wait()
        self._stderr_thread.join(timeout=5)
        if code != 0:
            raise self._failure(label)
        if self._feed_error is not None:
            # Bytes were dropped without a nonzero exit — the segment would
            # quietly come out short.
            raise RuntimeError(f"ffmpeg 编码线程异常（{label}）：{self._feed_error!r}")

    def abort(self):
        # Kill FIRST: close()'s sentinel would block on a full queue whose
        # consumer is the process being killed.
        self._closed = True
        try:
            self._process.kill()
        except OSError:
            pass
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        if self._feed_thread.is_alive():
            try:
                self._queue.put_nowait(_FEED_DONE)
            except queue.Full:
                pass  # the feeder dies on the broken pipe anyway
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


class _TemporalChunkStream:
    """Stream-decode one packed segment through the H3 VAE's own temporal
    chunk protocol, yielding rgb24 byte slabs.

    ``MiniMaxH3VideoVAE.decode_temporal`` decodes causally in
    ``tokens_chunk_size``-latent chunks with a ``token_overlap``-latent /
    ``frame_overlap``-frame cross-fade between neighbours — but it writes
    into one full-segment CPU buffer.  This class walks the SAME loop at the
    node level: the segment is padded up front exactly like decode_temporal
    (``_decode_temporal_chunks`` decides the chunk count), each protocol
    chunk is fetched with one ``video_vae.decode()`` call on its full
    ``chunk+overlap``-token slice from the padded stream, joins cross-fade
    through the model's own ``blend``, and the final tail overlap is written
    verbatim with the same plan clipping — so the assembled frames match a
    whole-segment decode while RAM stays at one decoded chunk (~22 frames
    for the stock 5+2 protocol).  A whole film never exists as one tensor,
    which removes the in-RAM decode's film-length ceiling.

    One known, bounded divergence from a whole-segment decode: the model
    blends raw pre-normalization part values at joins, while this stream
    blends the per-chunk finalized frames the wrapper returns — pixels that
    saturate the decoder's output range inside a join's ``frame_overlap``
    band can differ slightly.  Everything else (counts, chunk placement,
    blends, clipping) is protocol-identical.

    Models without the chunk protocol (or Sol-style non-causal decoders,
    which attend across the whole clip and must not be sliced) fall back to
    a single whole-segment decode — byte-identical to the plain path.
    """

    def __init__(self, video_vae, seg_video, slab_frames=8):
        self._vae = video_vae
        self._seg = seg_video
        self._slab = max(1, int(slab_frames))
        self.height = 0
        self.width = 0
        self.frame_count = 0
        self.max_chunk_frames = 0
        self.source_tokens = int(seg_video.shape[2])
        self.source_channels = int(seg_video.shape[1])
        self.source_h = int(seg_video.shape[3])
        self.source_w = int(seg_video.shape[4])

    def _decode(self, chunk):
        images = self._vae.decode(chunk)
        if images.ndim == 5:
            images = images.reshape(-1, *images.shape[-3:])
        return images.detach().to("cpu").to(torch.float32)

    def _emit(self, frames):
        """Finalize decoded frames into rgb24 byte slabs (bounded RAM)."""
        array = (frames.clamp_(0.0, 1.0).mul_(255.0).round_().to(torch.uint8)
                 .contiguous())
        self.height, self.width = int(array.shape[1]), int(array.shape[2])
        for start in range(0, int(array.shape[0]), self._slab):
            block = array[start:start + self._slab]
            self.frame_count += int(block.shape[0])
            yield block.numpy().tobytes()

    def __iter__(self):
        seg = self._seg
        tokens = int(seg.shape[2])
        model = getattr(self._vae, "first_stage_model", None)
        chunk_tokens = getattr(model, "tokens_chunk_size", None)
        overlap_tokens = getattr(model, "token_overlap", None)
        frame_overlap = getattr(model, "frame_overlap", None)
        # 协议属性必须齐全才走分块：缺任何一个（第三方/部分实现的 VAE）都
        # 回退整段解码——与普通路径逐字节一致，绝不带着半套协议硬切。
        if (not chunk_tokens or overlap_tokens is None or frame_overlap is None
                or getattr(model, "noncausal", False) or tokens <= 1
                or not callable(getattr(model, "blend", None))
                or not hasattr(model, "vae_ratio_t")
                or not hasattr(model, "frame_pre_padding")
                or not hasattr(model, "_decode_temporal_chunks")
                or not hasattr(model, "decode_output_shape")
                or getattr(model, "token_drop", 0) > chunk_tokens):
            # No chunk protocol, or a Sol-style whole-clip decoder that must
            # not be sliced: one whole-segment decode, exactly the plain path.
            yield from self._emit(self._decode(seg))
            return
        body_frames = chunk_tokens * model.vae_ratio_t - model.frame_pre_padding
        pad_tokens, num_chunks = model._decode_temporal_chunks(tokens)
        plan_total = int(model.decode_output_shape(
            (1, int(self.source_channels), tokens, int(self.source_h), int(self.source_w))
        )[2])
        padded = seg
        if pad_tokens > 0:
            padded = torch.cat([seg, seg[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)

        emitted = 0
        carried = None  # previous chunk's tail overlap: [frame_overlap, H, W, C]
        for index in range(num_chunks):
            start = index * chunk_tokens
            frames = self._decode(padded[:, :, start:start + chunk_tokens + overlap_tokens])
            self.max_chunk_frames = max(self.max_chunk_frames, int(frames.shape[0]))
            body_len = min(body_frames, int(frames.shape[0]))
            head = frames[:body_len]
            tail = frames[body_len:]
            if carried is not None and head.shape[0] > 0:
                # Same cross-fade decode_temporal applies between neighbours.
                head = model.blend(carried, head, frame_overlap, dim=0)
            if index == num_chunks - 1:
                # decode_temporal writes the final tail overlap verbatim,
                # clipped by the segment's decode plan.
                out = torch.cat([head, tail], dim=0) if tail.shape[0] else head
                remaining = plan_total - emitted
                if int(out.shape[0]) > remaining:
                    out = out[:remaining]
                yield from self._emit(out)
                emitted += int(out.shape[0])
            else:
                yield from self._emit(head)
                emitted += int(head.shape[0])
                carried = tail.clone() if tail.shape[0] else None


class MiniMaxH3AVDiskDecodeGH(io.ComfyNode):
    """Decodes the packed AV latent segment-by-segment straight to disk.

    Same contract as LKH3AVDecode, but the film never lives in RAM as one
    tensor: each ``lk_layout`` segment is decoded through the H3 VAE's own
    temporal chunk protocol (one decode call per protocol chunk, the model's
    own cross-fade across chunk joins — protocol-identical to a
    whole-segment decode, except pixels saturating the decoder range inside
    a join's overlap band) and piped chunk-by-chunk into an incremental
    H.264 encode that runs beside the GPU — the tail of one segment keeps
    encoding while the next one decodes.  The concat demuxer then assembles
    the final MP4 with a video stream copy.  This removes the in-RAM
    decode's film-length ceiling and keeps GPU and CPU busy in parallel.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LKH3AVDiskDecode",
            display_name="MiniMax-H3 AV落盘解码（LK）",
            category="LK/MiniMax H3 Integration",
            description=(
                "Decodes the packed AV latent segment-by-segment into one MP4 "
                "on disk (model-protocol chunking, decode/encode pipelined)."
            ),
            inputs=[
                io.Latent.Input(
                    "av_latent",
                    tooltip="长视频分段采样（或二跑精修）输出的打包 AV 潜空间，需携带 lk_layout。",
                ),
                io.Custom(RUNTIME_TYPE).Input(
                    "runtime", display_name="解码设置",
                    tooltip="CLIP + 视频/音频 VAE 捆绑 — 从 MiniMaxH3 长视频编排 的 解码设置 连一根线。",
                ),
                io.String.Input(
                    "filename_prefix", default="LK_H3/MiniMaxH3",
                    tooltip="输出文件名前缀，可含子目录（如 LK_H3/MiniMaxH3），遵循 ComfyUI 计数命名。",
                ),
                io.Int.Input(
                    "crf", default=18, min=0, max=51, step=1,
                    tooltip="H.264 质量（越低画质越好、文件越大），0 为无损。",
                ),
                io.Boolean.Input(
                    "keep_segments", default=False,
                    tooltip="保留逐段编码的临时 mp4 与 concat 列表（排查接缝问题时打开）。",
                ),
            ],
            outputs=[
                io.Video.Output(
                    "video", display_name="视频",
                    tooltip="落盘完成的整片 MP4。文件已写入输出目录且节点自带预览；"
                            "此输出口只给需要视频对象的下游（再接 保存视频 会二次转码，没必要）。",
                ),
            ],
        )

    @classmethod
    def execute(cls, av_latent, runtime, filename_prefix="LK_H3/MiniMaxH3",
                crf=18, keep_segments=False):
        _clip, video_vae, audio_vae = runtime_parts(runtime)
        if video_vae is None or audio_vae is None:
            raise ValueError(
                "AV落盘解码 缺少 解码设置 —— 请把 MiniMaxH3 长视频编排 的 解码设置 输出连到本节点。"
            )
        layout = av_latent.get("lk_layout") if isinstance(av_latent, dict) else None
        entries = layout.get("segments") if isinstance(layout, dict) else None
        if not entries:
            raise ValueError(
                "AV落盘解码 需要本包分段采样输出的打包 AV latent（携带 lk_layout）。"
                "非 LK latent 请改用 MiniMax-H3 AV解码（LK）。"
            )
        video, audio = nested_av_parts(av_latent)
        audio_mode = av_latent.get("gh_h3_audio_mode")
        ffmpeg = _resolve_ffmpeg()
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory()
        )
        os.makedirs(full_output_folder, exist_ok=True)
        base_name = f"{filename}_{counter:05}"
        final_path = os.path.join(full_output_folder, f"{base_name}.mp4")
        list_path = os.path.join(full_output_folder, f"{base_name}_concat.txt")
        progress = comfy.utils.ProgressBar(len(entries) + 1)

        # At most two encodes stay in flight: the oldest is reaped before a
        # new one starts.  Video chunks decode on the main thread while the
        # previous segments' x264 processes keep encoding in the background,
        # so the GPU never waits on the encoder.
        active = deque()

        def reap(item):
            handle, stream, expected, seg_index = item
            handle.finish(f"镜头{seg_index + 1}")
            actual = int(stream.frame_count)
            if expected is not None and int(expected) == actual:
                return
            # The chunk protocol is frame-exact, so first compare against the
            # model's own decode plan: a mismatch there is a real defect.
            model = getattr(video_vae, "first_stage_model", None)
            model_truth = None
            if model is not None and stream.source_tokens > 0:
                try:
                    shape = (1, int(stream.source_channels), stream.source_tokens,
                             int(stream.source_h), int(stream.source_w))
                    model_truth = int(model.decode_output_shape(shape)[2])
                except Exception:
                    model_truth = None
            if model_truth is not None and model_truth != actual:
                raise RuntimeError(
                    f"AV落盘解码 镜头{seg_index + 1} 解码帧数 {actual} 与模型时间维协议应有帧数 "
                    f"{model_truth} 不一致 —— 分块解码与模型整段解码协议必须帧数一致，"
                    "请检查 VAE/latent 后重试（可用 MiniMax-H3 AV解码（LK）对照定位）。"
                )
            logging.info(
                "[LKH3] 落盘解码 镜头%d 解码帧数 %d 与 layout 记录 %d 不同（模型时间维协议 %s，按实际帧数编码）",
                seg_index + 1, actual, int(expected),
                "一致" if model_truth == actual else model_truth,
            )

        try:
            seg_paths = []
            waveforms = []
            sample_rate = None
            for index, entry in enumerate(entries):
                v0, vt = int(entry["video_offset"]), int(entry["video_tokens"])
                a0, at = int(entry["audio_offset"]), int(entry["audio_steps"])
                seg_video = video[..., v0:v0 + vt, :, :].contiguous()
                seg_audio = audio[..., a0:a0 + at].contiguous()

                stream = _TemporalChunkStream(video_vae, seg_video)
                blocks = iter(stream)
                try:
                    probe = next(blocks)
                except StopIteration:
                    raise ValueError(f"AV落盘解码：镜头{index + 1} 解码出 0 帧。") from None

                # The first chunk fixes the canvas; from here on every
                # following chunk decodes while ffmpeg drains and encodes in
                # the background.
                seg_path = os.path.join(full_output_folder, f"{base_name}_seg{index + 1:03d}.mp4")
                while len(active) >= 2:
                    reap(active.popleft())
                if active and active[0][0].is_dead():
                    reap(active.popleft())  # surface an early ffmpeg death now
                handle = _BackgroundEncode([
                    ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{stream.width}x{stream.height}", "-r", str(FPS),
                    "-i", "pipe:",
                    "-an", "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", str(int(crf)), "-pix_fmt", "yuv420p",
                    seg_path,
                ])
                active.append((handle, stream, entry.get("frames"), index))
                handle.put(probe)
                del probe
                for block in blocks:
                    handle.put(block)
                handle.close()
                # Release the stream's latent reference before the segment
                # release below can actually free anything.
                blocks = None
                stream = None

                # Audio waveforms are tiny; keep them for ONE whole-film
                # clean + encode: the start-burst cleaner analyses the first
                # 0.9s only, so a per-segment application would wipe every
                # carried audio head.
                decoded = vae_decode_audio(audio_vae, {"samples": seg_audio})
                wave, rate = validate_audio(decoded, "decoded_audio")
                waveforms.append(wave.detach().to("cpu").to(torch.float32))
                sample_rate = int(rate)
                seg_paths.append(seg_path)
                del seg_video, seg_audio, decoded, wave, rate

                # Per-segment release keeps VRAM/RAM flat across arbitrarily
                # long films; background encodes hold no torch state.
                gc.collect()
                model_management.soft_empty_cache()
                progress.update(1)

            while active:
                reap(active.popleft())

            full_audio = {"waveform": torch.cat(waveforms, dim=-1), "sample_rate": sample_rate}
            del waveforms
            if audio_mode in {"native", "reference_only"}:
                full_audio = clean_generated_audio_start(full_audio)
            wave, sample_rate = validate_audio(full_audio, "decoded_audio")

            with open(list_path, "w", encoding="utf-8") as list_handle:
                for seg_path in seg_paths:
                    list_handle.write(_concat_entry(seg_path))

            _run_ffmpeg(
                [
                    ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "concat", "-safe", "0", "-i", list_path,
                    "-f", "f32le", "-ar", str(sample_rate),
                    "-ac", str(int(wave.shape[1])), "-i", "pipe:",
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart",
                    final_path,
                ],
                feed=_pcm_bytes(wave),
            )

            # Cleanup only on success — on failure the segment files stay on
            # disk as forensic material for whichever segment broke.
            if not keep_segments:
                for stale in seg_paths + [list_path]:
                    try:
                        os.remove(stale)
                    except OSError:
                        pass
        except BaseException:
            while active:
                active.popleft()[0].abort()
            raise

        progress.update(1)
        return io.NodeOutput(
            VideoFromFile(final_path),
            ui=ui.PreviewVideo([ui.SavedResult(f"{base_name}.mp4", subfolder, io.FolderType.output)]),
        )
