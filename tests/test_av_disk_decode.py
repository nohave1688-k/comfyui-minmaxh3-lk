"""AV落盘解码 tests: model-protocol chunk streaming + background encode.

Run with the ComfyUI environment's python (no pytest needed):

    <aki>/python/python.exe tests/test_av_disk_decode.py

Pinned claims:

1. ``_TemporalChunkStream`` (per-chunk ``video_vae.decode`` calls through the
   model's own 5+2 temporal protocol) reproduces a whole-segment
   ``decode_temporal`` pass at the wrapper level — frame counts from the
   model's real ``decode_output_shape`` formulas, frame content checked
   with token-tagged latents through a CONTEXT-SENSITIVE decoder stand-in
   (any hole/drift/duplication/extra-chunk breaks the equality).  Lengths
   cover every residue mod 5.
2. Per-chunk decode stays bounded: exactly chunk+overlap tokens requested,
   <= one chunk's plan frames returned (the "RAM stays O(one chunk)" claim).
3. Whole-segment / non-causal fallbacks return the plain single decode.
4. ``_BackgroundEncode`` round-trips bytes, surfaces dead encoders from
   ``put`` (no infinite block), and ``abort`` cannot wedge on a full queue.

Known wrapper-level divergence (documented in the stream docstring, not
observable through the wrapper contract): the model blends raw
pre-normalization part values at joins; the stream blends per-chunk
finalized frames, so decoder-range saturation inside join bands can differ
slightly from a whole-segment decode.
"""
import os
import subprocess
import sys
import tempfile
import threading
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_COMFY = os.path.abspath(os.path.join(_REPO, "..", ".."))
for path in (_COMFY, _REPO):
    if path not in sys.path:
        sys.path.insert(0, path)

from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE  # noqa: E402  (real formulas + blend)
from gh import audio_ops  # noqa: E402


# ---------------------------------------------------------------- fakes ----

class FakeH3Model:
    """Stand-in for MiniMaxH3VideoVAE with the real protocol constants.

    decode_output_shape / _decode_temporal_chunks / _decode_temporal_frame_plan
    are the REAL unbound methods (formula duplication = zero); blend is the
    real unbound blend.  decode() (wrapper level) reproduces decode_temporal's
    semantics for a single <=7-token chunk call: nearest-interp clip frames,
    pre-trim, j=0 body + j=1 tail overlap, buffer clipped to the plan.
    """

    noncausal = False
    vae_ratio = 2          # spatial
    vae_ratio_t = 4
    clip_length = 17
    token_drop = 3
    frame_pre_padding = (-17) % 4
    tokens_chunk_size = 5
    token_overlap = (-3) % 5
    frame_overlap = max(token_overlap * vae_ratio_t - frame_pre_padding, 0)
    # >0 makes the decoder CONTEXT-SENSITIVE like the real ViT (each clip's
    # output depends on every token in it) — this is what exposes extra or
    # misplaced protocol chunks; a purely per-token decode would hide them.
    context_gain = 0.05

    blend = MiniMaxH3VideoVAE.blend
    decode_output_shape = MiniMaxH3VideoVAE.decode_output_shape
    _decode_temporal_chunks = MiniMaxH3VideoVAE._decode_temporal_chunks
    _decode_temporal_frame_plan = MiniMaxH3VideoVAE._decode_temporal_frame_plan
    _decode_temporal_pad_frames = MiniMaxH3VideoVAE._decode_temporal_pad_frames

    class _DecoderStub:  # decode_output_shape reads out_channels (+ noncausal formula attrs)
        out_channels = 3
        frames_to_trim = 3
        block = 5

    decoder = _DecoderStub()

    def __init__(self, spatial=2):
        self.spatial = spatial
        self.max_request_tokens = 0
        self.decode_calls = 0

    # -- content: nearest-interp upsampling of token-tagged latents ---------
    def _interp(self, samples, tokens_span):
        t0, t1 = tokens_span
        frames = (t1 - t0) * self.vae_ratio_t
        clip = samples[:, :, t0:t1].float()
        out = torch.nn.functional.interpolate(
            clip,
            size=(frames, samples.shape[3] * self.spatial, samples.shape[4] * self.spatial),
            mode="nearest",
        )
        # Context sensitivity: every frame of this clip shifts by the clip's
        # own token content, exactly like a clip-local attention decoder.
        if self.context_gain:
            out = out + clip.mean(dim=(1, 2, 3, 4), keepdim=True) * self.context_gain
        return out  # [1, C, F, H, W] channel-second (model-native layout)

    def decode(self, samples):
        """Wrapper-level decode: for noncausal models the whole clip in one
        pass (simple frame formula); otherwise decode_temporal's single-call
        semantics (pad -> clip -> trim/overlap -> plan clipping)."""
        self.decode_calls += 1
        self.max_request_tokens = max(self.max_request_tokens, int(samples.shape[2]))
        t = int(samples.shape[2])
        if self.noncausal:
            frames = max(1, t * self.vae_ratio_t - self.decoder.frames_to_trim * -(-t // self.decoder.block))
            return torch.nn.functional.interpolate(
                samples.float(),
                size=(frames, samples.shape[3] * self.spatial, samples.shape[4] * self.spatial),
                mode="nearest",
            ).movedim(1, -1).contiguous()
        pad_tokens, num_chunks = self._decode_temporal_chunks(t)
        padded = samples
        if pad_tokens > 0:
            padded = torch.cat([samples, samples[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)
        chunk_dec = self.tokens_chunk_size * self.vae_ratio_t
        clip_token_len = min(self.tokens_chunk_size + self.token_overlap, t + pad_tokens)
        clip = self._interp(padded, (0, clip_token_len))  # [1, C, clip_frame_len, H, W]
        clip_frame_len = int(clip.shape[2])

        plan = self.decode_output_shape(
            (samples.shape[0], samples.shape[1], t, samples.shape[3], samples.shape[4])
        )[2]
        buf = torch.zeros(
            (1, clip.shape[1], plan, clip.shape[3], clip.shape[4]), dtype=torch.float32
        )
        write_pos = 0
        for j in range(2):  # split_count == 2 for token_drop > 0
            f_start = j * chunk_dec
            f_end = min(f_start + chunk_dec, clip_frame_len)
            part = clip[:, :, f_start + self.frame_pre_padding:f_end]
            take = min(int(part.shape[2]), plan - write_pos)
            if take > 0:
                buf[:, :, write_pos:write_pos + take] = part[:, :, :take]
                write_pos += take
        return buf.movedim(1, -1).contiguous()  # wrapper contract [1, F, H, W, C]


class FakeWrapper:
    """comfy.sd.VAE stand-in exposing `.decode` + `.first_stage_model`."""

    def __init__(self, model):
        self.first_stage_model = model

    def decode(self, samples):
        return self.first_stage_model.decode(samples)


class PlainWrapper:
    """A VAE without any chunk protocol: the whole-segment fallback path."""

    def __init__(self, spatial=2):
        self.spatial = spatial

    def decode(self, samples):
        frames = int(samples.shape[2]) * 4
        out = torch.nn.functional.interpolate(
            samples.float(),
            size=(frames, samples.shape[3] * self.spatial, samples.shape[4] * self.spatial),
            mode="nearest",
        )
        return out.movedim(1, -1).contiguous()


def tagged_latent(tokens, channels=3, latent_hw=(4, 3)):
    """[1, C, T, H, W] latent where spatial planes carry per-token tags that
    survive the uint8 conversion exactly: v(t)*255 = 2t + 10."""
    tags = torch.full((tokens,), ((10 % 256) / 255.0))
    for t in range(tokens):
        tags[t] = ((2 * t + 10) % 256) / 255.0
    plane = tags.view(1, 1, tokens, 1, 1).expand(1, channels, tokens, *latent_hw)
    return plane.contiguous()


def to_uint8(frames):
    return (frames.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)


# ------------------------------------------------- ground truth (truth) ----

def whole_decode_reference(model, samples):
    """Faithful transcription of MiniMaxH3VideoVAE.decode_temporal over the
    whole segment — the reference the chunk-call decomposition must match.
    Ordering matters: each part is pre-trimmed BEFORE the blend, exactly as
    in decode_temporal."""
    t = int(samples.shape[2])
    if model.noncausal or t <= 1:
        return model.decode(samples).movedim(-1, 1)  # plain path, model-native
    pad_tokens, num_chunks = model._decode_temporal_chunks(t)
    padded = samples
    if pad_tokens > 0:
        padded = torch.cat([samples, samples[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)
    plan = model.decode_output_shape((1, samples.shape[1], t, samples.shape[3], samples.shape[4]))[2]
    chunk_dec = model.tokens_chunk_size * model.vae_ratio_t
    dec = torch.zeros(
        (1, samples.shape[1], plan, samples.shape[3] * model.spatial, samples.shape[4] * model.spatial)
    )
    write_pos = 0
    dec_overlap = None
    z_padded_len = t + pad_tokens

    def write(part):
        nonlocal write_pos
        take = min(int(part.shape[2]), plan - write_pos)
        if take > 0:
            dec[:, :, write_pos:write_pos + take] = part[:, :, :take]
            write_pos += take

    for i in range(num_chunks):
        s = i * model.tokens_chunk_size
        e = min(s + model.tokens_chunk_size + model.token_overlap, z_padded_len)
        clip = model._interp(padded, (s, e))
        # decode_temporal: slice -> pre-trim -> blend -> write-clip.
        j0 = clip[:, :, :chunk_dec, :, :][:, :, model.frame_pre_padding:, :, :]
        if dec_overlap is not None:
            j0 = model.blend(dec_overlap, j0, model.frame_overlap, dim=2)
            dec_overlap = None
        write(j0)
        j1 = clip[:, :, chunk_dec:min(2 * chunk_dec, int(clip.shape[2])), :, :]
        j1 = j1[:, :, model.frame_pre_padding:, :, :]
        if int(j1.shape[2]) > 0:
            dec_overlap = j1.contiguous()
    if dec_overlap is not None:
        write(dec_overlap)
    return dec


# --------------------------------------------------------- stream tests ----

def check_protocol_case(tokens):
    model = FakeH3Model()
    wrapper = FakeWrapper(model)
    latent = tagged_latent(tokens)
    stream = audio_ops._TemporalChunkStream(wrapper, latent)
    raw = b"".join(iter(stream))
    mine = torch.frombuffer(
        bytearray(raw), dtype=torch.uint8
    ).view(-1, stream.height, stream.width, 3)

    reference = to_uint8(whole_decode_reference(model, latent).movedim(1, -1)[0])
    assert mine.shape[0] == reference.shape[0], (
        f"T={tokens}: {mine.shape[0]} frames != whole-decode {reference.shape[0]}"
    )
    assert torch.equal(mine, reference), f"T={tokens}: frame content differs"

    truth = int(model.decode_output_shape((1, latent.shape[1], tokens, latent.shape[3], latent.shape[4]))[2])
    assert mine.shape[0] == truth, f"T={tokens}: {mine.shape[0]} != decode_output_shape {truth}"
    assert stream.frame_count == mine.shape[0]
    assert stream.max_chunk_frames <= 22, f"T={tokens}: chunk decode {stream.max_chunk_frames} frames"
    assert model.max_request_tokens <= 7, f"T={tokens}: requested {model.max_request_tokens} tokens"
    _, num_chunks = model._decode_temporal_chunks(tokens)
    assert model.decode_calls == num_chunks, (
        f"T={tokens}: {model.decode_calls} decode calls != protocol chunks {num_chunks} "
        "(an off-protocol extra chunk cross-fades wrong content into the tail)"
    )
    print(f"  ok  T={tokens:>3}: {mine.shape[0]} frames == whole decode == plan, "
          f"{num_chunks} chunks, max {model.max_request_tokens}tok/{stream.max_chunk_frames}f")


def check_fallback_plain():
    wrapper = PlainWrapper()
    latent = tagged_latent(9)
    stream = audio_ops._TemporalChunkStream(wrapper, latent)
    raw = b"".join(iter(stream))
    mine = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(-1, stream.height, stream.width, 3)
    reference = to_uint8(wrapper.decode(latent)[0])  # wrapper contract: already [F,H,W,C]
    assert torch.equal(mine, reference), "plain fallback content differs"
    print("  ok  no-protocol fallback == plain whole decode")


def check_fallback_noncausal():
    model = FakeH3Model()
    model.noncausal = True
    wrapper = FakeWrapper(model)
    latent = tagged_latent(12)
    stream = audio_ops._TemporalChunkStream(wrapper, latent)
    raw = b"".join(iter(stream))
    mine = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(-1, stream.height, stream.width, 3)
    reference = to_uint8(model.decode(latent)[0])  # fake returns moved [F,H,W,C]
    assert torch.equal(mine, reference), "noncausal fallback content differs"
    assert model.max_request_tokens == 12, "noncausal model must not be sliced"
    print("  ok  noncausal (Sol) fallback decodes whole segment unsliced")


# ------------------------------------------------ background-encode tests --

def test_background_encode_roundtrip():
    data = os.urandom(3 * 1024 * 1024 + 12345)
    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "out.bin")
        handle = audio_ops._BackgroundEncode([
            sys.executable,
            "-c",
            "import sys; open(sys.argv[1], 'wb').write(sys.stdin.buffer.read())",
            out_path,
        ])
        for start in range(0, len(data), 65536):
            handle.put(data[start:start + 65536])
        handle.close()
        handle.finish("roundtrip")
        with open(out_path, "rb") as fh:
            written = fh.read()
    assert written == data, "background encode dropped or corrupted bytes"
    print("  ok  _BackgroundEncode round-trips 3MB through the bounded queue")


def test_background_encode_failure():
    handle = audio_ops._BackgroundEncode([
        sys.executable,
        "-c",
        "import sys; sys.stdin.buffer.read(); sys.stderr.write('boom'); sys.exit(3)",
    ])
    handle.put(b"hello")
    handle.close()
    try:
        handle.finish("failure")
        raise AssertionError("finish() should have raised on nonzero exit")
    except RuntimeError as exc:
        assert "code 3" in str(exc) and "boom" in str(exc), f"bad failure report: {exc}"
    print("  ok  _BackgroundEncode surfaces nonzero exits with stderr digest")


def test_background_encode_put_after_death():
    # Child reads ~256KB then exits: further puts must raise, never block.
    handle = audio_ops._BackgroundEncode([
        sys.executable,
        "-c",
        "import sys; sys.stdin.buffer.read(256 * 1024); sys.exit(9)",
    ])
    raised = None
    for index in range(64):
        try:
            handle.put(b"x" * 1024 * 1024)
        except RuntimeError as exc:
            raised = exc
            break
    assert raised is not None, "put() blocked forever on a dead encoder"
    assert "code 9" in str(raised), f"failure lacks exit code: {raised}"
    handle.abort()
    print("  ok  put() raises with stderr digest once the encoder dies")


def test_background_encode_abort_while_full():
    # Child never reads stdin: queue + pipe fill, feeder wedges in write().
    handle = audio_ops._BackgroundEncode([sys.executable, "-c", "import time; time.sleep(30)"])
    for _ in range(6):
        try:
            handle.put(b"x" * 1024 * 1024)
        except RuntimeError:
            break  # acceptable: put's death-poll can't fire here, queue fills first
    done = threading.Event()
    threading.Thread(target=lambda: (handle.abort(), done.set()), daemon=True).start()
    assert done.wait(timeout=15), "abort() wedged behind a full queue"
    print("  ok  abort() kills first and never wedges on a full queue")


def main():
    print("temporal chunk protocol vs whole decode:")
    # Every residue mod 5 (2 = the canonical sampler layout via 5k+2), plus
    # multi-chunk lengths and tail-padding edges.
    for tokens in (2, 5, 6, 7, 8, 9, 10, 12, 13, 14, 19, 22, 23, 24, 37, 55,
                   107, 109, 110, 112):
        check_protocol_case(tokens)
    check_fallback_plain()
    check_fallback_noncausal()
    print("background encoder:")
    test_background_encode_roundtrip()
    test_background_encode_failure()
    test_background_encode_put_after_death()
    test_background_encode_abort_while_full()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
