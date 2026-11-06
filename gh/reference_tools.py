"""Reference-focus preprocessing for MiniMax H3 reference images.

H3's reference adherence is driven by how much usable identity signal the
packed reference carries.  Researching the T8 pack's practices showed two
things: its `semantic_bridge` is a trained adapter that ships as separate
weight files (not liftable as a generic node), and its reference strength in
the V3 template comes from packing MANY views of the character.  This node
automates the preprocessing part: detect the faces in an uploaded photo, crop
tight identity views and assemble them into a single identity sheet the
storyboard can pack as one <Picture n>.

Detection uses OpenCV's YuNet face detector (a 230 KB ONNX shipped with the
pack) — no extra model downloads, and every failure path degrades to a plain
center crop with a console warning.
"""

from __future__ import annotations

import logging
import math
import os

import numpy as np
import torch
from comfy_api.latest import io

_PACK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_YUNET_PATH = os.path.join(_PACK_ROOT, "models", "face_detection_yunet_2023mar.onnx")

_FACE_DETECTOR = None


def _detector():
    """Lazy singleton for the YuNet face detector."""
    global _FACE_DETECTOR
    if _FACE_DETECTOR is None:
        import cv2
        if not os.path.exists(_YUNET_PATH):
            raise FileNotFoundError(f"人脸检测模型缺失: {_YUNET_PATH}")
        _FACE_DETECTOR = cv2.FaceDetectorYN.create(_YUNET_PATH, "", (320, 320), 0.5)
    return _FACE_DETECTOR


def detect_faces(image_rgb: np.ndarray, confidence: float) -> list[tuple[float, float, float, float]]:
    """Return face boxes (x, y, w, h) sorted largest-first, in source pixels."""
    import cv2
    det = _detector()
    det.setScoreThreshold(float(confidence))
    h, w = image_rgb.shape[:2]
    scale = 1.0
    work = image_rgb
    if max(h, w) > 1280:
        scale = 1280.0 / max(h, w)
        work = cv2.resize(image_rgb, (max(32, int(w * scale)), max(32, int(h * scale))))
    det.setInputSize((work.shape[1], work.shape[0]))
    bgr = cv2.cvtColor(work, cv2.COLOR_RGB2BGR)
    _, faces = det.detect(bgr)
    boxes: list[tuple[float, float, float, float]] = []
    if faces is not None:
        for face in faces:
            x, y, fw, fh = (float(v) / scale for v in face[:4])
            if fw > 4 and fh > 4:
                boxes.append((x, y, fw, fh))
    boxes.sort(key=lambda box: box[2] * box[3], reverse=True)
    return boxes


def _square_pad_resize(crop: np.ndarray, cell: int) -> np.ndarray:
    """Pad a crop to square with its own edge color, then resize to cell²."""
    import cv2
    h, w = crop.shape[:2]
    side = max(h, w)
    padded = np.full((side, side, 3), 96, np.uint8)
    left = (side - w) // 2
    top = (side - h) // 2
    padded[top:top + h, left:left + w] = crop
    return cv2.resize(padded, (cell, cell), interpolation=cv2.INTER_LANCZOS4)


def _center_square(image_rgb: np.ndarray, cell: int) -> np.ndarray:
    import cv2
    h, w = image_rgb.shape[:2]
    side = min(h, w)
    left = (w - side) // 2
    top = (h - side) // 2
    crop = image_rgb[top:top + side, left:left + side]
    return cv2.resize(crop, (cell, cell), interpolation=cv2.INTER_LANCZOS4)


def _to_tensor(sheet_bgr_or_rgb: np.ndarray) -> torch.Tensor:
    array = sheet_bgr_or_rgb.astype(np.float32) / 255.0
    return torch.from_numpy(array)[None, ...]


class MiniMaxH3ReferenceFocusGH(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LKH3ReferenceFocus",
            display_name="MiniMax-H3 参考聚焦（LK）",
            category="LK/MiniMax H3 Integration",
            description=(
                "Preprocesses a reference photo for stronger identity adherence: detects "
                "faces, crops tight identity views and assembles an identity sheet to "
                "pack as one <Picture n>.  Detection is a bundled 230KB YuNet model; "
                "every failure degrades to a center crop."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="参考照片（取第一批第一张）。人物占比越小/越远，越需要这个预处理。",
                ),
                io.Combo.Input(
                    "mode",
                    options=["face_sheet", "face_main", "center_square", "passthrough"],
                    default="face_sheet",
                    tooltip=(
                        "face_sheet = 检测人脸并把多张特写拼成一张身份表（参考能力最强）；"
                        "face_main = 只保留最大的人脸特写；center_square = 中心方裁；"
                        "passthrough = 原样输出。"
                    ),
                ),
                io.Int.Input(
                    "faces", default=4, min=1, max=6, step=1,
                    tooltip="face_sheet 模式最多打包的人脸数（按人脸大小从大到小取）。",
                ),
                io.Float.Input(
                    "margin", default=1.4, min=1.0, max=2.5, step=0.05,
                    tooltip="人脸框向外扩边系数。1.2 紧贴五官，1.6 带肩颈，2.0+ 带环境。",
                ),
                io.Int.Input(
                    "cell", default=512, min=256, max=1024, step=64,
                    tooltip="身份表每格的边长（像素）。",
                ),
                io.Float.Input(
                    "confidence", default=0.5, min=0.1, max=0.9, step=0.05,
                    tooltip="人脸检测置信度阈值。",
                ),
            ],
            outputs=[
                io.Image.Output(
                    "image", display_name="image",
                    tooltip="预处理后的参考图 — 接分镜节点的 参考图 插座。",
                ),
            ],
        )

    @classmethod
    def execute(cls, image, mode="face_sheet", faces=4, margin=1.4,
                cell=512, confidence=0.5):
        source = image[0].detach().to("cpu").clamp(0, 1).mul(255.0).round().to(torch.uint8).numpy()
        cell = max(64, int(cell))
        margin = max(1.0, float(margin))
        max_faces = max(1, int(faces))

        if mode == "passthrough":
            return io.NodeOutput(image[:1].clone())

        if mode == "center_square":
            return io.NodeOutput(_to_tensor(_center_square(source, cell)))

        crops: list[np.ndarray] = []
        if mode in ("face_sheet", "face_main"):
            try:
                boxes = detect_faces(source, confidence)
            except Exception as error:
                logging.warning(
                    "[LKH3] 参考聚焦：人脸检测不可用（%s），退化为中心裁切。", error
                )
                boxes = []
            if not boxes:
                logging.warning(
                    "[LKH3] 参考聚焦：未检测到人脸，退化为中心裁切——"
                    "如果人物确实在画面里，试着调低置信度。"
                )
            if mode == "face_main":
                boxes = boxes[:1]
            else:
                boxes = boxes[:max_faces]
            h, w = source.shape[:2]
            for x, y, fw, fh in boxes:
                cx, cy = x + fw / 2.0, y + fh / 2.0
                side = max(fw, fh) * margin
                half = side / 2.0
                left = int(max(0.0, cx - half))
                top = int(max(0.0, cy - half))
                right = int(min(float(w), cx + half))
                bottom = int(min(float(h), cy + half))
                if right - left < 16 or bottom - top < 16:
                    continue
                crops.append(_square_pad_resize(source[top:bottom, left:right], cell))

        if not crops:
            crops = [_center_square(source, cell)]

        if mode == "face_main" or len(crops) == 1:
            sheet = crops[0]
        else:
            columns = 2
            rows = math.ceil(len(crops) / columns)
            sheet = np.full((rows * cell, columns * cell, 3), 128, np.uint8)
            for slot, crop in enumerate(crops):
                row, col = divmod(slot, columns)
                sheet[row * cell:(row + 1) * cell, col * cell:(col + 1) * cell] = crop

        return io.NodeOutput(_to_tensor(sheet))
