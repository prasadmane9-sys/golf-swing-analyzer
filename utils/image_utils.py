"""OpenCV-based drawing utilities: skeleton overlay, annotations."""

import cv2
import numpy as np
from typing import List, Optional, Dict

from core.pose_estimator import Landmark, POSE_CONNECTIONS
from core.metrics_calculator import METRIC_IDEALS, metric_status


# Color palette (BGR)
COLOR_JOINT_GOOD  = (0, 220, 0)      # green
COLOR_JOINT_FAIR  = (0, 200, 255)    # yellow
COLOR_JOINT_POOR  = (0, 0, 255)      # red
COLOR_BONE        = (255, 255, 255)  # white
COLOR_TEXT_BG     = (20, 60, 20)     # dark green
COLOR_TEXT_FG     = (255, 255, 255)  # white
COLOR_LOW_CONF    = (0, 128, 255)    # orange


def draw_skeleton(
    frame_bgr: np.ndarray,
    landmarks: Optional[List[Landmark]],
    confidence: float = 1.0,
) -> np.ndarray:
    """Draw skeleton overlay on frame. Returns annotated copy."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    if landmarks is None:
        # Low confidence watermark
        _draw_low_conf_warning(out)
        return out

    def to_px(lm: Landmark):
        return (int(lm.x * w), int(lm.y * h))

    # Draw bones
    for (i, j) in POSE_CONNECTIONS:
        if i >= len(landmarks) or j >= len(landmarks):
            continue
        a, b = landmarks[i], landmarks[j]
        if a.visibility < 0.3 or b.visibility < 0.3:
            continue
        cv2.line(out, to_px(a), to_px(b), COLOR_BONE, 2, cv2.LINE_AA)

    # Draw joints
    for lm in landmarks:
        if lm.visibility < 0.3:
            continue
        px = (int(lm.x * w), int(lm.y * h))
        color = COLOR_JOINT_GOOD if lm.visibility > 0.7 else COLOR_JOINT_FAIR
        cv2.circle(out, px, 5, color, -1, cv2.LINE_AA)
        cv2.circle(out, px, 5, (0, 0, 0), 1, cv2.LINE_AA)  # thin black border

    if confidence < 0.65:
        _draw_low_conf_warning(out)

    return out


def annotate_phase(
    frame_bgr: np.ndarray,
    phase_label: str,
    metrics: Optional[Dict],
    confidence: float = 1.0,
) -> np.ndarray:
    """
    Add phase name banner and a compact metrics summary on the frame.
    Returns annotated copy.
    """
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    # ── Phase name banner (top) ───────────────────────────────────────────
    banner_h = 36
    cv2.rectangle(out, (0, 0), (w, banner_h), COLOR_TEXT_BG, -1)
    cv2.putText(out, phase_label.upper(), (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_TEXT_FG, 2, cv2.LINE_AA)

    # Confidence badge
    conf_text = f"Conf: {confidence:.0%}"
    conf_color = COLOR_JOINT_GOOD if confidence >= 0.75 else COLOR_LOW_CONF
    cv2.putText(out, conf_text, (w - 120, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, conf_color, 1, cv2.LINE_AA)

    # ── Compact metric overlay (bottom strip) ─────────────────────────────
    if metrics:
        strip_h = 22
        y0 = h - strip_h
        cv2.rectangle(out, (0, y0), (w, h), (0, 0, 0), -1)
        metric_strs = _build_metric_strs(metrics)
        x = 6
        for text, color in metric_strs:
            cv2.putText(out, text, (x, h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
            x += len(text) * 8 + 10
            if x > w - 80:
                break

    return out


def encode_frame_jpeg(frame_bgr: np.ndarray, quality: int = 85) -> bytes:
    """Encode a BGR frame to JPEG bytes."""
    import cv2
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode(".jpg", frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return bytes(buffer)


def frame_to_pil(frame_bgr: np.ndarray):
    """Convert BGR numpy frame to PIL Image."""
    from PIL import Image
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# ── Internal helpers ──────────────────────────────────────────────────────────
def _build_metric_strs(metrics: Dict):
    """Return list of (text_str, color) for compact display."""
    items = []
    key_order = ["spine_angle", "hip_rotation", "lead_arm_angle", "head_drift", "weight_shift"]
    for key in key_order:
        val = metrics.get(key)
        if val is None:
            continue
        info = METRIC_IDEALS.get(key, {})
        label = info.get("label", key)
        unit  = info.get("unit", "")
        status = metric_status(key, val)
        text = f"{label}: {val:.1f}{unit}"
        if status == "good":
            color = (0, 230, 0)
        elif status == "fair":
            color = (0, 200, 255)
        else:
            color = (0, 80, 255)
        items.append((text, color))
    return items


def _draw_low_conf_warning(frame: np.ndarray):
    h, w = frame.shape[:2]
    cv2.putText(frame, "LOW CONFIDENCE", (w // 2 - 100, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_LOW_CONF, 2, cv2.LINE_AA)
