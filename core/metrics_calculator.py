"""Biomechanical metrics calculated from MediaPipe pose landmarks."""

from typing import Optional, Dict, List
from core.pose_estimator import (
    Landmark,
    NOSE, LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE,
)
from utils.geometry import midpoint, angle_at_vertex, angle_between, vector_angle_from_vertical


# ── Ideal range definitions for PDF display ──────────────────────────────────
METRIC_IDEALS = {
    "spine_angle":     {"label": "Spine Angle",        "unit": "°",  "ideal_min": 20,   "ideal_max": 35},
    "hip_rotation":    {"label": "Hip Rotation",       "unit": "°",  "ideal_min": 35,   "ideal_max": 55},
    "lead_arm_angle":  {"label": "Lead Arm",           "unit": "°",  "ideal_min": 160,  "ideal_max": 180},
    "head_drift":      {"label": "Head Stability",     "unit": "",   "ideal_min": 0,    "ideal_max": 0.05},
    "weight_shift":    {"label": "Weight Shift",       "unit": "",   "ideal_min": 0.02, "ideal_max": 0.15},
    "shoulder_tilt":   {"label": "Shoulder Tilt",      "unit": "°",  "ideal_min": 0,    "ideal_max": 15},
    "knee_flex":       {"label": "Knee Flex",          "unit": "°",  "ideal_min": 20,   "ideal_max": 35},
}


def _lm(landmarks: List[Landmark], idx: int, vis_threshold: float = 0.2) -> Optional[tuple]:
    """Return (x, y) tuple for a landmark if visible enough."""
    if landmarks is None or idx >= len(landmarks):
        return None
    lm = landmarks[idx]
    if lm.visibility < vis_threshold:
        return None
    return (lm.x, lm.y)


class MetricsCalculator:
    def __init__(self, handedness: str = "right"):
        """
        handedness: "right" or "left"
        For right-handed golfer, the lead side is left (11/13/15 for shoulder/elbow/wrist).
        """
        if handedness == "right":
            self.lead_shoulder = LEFT_SHOULDER
            self.lead_elbow    = LEFT_ELBOW
            self.lead_wrist    = LEFT_WRIST
            self.trail_wrist   = RIGHT_WRIST
        else:
            self.lead_shoulder = RIGHT_SHOULDER
            self.lead_elbow    = RIGHT_ELBOW
            self.lead_wrist    = RIGHT_WRIST
            self.trail_wrist   = LEFT_WRIST

        self._address_baseline: Optional[Dict] = None

    def set_address_baseline(self, landmarks: List[Landmark]):
        """Must be called with the address-frame landmarks before computing deltas."""
        self._address_baseline = self._extract_raw(landmarks)

    def compute_all(self, landmarks: List[Landmark]) -> Dict:
        """Return dict of all metric values. Values are None if landmarks missing."""
        raw = self._extract_raw(landmarks)
        result = {}

        # ── Spine angle from vertical ──────────────────────────────────────
        if raw["shoulder_mid"] and raw["hip_mid"]:
            result["spine_angle"] = vector_angle_from_vertical(
                raw["shoulder_mid"], raw["hip_mid"]
            )
        else:
            # Fallback: try individual shoulders and hips at lower visibility threshold
            ls_fb = _lm(landmarks, LEFT_SHOULDER, 0.1)
            rs_fb = _lm(landmarks, RIGHT_SHOULDER, 0.1)
            lh_fb = _lm(landmarks, LEFT_HIP, 0.1)
            rh_fb = _lm(landmarks, RIGHT_HIP, 0.1)
            s_mid = midpoint(ls_fb, rs_fb) if (ls_fb and rs_fb) else None
            h_mid = midpoint(lh_fb, rh_fb) if (lh_fb and rh_fb) else None
            if s_mid and h_mid:
                result["spine_angle"] = vector_angle_from_vertical(s_mid, h_mid)
            else:
                result["spine_angle"] = None

        # ── Hip rotation (delta from address baseline) ─────────────────────
        if raw["hip_angle"] is not None:
            if self._address_baseline and self._address_baseline["hip_angle"] is not None:
                delta = raw["hip_angle"] - self._address_baseline["hip_angle"]
                # Positive = open hips toward target (good at impact)
                result["hip_rotation"] = delta
            else:
                # No baseline yet (address frame itself) — report 0
                result["hip_rotation"] = 0.0
        else:
            result["hip_rotation"] = None

        # ── Lead arm straightness ──────────────────────────────────────────
        ls = _lm(landmarks, self.lead_shoulder)
        le = _lm(landmarks, self.lead_elbow)
        lw = _lm(landmarks, self.lead_wrist)
        if ls and le and lw:
            result["lead_arm_angle"] = angle_at_vertex(ls, le, lw)
        else:
            result["lead_arm_angle"] = None

        # ── Head lateral drift from address ───────────────────────────────
        nose = _lm(landmarks, NOSE)
        if nose and self._address_baseline and self._address_baseline["nose"]:
            result["head_drift"] = abs(nose[0] - self._address_baseline["nose"][0])
        elif not self._address_baseline:
            # Address frame itself — drift is zero by definition
            result["head_drift"] = 0.0
        else:
            result["head_drift"] = None

        # ── Weight shift (hip midpoint lateral drift) ──────────────────────
        if raw["hip_mid"] and self._address_baseline and self._address_baseline["hip_mid"]:
            # Positive = toward target (left in rear-view for right-hander)
            shift = self._address_baseline["hip_mid"][0] - raw["hip_mid"][0]
            result["weight_shift"] = shift
        elif not self._address_baseline:
            # Address frame itself — shift is zero by definition
            result["weight_shift"] = 0.0
        else:
            result["weight_shift"] = None

        # ── Shoulder tilt (angle of shoulder line from horizontal) ─────────
        ls2 = _lm(landmarks, LEFT_SHOULDER)
        rs2 = _lm(landmarks, RIGHT_SHOULDER)
        if ls2 and rs2:
            result["shoulder_tilt"] = angle_between(
                (rs2[0] - ls2[0], rs2[1] - ls2[1]), (1, 0)
            )
        else:
            result["shoulder_tilt"] = None

        # ── Knee flex (average flex angle at both knee joints) ─────────────
        lh = _lm(landmarks, LEFT_HIP)
        lk = _lm(landmarks, LEFT_KNEE)
        la = _lm(landmarks, LEFT_ANKLE)
        rh = _lm(landmarks, RIGHT_HIP)
        rk = _lm(landmarks, RIGHT_KNEE)
        ra = _lm(landmarks, RIGHT_ANKLE)

        left_knee_angle  = angle_at_vertex(lh, lk, la) if (lh and lk and la) else None
        right_knee_angle = angle_at_vertex(rh, rk, ra) if (rh and rk and ra) else None

        # Convert full-extension angle to flex amount (180° = straight leg)
        if left_knee_angle is not None:
            left_knee_flex = 180.0 - left_knee_angle
        else:
            left_knee_flex = None

        if right_knee_angle is not None:
            right_knee_flex = 180.0 - right_knee_angle
        else:
            right_knee_flex = None

        if left_knee_flex is not None and right_knee_flex is not None:
            result["knee_flex"] = (left_knee_flex + right_knee_flex) / 2.0
        elif left_knee_flex is not None:
            result["knee_flex"] = left_knee_flex
        elif right_knee_flex is not None:
            result["knee_flex"] = right_knee_flex
        else:
            result["knee_flex"] = None

        return result

    # ── Internal helpers ───────────────────────────────────────────────────
    def _extract_raw(self, landmarks: List[Landmark]) -> Dict:
        ls = _lm(landmarks, LEFT_SHOULDER)
        rs = _lm(landmarks, RIGHT_SHOULDER)
        lh = _lm(landmarks, LEFT_HIP)
        rh = _lm(landmarks, RIGHT_HIP)
        nose = _lm(landmarks, NOSE)

        shoulder_mid = midpoint(ls, rs) if (ls and rs) else None
        hip_mid = midpoint(lh, rh) if (lh and rh) else None

        # Hip angle from horizontal
        if lh and rh:
            hip_vec = (rh[0] - lh[0], rh[1] - lh[1])
            hip_angle = angle_between(hip_vec, (1, 0))
        else:
            hip_angle = None

        return {
            "shoulder_mid": shoulder_mid,
            "hip_mid": hip_mid,
            "hip_angle": hip_angle,
            "nose": nose,
        }


def metric_status(key: str, value: float) -> str:
    """Return 'good', 'fair', or 'poor' for a metric value."""
    if value is None:
        return "unknown"
    info = METRIC_IDEALS.get(key)
    if not info:
        return "unknown"
    lo, hi = info["ideal_min"], info["ideal_max"]
    # Allow 15% leeway on each side for "fair"
    margin = (hi - lo) * 0.15 if (hi - lo) > 0 else 1.0
    if lo <= value <= hi:
        return "good"
    elif (lo - margin) <= value <= (hi + margin):
        return "fair"
    return "poor"
