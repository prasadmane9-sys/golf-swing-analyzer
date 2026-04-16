"""
Video processor: reads a golf swing video, detects 6 swing phase frames,
and returns full-resolution frames + pose landmarks for each phase.
"""

import logging
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from core.pose_estimator import PoseEstimator, Landmark, RIGHT_WRIST, LEFT_WRIST, LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER

logger = logging.getLogger(__name__)

PHASES = ["address", "backswing", "top_of_backswing", "downswing", "impact", "follow_through"]

PHASE_LABELS = {
    "address": "Address",
    "backswing": "Backswing",
    "top_of_backswing": "Top of Backswing",
    "downswing": "Downswing",
    "impact": "Impact",
    "follow_through": "Follow Through",
}

# Confidence threshold below which a warning is emitted and re-detection is attempted
_CONFIDENCE_WARN_THRESHOLD = 0.5

# Centered moving-average window size for signal smoothing
_SMOOTH_WINDOW = 5


@dataclass
class PhaseFrame:
    phase: str
    frame_number: int
    frame_bgr: np.ndarray
    landmarks: Optional[List[Landmark]]
    confidence: float   # 0.0 – 1.0


class VideoProcessor:
    def __init__(self, pose_estimator: PoseEstimator, fps_sample: int = 10,
                 handedness: str = "right", verbose: bool = False):
        self.pose = pose_estimator
        self.fps_sample = fps_sample
        self.handedness = handedness
        self.verbose = verbose
        # For right-handed golfer, dominant wrist (trail) moves most during backswing
        self._dom_wrist = RIGHT_WRIST if handedness == "right" else LEFT_WRIST

    # ── Public API ────────────────────────────────────────────────────────────

    def scan_video(self, video_path: str) -> Dict[str, PhaseFrame]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(video_fps / self.fps_sample))

        self._log(f"Video: {total_frames} frames @ {video_fps:.1f}fps, sampling every {step} frames")

        sampled_frames = []
        frame_num = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_num % step == 0:
                landmarks = self.pose.process_frame(frame)
                sampled_frames.append((frame_num, landmarks))
            frame_num += 1

        cap.release()
        self._log(f"Sampled {len(sampled_frames)} frames in pass 1")

        wrist_y   = []
        wrist_x   = []
        hip_y_arr = []
        frame_nums = []

        for fn, lms in sampled_frames:
            if lms is None:
                continue
            wy = self._get_y(lms, self._dom_wrist)
            wx = self._get_x(lms, self._dom_wrist)
            lh_y = self._get_y(lms, LEFT_HIP)
            rh_y = self._get_y(lms, RIGHT_HIP)
            hip_y = (lh_y + rh_y) / 2 if (lh_y is not None and rh_y is not None) else None

            if wy is not None:
                wrist_y.append(wy)
                wrist_x.append(wx if wx is not None else 0.5)
                hip_y_arr.append(hip_y if hip_y is not None else wy)
                frame_nums.append(fn)

        if len(frame_nums) < 6:
            raise ValueError("Could not detect pose in enough frames. Ensure the golfer is visible.")

        wrist_y   = np.array(wrist_y,   dtype=float)
        wrist_x   = np.array(wrist_x,   dtype=float)
        hip_y_arr = np.array(hip_y_arr, dtype=float)
        frame_nums = np.array(frame_nums)
        n = len(frame_nums)

        # Smooth all motion signals before phase detection (improvement #1)
        wrist_y_s   = self._smooth(wrist_y,   _SMOOTH_WINDOW)
        wrist_x_s   = self._smooth(wrist_x,   _SMOOTH_WINDOW)
        hip_y_s     = self._smooth(hip_y_arr, _SMOOTH_WINDOW)

        phase_indices = self._detect_phases(wrist_y_s, wrist_x_s, hip_y_s, frame_nums, n)
        self._log("Phase detection:")
        for phase, (fn, conf) in phase_indices.items():
            self._log(f"  {phase:20s} -> frame {fn:4d}  (confidence: {conf:.2f})")

        result: Dict[str, PhaseFrame] = {}
        cap = cv2.VideoCapture(video_path)
        for phase, (target_fn, confidence) in phase_indices.items():
            frame, landmarks = self._extract_frame_with_retry(cap, target_fn, total_frames)
            result[phase] = PhaseFrame(
                phase=phase,
                frame_number=target_fn,
                frame_bgr=frame,
                landmarks=landmarks,
                confidence=confidence,
            )
        cap.release()
        return result

    # ── Phase detection ───────────────────────────────────────────────────────

    def _detect_phases(self, wrist_y, wrist_x, hip_y, frame_nums, n) -> Dict[str, Tuple[int, float]]:
        """
        Returns {phase: (frame_number, confidence)} for each of the 6 phases.
        MediaPipe Y: 0=top of image, 1=bottom. Low Y = physically HIGH.
        All signals are pre-smoothed before this method is called.
        """
        result = {}
        velocity = np.abs(np.gradient(wrist_y))

        # ── ADDRESS ──────────────────────────────────────────────────────────
        # Improvement #2: search first 30%, require min velocity AND golfer stopped
        addr_idx = self._find_address(wrist_y, hip_y, velocity, n)
        addr_conf = self._conf(velocity[addr_idx], 0, 0.02)
        if addr_conf < _CONFIDENCE_WARN_THRESHOLD:
            logger.warning(
                "Address confidence %.2f below threshold — retrying with relaxed criteria", addr_conf
            )
            addr_idx = self._find_address_relaxed(wrist_y, hip_y, velocity, n)
            addr_conf = self._conf(velocity[addr_idx], 0, 0.02)
        result["address"] = (int(frame_nums[addr_idx]), float(addr_conf))

        # ── TOP OF BACKSWING ─────────────────────────────────────────────────
        # Improvement #3: global wrist Y minimum in addr→70%, cross-check wrist X
        top_idx, top_conf = self._find_top_of_backswing(wrist_y, wrist_x, addr_idx, n)
        if top_conf < _CONFIDENCE_WARN_THRESHOLD:
            logger.warning(
                "Top-of-backswing confidence %.2f below threshold — retrying without X cross-check",
                top_conf,
            )
            top_idx, top_conf = self._find_top_of_backswing(wrist_y, wrist_x, addr_idx, n, use_x_check=False)
        result["top_of_backswing"] = (int(frame_nums[top_idx]), float(top_conf))

        # ── BACKSWING ────────────────────────────────────────────────────────
        # Improvement #4: 50% of the way from address to top (was 33%)
        bs_idx = addr_idx + max(1, (top_idx - addr_idx) // 2)
        bs_idx = min(bs_idx, top_idx - 1)
        bs_conf = 0.85
        if bs_conf < _CONFIDENCE_WARN_THRESHOLD:
            logger.warning("Backswing confidence %.2f below threshold", bs_conf)
        result["backswing"] = (int(frame_nums[bs_idx]), bs_conf)

        # ── DOWNSWING ────────────────────────────────────────────────────────
        # Improvement #5: frame with maximum velocity after top (fastest transition)
        ds_idx, ds_conf = self._find_downswing(wrist_y, velocity, top_idx, n)
        if ds_conf < _CONFIDENCE_WARN_THRESHOLD:
            logger.warning("Downswing confidence %.2f below threshold", ds_conf)
        result["downswing"] = (int(frame_nums[ds_idx]), float(ds_conf))

        # ── IMPACT ───────────────────────────────────────────────────────────
        # Improvement #6: wrist X returns within 0.03 of address X; fallback to wrist Y
        addr_wx = wrist_x[addr_idx]
        addr_wy = wrist_y[addr_idx]
        impact_idx, imp_conf = self._find_impact(wrist_y, wrist_x, ds_idx, n, addr_wy, addr_wx)
        if imp_conf < _CONFIDENCE_WARN_THRESHOLD:
            logger.warning(
                "Impact confidence %.2f below threshold — expanding search window", imp_conf
            )
            impact_idx, imp_conf = self._find_impact(wrist_y, wrist_x, ds_idx, n, addr_wy, addr_wx,
                                                      x_tol=0.06, search_pct=0.95)
        result["impact"] = (int(frame_nums[impact_idx]), float(imp_conf))

        # ── FOLLOW THROUGH ───────────────────────────────────────────────────
        # Improvement #7: frame where velocity drops below 30% of peak post-impact velocity
        ft_idx, ft_conf = self._find_follow_through(velocity, impact_idx, n)
        if ft_conf < _CONFIDENCE_WARN_THRESHOLD:
            logger.warning("Follow-through confidence %.2f below threshold", ft_conf)
        result["follow_through"] = (int(frame_nums[ft_idx]), float(ft_conf))

        return result

    # ── Individual phase finders ──────────────────────────────────────────────

    def _find_address(self, wrist_y, hip_y, velocity, n) -> int:
        """
        Improvement #2: search first 30% of frames.
        Best frame = lowest (vel * 2 + height_diff) where 3+ consecutive
        samples all have velocity below the median velocity in the search window.
        """
        search_end = min(n, max(5, int(n * 0.30)))
        vel_window = velocity[:search_end]
        vel_median = np.median(vel_window) if len(vel_window) > 0 else 0.02

        best_idx = 0
        best_score = float("inf")

        for i in range(search_end):
            # Check that golfer has actually stopped: 3+ consecutive low-velocity samples
            consecutive_low = 0
            for j in range(i, min(search_end, i + 5)):
                if velocity[j] <= vel_median:
                    consecutive_low += 1
            if consecutive_low < 3:
                continue

            vel_score = velocity[i]
            height_diff = abs(wrist_y[i] - hip_y[i])
            score = vel_score * 2 + height_diff
            if score < best_score:
                best_score = score
                best_idx = i

        return best_idx

    def _find_address_relaxed(self, wrist_y, hip_y, velocity, n) -> int:
        """Fallback: original scoring without the consecutive-stop requirement."""
        search_end = min(n, max(5, int(n * 0.30)))
        best_idx = 0
        best_score = float("inf")
        for i in range(search_end):
            vel_score = velocity[i]
            height_diff = abs(wrist_y[i] - hip_y[i])
            score = vel_score * 2 + height_diff
            if score < best_score:
                best_score = score
                best_idx = i
        return best_idx

    def _find_top_of_backswing(self, wrist_y, wrist_x, addr_idx, n,
                                use_x_check: bool = True) -> Tuple[int, float]:
        """
        Improvement #3: global wrist Y minimum (wrist highest) in [addr_idx, 70% of n].
        Cross-check with wrist X: for a right-handed golfer filmed from behind,
        the trail (right) wrist should be near its rightmost point at the top.
        """
        search_end = max(addr_idx + 2, int(n * 0.70))
        local_wrist_y = wrist_y[addr_idx:search_end]
        local_wrist_x = wrist_x[addr_idx:search_end]

        if len(local_wrist_y) == 0:
            return addr_idx + 1, 0.5

        # Primary: find the global minimum Y (highest physical position)
        top_rel_y = int(np.argmin(local_wrist_y))

        if use_x_check and len(local_wrist_x) > 0:
            # X cross-check: trail wrist should be near its rightmost point
            # We combine Y and X rankings to pick the best candidate
            # Look at the top-5 candidate frames by Y
            n_candidates = min(5, len(local_wrist_y))
            sorted_by_y = np.argsort(local_wrist_y)[:n_candidates]
            x_max = np.max(local_wrist_x)

            best_rel = top_rel_y
            best_combined = float("inf")
            for rel in sorted_by_y:
                y_rank = float(local_wrist_y[rel] - np.min(local_wrist_y))   # 0 = best
                x_dist = float(x_max - local_wrist_x[rel])                   # 0 = rightmost
                combined = y_rank + x_dist * 0.5  # Y is primary signal
                if combined < best_combined:
                    best_combined = combined
                    best_rel = int(rel)
            top_rel = best_rel
        else:
            top_rel = top_rel_y

        top_idx = addr_idx + top_rel

        # Confidence: how much lower is the wrist Y compared to address (bigger rise = more confident)
        rise = wrist_y[addr_idx] - wrist_y[top_idx]  # positive = physically higher
        top_conf = float(min(1.0, max(0.0, rise / (wrist_y[addr_idx] + 1e-6) * 3.0)))
        top_conf = max(0.5, top_conf)

        return top_idx, top_conf

    def _find_downswing(self, wrist_y, velocity, top_idx, n) -> Tuple[int, float]:
        """
        Improvement #5: after the top, find the frame with maximum velocity
        (the fastest moment of the downswing transition).
        """
        search_end = min(n, top_idx + max(5, int(n * 0.25)))
        if top_idx + 1 >= search_end:
            return min(n - 1, top_idx + 1), 0.6

        local_vel = velocity[top_idx + 1:search_end]
        if len(local_vel) == 0:
            return min(n - 1, top_idx + 1), 0.6

        ds_rel = int(np.argmax(local_vel))
        ds_idx = top_idx + 1 + ds_rel

        peak_vel = local_vel[ds_rel]
        ds_conf = float(min(1.0, peak_vel / (np.mean(velocity) + 1e-9) * 0.5))
        ds_conf = max(0.6, ds_conf)

        return ds_idx, ds_conf

    def _find_impact(self, wrist_y, wrist_x, ds_idx, n, addr_wy, addr_wx,
                     x_tol: float = 0.03, search_pct: float = 0.90) -> Tuple[int, float]:
        """
        Improvement #6: search from downswing to 90% of video.
        Primary: wrist X returns within x_tol of address wrist X.
        Fallback: wrist Y closest to address wrist Y.
        """
        search_end = min(n, int(n * search_pct))

        # Primary: find where wrist X crosses back through address X position
        for i in range(ds_idx, search_end):
            if abs(wrist_x[i] - addr_wx) <= x_tol:
                x_conf = float(max(0.5, 1.0 - abs(wrist_x[i] - addr_wx) / (x_tol + 1e-9)))
                return i, x_conf

        # Fallback: wrist Y closest to address wrist Y
        min_diff = float("inf")
        impact_idx = ds_idx
        for i in range(ds_idx, search_end):
            diff = abs(wrist_y[i] - addr_wy)
            if diff < min_diff:
                min_diff = diff
                impact_idx = i
        imp_conf = float(max(0.4, 1.0 - min_diff * 10))
        return impact_idx, imp_conf

    def _find_follow_through(self, velocity, impact_idx, n) -> Tuple[int, float]:
        """
        Improvement #7: after impact, find where motion decelerates again —
        velocity drops below 30% of peak velocity after impact.
        Fallback: 92% of video.
        """
        fallback_idx = min(n - 1, int(n * 0.92))

        post_impact_vel = velocity[impact_idx:]
        if len(post_impact_vel) < 2:
            return fallback_idx, 0.7

        peak_vel = float(np.max(post_impact_vel))
        decel_threshold = peak_vel * 0.30

        for j, v in enumerate(post_impact_vel[1:], start=1):
            if v < decel_threshold:
                ft_idx = impact_idx + j
                ft_conf = float(max(0.6, 1.0 - v / (peak_vel + 1e-9)))
                return ft_idx, ft_conf

        return fallback_idx, 0.70

    # ── Signal utilities ──────────────────────────────────────────────────────

    @staticmethod
    def _smooth(signal: np.ndarray, window: int) -> np.ndarray:
        """
        Improvement #1: centered moving-average smoothing.
        Uses np.convolve with 'same' mode and a uniform kernel; edges are
        handled by shrinking the effective window so boundary values are
        still reasonable rather than biased toward zero.
        """
        if len(signal) < window:
            return signal.copy()
        half = window // 2
        kernel = np.ones(window) / window
        # Full convolution then trim to original length
        padded = np.pad(signal, (half, half), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")
        return smoothed[:len(signal)]

    def _conf(self, velocity, idx, threshold):
        vel = velocity if np.isscalar(velocity) else velocity[idx]
        return float(max(0.5, 1.0 - vel / (threshold + 1e-9)))

    # ── Frame extraction ──────────────────────────────────────────────────────

    def _extract_frame_with_retry(self, cap, frame_num: int, total: int
                                  ) -> Tuple[np.ndarray, Optional[List[Landmark]]]:
        for delta in [0, 2, -2, 4, -4, 6]:
            fn = max(0, min(total - 1, frame_num + delta))
            cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
            ret, frame = cap.read()
            if not ret:
                continue
            landmarks = self.pose.process_frame(frame)
            if landmarks is not None:
                return frame, landmarks
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        return frame if ret else np.zeros((480, 640, 3), dtype=np.uint8), None

    # ── Landmark helpers ──────────────────────────────────────────────────────

    def _get_y(self, landmarks, idx) -> Optional[float]:
        if landmarks and idx < len(landmarks) and landmarks[idx].visibility >= 0.3:
            return landmarks[idx].y
        return None

    def _get_x(self, landmarks, idx) -> Optional[float]:
        if landmarks and idx < len(landmarks) and landmarks[idx].visibility >= 0.3:
            return landmarks[idx].x
        return None

    def _log(self, msg: str):
        if self.verbose:
            print(f"  {msg}")
