"""
Heuristic body measurements from 4 orthographic-style photos.
Uses MediaPipe Pose + known subject-to-camera distance for scale.
Circumferences: ellipse model from front width + side depth at each level.

Calibration: POST /api/train with the same four photos + Akka tape reference learns
per-field multipliers (saved under data/tape_calibration.json). Measurements are
always computed from the images; multipliers map raw geometry toward tape.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from backend.calibration import effective_mults
from backend.image_io import decode_image_bytes_to_bgr


def _env_float(name: str, default: str) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


# Typical phone / studio cam horizontal FOV is wider than old default; narrow FOV ⇒ tiny cm/px.
# BODY_MEASURE_HFOV / BODY_MEASURE_CM_CALIB let you tune per device (see Dockerfile).
def _default_hfov() -> float:
    return _env_float("BODY_MEASURE_HFOV", "72")


def _cm_px_calibration_mult() -> float:
    """Optional global cm/px multiplier; keep at 1 when using tailor session calibration."""
    return max(0.5, _env_float("BODY_MEASURE_CM_CALIB", "1.0"))


def _inch_decimal_places() -> int:
    """Linear measurements: default 2 decimals (set BODY_MEASURE_INCH_DECIMALS=1 for 29.9\" style)."""
    try:
        return max(0, min(4, int(os.environ.get("BODY_MEASURE_INCH_DECIMALS", "1"))))
    except ValueError:
        return 1


def _fmt_inch(v: float) -> str:
    d = _inch_decimal_places()
    return f"{v:.{d}f}\""


def build_jean_style(display: dict[str, str]) -> dict[str, str]:
    """Jean-style rows derived from calibrated full-body display (same pipeline)."""
    jean: dict[str, str] = {}
    mapping = (
        ("Crotch Depth", "Crotch Depth"),
        ("Waist", "Waist"),
        ("Full Hip", "Hip"),
        ("Thigh", "Thigh"),
        ("Knee", "Knee"),
        ("Calf", "Calf"),
        ("Ankle", "Ankle"),
        ("Outseam", "Full Length"),
    )
    for src, dst in mapping:
        if src in display:
            jean[dst] = display[src]
    return jean


def _merged_to_display_strings(merged: dict[str, float], mults: dict[str, float]) -> dict[str, str]:
    display: dict[str, str] = {}
    order = [
        "Bust",
        "Underbust",
        "Waist",
        "High Hip",
        "Full Hip",
        "Shoulder Width",
        "Neck",
        "Arm Length",
        "Bicep",
        "Wrist",
        "Thigh",
        "Knee",
        "Calf",
        "Ankle",
        "Inseam",
        "Outseam",
        "Torso Length",
        "Crotch Depth",
        "Total Height",
    ]
    for label in order:
        if label not in merged:
            continue
        raw_in = cm_to_inches(merged[label])
        m = mults.get(label, 1.0)
        adj = raw_in * m
        if label == "Total Height":
            display[label] = format_inches(adj)
        else:
            display[label] = _fmt_inch(adj)
    return display


# MediaPipe Pose landmark indices
LM = mp.solutions.pose.PoseLandmark


@dataclass
class ViewConfig:
    name: str
    distance_cm: float  # subject to camera


def _focal_px(image_width: int, hfov_deg: float) -> float:
    """Horizontal focal length in pixels (pinhole)."""
    half = math.radians(hfov_deg / 2.0)
    return (image_width / 2.0) / math.tan(half)


def _cm_per_px_at_depth(image_width: int, distance_cm: float, hfov_deg: float) -> float:
    fx = _focal_px(image_width, hfov_deg)
    return (distance_cm / fx) * _cm_px_calibration_mult()


def _ellipse_perimeter_cm(a_cm: float, b_cm: float) -> float:
    """Ramanujan-ish approximation; a,b are semi-axes (half-width, half-depth)."""
    if a_cm <= 0 or b_cm <= 0:
        return 0.0
    h = ((a_cm - b_cm) ** 2) / ((a_cm + b_cm) ** 2)
    return math.pi * (a_cm + b_cm) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))


def _run_pose(rgb: np.ndarray) -> Any:
    pose = mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.3,
    )
    try:
        return pose.process(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    finally:
        pose.close()


def _y_norm(lms, idx: int, ih: int) -> float | None:
    if idx >= len(lms.landmark):
        return None
    lm = lms.landmark[idx]
    if lm.visibility < 0.3:
        return None
    return lm.y * ih


def _x_span_at_y(
    mask: np.ndarray,
    y: int,
    margin: int = 2,
) -> int | None:
    """Horizontal span of foreground at row y."""
    ih, iw = mask.shape[:2]
    y = int(np.clip(y, margin, ih - 1 - margin))
    row = mask[y, :] > 127
    xs = np.where(row)[0]
    if xs.size < 5:
        return None
    return int(xs.max() - xs.min())


def _half_width_cm_at_y(
    image_bgr: np.ndarray,
    segmentation_mask: np.ndarray,
    y_px: float,
    cm_per_px: float,
) -> float | None:
    ih, _ = image_bgr.shape[:2]
    span = _x_span_at_y(segmentation_mask, int(y_px))
    if span is None:
        return None
    return (span / 2.0) * cm_per_px


def _vertical_span_cm(
    mask: np.ndarray,
    cm_per_px_y: float,
) -> tuple[float, int, int] | None:
    """Returns (height_cm, y_top, y_bottom) from segmentation."""
    ys = np.where(mask.max(axis=1) > 127)[0]
    if ys.size < 10:
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    h_px = y1 - y0
    return h_px * cm_per_px_y, y0, y1


def _process_view(
    image_bgr: np.ndarray,
    distance_cm: float,
    hfov_deg: float,
) -> dict[str, Any]:
    h, w = image_bgr.shape[:2]
    cm_px = _cm_per_px_at_depth(w, distance_cm, hfov_deg)
    res = _run_pose(image_bgr)
    if not res.pose_landmarks:
        return {"ok": False, "error": "No pose detected"}
    lms = res.pose_landmarks
    mask = None
    if res.segmentation_mask is not None:
        m = (res.segmentation_mask * 255).astype(np.uint8)
        mask = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)

    if mask is None:
        return {"ok": False, "error": "No segmentation mask"}

    vert = _vertical_span_cm(mask, cm_px)
    if vert is None:
        return {"ok": False, "error": "Could not estimate body height from mask"}
    height_cm, y_top, y_bot = vert
    ih = h

    def rel_y(t: float) -> float:
        return y_top + t * (y_bot - y_top)

    # Landmark-based vertical anchors (fallback / blend)
    nose_y = _y_norm(lms, LM.NOSE.value, ih)
    sh_l = _y_norm(lms, LM.LEFT_SHOULDER.value, ih)
    sh_r = _y_norm(lms, LM.RIGHT_SHOULDER.value, ih)
    hip_l = _y_norm(lms, LM.LEFT_HIP.value, ih)
    hip_r = _y_norm(lms, LM.RIGHT_HIP.value, ih)
    ankle_l = _y_norm(lms, LM.LEFT_ANKLE.value, ih)
    ankle_r = _y_norm(lms, LM.RIGHT_ANKLE.value, ih)

    shoulder_y = None
    if sh_l is not None and sh_r is not None:
        shoulder_y = (sh_l + sh_r) / 2
    hip_y = None
    if hip_l is not None and hip_r is not None:
        hip_y = (hip_l + hip_r) / 2
    ankle_y = None
    if ankle_l is not None and ankle_r is not None:
        ankle_y = (ankle_l + ankle_r) / 2

    # Boost height when mask span is shorter than nose→ankle (crown + floor fudge).
    if nose_y is not None and ankle_y is not None:
        landmark_h_cm = abs(ankle_y - nose_y) * cm_px * 1.12
        height_cm = max(height_cm, landmark_h_cm)

    # Levels as fraction from top of mask to bottom (feet)
    levels: dict[str, float] = {
        "neck": 0.08,
        "bust": 0.22,
        "underbust": 0.30,
        "waist": 0.45,
        "high_hip": 0.52,
        "full_hip": 0.56,
        "thigh": 0.62,
        "knee": 0.72,
        "calf": 0.82,
        "ankle": 0.93,
    }
    if shoulder_y is not None:
        levels["neck"] = max(0.04, (shoulder_y - y_top) / (y_bot - y_top) - 0.06)
        levels["bust"] = min(0.35, (shoulder_y - y_top) / (y_bot - y_top) + 0.12)
    if hip_y is not None:
        levels["waist"] = max(0.35, (hip_y - y_top) / (y_bot - y_top) - 0.10)
        levels["high_hip"] = (hip_y - y_top) / (y_bot - y_top) - 0.03
        levels["full_hip"] = (hip_y - y_top) / (y_bot - y_top) + 0.02
    if ankle_y is not None:
        levels["ankle"] = max(0.88, min(0.97, (ankle_y - y_top) / (y_bot - y_top)))

    half_widths: dict[str, float] = {}
    for name, t in levels.items():
        yp = rel_y(t)
        hw = _half_width_cm_at_y(image_bgr, mask, yp, cm_px)
        if hw is not None and hw > 0:
            half_widths[name] = hw

    # Shoulder width: landmarks often read a bit narrow vs tape "edge to edge"; blend with mask span.
    shoulder_width_cm = None
    sl = lms.landmark[LM.LEFT_SHOULDER.value]
    sr = lms.landmark[LM.RIGHT_SHOULDER.value]
    lm_sw = None
    if sl.visibility > 0.5 and sr.visibility > 0.5:
        lm_sw = abs(sl.x - sr.x) * w * cm_px
    mask_sw = None
    if shoulder_y is not None:
        sp = _x_span_at_y(mask, int(shoulder_y))
        if sp is not None and sp > 10:
            # A-pose silhouette is wider than bone-to-bone; down-weight vs full span.
            mask_sw = sp * cm_px * 0.62
    if lm_sw is not None and mask_sw is not None:
        shoulder_width_cm = max(lm_sw, mask_sw)
    elif lm_sw is not None:
        shoulder_width_cm = lm_sw
    else:
        shoulder_width_cm = mask_sw

    # Arm length (shoulder to wrist) — average both sides
    arm_lengths = []
    for side_sh, side_el, side_wr in (
        (LM.LEFT_SHOULDER, LM.LEFT_ELBOW, LM.LEFT_WRIST),
        (LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, LM.RIGHT_WRIST),
    ):
        a, b, c = lms.landmark[side_sh.value], lms.landmark[side_el.value], lms.landmark[side_wr.value]
        if min(a.visibility, b.visibility, c.visibility) < 0.4:
            continue
        d1 = math.hypot((a.x - b.x) * w, (a.y - b.y) * h)
        d2 = math.hypot((b.x - c.x) * w, (b.y - c.y) * h)
        arm_lengths.append((d1 + d2) * cm_px)
    arm_length_cm = float(np.mean(arm_lengths)) if arm_lengths else None

    # Bicep / forearm / wrist from short segments at elbow level (very rough)
    def _seg_len(i, j):
        p, q = lms.landmark[i], lms.landmark[j]
        if p.visibility < 0.4 or q.visibility < 0.4:
            return None
        return math.hypot((p.x - q.x) * w, (p.y - q.y) * h) * cm_px

    bicep_cm = None
    upper_lens = []
    for sh, el in (
        (LM.LEFT_SHOULDER, LM.LEFT_ELBOW),
        (LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW),
    ):
        L = _seg_len(sh.value, el.value)
        if L:
            upper_lens.append(L)
    if upper_lens:
        u = float(np.mean(upper_lens))
        bicep_cm = math.pi * u * 0.22
    wrist_cm = None
    lw = _seg_len(LM.LEFT_WRIST.value, LM.LEFT_PINKY.value)
    rw = _seg_len(LM.RIGHT_WRIST.value, LM.RIGHT_PINKY.value)
    if lw and rw:
        wrist_cm = math.pi * ((lw + rw) / 4) * 0.9

    # Inseam: inner hip to ankle
    inseam_cm = None
    for hip, ank in ((LM.LEFT_HIP, LM.LEFT_ANKLE), (LM.RIGHT_HIP, LM.RIGHT_ANKLE)):
        hi, an = lms.landmark[hip.value], lms.landmark[ank.value]
        if hi.visibility > 0.4 and an.visibility > 0.4:
            inseam_cm = math.hypot((hi.x - an.x) * w, (hi.y - an.y) * h) * cm_px
            break

    # Outseam: waist level hip to ankle (side of leg)
    outseam_cm = None
    if hip_y is not None and ankle_y is not None:
        outseam_cm = abs(ankle_y - hip_y) * cm_px

    # Torso: shoulder to waist vertical
    torso_cm = None
    if shoulder_y is not None:
        wy = rel_y(levels["waist"])
        torso_cm = abs(wy - shoulder_y) * cm_px

    # Neck: estimate from head width at nose level
    neck_cm = None
    if nose_y is not None:
        hw_n = _half_width_cm_at_y(image_bgr, mask, nose_y, cm_px)
        if hw_n:
            neck_cm = _ellipse_perimeter_cm(hw_n * 0.55, hw_n * 0.45)

    # Crotch depth (rise): waist to hip vertical (jean-style)
    crotch_depth_cm = None
    if hip_y is not None:
        wy = rel_y(levels["waist"])
        crotch_depth_cm = abs(hip_y - wy) * cm_px

    return {
        "ok": True,
        "height_cm": height_cm,
        "half_widths": half_widths,
        "shoulder_width_cm": shoulder_width_cm,
        "arm_length_cm": arm_length_cm,
        "bicep_cm": bicep_cm,
        "wrist_cm": wrist_cm,
        "inseam_cm": inseam_cm,
        "outseam_cm": outseam_cm,
        "torso_cm": torso_cm,
        "neck_cm": neck_cm,
        "crotch_depth_cm": crotch_depth_cm,
        "levels": levels,
    }


def merge_measurements(
    front: dict,
    back: dict,
    left: dict,
    right: dict,
) -> dict[str, float]:
    """Combine front/back (width) with left/right (depth) for circumferences."""
    out: dict[str, float] = {}

    def avg_hw(side_views: list[dict], key: str) -> float | None:
        vals = []
        for v in side_views:
            if v.get("ok") and key in v.get("half_widths", {}):
                vals.append(v["half_widths"][key])
        if not vals:
            return None
        return float(np.mean(vals))

    # Front + back average for "width"; left + right for "depth"
    keys = ["neck", "bust", "underbust", "waist", "high_hip", "full_hip", "thigh", "knee", "calf", "ankle"]
    circum_keys = {
        "neck": "Neck",
        "bust": "Bust",
        "underbust": "Underbust",
        "waist": "Waist",
        "high_hip": "High Hip",
        "full_hip": "Full Hip",
        "thigh": "Thigh",
        "knee": "Knee",
        "calf": "Calf",
        "ankle": "Ankle",
    }
    for k, label in circum_keys.items():
        w = avg_hw([front, back], k)
        d = avg_hw([left, right], k)
        if w and d:
            out[label] = _ellipse_perimeter_cm(w, d)
        elif w:
            out[label] = math.pi * w  # half circle fallback
        elif d:
            out[label] = math.pi * d

    if "Neck" not in out or out.get("Neck", 0) < 12:
        for v in (front, back):
            nc = v.get("neck_cm") if v.get("ok") else None
            if nc and nc > 15:
                out["Neck"] = nc
                break

    # Height: prefer average of all successful views
    heights = [v["height_cm"] for v in (front, back, left, right) if v.get("ok") and v.get("height_cm")]
    if heights:
        out["Total Height"] = float(np.median(heights))

    # Shoulder: from front (or average front/back)
    sw = []
    for v in (front, back):
        if v.get("ok") and v.get("shoulder_width_cm"):
            sw.append(v["shoulder_width_cm"])
    if sw:
        out["Shoulder Width"] = float(np.mean(sw))

    # Arm length — prefer front
    for v in (front, back, left, right):
        if v.get("ok") and v.get("arm_length_cm"):
            out["Arm Length"] = v["arm_length_cm"]
            break

    # Bicep, wrist
    for v in (front, back):
        if v.get("ok") and v.get("bicep_cm"):
            out["Bicep"] = v["bicep_cm"]
            break
    for v in (front, back):
        if v.get("ok") and v.get("wrist_cm"):
            out["Wrist"] = v["wrist_cm"]
            break

    # Leg lengths — prefer side views for inseam/outseam verticals
    for v in (left, right, front):
        if v.get("ok") and v.get("inseam_cm"):
            out["Inseam"] = v["inseam_cm"]
            break
    for v in (left, right, front):
        if v.get("ok") and v.get("outseam_cm"):
            out["Outseam"] = v["outseam_cm"]
            break

    for v in (front, back):
        if v.get("ok") and v.get("torso_cm"):
            out["Torso Length"] = v["torso_cm"]
            break

    for v in (front, back):
        if v.get("ok") and v.get("crotch_depth_cm"):
            out["Crotch Depth"] = v["crotch_depth_cm"]
            break

    return out


def cm_to_inches(cm: float) -> float:
    return cm / 2.54


def format_inches(v: float) -> str:
    """Height in feet/inches; fractional inches follow BODY_MEASURE_INCH_DECIMALS."""
    d = _inch_decimal_places()
    if v > 48:
        feet = int(v // 12)
        inch = v - feet * 12
        if d == 0:
            inch = round(inch)
            if inch >= 12:
                feet += 1
                inch = 0
            return f"{feet}'{inch}\""
        return f"{feet}'{inch:.{d}f}\""
    return _fmt_inch(v)


def run_all(
    images: dict[str, bytes],
    hfov_deg: float | None = None,
) -> tuple[dict[str, str], list[str]]:
    """
    images keys: front, back, left, right
    distances: front/back 142cm, left/right 130cm
    """
    if hfov_deg is None:
        hfov_deg = _default_hfov()
    dist = {"front": 142.0, "back": 142.0, "left": 130.0, "right": 130.0}
    processed: dict[str, dict] = {}
    errors: list[str] = []

    for name in ("front", "back", "left", "right"):
        raw = images.get(name)
        if not raw:
            errors.append(f"Missing image: {name}")
            processed[name] = {"ok": False, "error": "missing"}
            continue
        im = decode_image_bytes_to_bgr(raw)
        if im is None:
            errors.append(f"Could not decode: {name}")
            processed[name] = {"ok": False, "error": "decode"}
            continue
        processed[name] = _process_view(im, dist[name], hfov_deg)
        if not processed[name].get("ok"):
            errors.append(f"{name}: {processed[name].get('error', 'failed')}")

    merged = merge_measurements(
        processed["front"],
        processed["back"],
        processed["left"],
        processed["right"],
    )

    mults = effective_mults()
    display = _merged_to_display_strings(merged, mults)
    return display, errors


def compute_raw_inches(
    images: dict[str, bytes],
    hfov_deg: float | None = None,
) -> tuple[dict[str, float], list[str]]:
    """Raw model inches (no tape calibration) — for POST /api/train."""
    if hfov_deg is None:
        hfov_deg = _default_hfov()
    dist = {"front": 142.0, "back": 142.0, "left": 130.0, "right": 130.0}
    processed: dict[str, dict] = {}
    errors: list[str] = []

    for name in ("front", "back", "left", "right"):
        raw = images.get(name)
        if not raw:
            errors.append(f"Missing image: {name}")
            processed[name] = {"ok": False, "error": "missing"}
            continue
        im = decode_image_bytes_to_bgr(raw)
        if im is None:
            errors.append(f"Could not decode: {name}")
            processed[name] = {"ok": False, "error": "decode"}
            continue
        processed[name] = _process_view(im, dist[name], hfov_deg)
        if not processed[name].get("ok"):
            errors.append(f"{name}: {processed[name].get('error', 'failed')}")

    merged = merge_measurements(
        processed["front"],
        processed["back"],
        processed["left"],
        processed["right"],
    )
    return {k: cm_to_inches(v) for k, v in merged.items()}, errors
