#!/usr/bin/env python
"""LiTraQ movement analysis for linear-track DeepLabCut data.

Workflow:
  1. Calibrate each video once with arena corners and a 9 cm reference block.
  2. Analyze DeepLabCut bodycenter tracks in calibrated cm coordinates.

The calibration maps DLC/video pixel coordinates to a rectified arena plane, then
uses the reference block in that rectified plane to convert pixels to cm.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd


Point = tuple[float, float]
DEFAULT_PLOT_DPI = 300
DEFAULT_KNOWN_PX_PER_CM = 19.0
DEFAULT_EDGE_WIDTH_CM = 6.5
# A bodycenter can stop just outside the nominal edge region even though the
# mouse has visibly completed the end-to-end transit.  Use a small tolerance
# only for transit segmentation; edge occupancy summaries keep the configured
# edge width unchanged.
DEFAULT_TRANSIT_EDGE_TOLERANCE_CM = 0.5
# Calibrated against 20 visually reviewed transit clips (9 pass, 11 exclude).
# Processed tracks contain frequent single-frame coordinate duplicates, so the
# per-frame low-speed fraction is not a reliable rejection feature.  A stricter
# geometry threshold plus a short *consecutive* low-speed run separates the QC
# labels without treating those isolated duplicate frames as pauses.
DEFAULT_STRAIGHT_PATH_EFFICIENCY_THRESHOLD = 0.95
DEFAULT_STRAIGHT_STOP_MIN_SEC = 0.10
DEFAULT_STRAIGHT_LOW_SPEED_FRACTION_THRESHOLD = 1.0
# A wall-oriented posture can look like a continuous straight transit when only
# the bodycenter marker is used.  The posture check uses the calibrated nose and
# bodycenter positions, with tailbase required as a third validity anchor.  The
# defaults are deliberately high-specificity: more than half of the transit
# must show the nose at a long wall while the bodycenter remains away from it.
DEFAULT_WALL_POSTURE_CHECK = True
DEFAULT_POSTURE_WALL_DISTANCE_CM = 1.0
DEFAULT_POSTURE_CENTER_AWAY_CM = 1.5
DEFAULT_POSTURE_MIN_FRACTION = 0.60
DEFAULT_POSTURE_BORDERLINE_FRACTION = 0.50
DEFAULT_POSTURE_MIN_VALID_FRACTION = 0.80
# Wall orientation alone also occurs during uninterrupted wall-following.
# Auto-rejection therefore requires a short low-speed overlap, while sustained
# wall posture without that overlap remains available as a borderline QC flag.
DEFAULT_POSTURE_INTERRUPTION_MAX_SPEED_CM_S = 5.0
DEFAULT_POSTURE_INTERRUPTION_MIN_SEC = 0.10
DEFAULT_FRAME_COUNT_TOLERANCE = 1
EDGE_REGION_LABELS = ("left_edge", "center", "right_edge")
EDGE_AREA_LABELS = ("left_edge", "right_edge")


@dataclass
class Calibration:
    path: Path
    homography: np.ndarray
    rectified_size_px: tuple[float, float]
    reference_length_cm: float
    px_per_cm: float
    cm_per_px: float
    arena_corners_px: np.ndarray
    reference_segment_px: np.ndarray | None
    crop_from_reference_image_px: dict | None
    video_size_px: tuple[int, int] | None


def parse_point_list(text: str, expected: int) -> np.ndarray:
    """Parse 'x,y;x,y;...' into an Nx2 float array."""
    try:
        pts = []
        for item in text.split(";"):
            x_text, y_text = item.split(",")
            pts.append((float(x_text), float(y_text)))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "points must look like 'x1,y1;x2,y2;...'"
        ) from exc
    if len(pts) != expected:
        raise argparse.ArgumentTypeError(f"expected {expected} points, got {len(pts)}")
    return np.asarray(pts, dtype=np.float32)


def order_arena_corners(points: np.ndarray) -> np.ndarray:
    """Return points in TL,TR,BR,BL order from four clicked arena corners."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("arena corners must be a 4x2 array")
    ordered = np.empty((4, 2), dtype=np.float32)
    sums = pts[:, 0] + pts[:, 1]
    diffs = pts[:, 0] - pts[:, 1]
    ordered[0] = pts[np.argmin(sums)]  # top-left
    ordered[2] = pts[np.argmax(sums)]  # bottom-right
    ordered[1] = pts[np.argmax(diffs)]  # top-right
    ordered[3] = pts[np.argmin(diffs)]  # bottom-left
    if len({tuple(point) for point in ordered}) != 4:
        raise ValueError("could not order arena corners; please click four distinct corners")
    return ordered


def polygon_signed_area(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def is_self_crossed_quad(points: np.ndarray) -> bool:
    pts = np.asarray(points, dtype=float)

    def ccw(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
        return bool((c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0]))

    def intersects(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
        return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)

    return intersects(pts[0], pts[1], pts[2], pts[3]) or intersects(
        pts[1], pts[2], pts[3], pts[0]
    )


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return image


def video_metadata(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {path}")
    meta = {
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return meta


def validate_tracking_video_frame_count(
    tracking_frame_count: int,
    video_frame_count: int,
    fps: float,
    tolerance_frames: int = DEFAULT_FRAME_COUNT_TOLERANCE,
) -> dict[str, int | float | bool]:
    """Validate that a DLC table and its source video cover the same frames."""
    tracking_frame_count = int(tracking_frame_count)
    video_frame_count = int(video_frame_count)
    tolerance_frames = int(tolerance_frames)
    if tracking_frame_count <= 0:
        raise ValueError("DLC tracking table contains no frames")
    if video_frame_count <= 0:
        raise ValueError("could not determine the processed-video frame count")
    if fps <= 0:
        raise ValueError("fps must be > 0 for frame-count validation")
    if tolerance_frames < 0:
        raise ValueError("frame-count tolerance must be >= 0")

    difference = tracking_frame_count - video_frame_count
    if abs(difference) > tolerance_frames:
        relation = "shorter" if difference < 0 else "longer"
        difference_seconds = abs(difference) / float(fps)
        raise ValueError(
            "DLC/video frame-count mismatch: "
            f"tracking={tracking_frame_count}, video={video_frame_count}; "
            f"tracking is {abs(difference)} frame(s) ({difference_seconds:.3f} s) {relation}. "
            "Select the matching files or rerun DLC tracking."
        )

    return {
        "tracking_frame_count": tracking_frame_count,
        "video_frame_count": video_frame_count,
        "difference_frames": difference,
        "difference_seconds": difference / float(fps),
        "tolerance_frames": tolerance_frames,
        "exact_match": difference == 0,
    }


def read_video_frame(path: Path, frame_index: int = 0) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"could not read frame {frame_index} from {path}")
    return frame


def load_reference_frame(
    image_arg: str | None,
    reference_video_arg: str | None,
    reference_frame: int,
) -> tuple[np.ndarray, dict]:
    """Load the frame used to measure the known 9 cm reference block."""
    if image_arg and reference_video_arg:
        raise ValueError("provide either --image or --reference-video, not both")
    if image_arg:
        image_path = Path(image_arg)
        return read_image(image_path), {"type": "image", "image": str(image_path)}
    if reference_video_arg:
        video_path = Path(reference_video_arg)
        return read_video_frame(video_path, reference_frame), {
            "type": "video_frame",
            "video": str(video_path),
            "frame": int(reference_frame),
        }
    raise ValueError("provide --image or --reference-video")


def load_reference_frame_from_args_or_payload(
    args: argparse.Namespace,
    payload: dict | None = None,
) -> tuple[np.ndarray, dict]:
    payload = payload or {}
    if getattr(args, "image", None) or getattr(args, "reference_video", None):
        return load_reference_frame(
            getattr(args, "image", None),
            getattr(args, "reference_video", None),
            getattr(args, "reference_frame", 0),
        )
    source = payload.get("reference_source") or {}
    if source.get("type") == "image" and source.get("image"):
        return load_reference_frame(source["image"], None, 0)
    if source.get("type") == "video_frame" and source.get("video") is not None:
        return load_reference_frame(None, source["video"], int(source.get("frame", 0)))
    if payload.get("source_image"):
        return load_reference_frame(payload["source_image"], None, 0)
    raise ValueError("provide --image, --reference-video, or --calibration with reference_source")


def estimate_crop(reference_image: np.ndarray, video_frame: np.ndarray) -> dict:
    """Estimate the video crop inside a larger reference image by template matching."""
    ref_h, ref_w = reference_image.shape[:2]
    vid_h, vid_w = video_frame.shape[:2]
    if (ref_w, ref_h) == (vid_w, vid_h):
        return {
            "x": 0,
            "y": 0,
            "width": vid_w,
            "height": vid_h,
            "method": "same_size",
            "match_score": 1.0,
        }
    if vid_w > ref_w or vid_h > ref_h:
        raise ValueError(
            "video frame is larger than the reference image; provide --crop manually"
        )
    ref_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    vid_gray = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(ref_gray, vid_gray, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(result)
    x, y = loc
    return {
        "x": int(x),
        "y": int(y),
        "width": int(vid_w),
        "height": int(vid_h),
        "method": "cv2.matchTemplate_TM_CCOEFF_NORMED",
        "match_score": float(score),
    }


def crop_image(image: np.ndarray, crop: dict | None) -> np.ndarray:
    if not crop:
        return image
    x = int(crop["x"])
    y = int(crop["y"])
    width = int(crop["width"])
    height = int(crop["height"])
    return image[y : y + height, x : x + width]


def get_screen_size() -> tuple[int, int] | None:
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        width = int(root.winfo_screenwidth())
        height = int(root.winfo_screenheight())
        root.destroy()
        return width, height
    except Exception:
        return None


def centered_display_scale(image_bgr: np.ndarray) -> tuple[float, tuple[int, int] | None]:
    screen = get_screen_size()
    if not screen:
        return 1.0, None
    screen_w, screen_h = screen
    image_h, image_w = image_bgr.shape[:2]
    max_w = int(screen_w * 0.92)
    max_h = int(screen_h * 0.82)
    scale = min(1.0, max_w / image_w, max_h / image_h)
    display_w = max(1, int(round(image_w * scale)))
    display_h = max(1, int(round(image_h * scale)))
    x = max(0, int((screen_w - display_w) / 2))
    y = max(0, int((screen_h - display_h) / 2))
    return scale, (x, y)


def roi_from_points(points: np.ndarray, image_shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
    pts = np.asarray(points, dtype=float)
    if pts.shape != (2, 2):
        raise ValueError("ROI point selection must contain two points")
    image_h, image_w = image_shape[:2]
    x1, y1 = np.floor(np.min(pts, axis=0)).astype(int)
    x2, y2 = np.ceil(np.max(pts, axis=0)).astype(int)
    x1 = max(0, min(image_w - 1, x1))
    y1 = max(0, min(image_h - 1, y1))
    x2 = max(0, min(image_w, x2))
    y2 = max(0, min(image_h, y2))
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        raise ValueError("ROI has zero width or height")
    return x1, y1, width, height


def compute_rectification(arena_corners_px: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """Build a homography from clicked TL,TR,BR,BL corners to a rectangle."""
    pts = np.asarray(arena_corners_px, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("arena_corners_px must be 4x2 in TL,TR,BR,BL order")
    if is_self_crossed_quad(pts):
        raise ValueError(
            "arena corners form a self-crossed quadrilateral; expected TL,TR,BR,BL order"
        )
    if abs(polygon_signed_area(pts)) < 1.0:
        raise ValueError("arena corner geometry is degenerate")
    top = np.linalg.norm(pts[1] - pts[0])
    bottom = np.linalg.norm(pts[2] - pts[3])
    right = np.linalg.norm(pts[2] - pts[1])
    left = np.linalg.norm(pts[3] - pts[0])
    width = float((top + bottom) / 2.0)
    height = float((left + right) / 2.0)
    if width <= 0 or height <= 0:
        raise ValueError("arena corner geometry has zero width or height")
    dst = np.asarray(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
    )
    homography = cv2.getPerspectiveTransform(pts, dst)
    return homography, (width, height)


def transform_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, homography)
    return transformed.reshape(-1, 2)


def build_calibration(
    arena_corners_px: np.ndarray,
    reference_segment_px: np.ndarray,
    reference_length_cm: float,
) -> tuple[np.ndarray, tuple[float, float], float, float]:
    homography, rectified_size_px = compute_rectification(arena_corners_px)
    ref_rectified = transform_points(reference_segment_px, homography)
    ref_length_px = float(np.linalg.norm(ref_rectified[1] - ref_rectified[0]))
    if ref_length_px <= 0:
        raise ValueError("reference segment length is zero after rectification")
    px_per_cm = ref_length_px / reference_length_cm
    cm_per_px = reference_length_cm / ref_length_px
    return homography, rectified_size_px, px_per_cm, cm_per_px


def draw_points_overlay(image_bgr: np.ndarray, points: Sequence[Point], scale: float) -> np.ndarray:
    if scale != 1.0:
        image = cv2.resize(
            image_bgr,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        )
    else:
        image = image_bgr.copy()
    display_points = [
        (int(round(point[0] * scale)), int(round(point[1] * scale))) for point in points
    ]
    for idx, point in enumerate(display_points, start=1):
        cv2.circle(image, point, 5, (0, 255, 255), -1)
        cv2.putText(
            image,
            str(idx),
            (point[0] + 7, point[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    if len(display_points) > 1:
        for a, b in zip(display_points[:-1], display_points[1:]):
            cv2.line(image, a, b, (0, 255, 255), 1)
    return image


def ginput_points(image_bgr: np.ndarray, title: str, n: int) -> np.ndarray:
    """Collect image points with a centered OpenCV click window."""
    points: list[Point] = []
    scale, window_position = centered_display_scale(image_bgr)
    window_name = f"LiTraQ calibration - {title}"

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < n:
            points.append((x / scale, y / scale))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)
    if window_position:
        cv2.moveWindow(window_name, window_position[0], window_position[1])
    print(title)
    print("  Left click: add point | Right click/u/backspace: undo | r: reset | q/esc: cancel")
    try:
        while len(points) < n:
            display = draw_points_overlay(image_bgr, points, scale)
            cv2.imshow(window_name, display)
            if window_position:
                cv2.moveWindow(window_name, window_position[0], window_position[1])
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                raise RuntimeError("point selection cancelled")
            if key in (8, ord("u")) and points:
                points.pop()
            elif key == ord("r"):
                points.clear()
        display = draw_points_overlay(image_bgr, points, scale)
        cv2.imshow(window_name, display)
        cv2.waitKey(150)
    finally:
        cv2.destroyWindow(window_name)
    return np.asarray(points, dtype=np.float32)


def save_calibration_preview(
    image_bgr: np.ndarray,
    arena_corners_px: np.ndarray,
    reference_segment_px: np.ndarray | None,
    out_path: Path,
) -> None:
    preview = image_bgr.copy()
    corners = np.asarray(arena_corners_px, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(preview, [corners], isClosed=True, color=(0, 255, 255), thickness=2)
    if reference_segment_px is not None:
        ref = np.asarray(reference_segment_px, dtype=np.int32)
        cv2.line(preview, tuple(ref[0]), tuple(ref[1]), color=(0, 255, 0), thickness=3)
    for i, point in enumerate(corners.reshape(-1, 2), start=1):
        cv2.circle(preview, tuple(point), 5, (0, 255, 255), -1)
        cv2.putText(
            preview,
            str(i),
            tuple(point + np.array([6, -6])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), preview)


def draw_reference_detection_preview(
    image_bgr: np.ndarray,
    roi: tuple[int, int, int, int],
    segment: np.ndarray,
    bbox: tuple[int, int, int, int],
    out_path: Path,
) -> None:
    preview = image_bgr.copy()
    x, y, w, h = roi
    bx, by, bw, bh = bbox
    cv2.rectangle(preview, (x, y), (x + w, y + h), (255, 180, 0), 2)
    cv2.rectangle(preview, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)
    p0 = tuple(np.asarray(segment[0], dtype=int))
    p1 = tuple(np.asarray(segment[1], dtype=int))
    cv2.line(preview, p0, p1, (0, 255, 0), 3)
    for label, point in (("A", p0), ("B", p1)):
        cv2.circle(preview, point, 5, (0, 255, 0), -1)
        cv2.putText(
            preview,
            label,
            (point[0] + 7, point[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), preview)


def detect_reference_block(
    image_bgr: np.ndarray,
    roi: tuple[int, int, int, int],
    min_aspect_ratio: float = 1.8,
) -> dict:
    """Detect a bright horizontal reference block in a user-specified ROI."""
    image_h, image_w = image_bgr.shape[:2]
    x, y, w, h = roi
    x = max(0, min(image_w - 1, int(x)))
    y = max(0, min(image_h - 1, int(y)))
    w = max(1, min(image_w - x, int(w)))
    h = max(1, min(image_h - y, int(h)))
    roi_image = image_bgr[y : y + h, x : x + w]
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
    otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    bright = cv2.inRange(gray, 130, 255)
    bright_low_saturation = cv2.inRange(
        hsv,
        np.asarray([0, 0, 105], dtype=np.uint8),
        np.asarray([179, 130, 255], dtype=np.uint8),
    )
    masks = [
        ("otsu", otsu),
        ("bright_gray_130", bright),
        ("bright_low_saturation", bright_low_saturation),
    ]

    candidates = []
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    roi_area = float(w * h)
    for method, mask in masks:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < max(100.0, roi_area * 0.01):
                continue
            bx, by, bw, bh = cv2.boundingRect(contour)
            aspect = max(bw / max(1, bh), bh / max(1, bw))
            if aspect < min_aspect_ratio:
                continue
            extent = area / float(max(1, bw * bh))
            rect = cv2.minAreaRect(contour)
            rect_w, rect_h = rect[1]
            long_side = max(rect_w, rect_h)
            short_side = min(rect_w, rect_h)
            if long_side <= 0 or short_side <= 0:
                continue
            box = cv2.boxPoints(rect)
            edges = [(box[i], box[(i + 1) % 4]) for i in range(4)]
            first, second = max(edges, key=lambda edge: np.linalg.norm(edge[1] - edge[0]))
            segment = np.asarray(
                [[first[0] + x, first[1] + y], [second[0] + x, second[1] + y]],
                dtype=np.float32,
            )
            if segment[0, 0] > segment[1, 0]:
                segment = segment[::-1]
            dx = float(second[0] - first[0])
            dy = float(second[1] - first[1])
            angle_deg = abs(math.degrees(math.atan2(dy, dx)))
            angle_deg = min(angle_deg, abs(180.0 - angle_deg))
            horizontal_score = max(0.2, 1.0 - angle_deg / 45.0)
            score = area * aspect * max(0.25, extent) * horizontal_score
            candidates.append(
                {
                    "method": method,
                    "score": score,
                    "area": area,
                    "extent": float(extent),
                    "bbox_local": (int(bx), int(by), int(bw), int(bh)),
                    "bbox_px": (int(x + bx), int(y + by), int(bw), int(bh)),
                    "min_area_rect_long_px": float(long_side),
                    "min_area_rect_short_px": float(short_side),
                    "angle_deg": float(angle_deg),
                    "segment_px": segment.tolist(),
                }
            )
    if not candidates:
        raise ValueError("could not detect a horizontal reference block in the ROI")
    best = max(candidates, key=lambda item: item["score"])
    best["roi_px"] = (x, y, w, h)
    return best


def write_calibration(
    out_path: Path,
    reference_source: dict,
    video_path: Path | None,
    image_size: tuple[int, int],
    video_size: tuple[int, int] | None,
    crop: dict | None,
    arena_corners_px: np.ndarray,
    reference_segment_px: np.ndarray | None,
    reference_length_cm: float,
    arena_corners_clicked_px: np.ndarray | None = None,
    arena_source: dict | None = None,
    reference_detection: dict | None = None,
    known_px_per_cm: float | None = None,
) -> dict:
    if known_px_per_cm is not None:
        if known_px_per_cm <= 0:
            raise ValueError("known_px_per_cm must be > 0")
        homography, rectified_size_px = compute_rectification(arena_corners_px)
        px_per_cm = float(known_px_per_cm)
        cm_per_px = 1.0 / px_per_cm
        scale_source = "known_px_per_cm"
    else:
        if reference_segment_px is None:
            raise ValueError("reference_segment_px is required unless known_px_per_cm is provided")
        homography, rectified_size_px, px_per_cm, cm_per_px = build_calibration(
            arena_corners_px, reference_segment_px, reference_length_cm
        )
        scale_source = "manual_reference_segment"
    payload = {
        "version": 1,
        "coordinate_frame": "DLC/processed-video pixels",
        "notes": [
            "arena_corners_px must be in top-left, top-right, bottom-right, bottom-left order",
            "reference_segment_px is the clicked known-length block in the same coordinate frame",
        ],
        "source_image": reference_source.get("image"),
        "reference_source": reference_source,
        "source_image_size_px": list(image_size),
        "video": str(video_path) if video_path else None,
        "video_size_px": list(video_size) if video_size else None,
        "arena_source": arena_source,
        "crop_from_reference_image_px": crop,
        "arena_corners_clicked_px": np.asarray(
            arena_corners_clicked_px if arena_corners_clicked_px is not None else arena_corners_px,
            dtype=float,
        ).tolist(),
        "arena_corners_px": np.asarray(arena_corners_px, dtype=float).tolist(),
        "reference_segment_px": (
            np.asarray(reference_segment_px, dtype=float).tolist()
            if reference_segment_px is not None
            else None
        ),
        "reference_detection": reference_detection,
        "reference_length_cm": float(reference_length_cm),
        "scale_source": scale_source,
        "homography_to_rectified_px": homography.tolist(),
        "rectified_size_px": [rectified_size_px[0], rectified_size_px[1]],
        "px_per_cm_rectified": float(px_per_cm),
        "cm_per_px_rectified": float(cm_per_px),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def load_calibration(path: Path) -> Calibration:
    payload = json.loads(path.read_text(encoding="utf-8"))
    reference_segment = payload.get("reference_segment_px")
    return Calibration(
        path=path,
        homography=np.asarray(payload["homography_to_rectified_px"], dtype=np.float32),
        rectified_size_px=tuple(float(x) for x in payload["rectified_size_px"]),
        reference_length_cm=float(payload["reference_length_cm"]),
        px_per_cm=float(payload["px_per_cm_rectified"]),
        cm_per_px=float(payload["cm_per_px_rectified"]),
        arena_corners_px=np.asarray(payload["arena_corners_px"], dtype=np.float32),
        reference_segment_px=(
            np.asarray(reference_segment, dtype=np.float32)
            if reference_segment is not None
            else None
        ),
        crop_from_reference_image_px=payload.get("crop_from_reference_image_px"),
        video_size_px=tuple(payload["video_size_px"]) if payload.get("video_size_px") else None,
    )


def read_dlc_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".h5":
        return pd.read_hdf(path)
    if suffix == ".csv":
        return pd.read_csv(path, header=[0, 1, 2], index_col=0)
    raise ValueError(f"unsupported tracking file type: {path.suffix}")


def extract_bodypart(df: pd.DataFrame, bodypart: str) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        expected = [f"{bodypart}_x", f"{bodypart}_y"]
        if not all(col in df.columns for col in expected):
            raise ValueError(
                "non-MultiIndex DLC table needs columns like "
                f"{bodypart}_x and {bodypart}_y"
            )
        out = pd.DataFrame({"x": df[expected[0]], "y": df[expected[1]]})
        like_col = f"{bodypart}_likelihood"
        out["likelihood"] = df[like_col] if like_col in df.columns else 1.0
        return out

    tuples = list(df.columns)
    matches: dict[str, pd.Series] = {}
    for col in tuples:
        col_values = [str(x) for x in col]
        if bodypart not in col_values:
            continue
        for coord in ("x", "y", "likelihood"):
            if coord in col_values:
                series = df[col]
                if isinstance(series, pd.DataFrame):
                    series = series.iloc[:, 0]
                matches[coord] = series
                break
    missing = [coord for coord in ("x", "y") if coord not in matches]
    if missing:
        bodyparts = sorted(
            {
                str(level_value)
                for col in tuples
                for level_value in col
                if str(level_value) not in {"x", "y", "likelihood"}
            }
        )
        raise ValueError(
            f"could not find {bodypart} {missing}; available labels include: {bodyparts}"
        )
    out = pd.DataFrame({"x": matches["x"], "y": matches["y"]})
    out["likelihood"] = matches.get("likelihood", 1.0)
    return out.apply(pd.to_numeric, errors="coerce")


def contiguous_true_runs(mask: np.ndarray) -> Iterable[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    n = len(mask)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        start = i
        while i < n and mask[i]:
            i += 1
        yield start, i


def fill_short_false_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    if max_gap <= 0:
        return mask
    out = np.asarray(mask, dtype=bool).copy()
    n = len(out)
    i = 0
    while i < n:
        if out[i]:
            i += 1
            continue
        start = i
        while i < n and not out[i]:
            i += 1
        end = i
        if (
            end - start <= max_gap
            and start > 0
            and end < n
            and out[start - 1]
            and out[end]
        ):
            out[start:end] = True
    return out


def remove_short_or_small_bouts(
    moving: np.ndarray,
    x_cm: np.ndarray,
    y_cm: np.ndarray,
    min_frames: int,
    min_displacement_cm: float,
) -> np.ndarray:
    out = np.asarray(moving, dtype=bool).copy()
    for start, end in list(contiguous_true_runs(out)):
        duration_ok = (end - start) >= min_frames
        # moving[start] describes the step from start - 1 to start, so the bout
        # begins at the preceding position. Using x_cm[start] drops its first step.
        anchor = max(0, start - 1)
        dx = x_cm[end - 1] - x_cm[anchor]
        dy = y_cm[end - 1] - y_cm[anchor]
        displacement = math.hypot(float(dx), float(dy))
        displacement_ok = displacement >= min_displacement_cm
        if not duration_ok or not displacement_ok:
            out[start:end] = False
    return out


def interpolate_short_nan_runs(values: np.ndarray, valid: np.ndarray, max_gap: int) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    invalid = ~np.asarray(valid, dtype=bool)
    n = len(out)
    i = 0
    while i < n:
        if not invalid[i]:
            i += 1
            continue
        start = i
        while i < n and invalid[i]:
            i += 1
        end = i
        if end - start <= max_gap and start > 0 and end < n:
            left = out[start - 1]
            right = out[end]
            if np.isfinite(left) and np.isfinite(right):
                out[start:end] = np.linspace(left, right, end - start + 2)[1:-1]
    return out


def rolling_median_by_valid_segments(values: np.ndarray, valid: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    out = np.asarray(values, dtype=float).copy()
    for start, end in contiguous_true_runs(valid):
        segment = pd.Series(out[start:end])
        smoothed = segment.rolling(window, center=True, min_periods=1).median().to_numpy()
        out[start:end] = smoothed
    return out


def calibrated_positions(
    track: pd.DataFrame,
    calibration: Calibration,
    likelihood_threshold: float,
    arena_margin_cm: float,
) -> pd.DataFrame:
    raw = track[["x", "y"]].to_numpy(dtype=np.float32)
    rectified = transform_points(raw, calibration.homography)
    x_rect = rectified[:, 0]
    y_rect = rectified[:, 1]
    width_px, height_px = calibration.rectified_size_px
    margin_px = arena_margin_cm / calibration.cm_per_px
    inside = (
        (x_rect >= -margin_px)
        & (x_rect <= width_px + margin_px)
        & (y_rect >= -margin_px)
        & (y_rect <= height_px + margin_px)
    )
    likelihood = track["likelihood"].to_numpy(dtype=float)
    finite = np.isfinite(raw[:, 0]) & np.isfinite(raw[:, 1]) & np.isfinite(likelihood)
    valid = finite & (likelihood >= likelihood_threshold) & inside
    out = pd.DataFrame(
        {
            "x_px": raw[:, 0],
            "y_px": raw[:, 1],
            "likelihood": likelihood,
            "x_rectified_px": x_rect,
            "y_rectified_px": y_rect,
            "x_cm": x_rect * calibration.cm_per_px,
            "y_cm": y_rect * calibration.cm_per_px,
            "valid_raw": valid,
        }
    )
    out.loc[~valid, ["x_cm", "y_cm", "x_rectified_px", "y_rectified_px"]] = np.nan
    return out


def compute_wall_posture_frame_metrics(
    dlc_table: pd.DataFrame,
    calibration: Calibration,
    likelihood_threshold: float,
    arena_margin_cm: float,
    nose_bodypart: str = "nose",
    center_bodypart: str = "bodycenter",
    tailbase_bodypart: str = "tailbase",
) -> pd.DataFrame:
    """Return per-frame whole-body features for sustained long-wall posture.

    This intentionally does not call the behavior "rearing": a single top-down
    camera cannot prove vertical posture.  It detects the observable proxy that
    motivated the visual QC label--the nose remains at a long wall while the
    bodycenter is still well inside the arena.  Tailbase is included in the
    joint validity mask so a nominal posture decision requires the whole tracked
    body to be available.
    """
    bodyparts = {
        "nose": nose_bodypart,
        "bodycenter": center_bodypart,
        "tailbase": tailbase_bodypart,
    }
    positions: dict[str, pd.DataFrame] = {}
    expected_length: int | None = None
    for label, bodypart in bodyparts.items():
        track = extract_bodypart(dlc_table, bodypart)
        if expected_length is None:
            expected_length = len(track)
        elif len(track) != expected_length:
            raise ValueError(
                "posture bodypart row counts differ: "
                f"expected {expected_length}, got {len(track)} for {bodypart}"
            )
        positions[label] = calibrated_positions(
            track,
            calibration=calibration,
            likelihood_threshold=likelihood_threshold,
            arena_margin_cm=arena_margin_cm,
        )

    arena_height_cm = calibration.rectified_size_px[1] * calibration.cm_per_px
    out = pd.DataFrame({"frame": np.arange(expected_length or 0, dtype=int)})
    valid_columns: list[np.ndarray] = []
    for label, position in positions.items():
        out[f"{label}_x_cm"] = position["x_cm"].to_numpy(dtype=float)
        out[f"{label}_y_cm"] = position["y_cm"].to_numpy(dtype=float)
        out[f"{label}_likelihood"] = position["likelihood"].to_numpy(dtype=float)
        valid = position["valid_raw"].to_numpy(dtype=bool)
        out[f"{label}_valid"] = valid
        valid_columns.append(valid)
        y_cm = out[f"{label}_y_cm"].to_numpy(dtype=float)
        out[f"{label}_wall_distance_cm"] = np.minimum(y_cm, arena_height_cm - y_cm)
    out["posture_valid"] = np.logical_and.reduce(valid_columns)
    return out


def add_wall_posture_classification(
    events: pd.DataFrame,
    posture_metrics: pd.DataFrame,
    wall_distance_cm: float = DEFAULT_POSTURE_WALL_DISTANCE_CM,
    center_away_cm: float = DEFAULT_POSTURE_CENTER_AWAY_CM,
    min_fraction: float = DEFAULT_POSTURE_MIN_FRACTION,
    borderline_fraction: float = DEFAULT_POSTURE_BORDERLINE_FRACTION,
    min_valid_fraction: float = DEFAULT_POSTURE_MIN_VALID_FRACTION,
    motion_metrics: pd.DataFrame | None = None,
    fps: float | None = None,
    interruption_max_speed_cm_s: float = DEFAULT_POSTURE_INTERRUPTION_MAX_SPEED_CM_S,
    interruption_min_sec: float = DEFAULT_POSTURE_INTERRUPTION_MIN_SEC,
) -> pd.DataFrame:
    """Add a motion-supported wall-posture rejection to transit candidates.

    When aligned bodycenter motion metrics and FPS are supplied, sustained wall
    orientation is rejected only if it overlaps a short low-speed run.  This
    separates a posture-related interruption from uninterrupted wall-following.
    Calls without motion metrics retain the legacy posture-only behavior.
    """
    if wall_distance_cm < 0:
        raise ValueError("posture wall distance must be >= 0")
    if center_away_cm < 0:
        raise ValueError("posture center-away distance must be >= 0")
    if not 0 <= min_fraction <= 1:
        raise ValueError("posture minimum fraction must be between 0 and 1")
    if not 0 <= borderline_fraction <= min_fraction:
        raise ValueError(
            "posture borderline fraction must be between 0 and the rejection fraction"
        )
    if not 0 <= min_valid_fraction <= 1:
        raise ValueError("posture minimum valid fraction must be between 0 and 1")
    if interruption_max_speed_cm_s < 0:
        raise ValueError("posture interruption maximum speed must be >= 0")
    if interruption_min_sec < 0:
        raise ValueError("posture interruption minimum duration must be >= 0")
    motion_gate_enabled = motion_metrics is not None
    if motion_gate_enabled:
        if fps is None or not np.isfinite(fps) or fps <= 0:
            raise ValueError("positive FPS is required with posture motion metrics")
        if len(motion_metrics) != len(posture_metrics):
            raise ValueError(
                "posture and motion metrics must have the same row count: "
                f"{len(posture_metrics)} vs {len(motion_metrics)}"
            )
        if "speed_cm_s" not in motion_metrics.columns:
            raise ValueError("posture motion metrics need a speed_cm_s column")

    out = events.copy()
    feature_columns: dict[str, object] = {
        "posture_valid_fraction": np.nan,
        "nose_wall_distance_median_cm": np.nan,
        "bodycenter_wall_distance_median_cm": np.nan,
        "tailbase_wall_distance_median_cm": np.nan,
        "wall_posture_fraction": np.nan,
        "wall_posture_frame_count": 0,
        "wall_posture_start_frame": np.nan,
        "wall_posture_end_frame": np.nan,
        "wall_posture_low_speed_frame_count": 0,
        "wall_posture_low_speed_longest_s": np.nan,
        "wall_posture_motion_supported": False,
        "wall_posture_borderline": False,
        "wall_posture_interruption": False,
    }
    for column, default in feature_columns.items():
        out[column] = default
    if out.empty:
        return out

    for index, event in out.iterrows():
        start_frame = int(event["start_frame"])
        end_frame = int(event["end_frame"])
        segment = posture_metrics.iloc[start_frame : end_frame + 1]
        if segment.empty:
            continue
        valid = segment["posture_valid"].to_numpy(dtype=bool)
        valid_count = int(valid.sum())
        valid_fraction = valid_count / len(segment)
        out.at[index, "posture_valid_fraction"] = valid_fraction
        if valid_count == 0:
            continue

        for label in ("nose", "bodycenter", "tailbase"):
            distances = segment.loc[
                segment["posture_valid"], f"{label}_wall_distance_cm"
            ]
            out.at[index, f"{label}_wall_distance_median_cm"] = float(
                distances.median()
            )

        posture_frame = (
            valid
            & (
                segment["nose_wall_distance_cm"].to_numpy(dtype=float)
                < wall_distance_cm
            )
            & (
                segment["bodycenter_wall_distance_cm"].to_numpy(dtype=float)
                > center_away_cm
            )
        )
        posture_count = int(posture_frame.sum())
        posture_fraction = posture_count / valid_count
        out.at[index, "wall_posture_fraction"] = posture_fraction
        out.at[index, "wall_posture_frame_count"] = posture_count
        if posture_count:
            posture_frames = segment.loc[posture_frame, "frame"]
            out.at[index, "wall_posture_start_frame"] = int(posture_frames.iloc[0])
            out.at[index, "wall_posture_end_frame"] = int(posture_frames.iloc[-1])

        if motion_gate_enabled:
            motion_segment = motion_metrics.iloc[start_frame : end_frame + 1]
            speed = motion_segment["speed_cm_s"].to_numpy(dtype=float)
            low_speed_posture = (
                posture_frame
                & np.isfinite(speed)
                & (speed <= interruption_max_speed_cm_s)
            )
            low_speed_count = int(low_speed_posture.sum())
            longest_low_speed_frames = longest_true_run_frames(low_speed_posture)
            longest_low_speed_s = longest_low_speed_frames / float(fps)
            motion_supported = bool(
                longest_low_speed_s + 1e-12 >= interruption_min_sec
            )
            out.at[index, "wall_posture_low_speed_frame_count"] = low_speed_count
            out.at[index, "wall_posture_low_speed_longest_s"] = longest_low_speed_s
            out.at[index, "wall_posture_motion_supported"] = motion_supported
        else:
            motion_supported = True
            out.at[index, "wall_posture_motion_supported"] = True

        valid_for_posture = valid_fraction >= min_valid_fraction
        interrupted = bool(
            valid_for_posture
            and posture_fraction >= min_fraction
            and motion_supported
        )
        borderline = bool(
            valid_for_posture
            and posture_fraction >= borderline_fraction
            and not interrupted
        )
        out.at[index, "wall_posture_borderline"] = borderline
        out.at[index, "wall_posture_interruption"] = interrupted
        reasons = [
            reason.strip()
            for reason in str(event.get("rejection_reasons", "")).split(";")
            if reason.strip() and reason.strip().lower() != "nan"
        ]
        if interrupted:
            if "wall posture interruption" not in reasons:
                reasons.append("wall posture interruption")
            out.at[index, "rejection_reasons"] = "; ".join(reasons)
            out.at[index, "decision_label"] = f"rejected: {reasons[0]}"
            out.at[index, "is_straight"] = False
        elif borderline:
            borderline_reasons = [
                reason.strip()
                for reason in str(event.get("borderline_reasons", "")).split(";")
                if reason.strip() and reason.strip().lower() != "nan"
            ]
            if "wall posture" not in borderline_reasons:
                borderline_reasons.append("wall posture")
            out.at[index, "borderline_reasons"] = "; ".join(borderline_reasons)
            if not reasons:
                out.at[index, "decision_label"] = "borderline"

    out["wall_posture_frame_count"] = out["wall_posture_frame_count"].astype(int)
    out["wall_posture_low_speed_frame_count"] = out[
        "wall_posture_low_speed_frame_count"
    ].astype(int)
    out["wall_posture_motion_supported"] = out[
        "wall_posture_motion_supported"
    ].astype(bool)
    out["wall_posture_borderline"] = out["wall_posture_borderline"].astype(bool)
    out["wall_posture_interruption"] = out["wall_posture_interruption"].astype(bool)
    out["is_straight"] = out["is_straight"].astype(bool)
    return out


def compute_motion_metrics(
    positions: pd.DataFrame,
    fps: float,
    max_gap_sec: float,
    smooth_window_sec: float,
    speed_threshold_cm_s: float,
    stop_gap_sec: float,
    min_bout_sec: float,
    min_bout_displacement_cm: float,
) -> pd.DataFrame:
    n = len(positions)
    frame = np.arange(n)
    max_gap_frames = int(round(max_gap_sec * fps))
    smooth_window_frames = max(1, int(round(smooth_window_sec * fps)))
    if smooth_window_frames % 2 == 0:
        smooth_window_frames += 1
    stop_gap_frames = int(round(stop_gap_sec * fps))
    min_bout_frames = max(1, int(math.ceil(min_bout_sec * fps)))

    valid_raw = positions["valid_raw"].to_numpy(dtype=bool)
    x_interp = interpolate_short_nan_runs(positions["x_cm"].to_numpy(), valid_raw, max_gap_frames)
    y_interp = interpolate_short_nan_runs(positions["y_cm"].to_numpy(), valid_raw, max_gap_frames)
    valid_interp = np.isfinite(x_interp) & np.isfinite(y_interp)

    x_smooth = rolling_median_by_valid_segments(x_interp, valid_interp, smooth_window_frames)
    y_smooth = rolling_median_by_valid_segments(y_interp, valid_interp, smooth_window_frames)

    dx = np.diff(x_smooth, prepend=np.nan)
    dy = np.diff(y_smooth, prepend=np.nan)
    step_distance_cm = np.hypot(dx, dy)
    valid_step = valid_interp & np.r_[False, valid_interp[:-1]]
    step_distance_cm[~valid_step] = np.nan
    speed_cm_s = step_distance_cm * fps

    moving = (speed_cm_s >= speed_threshold_cm_s) & valid_step
    moving = fill_short_false_gaps(moving, stop_gap_frames)
    moving = remove_short_or_small_bouts(
        moving,
        x_smooth,
        y_smooth,
        min_bout_frames,
        min_bout_displacement_cm,
    )

    movement_distance_cm = np.where(moving, step_distance_cm, 0.0)
    raw_distance_cm = np.where(valid_step, step_distance_cm, 0.0)

    metrics = positions.copy()
    metrics.insert(0, "frame", frame)
    metrics.insert(1, "time_s", frame / fps)
    metrics["x_cm_interpolated"] = x_interp
    metrics["y_cm_interpolated"] = y_interp
    metrics["valid_interpolated"] = valid_interp
    metrics["x_cm_smooth"] = x_smooth
    metrics["y_cm_smooth"] = y_smooth
    metrics["step_distance_cm"] = step_distance_cm
    metrics["speed_cm_s"] = speed_cm_s
    metrics["moving"] = moving
    metrics["raw_distance_cm"] = raw_distance_cm
    metrics["movement_distance_cm"] = movement_distance_cm
    return metrics


def summarize_metrics(metrics: pd.DataFrame, fps: float, bin_seconds: int | None = None) -> pd.DataFrame:
    if bin_seconds is None:
        grouped = [(0, metrics)]
        starts = {0: 0.0}
        ends = {0: len(metrics) / fps}
    else:
        bin_index = np.floor(metrics["time_s"] / bin_seconds).astype(int)
        grouped = list(metrics.groupby(bin_index, sort=True))
        starts = {idx: idx * bin_seconds for idx, _ in grouped}
        ends = {
            idx: min((idx + 1) * bin_seconds, float(metrics["time_s"].iloc[-1] + 1 / fps))
            for idx, _ in grouped
        }

    rows = []
    for idx, group in grouped:
        duration_s = ends[idx] - starts[idx]
        valid_time_s = float(group["valid_interpolated"].sum() / fps)
        moving_time_s = float(group["moving"].sum() / fps)
        raw_distance = float(np.nansum(group["raw_distance_cm"]))
        movement_distance = float(np.nansum(group["movement_distance_cm"]))
        active_speed = group.loc[group["moving"], "speed_cm_s"].replace([np.inf, -np.inf], np.nan)
        all_speed = group["speed_cm_s"].replace([np.inf, -np.inf], np.nan)
        rows.append(
            {
                "bin_index": int(idx),
                "bin_start_s": float(starts[idx]),
                "bin_end_s": float(ends[idx]),
                "duration_s": float(duration_s),
                "frame_count": int(len(group)),
                "valid_fraction": float(group["valid_interpolated"].mean()),
                "moving_time_s": moving_time_s,
                "moving_fraction_of_valid": moving_time_s / valid_time_s if valid_time_s else np.nan,
                "raw_distance_cm": raw_distance,
                "movement_distance_cm": movement_distance,
                "mean_speed_including_stops_cm_s": movement_distance / duration_s
                if duration_s
                else np.nan,
                "mean_speed_when_moving_cm_s": float(active_speed.mean())
                if len(active_speed)
                else np.nan,
                "median_speed_cm_s": float(all_speed.median()) if all_speed.notna().any() else np.nan,
                "p95_speed_cm_s": float(all_speed.quantile(0.95)) if all_speed.notna().any() else np.nan,
                "x_cm_mean": float(group["x_cm_smooth"].mean()),
                "x_cm_median": float(group["x_cm_smooth"].median()),
                "y_cm_mean": float(group["y_cm_smooth"].mean()),
                "y_cm_median": float(group["y_cm_smooth"].median()),
            }
        )
    return pd.DataFrame(rows)


def add_edge_regions(
    metrics: pd.DataFrame,
    arena_size_cm: tuple[float, float],
    edge_width_cm: float,
) -> pd.DataFrame:
    """Label each frame as left edge, center, right edge, or invalid."""
    out = metrics.copy()
    width_cm, height_cm = arena_size_cm
    usable_edge_width = max(0.0, min(float(edge_width_cm), width_cm / 2.0))
    x = out["x_cm_smooth"].to_numpy(dtype=float)
    y = out["y_cm_smooth"].to_numpy(dtype=float)
    valid = (
        out["valid_interpolated"].to_numpy(dtype=bool)
        & np.isfinite(x)
        & np.isfinite(y)
        & (x >= 0)
        & (x <= width_cm)
        & (y >= 0)
        & (y <= height_cm)
    )

    labels = np.full(len(out), "invalid", dtype=object)
    left = valid & (x <= usable_edge_width)
    right = valid & (x >= width_cm - usable_edge_width)
    center = valid & ~(left | right)
    labels[left] = "left_edge"
    labels[center] = "center"
    labels[right] = "right_edge"
    out["edge_region"] = labels
    return out


def region_entry_exit_counts(labels: np.ndarray, region: str) -> tuple[int, int]:
    in_region = labels == region
    if len(in_region) == 0:
        return 0, 0
    entries = int(np.sum(in_region & np.r_[True, ~in_region[:-1]]))
    exits = int(np.sum(in_region & np.r_[~in_region[1:], True]))
    return entries, exits


def summarize_edge_regions(
    metrics: pd.DataFrame,
    fps: float,
    arena_size_cm: tuple[float, float],
    edge_width_cm: float,
) -> pd.DataFrame:
    labels = metrics["edge_region"].to_numpy(dtype=object)
    valid_region = np.isin(labels, EDGE_REGION_LABELS)
    total_valid_time_s = float(valid_region.sum() / fps)
    rows = []
    for region in EDGE_REGION_LABELS:
        mask = labels == region
        region_time_s = float(mask.sum() / fps)
        region_rows = metrics.loc[mask]
        movement_distance = float(np.nansum(region_rows["movement_distance_cm"]))
        raw_distance = float(np.nansum(region_rows["raw_distance_cm"]))
        moving_speed = region_rows.loc[region_rows["moving"], "speed_cm_s"].replace(
            [np.inf, -np.inf], np.nan
        )
        entries, exits = region_entry_exit_counts(labels, region)
        rows.append(
            {
                "region": region,
                "edge_width_cm": float(edge_width_cm),
                "arena_width_cm": float(arena_size_cm[0]),
                "arena_height_cm": float(arena_size_cm[1]),
                "frame_count": int(mask.sum()),
                "time_s": region_time_s,
                "fraction_of_valid_time": (
                    region_time_s / total_valid_time_s if total_valid_time_s else np.nan
                ),
                "raw_distance_cm": raw_distance,
                "movement_distance_cm": movement_distance,
                "mean_speed_including_stops_cm_s": (
                    movement_distance / region_time_s if region_time_s else np.nan
                ),
                "mean_speed_when_moving_cm_s": (
                    float(moving_speed.mean()) if moving_speed.notna().any() else np.nan
                ),
                "entry_count": entries,
                "exit_count": exits,
            }
        )
    return pd.DataFrame(rows)


def edge_area_runs(labels: Sequence[object]) -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    n = len(labels)
    index = 0
    while index < n:
        label = str(labels[index])
        if label not in EDGE_AREA_LABELS:
            index += 1
            continue
        start = index
        while index < n and str(labels[index]) == label:
            index += 1
        runs.append({"area": label, "start_frame": start, "end_frame": index - 1})
    return runs


def longest_true_run_frames(mask: np.ndarray) -> int:
    longest = 0
    for start, end in contiguous_true_runs(mask):
        longest = max(longest, end - start)
    return longest


def longest_true_run_bounds(mask: np.ndarray) -> tuple[int | None, int | None, int]:
    best_start: int | None = None
    best_end: int | None = None
    best_length = 0
    for start, end in contiguous_true_runs(mask):
        length = end - start
        if length > best_length:
            best_start = start
            best_end = end
            best_length = length
    return best_start, best_end, best_length


def segment_step_distance(metrics: pd.DataFrame, start_frame: int, end_frame: int) -> float:
    if end_frame <= start_frame:
        return 0.0
    return float(
        np.nansum(metrics["step_distance_cm"].iloc[start_frame + 1 : end_frame + 1])
    )


def line_deviation_cm(points: np.ndarray) -> float:
    finite = np.isfinite(points[:, 0]) & np.isfinite(points[:, 1])
    points = points[finite]
    if len(points) < 2:
        return np.nan
    start = points[0]
    end = points[-1]
    line = end - start
    length = float(np.linalg.norm(line))
    if length <= 0:
        return np.nan
    offsets = points - start
    distances = np.abs(line[0] * offsets[:, 1] - line[1] * offsets[:, 0]) / length
    return float(np.nanmax(distances)) if len(distances) else np.nan


def first_last_finite_points(points: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    finite = np.isfinite(points[:, 0]) & np.isfinite(points[:, 1])
    if not finite.any():
        return None, None
    finite_points = points[finite]
    return finite_points[0], finite_points[-1]


def transit_event_row(
    event_id: int,
    metrics: pd.DataFrame,
    fps: float,
    source_run: dict[str, object],
    target_run: dict[str, object],
    arena_size_cm: tuple[float, float],
    edge_width_cm: float,
    path_efficiency_threshold: float,
    max_deviation_cm: float,
    stop_speed_threshold_cm_s: float,
    stop_min_sec: float,
    low_speed_fraction_threshold: float,
    max_step_jump_cm: float,
    min_valid_fraction: float,
    borderline_margin: float,
) -> dict[str, object]:
    start_frame = int(source_run["end_frame"])
    end_frame = int(target_run["start_frame"])
    segment = metrics.iloc[start_frame : end_frame + 1]
    points = segment[["x_cm_smooth", "y_cm_smooth"]].to_numpy(dtype=float)
    first_point, last_point = first_last_finite_points(points)
    if first_point is None or last_point is None:
        straight_distance = np.nan
    else:
        straight_distance = float(np.linalg.norm(last_point - first_point))
    actual_distance = segment_step_distance(metrics, start_frame, end_frame)
    path_efficiency = straight_distance / actual_distance if actual_distance > 0 else np.nan
    max_line_deviation = line_deviation_cm(points)
    center_axis_deviation = (
        float(np.nanmax(np.abs(points[:, 1] - arena_size_cm[1] / 2.0)))
        if np.isfinite(points[:, 1]).any()
        else np.nan
    )

    x = points[:, 0]
    finite_x = x[np.isfinite(x)]
    dx = np.diff(finite_x) if len(finite_x) > 1 else np.asarray([], dtype=float)
    start_area = str(source_run["area"])
    end_area = str(target_run["area"])
    direction = "left_to_right" if start_area == "left_edge" else "right_to_left"
    if direction == "left_to_right":
        reverse_distance = float(np.sum(np.maximum(-dx, 0.0)))
    else:
        reverse_distance = float(np.sum(np.maximum(dx, 0.0)))
    max_reverse_cm = max(float(edge_width_cm) / 2.0, 1.0)
    has_uturn = bool(reverse_distance > max_reverse_cm)

    speed = segment["speed_cm_s"].replace([np.inf, -np.inf], np.nan)
    valid_segment = segment["valid_interpolated"].to_numpy(dtype=bool)
    low_speed = (speed.to_numpy(dtype=float) <= float(stop_speed_threshold_cm_s)) & valid_segment
    low_speed_start, low_speed_end, longest_low_speed_frames = longest_true_run_bounds(low_speed)
    longest_stop_s = longest_low_speed_frames / fps
    has_stop = bool(longest_stop_s >= stop_min_sec)
    valid_fraction = float(
        np.mean(np.isfinite(points[:, 0]) & np.isfinite(points[:, 1]))
    ) if len(points) else np.nan
    low_speed_fraction = (
        float(np.sum(low_speed) / max(1, np.sum(valid_segment))) if len(segment) else np.nan
    )
    step_distance = segment["step_distance_cm"].replace([np.inf, -np.inf], np.nan)
    max_step_distance = float(step_distance.max()) if step_distance.notna().any() else np.nan
    p95_step_distance = (
        float(step_distance.quantile(0.95)) if step_distance.notna().any() else np.nan
    )
    tracking_anomaly = bool(
        (np.isfinite(valid_fraction) and valid_fraction < min_valid_fraction)
        or (np.isfinite(max_step_distance) and max_step_distance > max_step_jump_cm)
    )
    duration_s = (end_frame - start_frame) / fps if end_frame >= start_frame else 0.0
    moving_speed = segment.loc[segment["moving"], "speed_cm_s"].replace(
        [np.inf, -np.inf], np.nan
    )
    rejection_reasons: list[str] = []
    if not np.isfinite(path_efficiency) or path_efficiency < path_efficiency_threshold:
        rejection_reasons.append("path efficiency")
    if not np.isfinite(max_line_deviation) or max_line_deviation > max_deviation_cm:
        rejection_reasons.append("deviation")
    if has_stop:
        rejection_reasons.append("pause")
    if np.isfinite(low_speed_fraction) and low_speed_fraction > low_speed_fraction_threshold:
        rejection_reasons.append("low-speed segment")
    if tracking_anomaly:
        rejection_reasons.append("posture/tracking anomaly")
    if has_uturn:
        rejection_reasons.append("direction reversal")

    borderline_reasons: list[str] = []
    # Express the efficiency guard band as a fraction of the remaining
    # headroom to the theoretical maximum (1.0).  The previous
    # threshold*margin rule made every event borderline when the calibrated
    # threshold was raised to 0.95.
    efficiency_margin = max(
        0.005,
        (1.0 - path_efficiency_threshold) * borderline_margin,
    )
    deviation_margin = max(0.2, max_deviation_cm * borderline_margin)
    low_speed_margin = max(0.01, low_speed_fraction_threshold * borderline_margin)
    stop_margin = max(1.0 / fps, stop_min_sec * borderline_margin)
    if (
        np.isfinite(path_efficiency)
        and path_efficiency_threshold <= path_efficiency < path_efficiency_threshold + efficiency_margin
    ):
        borderline_reasons.append("path efficiency")
    if (
        np.isfinite(max_line_deviation)
        and max_deviation_cm - deviation_margin < max_line_deviation <= max_deviation_cm
    ):
        borderline_reasons.append("deviation")
    if (
        stop_min_sec > 0
        and not has_stop
        and stop_min_sec - stop_margin - 1e-9 <= longest_stop_s < stop_min_sec
    ):
        borderline_reasons.append("pause")
    if (
        np.isfinite(low_speed_fraction)
        and low_speed_fraction_threshold - low_speed_margin
        < low_speed_fraction
        <= low_speed_fraction_threshold
    ):
        borderline_reasons.append("low-speed segment")

    if rejection_reasons:
        decision_label = f"rejected: {rejection_reasons[0]}"
    elif borderline_reasons:
        decision_label = "borderline"
    else:
        decision_label = "accepted"
    # Borderline means that all acceptance thresholds passed but the event is
    # close enough to a boundary to merit visual QC.  It is still a straight
    # transit; only explicit rejection reasons make is_straight false.
    is_straight = not rejection_reasons

    return {
        "event_id": int(event_id),
        "start_area": start_area,
        "end_area": end_area,
        "direction": direction,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_time_s": start_frame / fps,
        "end_time_s": end_frame / fps,
        "duration_s": float(duration_s),
        "actual_distance_cm": actual_distance,
        "straight_distance_cm": straight_distance,
        "path_efficiency": path_efficiency,
        "max_line_deviation_cm": max_line_deviation,
        "max_center_axis_deviation_cm": center_axis_deviation,
        "reverse_distance_cm": reverse_distance,
        "has_stop": has_stop,
        "longest_low_speed_s": float(longest_stop_s),
        "low_speed_fraction": low_speed_fraction,
        "low_speed_start_frame": (
            int(start_frame + low_speed_start) if low_speed_start is not None else np.nan
        ),
        "low_speed_end_frame": (
            int(start_frame + low_speed_end - 1) if low_speed_end is not None else np.nan
        ),
        "has_uturn": has_uturn,
        "valid_fraction": valid_fraction,
        "tracking_anomaly": tracking_anomaly,
        "max_step_distance_cm": max_step_distance,
        "p95_step_distance_cm": p95_step_distance,
        "mean_speed_cm_s": actual_distance / duration_s if duration_s else np.nan,
        "mean_moving_speed_cm_s": (
            float(moving_speed.mean()) if moving_speed.notna().any() else np.nan
        ),
        "max_speed_cm_s": float(speed.max()) if speed.notna().any() else np.nan,
        "decision_label": decision_label,
        "rejection_reasons": "; ".join(rejection_reasons),
        "borderline_reasons": "; ".join(borderline_reasons),
        "is_straight": is_straight,
    }


def detect_straight_transit_events(
    metrics: pd.DataFrame,
    fps: float,
    arena_size_cm: tuple[float, float],
    edge_width_cm: float,
    path_efficiency_threshold: float,
    max_deviation_cm: float,
    stop_speed_threshold_cm_s: float,
    stop_min_sec: float,
    low_speed_fraction_threshold: float,
    max_step_jump_cm: float,
    min_valid_fraction: float,
    borderline_margin: float,
) -> pd.DataFrame:
    labels = metrics["edge_region"].to_numpy(dtype=object)
    runs = edge_area_runs(labels)
    rows = []
    event_id = 1
    for source_run, target_run in zip(runs[:-1], runs[1:]):
        if source_run["area"] == target_run["area"]:
            continue
        rows.append(
            transit_event_row(
                event_id,
                metrics,
                fps,
                source_run,
                target_run,
                arena_size_cm,
                edge_width_cm,
                path_efficiency_threshold,
                max_deviation_cm,
                stop_speed_threshold_cm_s,
                stop_min_sec,
                low_speed_fraction_threshold,
                max_step_jump_cm,
                min_valid_fraction,
                borderline_margin,
            )
        )
        event_id += 1
    columns = [
        "event_id",
        "start_area",
        "end_area",
        "direction",
        "start_frame",
        "end_frame",
        "start_time_s",
        "end_time_s",
        "duration_s",
        "actual_distance_cm",
        "straight_distance_cm",
        "path_efficiency",
        "max_line_deviation_cm",
        "max_center_axis_deviation_cm",
        "reverse_distance_cm",
        "has_stop",
        "longest_low_speed_s",
        "low_speed_fraction",
        "low_speed_start_frame",
        "low_speed_end_frame",
        "has_uturn",
        "valid_fraction",
        "tracking_anomaly",
        "max_step_distance_cm",
        "p95_step_distance_cm",
        "mean_speed_cm_s",
        "mean_moving_speed_cm_s",
        "max_speed_cm_s",
        "decision_label",
        "rejection_reasons",
        "borderline_reasons",
        "is_straight",
    ]
    return pd.DataFrame(rows, columns=columns)


def summarize_straight_transits(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(
            [
                {
                    "transit_count": 0,
                    "straight_transit_count": 0,
                    "strict_accepted_transit_count": 0,
                    "borderline_transit_count": 0,
                    "rejected_transit_count": 0,
                    "wall_posture_interruption_count": 0,
                    "wall_posture_borderline_count": 0,
                    "straight_fraction": np.nan,
                    "mean_straight_speed_cm_s": np.nan,
                    "median_straight_speed_cm_s": np.nan,
                    "mean_straight_path_efficiency": np.nan,
                }
            ]
        )
    straight = events.loc[events["is_straight"].astype(bool)]
    strict_accepted = events.loc[events["decision_label"] == "accepted"]
    borderline = events.loc[events["decision_label"] == "borderline"]
    rejected = events.loc[events["decision_label"].astype(str).str.startswith("rejected")]
    wall_posture_count = (
        int(events["wall_posture_interruption"].astype(bool).sum())
        if "wall_posture_interruption" in events.columns
        else 0
    )
    wall_posture_borderline_count = (
        int(events["wall_posture_borderline"].astype(bool).sum())
        if "wall_posture_borderline" in events.columns
        else 0
    )
    return pd.DataFrame(
        [
            {
                "transit_count": int(len(events)),
                "straight_transit_count": int(len(straight)),
                "strict_accepted_transit_count": int(len(strict_accepted)),
                "borderline_transit_count": int(len(borderline)),
                "rejected_transit_count": int(len(rejected)),
                "wall_posture_interruption_count": wall_posture_count,
                "wall_posture_borderline_count": wall_posture_borderline_count,
                "straight_fraction": float(len(straight) / len(events)) if len(events) else np.nan,
                "mean_straight_speed_cm_s": float(straight["mean_speed_cm_s"].mean())
                if len(straight)
                else np.nan,
                "median_straight_speed_cm_s": float(straight["mean_speed_cm_s"].median())
                if len(straight)
                else np.nan,
                "mean_straight_path_efficiency": float(straight["path_efficiency"].mean())
                if len(straight)
                else np.nan,
            }
        ]
    )


def detect_back_and_forth_events(
    metrics: pd.DataFrame,
    transit_events: pd.DataFrame,
    fps: float,
    time_window_sec: float,
    min_round_trips: int,
) -> pd.DataFrame:
    min_transitions = max(1, int(min_round_trips)) * 2
    if transit_events.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "start_frame",
                "end_frame",
                "start_time_s",
                "end_time_s",
                "duration_s",
                "transition_count",
                "round_trip_count",
                "round_trips_per_min",
                "area_sequence",
                "movement_distance_cm",
                "mean_speed_cm_s",
                "mean_moving_speed_cm_s",
            ]
        )

    transitions = [
        row._asdict()
        for row in transit_events.sort_values("start_frame").itertuples(index=False)
    ]
    chains: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    for transition in transitions:
        if not current:
            current = [transition]
            continue
        previous = current[-1]
        gap = float(transition["start_time_s"]) - float(previous["end_time_s"])
        alternating = previous["end_area"] == transition["start_area"]
        if alternating and gap <= time_window_sec:
            current.append(transition)
        else:
            if len(current) >= min_transitions:
                chains.append(current)
            current = [transition]
    if len(current) >= min_transitions:
        chains.append(current)

    rows = []
    for event_id, chain in enumerate(chains, start=1):
        start_frame = int(chain[0]["start_frame"])
        end_frame = int(chain[-1]["end_frame"])
        segment = metrics.iloc[start_frame : end_frame + 1]
        duration_s = (end_frame - start_frame) / fps if end_frame >= start_frame else 0.0
        distance = segment_step_distance(metrics, start_frame, end_frame)
        moving_speed = segment.loc[segment["moving"], "speed_cm_s"].replace(
            [np.inf, -np.inf], np.nan
        )
        sequence = [str(chain[0]["start_area"])] + [str(item["end_area"]) for item in chain]
        round_trip_count = len(chain) // 2
        rows.append(
            {
                "event_id": int(event_id),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time_s": start_frame / fps,
                "end_time_s": end_frame / fps,
                "duration_s": float(duration_s),
                "transition_count": int(len(chain)),
                "round_trip_count": int(round_trip_count),
                "round_trips_per_min": (
                    round_trip_count / (duration_s / 60.0) if duration_s else np.nan
                ),
                "area_sequence": ">".join(sequence),
                "movement_distance_cm": distance,
                "mean_speed_cm_s": distance / duration_s if duration_s else np.nan,
                "mean_moving_speed_cm_s": (
                    float(moving_speed.mean()) if moving_speed.notna().any() else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_back_and_forth_events(events: pd.DataFrame, duration_s: float) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(
            [
                {
                    "back_and_forth_event_count": 0,
                    "total_transition_count": 0,
                    "total_round_trip_count": 0.0,
                    "events_per_min": 0.0 if duration_s else np.nan,
                    "round_trips_per_min": 0.0 if duration_s else np.nan,
                    "mean_event_duration_s": np.nan,
                    "mean_round_trips_per_event": np.nan,
                    "mean_event_speed_cm_s": np.nan,
                }
            ]
        )
    event_count = len(events)
    total_round_trips = float(events["round_trip_count"].sum())
    return pd.DataFrame(
        [
            {
                "back_and_forth_event_count": int(event_count),
                "total_transition_count": int(events["transition_count"].sum()),
                "total_round_trip_count": total_round_trips,
                "events_per_min": event_count / (duration_s / 60.0) if duration_s else np.nan,
                "round_trips_per_min": (
                    total_round_trips / (duration_s / 60.0) if duration_s else np.nan
                ),
                "mean_event_duration_s": float(events["duration_s"].mean()),
                "mean_round_trips_per_event": float(events["round_trip_count"].mean()),
                "mean_event_speed_cm_s": float(events["mean_speed_cm_s"].mean()),
            }
        ]
    )


def rectified_cm_to_video_px(points_cm: np.ndarray, calibration: Calibration) -> np.ndarray:
    rectified_px = np.asarray(points_cm, dtype=np.float32).copy()
    rectified_px[:, 0] = rectified_px[:, 0] / calibration.cm_per_px
    rectified_px[:, 1] = rectified_px[:, 1] / calibration.cm_per_px
    inverse_homography = np.linalg.inv(calibration.homography)
    return transform_points(rectified_px, inverse_homography)


def draw_polyline_px(
    frame: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 2,
    closed: bool = False,
) -> None:
    finite = np.isfinite(points[:, 0]) & np.isfinite(points[:, 1])
    points = points[finite]
    if len(points) < 2:
        return
    pts = np.round(points).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [pts], closed, color, thickness, cv2.LINE_AA)


def event_labels_by_frame(
    event_df: pd.DataFrame,
    frame_count: int,
    prefix: str,
    straight_only: bool = False,
) -> list[str]:
    labels = [""] * frame_count
    if event_df.empty:
        return labels
    rows = event_df
    if straight_only and "is_straight" in rows:
        rows = rows.loc[rows["is_straight"].astype(bool)]
    for row in rows.itertuples(index=False):
        start = max(0, int(getattr(row, "start_frame")))
        end = min(frame_count - 1, int(getattr(row, "end_frame")))
        if start > end:
            continue
        if prefix == "Straight":
            decision = str(getattr(row, "decision_label", "accepted"))
            text = (
                f"Transit #{int(getattr(row, 'event_id'))} {decision} "
                f"{float(getattr(row, 'mean_speed_cm_s')):.1f} cm/s "
                f"eff {float(getattr(row, 'path_efficiency')):.2f}"
            )
        else:
            text = (
                f"Back-and-forth #{int(getattr(row, 'event_id'))} "
                f"{float(getattr(row, 'round_trip_count')):.1f} trips"
            )
        for frame_index in range(start, end + 1):
            labels[frame_index] = text if not labels[frame_index] else labels[frame_index] + " | " + text
    return labels


def create_qc_video(
    video_path: Path,
    metrics: pd.DataFrame,
    calibration: Calibration,
    out_path: Path,
    fps: float,
    arena_size_cm: tuple[float, float],
    edge_width_cm: float,
    straight_events: pd.DataFrame,
    back_and_forth_events: pd.DataFrame,
    frame_start: int = 0,
    frame_end: int | None = None,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video for QC: {video_path}")
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or len(metrics)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_fps = float(cap.get(cv2.CAP_PROP_FPS) or fps)
    last_frame = min(video_frame_count, len(metrics)) - 1
    if frame_end is not None:
        last_frame = min(last_frame, int(frame_end))
    frame_start = max(0, min(int(frame_start), last_frame))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise ValueError(f"could not create QC video: {out_path}")

    width_cm, height_cm = arena_size_cm
    edge_width_cm = max(0.0, min(float(edge_width_cm), width_cm / 2.0))
    outline_cm = np.asarray(
        [[0, 0], [width_cm, 0], [width_cm, height_cm], [0, height_cm]], dtype=np.float32
    )
    left_boundary_cm = np.asarray([[edge_width_cm, 0], [edge_width_cm, height_cm]], dtype=np.float32)
    right_boundary_cm = np.asarray(
        [[width_cm - edge_width_cm, 0], [width_cm - edge_width_cm, height_cm]],
        dtype=np.float32,
    )
    outline_px = rectified_cm_to_video_px(outline_cm, calibration)
    left_boundary_px = rectified_cm_to_video_px(left_boundary_cm, calibration)
    right_boundary_px = rectified_cm_to_video_px(right_boundary_cm, calibration)
    straight_labels = event_labels_by_frame(straight_events, len(metrics), "Straight", False)
    shuttle_labels = event_labels_by_frame(back_and_forth_events, len(metrics), "Back-and-forth")
    trail_frames = max(1, int(round(output_fps * 3.0)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    try:
        for frame_index in range(frame_start, last_frame + 1):
            ok, frame = cap.read()
            if not ok:
                break
            draw_polyline_px(frame, outline_px, (30, 220, 255), 2, closed=True)
            draw_polyline_px(frame, left_boundary_px, (255, 190, 30), 2)
            draw_polyline_px(frame, right_boundary_px, (255, 190, 30), 2)

            trail = metrics.iloc[max(0, frame_index - trail_frames) : frame_index + 1]
            trail_points = trail[["x_px", "y_px"]].to_numpy(dtype=float)
            draw_polyline_px(frame, trail_points, (0, 255, 160), 2)

            row = metrics.iloc[frame_index]
            x_px = float(row["x_px"])
            y_px = float(row["y_px"])
            if np.isfinite(x_px) and np.isfinite(y_px):
                color = (0, 80, 255) if bool(row.get("valid_interpolated", False)) else (80, 80, 80)
                cv2.circle(frame, (int(round(x_px)), int(round(y_px))), 6, color, -1, cv2.LINE_AA)

            region = str(row.get("edge_region", "invalid"))
            speed = row.get("speed_cm_s", np.nan)
            labels = [
                f"Frame {frame_index}  Time {frame_index / fps:.2f}s",
                f"Region: {region}  Speed: {float(speed):.2f} cm/s" if np.isfinite(speed) else f"Region: {region}",
            ]
            if straight_labels[frame_index]:
                labels.append(straight_labels[frame_index])
            if shuttle_labels[frame_index]:
                labels.append(shuttle_labels[frame_index])
            for line_index, text in enumerate(labels):
                y = 28 + line_index * 28
                cv2.putText(
                    frame,
                    text,
                    (18, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (0, 0, 0),
                    4,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    text,
                    (18, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            writer.write(frame)
    finally:
        writer.release()
        cap.release()


def create_event_qc_clips(
    video_path: Path,
    metrics: pd.DataFrame,
    calibration: Calibration,
    out_dir: Path,
    prefix: str,
    fps: float,
    arena_size_cm: tuple[float, float],
    edge_width_cm: float,
    straight_events: pd.DataFrame,
    back_and_forth_events: pd.DataFrame,
    prepost_sec: float,
) -> list[Path]:
    clip_dir = out_dir / "event_clips"
    clip_paths: list[Path] = []
    padding = max(0, int(round(prepost_sec * fps)))
    straight_rows = straight_events
    for row in straight_rows.itertuples(index=False):
        decision = str(getattr(row, "decision_label", "candidate")).split(":", 1)[0]
        decision = "".join(char if char.isalnum() else "_" for char in decision).strip("_")
        path = clip_dir / f"{prefix}_straight_transit_candidate_{int(row.event_id):03d}_{decision}_qc.mp4"
        create_qc_video(
            video_path,
            metrics,
            calibration,
            path,
            fps,
            arena_size_cm,
            edge_width_cm,
            straight_events,
            back_and_forth_events,
            frame_start=int(row.start_frame) - padding,
            frame_end=int(row.end_frame) + padding,
        )
        clip_paths.append(path)
    for row in back_and_forth_events.itertuples(index=False):
        path = clip_dir / f"{prefix}_back_and_forth_{int(row.event_id):03d}_qc.mp4"
        create_qc_video(
            video_path,
            metrics,
            calibration,
            path,
            fps,
            arena_size_cm,
            edge_width_cm,
            straight_events,
            back_and_forth_events,
            frame_start=int(row.start_frame) - padding,
            frame_end=int(row.end_frame) + padding,
        )
        clip_paths.append(path)
    return clip_paths


def infer_prefix(path: Path) -> str:
    name = path.stem
    for marker in ("DLC_", "_filtered"):
        if marker in name:
            name = name.split(marker)[0].rstrip("_")
    return name


def analysis_output_paths(
    out_dir: Path,
    prefix: str,
    density_cmap: str,
    bin_seconds: Sequence[int],
) -> dict[str, object]:
    del density_cmap
    metrics_dir = out_dir / "metrics"
    events_dir = out_dir / "events"
    plots_dir = out_dir / "plots"
    qc_dir = out_dir / "qc"
    return {
        "metrics_dir": metrics_dir,
        "events_dir": events_dir,
        "plots_dir": plots_dir,
        "qc_dir": qc_dir,
        "speed_trace_png": plots_dir / f"{prefix}_speed_trace.png",
        "path_speed_png": plots_dir / f"{prefix}_path_speed_cm.png",
        "density_png": plots_dir / f"{prefix}_bodycenter_density.png",
        "distance_bins_pngs": {
            int(seconds): plots_dir / f"{prefix}_distance_bins_{int(seconds)}s.png"
            for seconds in bin_seconds
        },
    }


def get_density_colormap(name: str):
    requested = name.strip() or "coolwarm"
    if requested.lower() == "rocket":
        try:
            import seaborn as sns

            cmap = sns.color_palette("rocket", as_cmap=True)
        except Exception:
            cmap = LinearSegmentedColormap.from_list(
                "rocket",
                ["#03051a", "#35193e", "#701f57", "#ad1759", "#e13342", "#f37651", "#f6b48f"],
            )
    else:
        try:
            cmap = plt.get_cmap(requested)
        except ValueError:
            try:
                import seaborn as sns

                cmap = sns.color_palette(requested, as_cmap=True)
            except Exception as exc:
                raise ValueError(f"unknown density colormap: {requested}") from exc

    # Dense regions should read bright/red by default. Sequential maps already
    # get brighter toward high values; RdBu needs reversing so high values are red.
    if requested.lower() == "rdbu":
        cmap = cmap.reversed()
    return cmap


def draw_ideal_arena(ax, arena_size_cm: tuple[float, float]) -> None:
    width_cm, height_cm = arena_size_cm
    ax.plot(
        [0, width_cm, width_cm, 0, 0],
        [0, 0, height_cm, height_cm, 0],
        color="#101010",
        linewidth=1.8,
        zorder=5,
    )
    ax.set_xlim(0, width_cm)
    ax.set_ylim(height_cm, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Rectified arena X (cm)")
    ax.set_ylabel("Rectified arena Y (cm)")


def plot_speed_colored_path(
    metrics: pd.DataFrame,
    out_path: Path,
    arena_size_cm: tuple[float, float],
    speed_color_center_cm_s: float,
) -> None:
    x = metrics["x_cm_smooth"].to_numpy(dtype=float)
    y = metrics["y_cm_smooth"].to_numpy(dtype=float)
    speed = metrics["speed_cm_s"].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    segment_mask = valid[1:] & valid[:-1] & np.isfinite(speed[1:])
    points = np.column_stack([x, y])
    segments = np.stack([points[:-1], points[1:]], axis=1)[segment_mask]
    segment_speeds = speed[1:][segment_mask]

    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.set_facecolor("#777777")
    draw_ideal_arena(ax, arena_size_cm)
    if len(segments):
        center = max(float(speed_color_center_cm_s), 1e-6)
        vmax = float(np.nanpercentile(segment_speeds, 99.5))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = float(np.nanmax(segment_speeds)) if len(segment_speeds) else 1.0
        vmax = max(vmax, center * 2.0, 1.0)
        norm = (
            TwoSlopeNorm(vmin=0.0, vcenter=center, vmax=vmax)
            if center < vmax
            else Normalize(vmin=0.0, vmax=vmax)
        )
        line_collection = LineCollection(
            segments,
            cmap="gray",
            norm=norm,
            linewidth=1.8,
            capstyle="round",
            joinstyle="round",
            zorder=4,
        )
        line_collection.set_array(segment_speeds)
        ax.add_collection(line_collection)
        cbar = fig.colorbar(line_collection, ax=ax, pad=0.02, fraction=0.035)
        cbar.set_label("Speed (cm/s)")
    ax.set_title("Bodycenter path")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_PLOT_DPI)
    plt.close(fig)


def plot_bodycenter_density(
    metrics: pd.DataFrame,
    out_path: Path,
    arena_size_cm: tuple[float, float],
    cmap_name: str,
    bins: int,
) -> None:
    width_cm, height_cm = arena_size_cm
    x = metrics["x_cm_smooth"].to_numpy(dtype=float)
    y = metrics["y_cm_smooth"].to_numpy(dtype=float)
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= 0)
        & (x <= width_cm)
        & (y >= 0)
        & (y <= height_cm)
    )
    x = x[valid]
    y = y[valid]
    bins = max(10, int(bins))
    x_bins = bins
    y_bins = max(6, int(round(bins * height_cm / max(width_cm, 1e-9))))
    hist, x_edges, y_edges = np.histogram2d(
        x,
        y,
        bins=[x_bins, y_bins],
        range=[[0, width_cm], [0, height_cm]],
    )
    hist = hist.T
    if hist.max() <= 0:
        fig, ax = plt.subplots(figsize=(12, 4.2))
        ax.set_facecolor("#f2f2f2")
        draw_ideal_arena(ax, arena_size_cm)
        ax.set_title("Bodycenter density (no valid positions)")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(out_path, dpi=DEFAULT_PLOT_DPI)
        plt.close(fig)
        return
    masked = np.ma.masked_where(hist <= 0, hist)
    cmap = get_density_colormap(cmap_name)
    cmap = cmap.copy()
    cmap.set_bad(color="#101010")
    norm = LogNorm(vmin=1, vmax=max(float(hist.max()), 1.0))

    fig, ax = plt.subplots(figsize=(12, 4.2))
    image = ax.imshow(
        masked,
        extent=[0, width_cm, height_cm, 0],
        origin="upper",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        aspect="equal",
    )
    draw_ideal_arena(ax, arena_size_cm)
    ax.set_title("Bodycenter density")
    ax.spines[["top", "right"]].set_visible(False)
    cbar = fig.colorbar(image, ax=ax, pad=0.02, fraction=0.035)
    cbar.set_label("Bodycenter count per bin")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_PLOT_DPI)
    plt.close(fig)


def plot_outputs(
    metrics: pd.DataFrame,
    summaries: dict[int, pd.DataFrame],
    out_dir: Path,
    prefix: str,
    arena_size_cm: tuple[float, float],
    density_cmap: str,
    density_bins: int,
    speed_color_center_cm_s: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths = analysis_output_paths(out_dir, prefix, density_cmap, summaries.keys())
    Path(output_paths["plots_dir"]).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(metrics["time_s"] / 60.0, metrics["speed_cm_s"], color="#3b6ea8", linewidth=0.8)
    ax.fill_between(
        metrics["time_s"] / 60.0,
        0,
        metrics["speed_cm_s"].fillna(0),
        where=metrics["moving"].to_numpy(dtype=bool),
        color="#ef8a35",
        alpha=0.25,
        linewidth=0,
    )
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Speed (cm/s)")
    ax.set_title("Bodycenter speed; orange areas meet the movement definition")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_paths["speed_trace_png"], dpi=DEFAULT_PLOT_DPI)
    plt.close(fig)

    plot_speed_colored_path(
        metrics,
        output_paths["path_speed_png"],
        arena_size_cm,
        speed_color_center_cm_s=speed_color_center_cm_s,
    )
    plot_bodycenter_density(
        metrics,
        output_paths["density_png"],
        arena_size_cm,
        density_cmap,
        density_bins,
    )

    for bin_seconds, summary in summaries.items():
        fig, ax = plt.subplots(figsize=(10, 4))
        minutes = summary["bin_start_s"] / 60.0
        width = bin_seconds / 60.0 * 0.85
        ax.bar(minutes, summary["movement_distance_cm"], width=width, color="#3b6ea8", align="edge")
        ax.set_xlabel("Bin start (min)")
        ax.set_ylabel("Movement distance (cm)")
        ax.set_title(f"Distance per {bin_seconds} s bin")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(output_paths["distance_bins_pngs"][int(bin_seconds)], dpi=DEFAULT_PLOT_DPI)
        plt.close(fig)


def command_calibrate(args: argparse.Namespace) -> None:
    if getattr(args, "known_px_per_cm", None) is not None:
        video_path = Path(args.video) if args.video else None
        if video_path is None:
            raise SystemExit("--known-px-per-cm requires --video")
        if args.known_px_per_cm <= 0:
            raise SystemExit("--known-px-per-cm must be > 0")
        meta = video_metadata(video_path)
        video_size = (meta["width"], meta["height"])
        arena_display_image = read_video_frame(video_path, args.arena_frame)
        if args.arena_corners:
            arena_corners_clicked = parse_point_list(args.arena_corners, 4)
        else:
            arena_corners_clicked = ginput_points(
                arena_display_image,
                "Click the four arena corners in any order",
                4,
            )
        arena_corners = order_arena_corners(arena_corners_clicked)
        out_path = Path(args.out)
        payload = write_calibration(
            out_path=out_path,
            reference_source={
                "type": "known_scale",
                "known_px_per_cm": float(args.known_px_per_cm),
            },
            video_path=video_path,
            image_size=video_size,
            video_size=video_size,
            crop=None,
            arena_corners_px=arena_corners,
            reference_segment_px=None,
            reference_length_cm=args.reference_length_cm,
            arena_corners_clicked_px=arena_corners_clicked,
            arena_source={
                "type": "video",
                "video": str(video_path),
                "frame": int(args.arena_frame),
            },
            known_px_per_cm=float(args.known_px_per_cm),
        )
        preview_path = Path(args.preview) if args.preview else out_path.with_suffix(".preview.png")
        save_calibration_preview(arena_display_image, arena_corners, None, preview_path)
        print(f"Wrote calibration: {out_path}")
        print(f"Wrote preview: {preview_path}")
        print(f"Arena corners were clicked on processed video frame {args.arena_frame}.")
        print(
            "Scale after rectification: "
            f"{payload['px_per_cm_rectified']:.3f} px/cm "
            f"({payload['cm_per_px_rectified']:.5f} cm/px)"
        )
        return

    image, reference_source = load_reference_frame(
        args.image,
        args.reference_video,
        args.reference_frame,
    )
    image_h, image_w = image.shape[:2]

    video_path = Path(args.video) if args.video else None
    video_size = None
    crop = None
    if args.crop:
        x, y, width, height = args.crop
        crop = {"x": x, "y": y, "width": width, "height": height, "method": "manual"}
        video_size = (width, height)
    elif video_path:
        meta = video_metadata(video_path)
        video_size = (meta["width"], meta["height"])
        video_frame = read_video_frame(video_path, args.crop_match_frame)
        crop = estimate_crop(image, video_frame)
        if crop["match_score"] < args.min_crop_match_score:
            print(
                "WARNING: crop match score is low "
                f"({crop['match_score']:.3f}); check the preview or pass --crop manually",
                file=sys.stderr,
            )

    reference_display_image = crop_image(image, crop)
    if args.arena_source == "video":
        if not video_path:
            raise SystemExit("--arena-source video requires --video")
        arena_display_image = read_video_frame(video_path, args.arena_frame)
    else:
        arena_display_image = reference_display_image

    if args.arena_corners:
        arena_corners_clicked = parse_point_list(args.arena_corners, 4)
    else:
        arena_corners_clicked = ginput_points(
            arena_display_image,
            "Click the four arena corners in any order",
            4,
        )
    arena_corners = order_arena_corners(arena_corners_clicked)
    reference_detection = None
    if args.reference_points:
        reference_segment = parse_point_list(args.reference_points, 2)
    elif args.reference_mode == "roi":
        if args.reference_roi:
            reference_roi = tuple(int(v) for v in args.reference_roi)
        else:
            roi_points = ginput_points(
                reference_display_image,
                f"Click the top-left and bottom-right ROI corners around the {args.reference_length_cm:g} cm block",
                2,
            )
            reference_roi = roi_from_points(roi_points, reference_display_image.shape)
        reference_detection = detect_reference_block(
            reference_display_image,
            roi=reference_roi,
            min_aspect_ratio=args.min_aspect_ratio,
        )
        reference_segment = np.asarray(reference_detection["segment_px"], dtype=np.float32)
    else:
        reference_segment = ginput_points(
            reference_display_image,
            f"Click the two endpoints of the {args.reference_length_cm:g} cm block",
            2,
        )

    out_path = Path(args.out)
    payload = write_calibration(
        out_path=out_path,
        reference_source=reference_source,
        video_path=video_path,
        image_size=(image_w, image_h),
        video_size=video_size,
        crop=crop,
        arena_corners_px=arena_corners,
        reference_segment_px=reference_segment,
        reference_length_cm=args.reference_length_cm,
        arena_corners_clicked_px=arena_corners_clicked,
        arena_source={
            "type": args.arena_source,
            "video": str(video_path) if args.arena_source == "video" and video_path else None,
            "frame": int(args.arena_frame) if args.arena_source == "video" else None,
        },
        reference_detection=reference_detection,
    )

    preview_path = Path(args.preview) if args.preview else out_path.with_suffix(".preview.png")
    save_calibration_preview(reference_display_image, arena_corners, reference_segment, preview_path)
    print(f"Wrote calibration: {out_path}")
    print(f"Wrote preview: {preview_path}")
    if not np.allclose(arena_corners_clicked, arena_corners):
        print("Arena corners were automatically ordered as TL,TR,BR,BL.")
    if args.arena_source == "video":
        print(f"Arena corners were clicked on processed video frame {args.arena_frame}.")
    if reference_source["type"] == "video_frame":
        print(
            "Reference block frame: "
            f"{reference_source['video']} frame {reference_source['frame']}."
        )
    if reference_detection:
        print(
            "Reference block detected from ROI: "
            f"bbox={reference_detection['bbox_px']} "
            f"segment={reference_detection['segment_px']}."
        )
    if crop:
        print(
            "Crop used: "
            f"x={crop['x']} y={crop['y']} width={crop['width']} height={crop['height']} "
            f"score={crop.get('match_score', 'manual')}"
        )
    print(
        "Scale after rectification: "
        f"{payload['px_per_cm_rectified']:.3f} px/cm "
        f"({payload['cm_per_px_rectified']:.5f} cm/px)"
    )


def command_analyze(args: argparse.Namespace) -> None:
    tracking_path = Path(args.tracking)
    calibration = load_calibration(Path(args.calibration))
    video_path = Path(args.video) if args.video else None
    video_meta = video_metadata(video_path) if video_path is not None else None
    fps = args.fps
    if fps is None:
        if video_meta is None:
            raise SystemExit("provide --video or --fps")
        fps = video_meta["fps"]
    if fps <= 0:
        raise SystemExit("fps must be > 0")

    df = read_dlc_table(tracking_path)
    track = extract_bodypart(df, args.bodypart)
    frame_count_validation = None
    if video_meta is not None:
        try:
            frame_count_validation = validate_tracking_video_frame_count(
                tracking_frame_count=len(track),
                video_frame_count=video_meta["frame_count"],
                fps=fps,
            )
        except ValueError as exc:
            raise ValueError(
                f"{exc}\nTracking: {tracking_path}\nVideo: {video_path}"
            ) from exc
        difference = int(frame_count_validation["difference_frames"])
        status = "exact match" if difference == 0 else "within 1-frame tolerance"
        print(
            "Frame-count validation passed: "
            f"tracking={len(track)}, video={video_meta['frame_count']} ({status})."
        )
    positions = calibrated_positions(
        track,
        calibration=calibration,
        likelihood_threshold=args.likelihood_threshold,
        arena_margin_cm=args.arena_margin_cm,
    )
    metrics = compute_motion_metrics(
        positions,
        fps=fps,
        max_gap_sec=args.max_gap_sec,
        smooth_window_sec=args.smooth_window_sec,
        speed_threshold_cm_s=args.speed_threshold_cm_s,
        stop_gap_sec=args.stop_gap_sec,
        min_bout_sec=args.min_bout_sec,
        min_bout_displacement_cm=args.min_bout_displacement_cm,
    )
    arena_size_cm = (
        calibration.rectified_size_px[0] * calibration.cm_per_px,
        calibration.rectified_size_px[1] * calibration.cm_per_px,
    )
    edge_width_cm = float(getattr(args, "edge_width_cm", DEFAULT_EDGE_WIDTH_CM))
    metrics = add_edge_regions(metrics, arena_size_cm, edge_width_cm)
    transit_edge_tolerance_cm = float(
        getattr(
            args,
            "transit_edge_tolerance_cm",
            DEFAULT_TRANSIT_EDGE_TOLERANCE_CM,
        )
    )
    if transit_edge_tolerance_cm < 0:
        raise ValueError("transit edge tolerance must be >= 0")
    transit_metrics = add_edge_regions(
        metrics,
        arena_size_cm,
        edge_width_cm + transit_edge_tolerance_cm,
    )

    wall_posture_check = bool(
        getattr(args, "wall_posture_check", DEFAULT_WALL_POSTURE_CHECK)
    )
    posture_metrics: pd.DataFrame | None = None
    if wall_posture_check:
        try:
            posture_metrics = compute_wall_posture_frame_metrics(
                df,
                calibration=calibration,
                likelihood_threshold=args.likelihood_threshold,
                arena_margin_cm=args.arena_margin_cm,
                nose_bodypart=str(getattr(args, "posture_nose_bodypart", "nose")),
                center_bodypart=str(
                    getattr(args, "posture_center_bodypart", "bodycenter")
                ),
                tailbase_bodypart=str(
                    getattr(args, "posture_tailbase_bodypart", "tailbase")
                ),
            )
        except ValueError as exc:
            raise ValueError(
                f"wall-posture check could not be prepared: {exc}. "
                "Provide nose/bodycenter/tailbase tracks or disable the check."
            ) from exc
        if len(posture_metrics) != len(metrics):
            raise ValueError(
                "wall-posture metrics and movement metrics have different row counts: "
                f"{len(posture_metrics)} vs {len(metrics)}"
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or infer_prefix(tracking_path)
    plot_paths = analysis_output_paths(
        out_dir,
        prefix,
        args.density_cmap,
        args.bin_seconds,
    )
    metrics_dir = Path(plot_paths["metrics_dir"])
    events_dir = Path(plot_paths["events_dir"])
    plots_dir = Path(plot_paths["plots_dir"])
    qc_dir = Path(plot_paths["qc_dir"])
    for directory in (metrics_dir, events_dir, plots_dir, qc_dir):
        directory.mkdir(parents=True, exist_ok=True)

    metrics_path = metrics_dir / f"{prefix}_frame_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    overall = summarize_metrics(metrics, fps=fps, bin_seconds=None)
    overall_path = metrics_dir / f"{prefix}_summary_overall.csv"
    overall.to_csv(overall_path, index=False)

    edge_summary = summarize_edge_regions(
        metrics,
        fps=fps,
        arena_size_cm=arena_size_cm,
        edge_width_cm=edge_width_cm,
    )
    edge_summary_path = metrics_dir / f"{prefix}_edge_region_summary.csv"
    edge_summary.to_csv(edge_summary_path, index=False)

    transit_candidates = detect_straight_transit_events(
        transit_metrics,
        fps=fps,
        arena_size_cm=arena_size_cm,
        edge_width_cm=edge_width_cm,
        path_efficiency_threshold=float(
            getattr(
                args,
                "straight_path_efficiency_threshold",
                DEFAULT_STRAIGHT_PATH_EFFICIENCY_THRESHOLD,
            )
        ),
        max_deviation_cm=float(getattr(args, "straight_max_deviation_cm", 6.0)),
        stop_speed_threshold_cm_s=float(
            getattr(args, "event_stop_speed_threshold_cm_s", args.speed_threshold_cm_s)
        ),
        stop_min_sec=float(
            getattr(args, "straight_stop_min_sec", DEFAULT_STRAIGHT_STOP_MIN_SEC)
        ),
        low_speed_fraction_threshold=float(
            getattr(
                args,
                "straight_low_speed_fraction_threshold",
                DEFAULT_STRAIGHT_LOW_SPEED_FRACTION_THRESHOLD,
            )
        ),
        max_step_jump_cm=float(getattr(args, "straight_max_step_jump_cm", 8.0)),
        min_valid_fraction=float(getattr(args, "straight_min_valid_fraction", 0.95)),
        borderline_margin=float(getattr(args, "straight_borderline_margin", 0.10)),
    )
    if posture_metrics is not None:
        transit_candidates = add_wall_posture_classification(
            transit_candidates,
            posture_metrics=posture_metrics,
            wall_distance_cm=float(
                getattr(
                    args,
                    "posture_wall_distance_cm",
                    DEFAULT_POSTURE_WALL_DISTANCE_CM,
                )
            ),
            center_away_cm=float(
                getattr(
                    args,
                    "posture_center_away_cm",
                    DEFAULT_POSTURE_CENTER_AWAY_CM,
                )
            ),
            min_fraction=float(
                getattr(
                    args,
                    "posture_min_fraction",
                    DEFAULT_POSTURE_MIN_FRACTION,
                )
            ),
            borderline_fraction=float(
                getattr(
                    args,
                    "posture_borderline_fraction",
                    DEFAULT_POSTURE_BORDERLINE_FRACTION,
                )
            ),
            min_valid_fraction=float(
                getattr(
                    args,
                    "posture_min_valid_fraction",
                    DEFAULT_POSTURE_MIN_VALID_FRACTION,
                )
            ),
            motion_metrics=metrics,
            fps=fps,
            interruption_max_speed_cm_s=float(
                getattr(
                    args,
                    "posture_interruption_max_speed_cm_s",
                    DEFAULT_POSTURE_INTERRUPTION_MAX_SPEED_CM_S,
                )
            ),
            interruption_min_sec=float(
                getattr(
                    args,
                    "posture_interruption_min_sec",
                    DEFAULT_POSTURE_INTERRUPTION_MIN_SEC,
                )
            ),
        )
    transit_candidates_path = events_dir / f"{prefix}_straight_transit_candidates.csv"
    transit_candidates.to_csv(transit_candidates_path, index=False)
    if transit_candidates.empty:
        transit_events = transit_candidates.copy()
    else:
        transit_events = transit_candidates.loc[
            transit_candidates["is_straight"].astype(bool)
        ].copy()
    transit_events_path = events_dir / f"{prefix}_straight_transit_events.csv"
    transit_events.to_csv(transit_events_path, index=False)
    transit_summary = summarize_straight_transits(transit_candidates)
    transit_summary_path = events_dir / f"{prefix}_straight_transit_summary.csv"
    transit_summary.to_csv(transit_summary_path, index=False)

    back_and_forth_events = detect_back_and_forth_events(
        metrics,
        transit_candidates,
        fps=fps,
        time_window_sec=float(getattr(args, "back_forth_time_window_sec", 30.0)),
        min_round_trips=int(getattr(args, "back_forth_min_round_trips", 1)),
    )
    back_and_forth_events_path = events_dir / f"{prefix}_back_and_forth_events.csv"
    back_and_forth_events.to_csv(back_and_forth_events_path, index=False)
    back_and_forth_summary = summarize_back_and_forth_events(
        back_and_forth_events,
        duration_s=len(metrics) / fps,
    )
    back_and_forth_summary_path = events_dir / f"{prefix}_back_and_forth_summary.csv"
    back_and_forth_summary.to_csv(back_and_forth_summary_path, index=False)

    summaries: dict[int, pd.DataFrame] = {}
    bin_summary_paths: dict[int, Path] = {}
    for bin_seconds in args.bin_seconds:
        summary = summarize_metrics(metrics, fps=fps, bin_seconds=bin_seconds)
        summaries[int(bin_seconds)] = summary
        bin_path = metrics_dir / f"{prefix}_summary_bins_{int(bin_seconds)}s.csv"
        bin_summary_paths[int(bin_seconds)] = bin_path
        summary.to_csv(bin_path, index=False)

    params = {
        "tracking": str(tracking_path),
        "video": str(args.video) if args.video else None,
        "calibration": str(calibration.path),
        "fps": fps,
        "frame_count_validation": frame_count_validation,
        "bodypart": args.bodypart,
        "likelihood_threshold": args.likelihood_threshold,
        "speed_threshold_cm_s": args.speed_threshold_cm_s,
        "max_gap_sec": args.max_gap_sec,
        "smooth_window_sec": args.smooth_window_sec,
        "stop_gap_sec": args.stop_gap_sec,
        "min_bout_sec": args.min_bout_sec,
        "min_bout_displacement_cm": args.min_bout_displacement_cm,
        "arena_margin_cm": args.arena_margin_cm,
        "edge_width_cm": edge_width_cm,
        "transit_edge_tolerance_cm": transit_edge_tolerance_cm,
        "straight_path_efficiency_threshold": float(
            getattr(
                args,
                "straight_path_efficiency_threshold",
                DEFAULT_STRAIGHT_PATH_EFFICIENCY_THRESHOLD,
            )
        ),
        "straight_max_deviation_cm": float(getattr(args, "straight_max_deviation_cm", 6.0)),
        "event_stop_speed_threshold_cm_s": float(
            getattr(args, "event_stop_speed_threshold_cm_s", args.speed_threshold_cm_s)
        ),
        "straight_stop_min_sec": float(
            getattr(args, "straight_stop_min_sec", DEFAULT_STRAIGHT_STOP_MIN_SEC)
        ),
        "straight_low_speed_fraction_threshold": float(
            getattr(
                args,
                "straight_low_speed_fraction_threshold",
                DEFAULT_STRAIGHT_LOW_SPEED_FRACTION_THRESHOLD,
            )
        ),
        "straight_max_step_jump_cm": float(getattr(args, "straight_max_step_jump_cm", 8.0)),
        "straight_min_valid_fraction": float(
            getattr(args, "straight_min_valid_fraction", 0.95)
        ),
        "straight_borderline_margin": float(getattr(args, "straight_borderline_margin", 0.10)),
        "wall_posture_check": wall_posture_check,
        "posture_nose_bodypart": str(getattr(args, "posture_nose_bodypart", "nose")),
        "posture_center_bodypart": str(
            getattr(args, "posture_center_bodypart", "bodycenter")
        ),
        "posture_tailbase_bodypart": str(
            getattr(args, "posture_tailbase_bodypart", "tailbase")
        ),
        "posture_wall_distance_cm": float(
            getattr(
                args,
                "posture_wall_distance_cm",
                DEFAULT_POSTURE_WALL_DISTANCE_CM,
            )
        ),
        "posture_center_away_cm": float(
            getattr(
                args,
                "posture_center_away_cm",
                DEFAULT_POSTURE_CENTER_AWAY_CM,
            )
        ),
        "posture_min_fraction": float(
            getattr(args, "posture_min_fraction", DEFAULT_POSTURE_MIN_FRACTION)
        ),
        "posture_borderline_fraction": float(
            getattr(
                args,
                "posture_borderline_fraction",
                DEFAULT_POSTURE_BORDERLINE_FRACTION,
            )
        ),
        "posture_min_valid_fraction": float(
            getattr(
                args,
                "posture_min_valid_fraction",
                DEFAULT_POSTURE_MIN_VALID_FRACTION,
            )
        ),
        "posture_interruption_max_speed_cm_s": float(
            getattr(
                args,
                "posture_interruption_max_speed_cm_s",
                DEFAULT_POSTURE_INTERRUPTION_MAX_SPEED_CM_S,
            )
        ),
        "posture_interruption_min_sec": float(
            getattr(
                args,
                "posture_interruption_min_sec",
                DEFAULT_POSTURE_INTERRUPTION_MIN_SEC,
            )
        ),
        "back_forth_time_window_sec": float(
            getattr(args, "back_forth_time_window_sec", 30.0)
        ),
        "back_forth_min_round_trips": int(getattr(args, "back_forth_min_round_trips", 1)),
        "density_cmap": args.density_cmap,
        "density_bins": args.density_bins,
        "speed_color_center_cm_s": args.speed_color_center_cm_s,
        "outputs": {
            "metrics_dir": str(metrics_dir),
            "events_dir": str(events_dir),
            "plots_dir": str(plots_dir),
            "qc_dir": str(qc_dir),
            "frame_metrics_csv": str(metrics_path),
            "overall_summary_csv": str(overall_path),
            "edge_region_summary_csv": str(edge_summary_path),
            "straight_transit_candidates_csv": str(transit_candidates_path),
            "straight_transit_events_csv": str(transit_events_path),
            "straight_transit_summary_csv": str(transit_summary_path),
            "back_and_forth_events_csv": str(back_and_forth_events_path),
            "back_and_forth_summary_csv": str(back_and_forth_summary_path),
            "bin_summary_csvs": {
                str(bin_seconds): str(path)
                for bin_seconds, path in bin_summary_paths.items()
            },
            "plot_pngs": {
                "speed_trace": str(plot_paths["speed_trace_png"]),
                "path_speed": str(plot_paths["path_speed_png"]),
                "density": str(plot_paths["density_png"]),
                "distance_bins": {
                    str(seconds): str(path)
                    for seconds, path in plot_paths["distance_bins_pngs"].items()
                },
            },
        },
    }
    qc_video_path = qc_dir / f"{prefix}_event_qc.mp4"
    qc_clip_paths: list[Path] = []
    if bool(getattr(args, "qc_video", False)):
        if not args.video:
            raise SystemExit("--qc-video requires --video")
        create_qc_video(
            Path(args.video),
            metrics,
            calibration,
            qc_video_path,
            fps,
            arena_size_cm,
            edge_width_cm,
            transit_candidates,
            back_and_forth_events,
        )
        params["outputs"]["qc_video"] = str(qc_video_path)
    if bool(getattr(args, "qc_event_clips", False)):
        if not args.video:
            raise SystemExit("--qc-event-clips requires --video")
        qc_clip_paths = create_event_qc_clips(
            Path(args.video),
            metrics,
            calibration,
            qc_dir,
            prefix,
            fps,
            arena_size_cm,
            edge_width_cm,
            transit_candidates,
            back_and_forth_events,
            prepost_sec=float(getattr(args, "qc_clip_padding_sec", 1.0)),
        )
        params["outputs"]["qc_event_clips"] = [str(path) for path in qc_clip_paths]
    if not args.no_plots:
        plot_outputs(
            metrics,
            summaries,
            out_dir,
            prefix,
            arena_size_cm=arena_size_cm,
            density_cmap=args.density_cmap,
            density_bins=args.density_bins,
            speed_color_center_cm_s=args.speed_color_center_cm_s,
        )
    parameters_path = metrics_dir / f"{prefix}_analysis_parameters.json"
    parameters_path.write_text(
        json.dumps(params, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Wrote analysis parameters: {parameters_path}")
    print(f"Wrote per-frame metrics: {metrics_path}")
    print(f"Wrote overall summary: {overall_path}")
    print(f"Wrote edge-region summary: {edge_summary_path}")
    print(f"Wrote straight transit candidates: {transit_candidates_path}")
    print(f"Wrote accepted straight transit events: {transit_events_path}")
    print(f"Wrote back-and-forth events: {back_and_forth_events_path}")
    if bool(getattr(args, "qc_video", False)):
        print(f"Wrote QC video: {qc_video_path}")
    if qc_clip_paths:
        print(f"Wrote {len(qc_clip_paths)} QC event clip(s): {qc_dir / 'event_clips'}")
    for bin_seconds in args.bin_seconds:
        print(f"Wrote {int(bin_seconds)} s bins: {bin_summary_paths[int(bin_seconds)]}")
    print(
        "Overall movement distance: "
        f"{float(overall['movement_distance_cm'].iloc[0]):.2f} cm; "
        "moving time: "
        f"{float(overall['moving_time_s'].iloc[0]):.2f} s"
    )
    straight_count = int(transit_summary["straight_transit_count"].iloc[0])
    shuttle_count = int(back_and_forth_summary["back_and_forth_event_count"].iloc[0])
    print(f"Straight transit events: {straight_count}; back-and-forth events: {shuttle_count}")


def command_detect_crop(args: argparse.Namespace) -> None:
    image = read_image(Path(args.image))
    frame = read_video_frame(Path(args.video), args.frame)
    crop = estimate_crop(image, frame)
    print(json.dumps(crop, indent=2, ensure_ascii=False))
    if args.preview:
        preview = image.copy()
        x, y, w, h = crop["x"], crop["y"], crop["width"], crop["height"]
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 255), 3)
        out = Path(args.preview)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), preview)


def command_detect_reference(args: argparse.Namespace) -> None:
    calibration = load_calibration(Path(args.calibration)) if args.calibration else None
    payload = (
        json.loads(Path(args.calibration).read_text(encoding="utf-8"))
        if args.calibration
        else {}
    )
    try:
        image, reference_source = load_reference_frame_from_args_or_payload(args, payload)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    crop = None
    if args.crop:
        x, y, width, height = args.crop
        crop = {"x": x, "y": y, "width": width, "height": height, "method": "manual"}
    elif payload.get("crop_from_reference_image_px"):
        crop = payload["crop_from_reference_image_px"]
    elif args.video:
        crop = estimate_crop(image, read_video_frame(Path(args.video), args.crop_match_frame))
    display_image = crop_image(image, crop)

    if args.roi:
        roi = tuple(int(v) for v in args.roi)
    else:
        roi_points = ginput_points(
            display_image,
            "Click the top-left and bottom-right corners of an ROI around the 9 cm block",
            2,
        )
        roi = roi_from_points(roi_points, display_image.shape)

    detection = detect_reference_block(
        display_image,
        roi=roi,
        min_aspect_ratio=args.min_aspect_ratio,
    )
    segment = np.asarray(detection["segment_px"], dtype=np.float32)
    raw_length_px = float(np.linalg.norm(segment[1] - segment[0]))
    result = {
        "reference_source": reference_source,
        "crop_from_reference_image_px": crop,
        "roi_px": list(detection["roi_px"]),
        "method": detection["method"],
        "detected_bbox_px": list(detection["bbox_px"]),
        "detected_segment_px": detection["segment_px"],
        "detected_raw_length_px": raw_length_px,
        "reference_length_cm": float(args.reference_length_cm),
        "detected_raw_px_per_cm": raw_length_px / args.reference_length_cm,
    }

    if calibration is not None:
        detected_rectified = transform_points(segment, calibration.homography)
        detected_rectified_length_px = float(
            np.linalg.norm(detected_rectified[1] - detected_rectified[0])
        )
        detected_px_per_cm = detected_rectified_length_px / args.reference_length_cm
        manual_px_per_cm = calibration.px_per_cm
        comparison = {
            "manual_px_per_cm_rectified": manual_px_per_cm,
            "detected_rectified_length_px": detected_rectified_length_px,
            "detected_px_per_cm_rectified": detected_px_per_cm,
            "difference_px_per_cm": detected_px_per_cm - manual_px_per_cm,
            "difference_percent": (detected_px_per_cm / manual_px_per_cm - 1.0) * 100.0,
        }
        if calibration.reference_segment_px is not None:
            comparison["manual_segment_px"] = calibration.reference_segment_px.tolist()
            comparison["manual_rectified_length_px"] = (
                calibration.px_per_cm * calibration.reference_length_cm
            )
        result.update(comparison)

    if args.preview:
        draw_reference_detection_preview(
            display_image,
            roi=tuple(result["roi_px"]),
            segment=segment,
            bbox=tuple(result["detected_bbox_px"]),
            out_path=Path(args.preview),
        )
        result["preview"] = args.preview
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(result, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate and analyze DeepLabCut bodycenter movement in a linear-track arena."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibrate = subparsers.add_parser("calibrate", help="create calibration JSON by clicking points")
    calibrate.add_argument("--image", help="reference image containing the known block")
    calibrate.add_argument(
        "--reference-video",
        help="original video containing the known block; use with --reference-frame",
    )
    calibrate.add_argument(
        "--reference-frame",
        type=int,
        default=0,
        help="frame index in --reference-video where the known block is visible",
    )
    calibrate.add_argument("--video", help="processed video whose coordinate frame matches DLC")
    calibrate.add_argument(
        "--crop",
        nargs=4,
        type=int,
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
        help="manual crop from reference image to DLC/video coordinates",
    )
    calibrate.add_argument(
        "--crop-match-frame",
        type=int,
        default=0,
        help="processed-video frame used to estimate crop in the reference image",
    )
    calibrate.add_argument(
        "--min-crop-match-score",
        type=float,
        default=0.65,
        help="warn if automatic crop template-match score is below this value",
    )
    calibrate.add_argument(
        "--arena-corners",
        help="optional noninteractive four arena corner points in any order: 'x,y;x,y;x,y;x,y'",
    )
    calibrate.add_argument(
        "--reference-points",
        help="optional noninteractive endpoints of known block: 'x,y;x,y'",
    )
    calibrate.add_argument(
        "--reference-mode",
        choices=("points", "roi"),
        default="points",
        help="measure the known block by endpoint clicks or by human ROI + automatic block detection",
    )
    calibrate.add_argument(
        "--reference-roi",
        nargs=4,
        type=int,
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
        help="ROI around the known block in displayed/DLC coordinates for --reference-mode roi",
    )
    calibrate.add_argument("--reference-length-cm", type=float, default=9.0)
    calibrate.add_argument("--min-aspect-ratio", type=float, default=1.8)
    calibrate.add_argument(
        "--known-px-per-cm",
        type=float,
        help="use this rectified px/cm scale and skip reference-block measurement",
    )
    calibrate.add_argument(
        "--arena-source",
        choices=("reference", "video"),
        default="reference",
        help="click arena corners on the cropped reference image or on --video directly",
    )
    calibrate.add_argument(
        "--arena-frame",
        type=int,
        default=0,
        help="processed-video frame used when --arena-source video",
    )
    calibrate.add_argument("--out", required=True, help="output calibration JSON")
    calibrate.add_argument("--preview", help="output preview image path")
    calibrate.set_defaults(func=command_calibrate)

    analyze = subparsers.add_parser("analyze", help="analyze a DLC CSV/H5 using a calibration JSON")
    analyze.add_argument("--tracking", required=True, help="DLC .csv or .h5")
    analyze.add_argument("--calibration", required=True, help="calibration JSON from calibrate")
    analyze.add_argument("--video", help="processed video for FPS")
    analyze.add_argument("--fps", type=float, help="FPS if no --video is provided")
    analyze.add_argument("--bodypart", default="bodycenter")
    analyze.add_argument("--likelihood-threshold", type=float, default=0.90)
    analyze.add_argument("--speed-threshold-cm-s", type=float, default=1.0)
    analyze.add_argument("--max-gap-sec", type=float, default=0.50)
    analyze.add_argument("--smooth-window-sec", type=float, default=0.20)
    analyze.add_argument("--stop-gap-sec", type=float, default=0.20)
    analyze.add_argument("--min-bout-sec", type=float, default=0.20)
    analyze.add_argument("--min-bout-displacement-cm", type=float, default=0.50)
    analyze.add_argument("--arena-margin-cm", type=float, default=2.0)
    analyze.add_argument("--edge-width-cm", type=float, default=DEFAULT_EDGE_WIDTH_CM)
    analyze.add_argument(
        "--transit-edge-tolerance-cm",
        type=float,
        default=DEFAULT_TRANSIT_EDGE_TOLERANCE_CM,
        help="extra edge reach used only to end transit events",
    )
    analyze.add_argument(
        "--straight-path-efficiency-threshold",
        type=float,
        default=DEFAULT_STRAIGHT_PATH_EFFICIENCY_THRESHOLD,
    )
    analyze.add_argument("--straight-max-deviation-cm", type=float, default=6.0)
    analyze.add_argument("--event-stop-speed-threshold-cm-s", type=float, default=1.0)
    analyze.add_argument(
        "--straight-stop-min-sec",
        type=float,
        default=DEFAULT_STRAIGHT_STOP_MIN_SEC,
    )
    analyze.add_argument(
        "--straight-low-speed-fraction-threshold",
        type=float,
        default=DEFAULT_STRAIGHT_LOW_SPEED_FRACTION_THRESHOLD,
    )
    analyze.add_argument("--straight-max-step-jump-cm", type=float, default=8.0)
    analyze.add_argument("--straight-min-valid-fraction", type=float, default=0.95)
    analyze.add_argument("--straight-borderline-margin", type=float, default=0.10)
    analyze.add_argument(
        "--wall-posture-check",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_WALL_POSTURE_CHECK,
        help=(
            "reject sustained nose-at-wall/bodycenter-away posture; "
            "use --no-wall-posture-check to disable"
        ),
    )
    analyze.add_argument("--posture-nose-bodypart", default="nose")
    analyze.add_argument("--posture-center-bodypart", default="bodycenter")
    analyze.add_argument("--posture-tailbase-bodypart", default="tailbase")
    analyze.add_argument(
        "--posture-wall-distance-cm",
        type=float,
        default=DEFAULT_POSTURE_WALL_DISTANCE_CM,
    )
    analyze.add_argument(
        "--posture-center-away-cm",
        type=float,
        default=DEFAULT_POSTURE_CENTER_AWAY_CM,
    )
    analyze.add_argument(
        "--posture-min-fraction",
        type=float,
        default=DEFAULT_POSTURE_MIN_FRACTION,
    )
    analyze.add_argument(
        "--posture-borderline-fraction",
        type=float,
        default=DEFAULT_POSTURE_BORDERLINE_FRACTION,
    )
    analyze.add_argument(
        "--posture-min-valid-fraction",
        type=float,
        default=DEFAULT_POSTURE_MIN_VALID_FRACTION,
    )
    analyze.add_argument(
        "--posture-interruption-max-speed-cm-s",
        type=float,
        default=DEFAULT_POSTURE_INTERRUPTION_MAX_SPEED_CM_S,
        help="maximum bodycenter speed supporting a wall-posture interruption",
    )
    analyze.add_argument(
        "--posture-interruption-min-sec",
        type=float,
        default=DEFAULT_POSTURE_INTERRUPTION_MIN_SEC,
        help="minimum consecutive low-speed wall-posture duration",
    )
    analyze.add_argument("--back-forth-time-window-sec", type=float, default=30.0)
    analyze.add_argument("--back-forth-min-round-trips", type=int, default=1)
    analyze.add_argument("--qc-video", action="store_true")
    analyze.add_argument("--qc-event-clips", action="store_true")
    analyze.add_argument("--qc-clip-padding-sec", type=float, default=1.0)
    analyze.add_argument("--bin-seconds", nargs="+", type=int, default=[60, 120])
    analyze.add_argument(
        "--density-cmap",
        default="coolwarm",
        help="density heatmap colormap, e.g. CMRmap, rocket, magma, viridis, coolwarm, RdBu",
    )
    analyze.add_argument("--density-bins", type=int, default=90)
    analyze.add_argument("--speed-color-center-cm-s", type=float, default=20.0)
    analyze.add_argument("--out-dir", default="outputs/motion")
    analyze.add_argument("--prefix")
    analyze.add_argument("--no-plots", action="store_true")
    analyze.set_defaults(func=command_analyze)

    detect_crop = subparsers.add_parser("detect-crop", help="estimate processed-video crop in a reference image")
    detect_crop.add_argument("--image", required=True)
    detect_crop.add_argument("--video", required=True)
    detect_crop.add_argument("--frame", type=int, default=0)
    detect_crop.add_argument("--preview", help="optional preview image with crop rectangle")
    detect_crop.set_defaults(func=command_detect_crop)

    detect_reference = subparsers.add_parser(
        "detect-reference",
        help="semi-automatically detect the 9 cm reference block inside a user ROI",
    )
    detect_reference.add_argument("--calibration", help="existing calibration JSON to compare against")
    detect_reference.add_argument("--image", help="reference image containing the known block")
    detect_reference.add_argument(
        "--reference-video",
        help="original video containing the known block; use with --reference-frame",
    )
    detect_reference.add_argument("--reference-frame", type=int, default=0)
    detect_reference.add_argument("--video", help="processed video used only to estimate crop if needed")
    detect_reference.add_argument(
        "--crop",
        nargs=4,
        type=int,
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
        help="manual crop from reference image to DLC/video coordinates",
    )
    detect_reference.add_argument("--crop-match-frame", type=int, default=0)
    detect_reference.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
        help="ROI around the 9 cm block in displayed/DLC coordinates",
    )
    detect_reference.add_argument("--reference-length-cm", type=float, default=9.0)
    detect_reference.add_argument("--min-aspect-ratio", type=float, default=1.8)
    detect_reference.add_argument("--preview", help="output preview image path")
    detect_reference.add_argument("--out", help="optional JSON result path")
    detect_reference.set_defaults(func=command_detect_reference)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
