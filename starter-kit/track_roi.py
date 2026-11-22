"""Unified Kalman + ByteTrack + CMC + Upscaled ROI Crop & Discovery Tracker for C-UAS MOT.

Unifies the best of ``track_roi.py`` and ``Kalman/kalman_tracker.py`` with zero external
filter dependencies (pure NumPy Kalman + SciPy Hungarian + optional OpenCV CMC):

Per frame:
  1. Estimate frame-to-frame camera motion (sparse LK optical flow + RANSAC affine CMC)
     and compensate all active Kalman track positions before prediction;
  2. Predict every track forward with a constant-velocity Kalman filter (with pixel
     noise floors so tiny 7x7 px drones can adapt velocity quickly);
  3. Run the detector on the full frame (384x384 global pass);
  4. For up to ``--max-crops`` small tracks, run the detector on a square ROI crop
     centred on the predicted position, allowing crops smaller than ``img_size``
     (e.g. 192x192 upscaled 2x -> 384x384) so 7-10 px drones in 1080p become 15-25 px;
  5. (Optional ``--discover``) When idle (no confirmed track) on high-res video, scan
     ``--discover-budget`` overlapping tiles per frame in round-robin to discover tiny
     targets invisible to the global pass at a flat latency budget; once hit, step 4
     takes over at 1-2 passes/frame;
  6. Merge detections via NMS, associate with ByteTrack-style two-stage Hungarian
     matching (IoU + adaptive centre distance), vote track class by score, backfill
     initial confirmation frames, and linearly interpolate short re-acquired gaps.

Supports both full ONNX (``--onnx``) and partitioned Metis-style split ONNX
(``--backbone-onnx`` + ``--decoder-onnx``), plus detection caching (``--save-dets`` /
``--load-dets``) for rapid hyperparameter sweeps in tracker-only mode.

Usage (from starter-kit/):
    python track_roi.py --onnx exports/rfdetr_20ep/full/inference_model.onnx \\
        --eval-dir ../tracking_eval --out-dir pred_mot_unified/ --split valid
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from predict_onnx_baseline import (
    _decode_rfdetr_topk,
    _iou_pairwise,
    _nms_dets,
    _preprocess_imagenet,
)

try:
    import cv2
except ImportError:  # pragma: no cover - optional fallback if OpenCV is absent
    cv2 = None

Det = tuple[int, float, list[float]]  # (class_id 0-based, score, [x, y, w, h] in frame px)
Region = tuple[int, int, int]  # (x0, y0, side) square crop in frame px


@dataclass
class Config:
    img_size: int = 384
    num_classes: int = 3
    top_k: int = 10
    crop_top_k: int = 5
    high_thr: float = 0.40  # detections allowed to start tracks / matched first
    low_thr: float = 0.10  # global detections below this are dropped
    crop_score_thr: float = 0.15
    nms_iou: float = 0.50
    iou_gate: float = 0.10
    dist_gate: float = 1.50  # centre distance gate in units of max(box_size, 12px)
    min_hits: int = 3
    max_age: int = 45  # frames a confirmed track survives occlusion (~1.5s at 30fps)
    tentative_max_misses: int = 2
    # Kalman noise floors (crucial for tiny 7x7 px targets) & occlusion damping
    min_pos_std_px: float = 1.0
    min_vel_std_px: float = 0.5
    vel_damping: float = 0.92  # gentle velocity decay while coasting behind an obstacle
    occlusion_reid_gate: float = 4.5  # OC-SORT style recovery radius (in box-size units)
    # Camera Motion Compensation (sparse LK + RANSAC affine)
    use_cmc: bool = True
    # Track-guided upscaled ROI crops
    use_crops: bool = True
    max_crops: int = 3
    crop_scale: float = 8.0  # crop side = max(min_crop_side, crop_scale * target size)
    min_crop_side: int = 192  # < img_size enables 2x upscaling for tiny drones!
    crop_skip_px: float = 32.0  # skip crop if target already >= this in global pass
    # Motion-prioritized + round-robin tile discovery while idle (opt-in --discover)
    use_discover: bool = False
    discover_tile_side: int = 256
    discover_overlap: float = 0.15
    discover_budget_idle: int = 6  # tiles/frame while no visible confirmed track exists
    discover_budget_tracking: int = 0  # tiles/frame once a confirmed track exists
    discover_top_k: int = 3
    discover_score_thr: float = 0.35
    discover_motion_priority: bool = True  # prioritize tiles with 5-frame CMC motion peaks
    discover_micro_zoom: int = 128  # immediate 3x zoom confirmation on tiny tile hits (0=off)
    # Legacy full-frame periodic tiling (opt-in)
    use_tiles: bool = False
    tile_side: int = 384
    tile_overlap: float = 0.2
    tile_every_idle: int = 5
    tile_every_tracking: int = 15
    tile_top_k: int = 3
    tile_score_thr: float | None = None
    # Output gating, gap interpolation & coasting
    out_score_thr: float = 0.35
    score_ema: float = 0.30
    backfill: bool = True
    interpolate_gaps: bool = True
    max_interp_gap: int = 20
    emit_coasted: bool = False
    coast_max: int = 3


# --------------------------------------------------------------------------- CMC


def _compose_affine(M_prev: np.ndarray, M_step: np.ndarray) -> np.ndarray:
    """Compose 2x3 affine transforms: apply M_prev first, then M_step."""
    A = np.vstack([M_prev, [0.0, 0.0, 1.0]])
    B = np.vstack([M_step, [0.0, 0.0, 1.0]])
    return (B @ A)[:2, :]


def warp_box_cmc(box: list[float], M: np.ndarray) -> list[float]:
    """Shift a box's centre by 2x3 camera motion matrix M while preserving width/height."""
    if not box:
        return box
    cx = box[0] + box[2] / 2.0
    cy = box[1] + box[3] / 2.0
    nx, ny = (M[:, :2] @ np.array([cx, cy], dtype=np.float64) + M[:, 2]).tolist()
    return [nx - box[2] / 2.0, ny - box[3] / 2.0, box[2], box[3]]


class CameraMotionCompensator:
    """Fast 2D similarity/affine camera motion estimator + 5-frame CMC motion energy map."""

    def __init__(self, downscale: float = 0.5, motion_gap: int = 5) -> None:
        self.downscale = downscale
        self.motion_gap = motion_gap
        self.prev_gray: np.ndarray | None = None
        self._ring_gray: list[np.ndarray] = []
        self._ring_M: list[np.ndarray] = []  # step transforms in downscaled coords
        self.motion_map: np.ndarray | None = None

    def estimate(self, image: Image.Image) -> np.ndarray | None:
        """Return 2x3 affine matrix mapping prev frame -> current frame (full-res px)."""
        if cv2 is None:
            return None
        rgb = np.asarray(image)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if self.downscale != 1.0:
            small = cv2.resize(
                gray,
                None,
                fx=self.downscale,
                fy=self.downscale,
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = gray

        M_small: np.ndarray | None = None
        M_full: np.ndarray | None = None
        if self.prev_gray is not None and self.prev_gray.shape == small.shape:
            pts0 = cv2.goodFeaturesToTrack(
                self.prev_gray, maxCorners=120, qualityLevel=0.01, minDistance=20
            )
            if pts0 is not None and len(pts0) >= 6:
                pts1, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, small, pts0, None
                )
                if pts1 is not None and status is not None:
                    ok = status.ravel() == 1
                    if int(ok.sum()) >= 6:
                        M, _ = cv2.estimateAffinePartial2D(
                            pts0[ok],
                            pts1[ok],
                            method=cv2.RANSAC,
                            ransacReprojThreshold=2.0,
                        )
                        if M is not None and np.all(np.isfinite(M)):
                            scale = float(np.hypot(M[0, 0], M[0, 1]))
                            if 0.85 <= scale <= 1.15:
                                M_small = M.astype(np.float64)
                                M_full = M_small.copy()
                                M_full[0, 2] /= self.downscale
                                M_full[1, 2] /= self.downscale

        # Update 5-frame motion buffer & compute CMC-compensated motion energy map
        step_M = M_small if M_small is not None else np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self._ring_gray.append(small)
        self._ring_M.append(step_M)
        if len(self._ring_gray) > self.motion_gap + 1:
            self._ring_gray.pop(0)
            self._ring_M.pop(0)

        if len(self._ring_gray) == self.motion_gap + 1:
            ref_gray = self._ring_gray[0]
            if ref_gray.shape == small.shape:
                M_cum = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
                for m_step in self._ring_M[1:]:
                    M_cum = _compose_affine(M_cum, m_step)
                h_s, w_s = small.shape
                warped = cv2.warpAffine(
                    ref_gray, M_cum, (w_s, h_s), borderMode=cv2.BORDER_REPLICATE
                )
                diff = cv2.absdiff(small, warped).astype(np.float32)
                # Suppress warp border artefacts & highlight compact moving blobs
                b = max(4, int(12 * self.downscale))
                diff[:b, :] = 0.0
                diff[-b:, :] = 0.0
                diff[:, :b] = 0.0
                diff[:, -b:] = 0.0
                fine = cv2.GaussianBlur(diff, (3, 3), 0)
                coarse = cv2.GaussianBlur(diff, (15, 15), 0)
                self.motion_map = np.maximum(0.0, fine - 0.7 * coarse)
            else:
                self.motion_map = None
        else:
            self.motion_map = None

        self.prev_gray = small
        return M_full


def select_discover_tiles(
    tiles: list[Region],
    motion_map: np.ndarray | None,
    downscale: float,
    cursor: int,
    budget: int,
    cooldown: dict[int, int],
    frame_idx: int,
    *,
    use_motion: bool = True,
) -> tuple[list[Region], int]:
    """Pick up to ``budget`` discovery tiles: motion-peak tiles + 1 round-robin tile."""
    n = len(tiles)
    if n == 0 or budget <= 0:
        return [], cursor
    n_run = min(budget, n)
    chosen_idx: list[int] = []

    # 1. Motion-prioritized tiles (targets moving drones like seq_0003 in 1-2 frames!)
    if use_motion and motion_map is not None and n_run > 1:
        h_m, w_m = motion_map.shape
        scored: list[tuple[float, int]] = []
        for idx, (x0, y0, side) in enumerate(tiles):
            if frame_idx - cooldown.get(idx, -999) <= 2:
                continue
            sx1 = max(0, min(w_m - 1, int(x0 * downscale)))
            sy1 = max(0, min(h_m - 1, int(y0 * downscale)))
            sx2 = max(sx1 + 1, min(w_m, int((x0 + side) * downscale)))
            sy2 = max(sy1 + 1, min(h_m, int((y0 + side) * downscale)))
            patch = motion_map[sy1:sy2, sx1:sx2]
            if patch.size > 0:
                score = float(np.percentile(patch, 99.5))
                if score >= 6.0:
                    scored.append((score, idx))
        scored.sort(reverse=True)
        for _, idx in scored[: max(0, n_run - 1)]:
            chosen_idx.append(idx)
            cooldown[idx] = frame_idx

    # 2. Fill remaining budget (at least 1 tile) with round-robin for static/hovering targets
    attempts = 0
    while len(chosen_idx) < n_run and attempts < n:
        idx = cursor % n
        cursor += 1
        attempts += 1
        if idx not in chosen_idx:
            chosen_idx.append(idx)
            cooldown[idx] = frame_idx

    return [tiles[i] for i in chosen_idx], cursor


# --------------------------------------------------------------------------- Kalman


class BoxKalman:
    """Constant-velocity Kalman filter on (cx, cy, w, h) with pixel noise floors."""

    _F = np.eye(8)
    _F[:4, 4:] = np.eye(4)
    _H = np.eye(4, 8)
    _STD_POS = 1.0 / 20.0
    _STD_VEL = 1.0 / 160.0

    def __init__(
        self,
        box_xywh: list[float],
        *,
        min_pos_std: float = 1.0,
        min_vel_std: float = 0.5,
    ) -> None:
        self.min_pos_std = min_pos_std
        self.min_vel_std = min_vel_std
        z = _xywh_to_cxcywh(box_xywh)
        self.mean = np.r_[z, np.zeros(4)]
        pos_s, vel_s = self._stds(z[2], z[3])
        std = np.r_[2.0 * pos_s, 10.0 * vel_s]
        self.cov = np.diag(std**2)

    def _stds(self, w: float, h: float) -> tuple[np.ndarray, np.ndarray]:
        s = max(float(w), float(h), 1.0)
        pos_s = np.maximum(np.full(4, self._STD_POS * s), self.min_pos_std)
        vel_s = np.maximum(np.full(4, self._STD_VEL * s), self.min_vel_std)
        return pos_s, vel_s

    def apply_cmc(self, M: np.ndarray) -> None:
        """Warp predicted centre (cx, cy) and velocity (vx, vy) by camera motion."""
        R = M[:, :2]
        t = M[:, 2]
        self.mean[:2] = R @ self.mean[:2] + t
        self.mean[4:6] = R @ self.mean[4:6]

    def predict(self, *, coasting: bool = False, vel_damping: float = 0.92) -> None:
        if coasting:
            # Gently damp velocity while occluded so the track doesn't run away off-screen
            self.mean[4:6] *= vel_damping
            self.mean[6:8] *= vel_damping * 0.85
        # Prevent negative width/height velocity from collapsing the box
        for i in (2, 3):
            if self.mean[i] + self.mean[i + 4] <= 1.0:
                self.mean[i + 4] = 0.0
        pos_s, vel_s = self._stds(self.mean[2], self.mean[3])
        std = np.r_[pos_s, vel_s]
        self.mean = self._F @ self.mean
        self.mean[2:4] = np.maximum(self.mean[2:4], 1.0)
        self.cov = self._F @ self.cov @ self._F.T + np.diag(std**2)

    def reanchor(
        self, box_xywh: list[float], *, dt: int, prev_cmc_box: list[float] | None
    ) -> None:
        """Observation-centric re-anchor after an occlusion gap (OC-SORT style)."""
        z = _xywh_to_cxcywh(box_xywh)
        if prev_cmc_box and dt >= 2:
            prev_z = _xywh_to_cxcywh(prev_cmc_box)
            vx = float(np.clip((z[0] - prev_z[0]) / dt, -40.0, 40.0))
            vy = float(np.clip((z[1] - prev_z[1]) / dt, -40.0, 40.0))
            self.mean[:4] = z
            self.mean[4:6] = [vx, vy]
            self.mean[6:8] = 0.0
            pos_s, vel_s = self._stds(z[2], z[3])
            self.cov = np.diag(np.r_[pos_s, 4.0 * vel_s] ** 2)
        else:
            self.update(box_xywh)

    def update(self, box_xywh: list[float]) -> None:
        z = _xywh_to_cxcywh(box_xywh)
        pos_s, _ = self._stds(z[2], z[3])
        r = np.diag(pos_s**2)
        innov_cov = self._H @ self.cov @ self._H.T + r
        gain = np.linalg.solve(innov_cov, self._H @ self.cov).T
        self.mean = self.mean + gain @ (z - self._H @ self.mean)
        self.mean[2:4] = np.maximum(self.mean[2:4], 1.0)
        self.cov = (np.eye(8) - gain @ self._H) @ self.cov

    def box(self) -> list[float]:
        cx, cy, w, h = self.mean[:4]
        w, h = max(float(w), 1.0), max(float(h), 1.0)
        return [float(cx) - w / 2.0, float(cy) - h / 2.0, w, h]


def _xywh_to_cxcywh(b: list[float]) -> np.ndarray:
    return np.array([b[0] + b[2] / 2.0, b[1] + b[3] / 2.0, b[2], b[3]], dtype=np.float64)


# --------------------------------------------------------------------------- tracker


@dataclass
class Track:
    kf: BoxKalman
    cls: int
    score: float
    avg_score: float = 0.0
    hits: int = 1
    misses: int = 0
    tid: int | None = None  # assigned on confirmation so output ids stay compact
    last_det_frame: int = 0
    last_det_box: list[float] = field(default_factory=list)
    cmc_last_box: list[float] = field(default_factory=list)  # last_det_box warped by CMC
    class_votes: dict[int, float] = field(default_factory=lambda: defaultdict(float))
    pending: list[tuple[int, list[float], float, int]] = field(default_factory=list)

    @property
    def confirmed(self) -> bool:
        return self.tid is not None

    @property
    def voted_cls(self) -> int:
        if not self.class_votes:
            return self.cls
        return max(self.class_votes.items(), key=lambda kv: kv[1])[0]


def _match(
    tracks: list[Track], dets: list[Det], cfg: Config
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian matching on IoU, with an adaptive centre-distance fallback for tiny targets."""
    if not tracks or not dets:
        return [], list(range(len(tracks))), list(range(len(dets)))
    tb = np.asarray([t.kf.box() for t in tracks], dtype=np.float64)
    db = np.asarray([d[2] for d in dets], dtype=np.float64)
    iou = _iou_pairwise(tb, db)
    tc = tb[:, :2] + tb[:, 2:] / 2.0
    dc = db[:, :2] + db[:, 2:] / 2.0
    scale = np.maximum(np.maximum(tb[:, 2], tb[:, 3]), 12.0)[:, None]
    dist = np.linalg.norm(tc[:, None, :] - dc[None, :, :], axis=2) / scale
    # Search radius grows while a track is coasting
    gate = cfg.dist_gate * (1.0 + 0.30 * np.asarray([t.misses for t in tracks]))[:, None]
    ok = (iou >= cfg.iou_gate) | (dist <= gate)
    cost = np.where(ok, (1.0 - iou) + 0.5 * np.minimum(dist, 3.0), 1e6)
    rows, cols = linear_sum_assignment(cost)
    matches = [(int(r), int(c)) for r, c in zip(rows, cols, strict=True) if ok[r, c]]
    mt, md = {r for r, _ in matches}, {c for _, c in matches}
    return (
        matches,
        [i for i in range(len(tracks)) if i not in mt],
        [j for j in range(len(dets)) if j not in md],
    )


def _recover_occluded(
    tracks: list[Track], dets: list[Det], cfg: Config
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Stage-3 OC-SORT-style recovery matching new detections to occluded confirmed tracks."""
    if not tracks or not dets:
        return [], list(range(len(tracks))), list(range(len(dets)))
    # Only match confirmed tracks that were missed (occluded behind an obstacle)
    occ_idx = [i for i, t in enumerate(tracks) if t.confirmed and t.misses >= 1]
    if not occ_idx:
        return [], list(range(len(tracks))), list(range(len(dets)))

    n_o, n_d = len(occ_idx), len(dets)
    cost = np.full((n_o, n_d), 1e6, dtype=np.float64)
    ok = np.zeros((n_o, n_d), dtype=bool)

    for r, ti in enumerate(occ_idx):
        t = tracks[ti]
        pb = t.kf.box()
        pcx, pcy = pb[0] + pb[2] / 2.0, pb[1] + pb[3] / 2.0
        lb = t.cmc_last_box or t.last_det_box or pb
        lcx, lcy = lb[0] + lb[2] / 2.0, lb[1] + lb[3] / 2.0
        t_sz = max(lb[2], lb[3], 16.0)
        # Gate grows generously with occlusion duration so the target re-attaches on exit
        gate = cfg.occlusion_reid_gate * (1.0 + 0.12 * min(t.misses, 30))
        v_cls = t.voted_cls
        for c, (cid, _, db) in enumerate(dets):
            d_sz = max(db[2], db[3], 4.0)
            ratio = d_sz / max(lb[2], lb[3], 4.0)
            if not (0.30 <= ratio <= 3.30):
                continue
            dcx, dcy = db[0] + db[2] / 2.0, db[1] + db[3] / 2.0
            d_pred = float(np.hypot(dcx - pcx, dcy - pcy)) / t_sz
            d_last = float(np.hypot(dcx - lcx, dcy - lcy)) / t_sz
            d_min = min(d_pred, d_last)
            if d_min <= gate:
                ok[r, c] = True
                cls_pen = 0.0 if cid == v_cls else 0.75
                size_pen = 0.5 * abs(np.log(max(ratio, 1e-3)))
                cost[r, c] = d_min + cls_pen + size_pen

    rows, cols = linear_sum_assignment(cost)
    matches = [
        (occ_idx[int(r)], int(c)) for r, c in zip(rows, cols, strict=True) if ok[r, c]
    ]
    mt = {r for r, _ in matches}
    md = {c for _, c in matches}
    return (
        matches,
        [i for i in range(len(tracks)) if i not in mt],
        [j for j in range(len(dets)) if j not in md],
    )


def crop_regions(tracks: list[Track], w_px: int, h_px: int, cfg: Config) -> list[Region]:
    """Square ROI windows around predicted positions of small tracks (supports upscaling)."""
    cands = sorted(tracks, key=lambda t: (t.confirmed, t.avg_score, t.score), reverse=True)
    regions: list[Region] = []
    for t in cands:
        if len(regions) >= cfg.max_crops:
            break
        # Don't waste crop passes on unconfirmed low-score clutter
        if not t.confirmed and t.score < cfg.high_thr * 0.75:
            continue
        x, y, w, h = t.kf.box()
        in_model = max(w * cfg.img_size / w_px, h * cfg.img_size / h_px)
        if in_model >= cfg.crop_skip_px:
            continue  # already well resolved by the global pass
        # Allow side < img_size (down to min_crop_side) to upscale tiny drones;
        # widen search window during occlusion so re-emerging drones are recaptured!
        expand = 1.0 + 0.18 * min(t.misses, 8)
        side = int(round(max(cfg.min_crop_side, cfg.crop_scale * max(w, h)) * expand))
        side = max(1, min(side, w_px, h_px))
        cx, cy = x + w / 2.0, y + h / 2.0
        x0 = int(np.clip(round(cx - side / 2.0), 0, w_px - side))
        y0 = int(np.clip(round(cy - side / 2.0), 0, h_px - side))
        regions.append((x0, y0, side))
    return regions


def _axis_starts(n: int, side: int, stride: int) -> list[int]:
    if n <= side:
        return [0]
    starts = list(range(0, n - side + 1, stride))
    if starts[-1] != n - side:
        starts.append(n - side)
    return starts


def tile_regions(
    w_px: int, h_px: int, side_cfg: int, overlap: float
) -> list[Region]:
    """Overlapping square tiles covering the frame; empty if one tile covers the frame."""
    side = max(1, min(side_cfg, w_px, h_px))
    if side >= w_px and side >= h_px:
        return []
    stride = max(1, int(side * (1.0 - overlap)))
    return [
        (x0, y0, side)
        for y0 in _axis_starts(h_px, side, stride)
        for x0 in _axis_starts(w_px, side, stride)
    ]


def should_tile(f1: int, tracks: list[Track], cfg: Config) -> bool:
    if not cfg.use_tiles:
        return False
    tracking = any(t.confirmed for t in tracks)
    period = cfg.tile_every_tracking if tracking else cfg.tile_every_idle
    return period > 0 and (f1 - 1) % period == 0


class Detector(Protocol):
    def detect(
        self, image: Any, region: Region | None, *, top_k: int, score_thr: float
    ) -> list[Det]: ...


@dataclass
class SeqResult:
    lines: list[str]
    passes: list[int]
    frame_ms: list[float]
    global_dets: dict[str, list[list[Any]]] = field(default_factory=dict)


def _nms_small_aware(
    dets: list[Det], *, iou_thr: float = 0.5, small_px: float = 20.0
) -> list[Det]:
    """Greedy NMS that also suppresses near-duplicate tiny boxes (IoU can be 0 when shifted by 4px)."""
    kept: list[Det] = []
    for cand in sorted(dets, key=lambda d: -d[1]):
        cb = np.asarray([cand[2]], dtype=np.float64)
        ccx = cand[2][0] + cand[2][2] / 2.0
        ccy = cand[2][1] + cand[2][3] / 2.0
        c_sz = max(cand[2][2], cand[2][3])
        dup = False
        for k in kept:
            if float(_iou_pairwise(cb, np.asarray([k[2]], dtype=np.float64))[0, 0]) >= iou_thr:
                dup = True
                break
            k_sz = max(k[2][2], k[2][3])
            if min(c_sz, k_sz) < small_px:
                kcx = k[2][0] + k[2][2] / 2.0
                kcy = k[2][1] + k[2][3] / 2.0
                if np.hypot(ccx - kcx, ccy - kcy) < max(c_sz, k_sz, 12.0):
                    dup = True
                    break
        if not dup:
            kept.append(cand)
    return kept


def _near_existing_track(box: list[float], tracks: list[Track]) -> bool:
    """Prevent spawning a duplicate track right on top of an already-tracked target."""
    if not tracks:
        return False
    bx, by, bw, bh = box
    bcx, bcy = bx + bw / 2.0, by + bh / 2.0
    b_arr = np.asarray([box], dtype=np.float64)
    for t in tracks:
        if t.misses > 3:
            continue
        tb = t.kf.box()
        if float(_iou_pairwise(b_arr, np.asarray([tb], dtype=np.float64))[0, 0]) >= 0.25:
            return True
        tcx, tcy = tb[0] + tb[2] / 2.0, tb[1] + tb[3] / 2.0
        rad = 1.5 * max(bw, bh, tb[2], tb[3], 12.0)
        if float(np.hypot(bcx - tcx, bcy - tcy)) <= rad:
            return True
    return False


def _clamp_box(b: list[float], w_px: int, h_px: int) -> list[float]:
    x1 = max(0.0, min(float(w_px - 1), float(b[0])))
    y1 = max(0.0, min(float(h_px - 1), float(b[1])))
    x2 = max(x1 + 1.0, min(float(w_px), float(b[0] + b[2])))
    y2 = max(y1 + 1.0, min(float(h_px), float(b[1] + b[3])))
    return [x1, y1, x2 - x1, y2 - y1]


def track_sequence(
    frames: Iterator[tuple[int, Any, int, int]],
    detector: Detector | None,
    cfg: Config,
    *,
    cached_global_dets: dict[str, list[list[Any]]] | None = None,
) -> SeqResult:
    """Run detection + unified Kalman/ByteTrack/ROI tracking over a frame iterator."""
    tracks: list[Track] = []
    next_tid = 1
    # Store output keyed by (frame, tid) to strictly guarantee zero duplicate rows
    emitted: dict[tuple[int, int], tuple[list[float], float, int]] = {}
    passes: list[int] = []
    frame_ms: list[float] = []
    saved_global: dict[str, list[list[Any]]] = {}
    cmc = CameraMotionCompensator() if cfg.use_cmc else None
    discover_cursor = 0
    discover_tiles_cache: list[Region] | None = None
    discover_dims: tuple[int, int] | None = None
    discover_cooldown: dict[int, int] = {}

    for f1, image, w_px, h_px in frames:
        t0 = time.perf_counter()

        # 1. Camera motion compensation + damped Kalman predict
        M = cmc.estimate(image) if (cmc is not None and image is not None) else None
        for t in tracks:
            if M is not None:
                t.kf.apply_cmc(M)
                if t.cmc_last_box:
                    t.cmc_last_box = warp_box_cmc(t.cmc_last_box, M)
            t.kf.predict(coasting=(t.misses > 0), vel_damping=cfg.vel_damping)

        # 2. Global pass (live or cached)
        n_pass = 0
        if cached_global_dets is not None:
            raw_g = cached_global_dets.get(str(f1), [])
            dets: list[Det] = [
                (int(c), float(s), [float(v) for v in b])
                for c, s, b in raw_g
                if float(s) >= cfg.low_thr
            ]
        else:
            assert detector is not None
            dets = detector.detect(image, None, top_k=cfg.top_k, score_thr=cfg.low_thr)
            n_pass += 1
        saved_global[str(f1)] = [[int(c), float(s), [float(v) for v in b]] for c, s, b in dets]

        # 3. Track-guided upscaled ROI crops
        if cfg.use_crops and detector is not None and image is not None:
            for region in crop_regions(tracks, w_px, h_px, cfg):
                c_dets = detector.detect(
                    image, region, top_k=cfg.crop_top_k, score_thr=cfg.crop_score_thr
                )
                # Reject huge clipped boxes filling most of the crop window
                _, _, side = region
                for cid, s, b in c_dets:
                    if b[2] < 0.85 * side and b[3] < 0.85 * side:
                        dets.append((cid, s, b))
                n_pass += 1

        # 4a. Motion-prioritized + round-robin tile discovery (flat latency while idle!)
        if cfg.use_discover and detector is not None and image is not None:
            has_visible = any(
                t.confirmed and t.avg_score >= cfg.out_score_thr for t in tracks
            )
            budget = cfg.discover_budget_tracking if has_visible else cfg.discover_budget_idle
            if budget > 0:
                if discover_dims != (w_px, h_px):
                    discover_dims = (w_px, h_px)
                    discover_tiles_cache = tile_regions(
                        w_px, h_px, cfg.discover_tile_side, cfg.discover_overlap
                    )
                    discover_cursor = 0
                    discover_cooldown.clear()
                tiles = discover_tiles_cache or []
                chosen_tiles, discover_cursor = select_discover_tiles(
                    tiles,
                    cmc.motion_map if cmc is not None else None,
                    cmc.downscale if cmc is not None else 0.5,
                    discover_cursor,
                    budget,
                    discover_cooldown,
                    f1,
                    use_motion=cfg.discover_motion_priority,
                )
                micro_cand: tuple[float, list[float]] | None = None
                for region in chosen_tiles:
                    _, _, side = region
                    t_dets = detector.detect(
                        image,
                        region,
                        top_k=cfg.discover_top_k,
                        score_thr=cfg.discover_score_thr * 0.75
                        if cfg.discover_micro_zoom > 0
                        else cfg.discover_score_thr,
                    )
                    for cid, s, b in t_dets:
                        if b[2] >= 0.85 * side or b[3] >= 0.85 * side:
                            continue
                        if s >= cfg.discover_score_thr:
                            dets.append((cid, s, b))
                        if (
                            cfg.discover_micro_zoom > 0
                            and cfg.discover_micro_zoom < side
                            and max(b[2], b[3]) <= 24.0
                            and s < cfg.high_thr
                            and (micro_cand is None or s > micro_cand[0])
                        ):
                            micro_cand = (s, b)
                    n_pass += 1
                # Immediate 3x zoom (128px) micro-confirmation on the best tiny tile hit!
                if micro_cand is not None:
                    _, mb = micro_cand
                    mside = max(1, min(cfg.discover_micro_zoom, w_px, h_px))
                    mcx, mcy = mb[0] + mb[2] / 2.0, mb[1] + mb[3] / 2.0
                    mx0 = int(np.clip(round(mcx - mside / 2.0), 0, w_px - mside))
                    my0 = int(np.clip(round(mcy - mside / 2.0), 0, h_px - mside))
                    for cid, s, b in detector.detect(
                        image,
                        (mx0, my0, mside),
                        top_k=cfg.crop_top_k,
                        score_thr=cfg.crop_score_thr,
                    ):
                        if b[2] < 0.85 * mside and b[3] < 0.85 * mside:
                            dets.append((cid, s, b))
                    n_pass += 1

        # 4b. Legacy full-frame periodic tiling (opt-in)
        if should_tile(f1, tracks, cfg) and detector is not None and image is not None:
            tile_thr = cfg.high_thr if cfg.tile_score_thr is None else cfg.tile_score_thr
            for region in tile_regions(w_px, h_px, cfg.tile_side, cfg.tile_overlap):
                dets += detector.detect(
                    image, region, top_k=cfg.tile_top_k, score_thr=tile_thr
                )
                n_pass += 1

        dets = _nms_small_aware(dets, iou_thr=cfg.nms_iou)

        # 5. ByteTrack two-stage association + Stage-3 OC-SORT occlusion recovery
        high = [d for d in dets if d[1] >= cfg.high_thr]
        low = [d for d in dets if d[1] < cfg.high_thr]
        m1, rest_t, rest_high = _match(tracks, high, cfg)
        rest_tracks = [tracks[i] for i in rest_t]
        m2, rest_t2, _ = _match(rest_tracks, low, cfg)

        # Stage 3: try re-attaching unmatched high-score detections to occluded confirmed tracks
        # BEFORE spawning a new track_id!
        occ_tracks = [rest_tracks[i] for i in rest_t2]
        unmatched_high_dets = [high[j] for j in rest_high]
        m3, rest_t3, rest_high_idx = _recover_occluded(occ_tracks, unmatched_high_dets, cfg)
        rest_high = [rest_high[j] for j in rest_high_idx]

        updates: list[tuple[Track, Det, bool]] = [(tracks[i], high[j], False) for i, j in m1]
        updates += [(rest_tracks[i], low[j], False) for i, j in m2]
        updates += [(occ_tracks[i], unmatched_high_dets[j], True) for i, j in m3]

        for trk, (cid, score, box), recovered in updates:
            prev_frame = trk.last_det_frame
            prev_box = trk.last_det_box
            prev_cmc_box = trk.cmc_last_box
            gap_dt = f1 - prev_frame if prev_frame > 0 else 1
            if recovered or trk.misses >= 2:
                trk.kf.reanchor(box, dt=gap_dt, prev_cmc_box=prev_cmc_box)
            else:
                trk.kf.update(box)
            trk.cls, trk.score = cid, score
            trk.class_votes[cid] += score
            trk.avg_score += cfg.score_ema * (score - trk.avg_score)
            trk.hits += 1
            trk.misses = 0
            trk.last_det_frame = f1
            trk.last_det_box = list(box)
            trk.cmc_last_box = list(box)

            v_cls = trk.voted_cls
            visible = trk.avg_score >= cfg.out_score_thr
            entry = (f1, list(box), score, v_cls)

            if trk.confirmed:
                assert trk.tid is not None
                if visible:
                    # Linear gap interpolation when re-acquiring a missed track
                    gap = f1 - prev_frame
                    if (
                        cfg.interpolate_gaps
                        and prev_frame > 0
                        and prev_box
                        and 1 < gap <= cfg.max_interp_gap
                    ):
                        p_arr = np.asarray(prev_box, dtype=np.float64)
                        c_arr = np.asarray(box, dtype=np.float64)
                        for gf in range(prev_frame + 1, f1):
                            if (gf, trk.tid) not in emitted:
                                a = (gf - prev_frame) / float(gap)
                                ibox = _clamp_box(
                                    ((1.0 - a) * p_arr + a * c_arr).tolist(),
                                    w_px,
                                    h_px,
                                )
                                emitted[(gf, trk.tid)] = (ibox, score * 0.75, v_cls)
                    emitted[(f1, trk.tid)] = (
                        _clamp_box(list(box), w_px, h_px),
                        score,
                        v_cls,
                    )
            else:
                trk.pending.append(entry)
                if trk.hits >= cfg.min_hits:
                    trk.tid, next_tid = next_tid, next_tid + 1
                    flush = trk.pending if cfg.backfill else trk.pending[-1:]
                    if visible:
                        for pf, pb, ps, _ in flush:
                            emitted[(pf, trk.tid)] = (
                                _clamp_box(pb, w_px, h_px),
                                ps,
                                v_cls,
                            )
                        # Interpolate any 1-frame gap inside pending buffer too
                        if cfg.interpolate_gaps and len(flush) >= 2:
                            for idx in range(1, len(flush)):
                                f_a, b_a, s_a, _ = flush[idx - 1]
                                f_b, b_b, s_b, _ = flush[idx]
                                gap = f_b - f_a
                                if 1 < gap <= cfg.max_interp_gap:
                                    pa = np.asarray(b_a, dtype=np.float64)
                                    pb = np.asarray(b_b, dtype=np.float64)
                                    for gf in range(f_a + 1, f_b):
                                        if (gf, trk.tid) not in emitted:
                                            a = (gf - f_a) / float(gap)
                                            ibox = _clamp_box(
                                                ((1.0 - a) * pa + a * pb).tolist(),
                                                w_px,
                                                h_px,
                                            )
                                            emitted[(gf, trk.tid)] = (
                                                ibox,
                                                0.5 * (s_a + s_b),
                                                v_cls,
                                            )
                    trk.pending = []

        for i in rest_t3:
            occ_tracks[i].misses += 1

        for j in rest_high:
            cid, score, box = high[j]
            if _near_existing_track(box, tracks):
                continue
            kf = BoxKalman(
                box,
                min_pos_std=cfg.min_pos_std_px,
                min_vel_std=cfg.min_vel_std_px,
            )
            trk = Track(
                kf=kf,
                cls=cid,
                score=score,
                avg_score=score,
                last_det_frame=f1,
                last_det_box=list(box),
                cmc_last_box=list(box),
            )
            trk.class_votes[cid] += score
            trk.pending.append((f1, list(box), score, cid))
            if cfg.min_hits <= 1 and score >= cfg.out_score_thr:
                trk.tid, next_tid = next_tid, next_tid + 1
                emitted[(f1, trk.tid)] = (
                    _clamp_box(list(box), w_px, h_px),
                    score,
                    cid,
                )
                trk.pending = []
            tracks.append(trk)

        if cfg.emit_coasted:
            for t in tracks:
                if (
                    t.confirmed
                    and t.tid is not None
                    and 1 <= t.misses <= cfg.coast_max
                    and t.avg_score >= cfg.out_score_thr
                ):
                    cbox = _clamp_box(t.kf.box(), w_px, h_px)
                    emitted[(f1, t.tid)] = (cbox, t.score * 0.5, t.voted_cls)

        tracks = [
            t
            for t in tracks
            if t.misses <= (cfg.max_age if t.confirmed else cfg.tentative_max_misses)
        ]
        passes.append(n_pass)
        frame_ms.append((time.perf_counter() - t0) * 1000.0)

    lines = [
        f"{f},{tid},{b[0]:.2f},{b[1]:.2f},{b[2]:.2f},{b[3]:.2f},{s:.4f},{c + 1},1.0"
        for (f, tid), (b, s, c) in sorted(emitted.items())
        if b[2] > 0.0 and b[3] > 0.0
    ]
    return SeqResult(
        lines=lines, passes=passes, frame_ms=frame_ms, global_dets=saved_global
    )


# --------------------------------------------------------------------------- I/O & Detectors


class OnnxDetector:
    """Static RF-DETR ONNX model on CPU ONNX Runtime; regions are cropped from the frame."""

    def __init__(self, onnx_path: Path, cfg: Config) -> None:
        import onnxruntime as ort

        self.cfg = cfg
        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.out_names = [o.name for o in self.sess.get_outputs()]

    def detect(
        self, image: Image.Image, region: Region | None, *, top_k: int, score_thr: float
    ) -> list[Det]:
        if region is None:
            src, x0, y0 = image, 0, 0
            w_px, h_px = image.size
        else:
            x0, y0, side = region
            src = image.crop((x0, y0, x0 + side, y0 + side))
            w_px = h_px = side
        inp = _preprocess_imagenet(src, self.cfg.img_size)
        outs = dict(
            zip(self.out_names, self.sess.run(self.out_names, {self.in_name: inp}), strict=True)
        )
        dets = _decode_rfdetr_topk(
            outs["dets"][0],
            outs["labels"][0],
            w_px,
            h_px,
            self.cfg.num_classes,
            top_k=top_k,
            score_thr=score_thr,
        )
        return [(c, s, [b[0] + x0, b[1] + y0, b[2], b[3]]) for c, s, b in dets]


class SplitOnnxDetector:
    """Partitioned RF-DETR detector: Backbone ONNX (Metis target) + Host Decoder ONNX."""

    def __init__(self, backbone_onnx: Path, decoder_onnx: Path, cfg: Config) -> None:
        import onnxruntime as ort

        self.cfg = cfg
        self.bb_sess = ort.InferenceSession(
            str(backbone_onnx), providers=["CPUExecutionProvider"]
        )
        self.bb_in = self.bb_sess.get_inputs()[0].name
        self.bb_out = self.bb_sess.get_outputs()[0].name

        self.dec_sess = ort.InferenceSession(
            str(decoder_onnx), providers=["CPUExecutionProvider"]
        )
        self.dec_in = self.dec_sess.get_inputs()[0].name
        self.dec_outs = [o.name for o in self.dec_sess.get_outputs()]

    def detect(
        self, image: Image.Image, region: Region | None, *, top_k: int, score_thr: float
    ) -> list[Det]:
        if region is None:
            src, x0, y0 = image, 0, 0
            w_px, h_px = image.size
        else:
            x0, y0, side = region
            src = image.crop((x0, y0, x0 + side, y0 + side))
            w_px = h_px = side
        inp = _preprocess_imagenet(src, self.cfg.img_size)
        feats = self.bb_sess.run([self.bb_out], {self.bb_in: inp})[0]
        outs = dict(
            zip(
                self.dec_outs,
                self.dec_sess.run(self.dec_outs, {self.dec_in: feats}),
                strict=True,
            )
        )
        dets = _decode_rfdetr_topk(
            outs["dets"][0],
            outs["labels"][0],
            w_px,
            h_px,
            self.cfg.num_classes,
            top_k=top_k,
            score_thr=score_thr,
        )
        return [(c, s, [b[0] + x0, b[1] + y0, b[2], b[3]]) for c, s, b in dets]


def iter_video_frames(video: Path, tmp_dir: Path) -> Iterator[tuple[int, Image.Image, int, int]]:
    if cv2 is not None:
        cap = cv2.VideoCapture(str(video))
        if cap.isOpened():
            f1 = 0
            try:
                while True:
                    ok, bgr = cap.read()
                    if not ok:
                        break
                    f1 += 1
                    rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                    yield f1, rgb, rgb.width, rgb.height
            finally:
                cap.release()
            if f1 > 0:
                return
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-fps_mode",
            "passthrough",
            str(tmp_dir / "f_%06d.jpg"),
        ],
        check=True,
    )
    for f1, path in enumerate(sorted(tmp_dir.glob("f_*.jpg")), 1):
        with Image.open(path) as im:
            rgb = im.convert("RGB")
        yield f1, rgb, rgb.width, rgb.height


def list_sequences(eval_dir: Path, split: str) -> list[Path]:
    manifest_path = eval_dir / "sequences_manifest.json"
    splits: dict[str, str] = {}
    if manifest_path.is_file():
        doc = json.loads(manifest_path.read_text(encoding="utf-8"))
        splits = {e["sequence"]: e.get("split", "") for e in doc.get("sequences", [])}
    seqs = sorted(p for p in eval_dir.glob("seq_*") if (p / "video.mp4").exists())
    if split != "all":
        seqs = [p for p in seqs if splits.get(p.name) == split]
    return seqs


def parse_args() -> argparse.Namespace:
    d = Config()
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--onnx", type=Path, default=None, help="Full RF-DETR ONNX model")
    p.add_argument("--backbone-onnx", type=Path, default=None, help="Partitioned backbone ONNX")
    p.add_argument("--decoder-onnx", type=Path, default=None, help="Partitioned host decoder ONNX")
    p.add_argument("--load-dets", type=Path, default=None, help="Reuse cached global detections JSON")
    p.add_argument("--save-dets", type=Path, default=None, help="Save global detections JSON")
    p.add_argument("--eval-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--split", choices=("all", "valid", "test"), default="all")
    p.add_argument("--num-classes", type=int, default=d.num_classes)
    p.add_argument("--img-size", type=int, default=d.img_size)
    p.add_argument("--high-thr", type=float, default=d.high_thr)
    p.add_argument("--low-thr", type=float, default=d.low_thr)
    p.add_argument("--crop-score-thr", type=float, default=d.crop_score_thr)
    p.add_argument("--dist-gate", type=float, default=d.dist_gate)
    p.add_argument("--min-hits", type=int, default=d.min_hits)
    p.add_argument("--max-age", type=int, default=d.max_age)
    p.add_argument("--vel-damping", type=float, default=d.vel_damping)
    p.add_argument("--occlusion-reid-gate", type=float, default=d.occlusion_reid_gate)
    p.add_argument("--no-cmc", action="store_true", help="Disable camera motion compensation")
    p.add_argument("--no-crops", action="store_true", help="Disable track-guided ROI crops")
    p.add_argument("--max-crops", type=int, default=d.max_crops)
    p.add_argument("--crop-scale", type=float, default=d.crop_scale)
    p.add_argument("--min-crop-side", type=int, default=d.min_crop_side)
    p.add_argument("--crop-skip-px", type=float, default=d.crop_skip_px)
    p.add_argument("--discover", action="store_true", help="Enable motion-prioritized idle tile discovery")
    p.add_argument("--no-discover", action="store_true", help="Disable idle tile discovery (default)")
    p.add_argument("--discover-tile-side", type=int, default=d.discover_tile_side)
    p.add_argument("--discover-overlap", type=float, default=d.discover_overlap)
    p.add_argument("--discover-budget-idle", type=int, default=d.discover_budget_idle)
    p.add_argument("--discover-budget-tracking", type=int, default=d.discover_budget_tracking)
    p.add_argument("--discover-score-thr", type=float, default=d.discover_score_thr)
    p.add_argument("--no-discover-motion", action="store_true", help="Use pure round-robin instead of motion priority")
    p.add_argument("--discover-micro-zoom", type=int, default=d.discover_micro_zoom, help="3x zoom confirmation crop side (0=off)")
    p.add_argument("--tiles", action="store_true", help="Enable legacy full-frame periodic tiling")
    p.add_argument("--tile-side", type=int, default=d.tile_side)
    p.add_argument("--tile-overlap", type=float, default=d.tile_overlap)
    p.add_argument("--tile-every-idle", type=int, default=d.tile_every_idle)
    p.add_argument("--tile-every-tracking", type=int, default=d.tile_every_tracking)
    p.add_argument("--tile-score-thr", type=float, default=None, help="default: --high-thr")
    p.add_argument("--out-score-thr", type=float, default=d.out_score_thr)
    p.add_argument("--no-backfill", action="store_true", help="Strictly online output")
    p.add_argument("--no-interpolate", action="store_true", help="Disable linear gap interpolation")
    p.add_argument("--max-interp-gap", type=int, default=d.max_interp_gap)
    p.add_argument("--emit-coasted", action="store_true", help="Write predicted boxes in short gaps")
    p.add_argument("--coast-max", type=int, default=d.coast_max)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if cv2 is None and shutil.which("ffmpeg") is None:
        raise SystemExit("Need either opencv-python or ffmpeg on PATH")

    cfg = Config(
        img_size=args.img_size,
        num_classes=args.num_classes,
        high_thr=args.high_thr,
        low_thr=args.low_thr,
        crop_score_thr=args.crop_score_thr,
        dist_gate=args.dist_gate,
        min_hits=args.min_hits,
        max_age=args.max_age,
        vel_damping=args.vel_damping,
        occlusion_reid_gate=args.occlusion_reid_gate,
        use_cmc=not args.no_cmc,
        use_crops=not args.no_crops,
        max_crops=args.max_crops,
        crop_scale=args.crop_scale,
        min_crop_side=args.min_crop_side,
        crop_skip_px=args.crop_skip_px,
        use_discover=args.discover and not args.no_discover,
        discover_tile_side=args.discover_tile_side,
        discover_overlap=args.discover_overlap,
        discover_budget_idle=args.discover_budget_idle,
        discover_budget_tracking=args.discover_budget_tracking,
        discover_score_thr=args.discover_score_thr,
        discover_motion_priority=not args.no_discover_motion,
        discover_micro_zoom=args.discover_micro_zoom,
        use_tiles=args.tiles,
        tile_side=args.tile_side,
        tile_overlap=args.tile_overlap,
        tile_every_idle=args.tile_every_idle,
        tile_every_tracking=args.tile_every_tracking,
        tile_score_thr=args.tile_score_thr,
        out_score_thr=args.out_score_thr,
        backfill=not args.no_backfill,
        interpolate_gaps=not args.no_interpolate,
        max_interp_gap=args.max_interp_gap,
        emit_coasted=args.emit_coasted,
        coast_max=args.coast_max,
    )

    detector: Detector | None = None
    if args.backbone_onnx is not None and args.decoder_onnx is not None:
        detector = SplitOnnxDetector(args.backbone_onnx, args.decoder_onnx, cfg)
    elif args.onnx is not None:
        detector = OnnxDetector(args.onnx, cfg)
    elif args.load_dets is None:
        raise SystemExit("Provide --onnx, --backbone-onnx + --decoder-onnx, or --load-dets")

    cached_all: dict[str, dict[str, list[list[Any]]]] | None = None
    if args.load_dets is not None:
        cached_all = json.loads(args.load_dets.read_text(encoding="utf-8"))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_passes: list[int] = []
    all_ms: list[float] = []
    per_seq: dict[str, dict[str, Any]] = {}
    all_saved_dets: dict[str, dict[str, list[list[Any]]]] = {}

    with tempfile.TemporaryDirectory() as td:
        for seq_dir in list_sequences(args.eval_dir, args.split):
            frames_dir = Path(td) / seq_dir.name
            frames_dir.mkdir()
            cached_seq = cached_all.get(seq_dir.name) if cached_all is not None else None
            res = track_sequence(
                iter_video_frames(seq_dir / "video.mp4", frames_dir),
                detector,
                cfg,
                cached_global_dets=cached_seq,
            )
            (args.out_dir / f"{seq_dir.name}.txt").write_text(
                "\n".join(res.lines) + ("\n" if res.lines else ""), encoding="utf-8"
            )
            all_passes += res.passes
            all_ms += res.frame_ms
            all_saved_dets[seq_dir.name] = res.global_dets
            per_seq[seq_dir.name] = {
                "frames": len(res.passes),
                "boxes": len(res.lines),
                "mean_passes": round(float(np.mean(res.passes)), 2) if res.passes else 0.0,
            }

    if args.save_dets is not None:
        args.save_dets.parent.mkdir(parents=True, exist_ok=True)
        args.save_dets.write_text(json.dumps(all_saved_dets), encoding="utf-8")

    print(
        json.dumps(
            {
                "config": cfg.__dict__,
                "sequences": per_seq,
                "mean_model_passes_per_frame": round(float(np.mean(all_passes)), 2)
                if all_passes
                else 0.0,
                "median_ms_per_frame": round(float(np.median(all_ms)), 2) if all_ms else 0.0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
