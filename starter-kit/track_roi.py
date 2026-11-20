"""Track-guided ROI crop tracker for the C-UAS MOT sequences.

Per frame:
  1. predict every track forward with a constant-velocity Kalman filter;
  2. run the detector on the full frame (squashed to 384x384, like the baseline);
  3. for up to ``--max-crops`` small tracks, run the detector again on a square
     crop centred on the predicted position, at native resolution (crops smaller
     than the model input are upscaled -- the only place upscaling helps);
  4. (opt-in ``--tiles``) every few frames (more often while nothing is tracked), sweep the frame with
     overlapping native-resolution tiles to discover targets the global pass is
     too coarse to see; tile hits start tentative tracks that step 3 then follows;
  5. merge detections (NMS), then ByteTrack-style two-stage association
     (high-score detections first, low-score ones only extend existing tracks).

Only confirmed tracks are written, which keeps false positives (costly for MOTA
and IDF1 on these single-target sequences) down. Reuses the baseline's
preprocessing / decoding so results stay comparable with ``predict_onnx_baseline.py mot``.

Usage (from starter-kit/):
    python track_roi.py --onnx exports/rfdetr/full/inference_model.onnx \
        --eval-dir ../tracking_eval --out-dir pred_mot_roi/
    python track_roi.py ... --no-crops        # tracker only, for ablation
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
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

Det = tuple[int, float, list[float]]  # (class_id 0-based, score, [x, y, w, h] in frame px)
Region = tuple[int, int, int]  # (x0, y0, side) square crop in frame px


@dataclass
class Config:
    img_size: int = 384
    num_classes: int = 3
    top_k: int = 10
    crop_top_k: int = 5
    high_thr: float = 0.4  # detections allowed to start tracks / matched first
    low_thr: float = 0.1  # global detections below this are dropped
    crop_score_thr: float = 0.15
    nms_iou: float = 0.5
    iou_gate: float = 0.1
    dist_gate: float = 1.0  # centre distance, in units of the track's box size
    min_hits: int = 3
    max_age: int = 15  # frames a confirmed track survives without a detection
    tentative_max_misses: int = 1
    use_crops: bool = True
    max_crops: int = 3
    crop_scale: float = 4.0  # crop side = max(img_size, crop_scale * target size)
    crop_skip_px: float = 32.0  # skip crop if target already >= this in the global pass
    # tile search: periodic native-resolution sweep to *discover* targets too small
    # for the global pass; hits start tentative tracks that crops then confirm
    use_tiles: bool = False  # opt-in: with a weak detector it mostly adds false positives
    tile_side: int = 384
    tile_overlap: float = 0.2
    tile_every_idle: int = 5  # period (frames) while no confirmed track exists
    tile_every_tracking: int = 15  # period once something is being tracked
    tile_top_k: int = 3
    tile_score_thr: float | None = None  # None -> high_thr
    # a track is only written while its running (EMA) score is above this; weak
    # clutter tracks kept alive by low-score detections stay internal
    out_score_thr: float = 0.35
    score_ema: float = 0.3
    backfill: bool = True
    emit_coasted: bool = False
    coast_max: int = 5


# --------------------------------------------------------------------------- Kalman


class BoxKalman:
    """Constant-velocity Kalman filter on (cx, cy, w, h), noise scaled by box size."""

    _F = np.eye(8)
    _F[:4, 4:] = np.eye(4)
    _H = np.eye(4, 8)
    _STD_POS = 1.0 / 20.0
    _STD_VEL = 1.0 / 160.0

    def __init__(self, box_xywh: list[float]) -> None:
        z = _xywh_to_cxcywh(box_xywh)
        self.mean = np.r_[z, np.zeros(4)]
        s = max(z[2], z[3], 1.0)
        std = np.r_[[2 * self._STD_POS * s] * 4, [10 * self._STD_VEL * s] * 4]
        self.cov = np.diag(std**2)

    def predict(self) -> None:
        s = max(self.mean[2], self.mean[3], 1.0)
        std = np.r_[[self._STD_POS * s] * 4, [self._STD_VEL * s] * 4]
        self.mean = self._F @ self.mean
        self.mean[2:4] = np.maximum(self.mean[2:4], 1.0)
        self.cov = self._F @ self.cov @ self._F.T + np.diag(std**2)

    def update(self, box_xywh: list[float]) -> None:
        z = _xywh_to_cxcywh(box_xywh)
        s = max(self.mean[2], self.mean[3], 1.0)
        r = np.diag(np.full(4, (self._STD_POS * s) ** 2))
        innov_cov = self._H @ self.cov @ self._H.T + r
        gain = np.linalg.solve(innov_cov, self._H @ self.cov).T
        self.mean = self.mean + gain @ (z - self._H @ self.mean)
        self.cov = (np.eye(8) - gain @ self._H) @ self.cov

    def box(self) -> list[float]:
        cx, cy, w, h = self.mean[:4]
        w, h = max(float(w), 1.0), max(float(h), 1.0)
        return [float(cx) - w / 2, float(cy) - h / 2, w, h]


def _xywh_to_cxcywh(b: list[float]) -> np.ndarray:
    return np.array([b[0] + b[2] / 2, b[1] + b[3] / 2, b[2], b[3]], dtype=np.float64)


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
    pending: list[tuple[int, list[float], float, int]] = field(default_factory=list)

    @property
    def confirmed(self) -> bool:
        return self.tid is not None


def _match(
    tracks: list[Track], dets: list[Det], cfg: Config
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian matching on IoU, with a centre-distance fallback for tiny targets."""
    if not tracks or not dets:
        return [], list(range(len(tracks))), list(range(len(dets)))
    tb = np.asarray([t.kf.box() for t in tracks], dtype=np.float64)
    db = np.asarray([d[2] for d in dets], dtype=np.float64)
    iou = _iou_pairwise(tb, db)
    tc = tb[:, :2] + tb[:, 2:] / 2
    dc = db[:, :2] + db[:, 2:] / 2
    scale = np.maximum(np.maximum(tb[:, 2], tb[:, 3]), 8.0)[:, None]
    dist = np.linalg.norm(tc[:, None, :] - dc[None, :, :], axis=2) / scale
    # the search radius grows while a track is coasting
    gate = cfg.dist_gate * (1.0 + 0.25 * np.asarray([t.misses for t in tracks]))[:, None]
    ok = (iou >= cfg.iou_gate) | (dist <= gate)
    cost = np.where(ok, (1.0 - iou) + 0.5 * np.minimum(dist, 2.0), 1e6)
    rows, cols = linear_sum_assignment(cost)
    matches = [(int(r), int(c)) for r, c in zip(rows, cols, strict=True) if ok[r, c]]
    mt, md = {r for r, _ in matches}, {c for _, c in matches}
    return (
        matches,
        [i for i in range(len(tracks)) if i not in mt],
        [j for j in range(len(dets)) if j not in md],
    )


def crop_regions(tracks: list[Track], w_px: int, h_px: int, cfg: Config) -> list[Region]:
    """Square native-resolution windows around the predicted position of small tracks."""
    cands = sorted(tracks, key=lambda t: (t.confirmed, t.score), reverse=True)
    regions: list[Region] = []
    for t in cands:
        if len(regions) >= cfg.max_crops:
            break
        x, y, w, h = t.kf.box()
        in_model = max(w * cfg.img_size / w_px, h * cfg.img_size / h_px)
        if in_model >= cfg.crop_skip_px:
            continue  # already well resolved by the global pass
        side = int(round(max(cfg.img_size, cfg.crop_scale * max(w, h))))
        side = max(1, min(side, w_px, h_px))
        cx, cy = x + w / 2, y + h / 2
        x0 = int(np.clip(round(cx - side / 2), 0, w_px - side))
        y0 = int(np.clip(round(cy - side / 2), 0, h_px - side))
        regions.append((x0, y0, side))
    return regions


def _axis_starts(n: int, side: int, stride: int) -> list[int]:
    if n <= side:
        return [0]
    starts = list(range(0, n - side + 1, stride))
    if starts[-1] != n - side:
        starts.append(n - side)
    return starts


def tile_regions(w_px: int, h_px: int, cfg: Config) -> list[Region]:
    """Overlapping square tiles covering the frame; empty if one tile would be the whole frame."""
    side = max(1, min(cfg.tile_side, w_px, h_px))
    if side >= w_px and side >= h_px:
        return []
    stride = max(1, int(side * (1.0 - cfg.tile_overlap)))
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


def track_sequence(
    frames: Iterator[tuple[int, Any, int, int]], detector: Detector, cfg: Config
) -> SeqResult:
    """Run detection + tracking over ``(frame_1based, image, width, height)`` tuples."""
    tracks: list[Track] = []
    next_tid = 1
    rows: list[tuple[int, int, list[float], float, int]] = []
    passes: list[int] = []
    frame_ms: list[float] = []

    for f1, image, w_px, h_px in frames:
        t0 = time.perf_counter()
        for t in tracks:
            t.kf.predict()

        dets = detector.detect(image, None, top_k=cfg.top_k, score_thr=cfg.low_thr)
        n_pass = 1
        if cfg.use_crops:
            for region in crop_regions(tracks, w_px, h_px, cfg):
                dets += detector.detect(
                    image, region, top_k=cfg.crop_top_k, score_thr=cfg.crop_score_thr
                )
                n_pass += 1
        if should_tile(f1, tracks, cfg):
            tile_thr = cfg.high_thr if cfg.tile_score_thr is None else cfg.tile_score_thr
            for region in tile_regions(w_px, h_px, cfg):
                dets += detector.detect(
                    image, region, top_k=cfg.tile_top_k, score_thr=tile_thr
                )
                n_pass += 1
        dets = _nms_dets(dets, iou_thr=cfg.nms_iou)

        high = [d for d in dets if d[1] >= cfg.high_thr]
        low = [d for d in dets if d[1] < cfg.high_thr]
        m1, rest_t, rest_high = _match(tracks, high, cfg)
        rest_tracks = [tracks[i] for i in rest_t]
        m2, rest_t2, _ = _match(rest_tracks, low, cfg)

        updates = [(tracks[i], high[j]) for i, j in m1]
        updates += [(rest_tracks[i], low[j]) for i, j in m2]
        for trk, (cid, score, box) in updates:
            trk.kf.update(box)
            trk.cls, trk.score = cid, score
            trk.avg_score += cfg.score_ema * (score - trk.avg_score)
            trk.hits += 1
            trk.misses = 0
            visible = trk.avg_score >= cfg.out_score_thr
            entry = (f1, list(box), score, cid)
            if trk.confirmed:
                if visible:
                    rows.append((f1, trk.tid, entry[1], score, cid))
            else:
                trk.pending.append(entry)
                if trk.hits >= cfg.min_hits:
                    trk.tid, next_tid = next_tid, next_tid + 1
                    flush = trk.pending if cfg.backfill else trk.pending[-1:]
                    if visible:
                        rows.extend((f, trk.tid, b, s, c) for f, b, s, c in flush)
                    trk.pending = []

        for i in rest_t2:
            rest_tracks[i].misses += 1

        for j in rest_high:
            cid, score, box = high[j]
            trk = Track(kf=BoxKalman(box), cls=cid, score=score, avg_score=score)
            trk.pending.append((f1, list(box), score, cid))
            if cfg.min_hits <= 1 and score >= cfg.out_score_thr:
                trk.tid, next_tid = next_tid, next_tid + 1
                rows.append((f1, trk.tid, list(box), score, cid))
                trk.pending = []
            tracks.append(trk)

        if cfg.emit_coasted:
            for t in tracks:
                if (
                    t.confirmed
                    and 1 <= t.misses <= cfg.coast_max
                    and t.avg_score >= cfg.out_score_thr
                ):
                    rows.append((f1, t.tid, t.kf.box(), t.score * 0.5, t.cls))

        tracks = [
            t
            for t in tracks
            if t.misses <= (cfg.max_age if t.confirmed else cfg.tentative_max_misses)
        ]
        passes.append(n_pass)
        frame_ms.append((time.perf_counter() - t0) * 1000.0)

    rows.sort(key=lambda r: (r[0], r[1]))
    lines = [
        f"{f},{tid},{b[0]:.2f},{b[1]:.2f},{b[2]:.2f},{b[3]:.2f},{s:.4f},{c + 1},1.0"
        for f, tid, b, s, c in rows
    ]
    return SeqResult(lines=lines, passes=passes, frame_ms=frame_ms)


# --------------------------------------------------------------------------- I/O


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


def iter_video_frames(video: Path, tmp_dir: Path) -> Iterator[tuple[int, Image.Image, int, int]]:
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
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--eval-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--split", choices=("all", "valid", "test"), default="all")
    p.add_argument("--num-classes", type=int, default=d.num_classes)
    p.add_argument("--img-size", type=int, default=d.img_size)
    p.add_argument("--high-thr", type=float, default=d.high_thr)
    p.add_argument("--low-thr", type=float, default=d.low_thr)
    p.add_argument("--crop-score-thr", type=float, default=d.crop_score_thr)
    p.add_argument("--min-hits", type=int, default=d.min_hits)
    p.add_argument("--max-age", type=int, default=d.max_age)
    p.add_argument("--max-crops", type=int, default=d.max_crops)
    p.add_argument("--crop-scale", type=float, default=d.crop_scale)
    p.add_argument("--crop-skip-px", type=float, default=d.crop_skip_px)
    p.add_argument("--no-crops", action="store_true", help="tracker only (ablation)")
    p.add_argument("--tiles", action="store_true", help="enable periodic tile search")
    p.add_argument("--tile-side", type=int, default=d.tile_side)
    p.add_argument("--tile-overlap", type=float, default=d.tile_overlap)
    p.add_argument("--tile-every-idle", type=int, default=d.tile_every_idle)
    p.add_argument("--tile-every-tracking", type=int, default=d.tile_every_tracking)
    p.add_argument("--tile-score-thr", type=float, default=None, help="default: --high-thr")
    p.add_argument("--out-score-thr", type=float, default=d.out_score_thr)
    p.add_argument("--no-backfill", action="store_true", help="strictly online output")
    p.add_argument("--emit-coasted", action="store_true", help="write predicted boxes in short gaps")
    p.add_argument("--coast-max", type=int, default=d.coast_max)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH (macOS: brew install ffmpeg)")
    cfg = Config(
        img_size=args.img_size,
        num_classes=args.num_classes,
        high_thr=args.high_thr,
        low_thr=args.low_thr,
        crop_score_thr=args.crop_score_thr,
        min_hits=args.min_hits,
        max_age=args.max_age,
        use_crops=not args.no_crops,
        max_crops=args.max_crops,
        crop_scale=args.crop_scale,
        crop_skip_px=args.crop_skip_px,
        use_tiles=args.tiles,
        tile_side=args.tile_side,
        tile_overlap=args.tile_overlap,
        tile_every_idle=args.tile_every_idle,
        tile_every_tracking=args.tile_every_tracking,
        tile_score_thr=args.tile_score_thr,
        out_score_thr=args.out_score_thr,
        backfill=not args.no_backfill,
        emit_coasted=args.emit_coasted,
        coast_max=args.coast_max,
    )
    detector = OnnxDetector(args.onnx, cfg)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_passes: list[int] = []
    all_ms: list[float] = []
    per_seq: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory() as td:
        for seq_dir in list_sequences(args.eval_dir, args.split):
            frames_dir = Path(td) / seq_dir.name
            frames_dir.mkdir()
            res = track_sequence(iter_video_frames(seq_dir / "video.mp4", frames_dir), detector, cfg)
            (args.out_dir / f"{seq_dir.name}.txt").write_text(
                "\n".join(res.lines) + ("\n" if res.lines else ""), encoding="utf-8"
            )
            all_passes += res.passes
            all_ms += res.frame_ms
            per_seq[seq_dir.name] = {
                "frames": len(res.passes),
                "boxes": len(res.lines),
                "mean_passes": round(float(np.mean(res.passes)), 2) if res.passes else 0.0,
            }
    print(
        json.dumps(
            {
                "config": cfg.__dict__,
                "sequences": per_seq,
                "mean_model_passes_per_frame": round(float(np.mean(all_passes)), 2) if all_passes else 0.0,
                "median_ms_per_frame": round(float(np.median(all_ms)), 2) if all_ms else 0.0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
