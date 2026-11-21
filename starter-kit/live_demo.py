"""Real-time video / webcam demo for C-UAS Detection + Unified Kalman/ROI Tracking.

Runs either:
  - PyTorch CUDA (``--weights runs/c1_nano_soup65/checkpoint_soup.pth``) for true
    real-time speed on GPU (30+ FPS on RTX 3050);
  - Static ONNX Runtime (``--onnx exports/rfdetr_soup65/full/inference_model.onnx``)
    for CPU deployment parity.

Draws bounding boxes, track IDs, class labels, trajectory trails, active zoomed
ROI crop windows, and a live FPS/latency HUD.

Examples (from starter-kit/):
    # Real-time on a video file using GPU PyTorch (fastest):
    .venv/bin/python live_demo.py \\
        --weights runs/c1_nano_soup65/checkpoint_soup.pth \\
        --source ../tracking_eval/seq_0004/video.mp4

    # Real-time on webcam (camera 0):
    .venv/bin/python live_demo.py \\
        --weights runs/c1_nano_soup65/checkpoint_soup.pth \\
        --source 0

    # Using exported ONNX model (CPU) and saving annotated output video:
    .venv/bin/python live_demo.py \\
        --onnx exports/rfdetr_soup65/full/inference_model.onnx \\
        --source ../tracking_eval/seq_0001/video.mp4 \\
        --save runs/demo_seq0001.mp4
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from predict_onnx_baseline import _decode_rfdetr_topk, _preprocess_imagenet
from track_roi import (
    BoxKalman,
    CameraMotionCompensator,
    Config,
    Det,
    OnnxDetector,
    Region,
    Track,
    _clamp_box,
    _match,
    _near_existing_track,
    _nms_small_aware,
    crop_regions,
)

CLASS_NAMES = ("drone_air", "airplane", "helicopter")
CLASS_COLORS = (
    (0, 220, 255),   # drone_air: amber/yellow (BGR)
    (255, 180, 0),   # airplane: cyan-blue
    (80, 255, 120),  # helicopter: green
)


class TorchGpuDetector:
    """Fast PyTorch CUDA RF-DETR detector with identical decoding to OnnxDetector."""

    def __init__(self, weights_path: Path, cfg: Config, device: str = "cuda") -> None:
        import torch
        from rfdetr import RFDETRNano

        self.cfg = cfg
        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_amp = self.device.type == "cuda"

        m = RFDETRNano(num_classes=cfg.num_classes, pretrain_weights=str(weights_path))
        lw = m.model.model.eval().to(self.device)
        lw.export()
        self.net = lw

        # Warmup once so first live frame is fast
        dummy = torch.zeros(1, 3, cfg.img_size, cfg.img_size, device=self.device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_amp):
            self.net(dummy)

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
        t = self.torch.from_numpy(inp).to(self.device, non_blocking=True)
        with (
            self.torch.no_grad(),
            self.torch.amp.autocast("cuda", dtype=self.torch.bfloat16, enabled=self.use_amp),
        ):
            dets_t, labels_t = self.net(t)
        dets_np = dets_t[0].float().cpu().numpy()
        labels_np = labels_t[0].float().cpu().numpy()
        dets = _decode_rfdetr_topk(
            dets_np,
            labels_np,
            w_px,
            h_px,
            self.cfg.num_classes,
            top_k=top_k,
            score_thr=score_thr,
        )
        return [(c, s, [b[0] + x0, b[1] + y0, b[2], b[3]]) for c, s, b in dets]


class OnlineTracker:
    """Strictly online step-by-step wrapper around the unified Kalman + CMC + ROI tracker."""

    def __init__(self, detector: Any, cfg: Config) -> None:
        self.detector = detector
        self.cfg = cfg
        self.cmc = CameraMotionCompensator() if cfg.use_cmc else None
        self.tracks: list[Track] = []
        self.next_tid = 1
        self.frame_idx = 0

    def step(
        self, bgr: np.ndarray
    ) -> tuple[list[tuple[int, int, float, list[float]]], list[Region], int]:
        self.frame_idx += 1
        h_px, w_px = bgr.shape[:2]
        rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        M = self.cmc.estimate(rgb) if self.cmc is not None else None
        for t in self.tracks:
            if M is not None:
                t.kf.apply_cmc(M)
            t.kf.predict()

        dets = self.detector.detect(
            rgb, None, top_k=self.cfg.top_k, score_thr=self.cfg.low_thr
        )
        n_pass = 1
        regions: list[Region] = []
        if self.cfg.use_crops:
            regions = crop_regions(self.tracks, w_px, h_px, self.cfg)
            for region in regions:
                _, _, side = region
                for cid, s, b in self.detector.detect(
                    rgb, region, top_k=self.cfg.crop_top_k, score_thr=self.cfg.crop_score_thr
                ):
                    if b[2] < 0.85 * side and b[3] < 0.85 * side:
                        dets.append((cid, s, b))
                n_pass += 1

        dets = _nms_small_aware(dets, iou_thr=self.cfg.nms_iou)
        high = [d for d in dets if d[1] >= self.cfg.high_thr]
        low = [d for d in dets if d[1] < self.cfg.high_thr]

        m1, rest_t, rest_high = _match(self.tracks, high, self.cfg)
        rest_tracks = [self.tracks[i] for i in rest_t]
        m2, rest_t2, _ = _match(rest_tracks, low, self.cfg)

        active_out: list[tuple[int, int, float, list[float]]] = []
        for trk, (cid, score, box) in [(self.tracks[i], high[j]) for i, j in m1] + [
            (rest_tracks[i], low[j]) for i, j in m2
        ]:
            trk.kf.update(box)
            trk.cls, trk.score = cid, score
            trk.class_votes[cid] += score
            trk.avg_score += self.cfg.score_ema * (score - trk.avg_score)
            trk.hits += 1
            trk.misses = 0
            trk.last_det_frame = self.frame_idx
            trk.last_det_box = list(box)
            if not trk.confirmed and trk.hits >= self.cfg.min_hits:
                trk.tid, self.next_tid = self.next_tid, self.next_tid + 1
            if trk.confirmed and trk.tid is not None and trk.avg_score >= self.cfg.out_score_thr:
                active_out.append(
                    (trk.tid, trk.voted_cls, score, _clamp_box(list(box), w_px, h_px))
                )

        for i in rest_t2:
            rest_tracks[i].misses += 1

        for j in rest_high:
            cid, score, box = high[j]
            if _near_existing_track(box, self.tracks):
                continue
            kf = BoxKalman(
                box,
                min_pos_std=self.cfg.min_pos_std_px,
                min_vel_std=self.cfg.min_vel_std_px,
            )
            trk = Track(
                kf=kf,
                cls=cid,
                score=score,
                avg_score=score,
                last_det_frame=self.frame_idx,
                last_det_box=list(box),
            )
            trk.class_votes[cid] += score
            self.tracks.append(trk)

        self.tracks = [
            t
            for t in self.tracks
            if t.misses <= (self.cfg.max_age if t.confirmed else self.cfg.tentative_max_misses)
        ]
        return active_out, regions, n_pass


def _draw_overlay(
    frame: np.ndarray,
    tracks_out: list[tuple[int, int, float, list[float]]],
    regions: list[Region],
    trails: dict[int, deque[tuple[int, int]]],
    fps: float,
    ms: float,
    n_pass: int,
) -> np.ndarray:
    vis = frame.copy()
    # Draw zoomed ROI crop regions
    for x0, y0, side in regions:
        cv2.rectangle(vis, (x0, y0), (x0 + side, y0 + side), (255, 255, 0), 1)
        cv2.putText(
            vis,
            f"ROI {side}px",
            (x0 + 4, max(14, y0 + 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

    for tid, cid, score, (x, y, w, h) in tracks_out:
        color = CLASS_COLORS[cid % len(CLASS_COLORS)]
        cx, cy = int(round(x + w / 2.0)), int(round(y + h / 2.0))
        trails[tid].append((cx, cy))
        pts = list(trails[tid])
        for i in range(1, len(pts)):
            cv2.line(vis, pts[i - 1], pts[i], color, 2, cv2.LINE_AA)

        x1, y1 = int(round(x)), int(round(y))
        x2, y2 = int(round(x + w)), int(round(y + h))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        name = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"cls{cid}"
        label = f"ID {tid} | {name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ty = max(th + 6, y1 - 4)
        cv2.rectangle(vis, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), color, -1)
        cv2.putText(
            vis,
            label,
            (x1 + 3, ty - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    hud = f"FPS: {fps:5.1f} | {ms:5.1f} ms | passes: {n_pass} | tracks: {len(tracks_out)}"
    cv2.rectangle(vis, (8, 8), (540, 38), (20, 20, 20), -1)
    cv2.putText(
        vis, hud, (16, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
    )
    return vis


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--source", required=True, help="Video path (.mp4) or webcam index (e.g. 0)")
    p.add_argument("--weights", type=Path, default=None, help="PyTorch .pth checkpoint (runs on CUDA GPU)")
    p.add_argument("--onnx", type=Path, default=None, help="Static ONNX model (runs on CPU ORT)")
    p.add_argument("--save", type=Path, default=None, help="Optional output annotated MP4 path")
    p.add_argument("--no-display", action="store_true", help="Disable OpenCV GUI window (headless)")
    p.add_argument("--no-crops", action="store_true", help="Disable zoomed 192px ROI crops")
    p.add_argument("--no-cmc", action="store_true", help="Disable camera motion compensation")
    p.add_argument("--high-thr", type=float, default=0.40)
    p.add_argument("--out-score-thr", type=float, default=0.40)
    p.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = full video)")
    p.add_argument("--max-display-height", type=int, default=900, help="Scale preview window if taller")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        high_thr=args.high_thr,
        out_score_thr=args.out_score_thr,
        use_crops=not args.no_crops,
        use_cmc=not args.no_cmc,
        use_discover=False,
    )

    if args.weights is not None:
        detector: Any = TorchGpuDetector(args.weights, cfg)
        print(f"Loaded PyTorch detector on {detector.device} from {args.weights}")
    elif args.onnx is not None:
        detector = OnnxDetector(args.onnx, cfg)
        print(f"Loaded ONNX Runtime CPU detector from {args.onnx}")
    else:
        raise SystemExit("Provide either --weights <checkpoint.pth> (GPU) or --onnx <model.onnx> (CPU)")

    src: int | str = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {args.source}")

    writer: cv2.VideoWriter | None = None
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.save), fourcc, fps_in, (w_in, h_in))

    tracker = OnlineTracker(detector, cfg)
    trails: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=30))
    fps_ema = 0.0
    frame_times: list[float] = []

    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            tracks_out, regions, n_pass = tracker.step(bgr)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            frame_times.append(dt_ms)
            inst_fps = 1000.0 / max(dt_ms, 1e-3)
            fps_ema = inst_fps if fps_ema == 0.0 else 0.85 * fps_ema + 0.15 * inst_fps

            vis = _draw_overlay(bgr, tracks_out, regions, trails, fps_ema, dt_ms, n_pass)
            if writer is not None:
                writer.write(vis)

            if not args.no_display:
                disp = vis
                if disp.shape[0] > args.max_display_height:
                    scale = args.max_display_height / disp.shape[0]
                    disp = cv2.resize(disp, None, fx=scale, fy=scale)
                cv2.imshow("C-UAS Live Tracking (q to quit)", disp)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

            if args.max_frames > 0 and tracker.frame_idx >= args.max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    if frame_times:
        med_ms = float(np.median(frame_times))
        print(
            f"Processed {len(frame_times)} frames | median={med_ms:.1f} ms/frame "
            f"({1000.0 / med_ms:.1f} FPS)"
        )


if __name__ == "__main__":
    main()
