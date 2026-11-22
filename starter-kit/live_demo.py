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
import threading
import time
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
    _recover_occluded,
    crop_regions,
    select_discover_tiles,
    tile_regions,
    warp_box_cmc,
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
        self._discover_cursor = 0
        self._discover_tiles: list[Region] = []
        self._discover_dims: tuple[int, int] | None = None
        self._discover_cooldown: dict[int, int] = {}

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
                if t.cmc_last_box:
                    t.cmc_last_box = warp_box_cmc(t.cmc_last_box, M)
            t.kf.predict(coasting=(t.misses > 0), vel_damping=self.cfg.vel_damping)

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

        if self.cfg.use_discover:
            has_visible = any(
                t.confirmed and t.avg_score >= self.cfg.out_score_thr for t in self.tracks
            )
            budget = (
                self.cfg.discover_budget_tracking
                if has_visible
                else self.cfg.discover_budget_idle
            )
            if budget > 0:
                if self._discover_dims != (w_px, h_px):
                    self._discover_dims = (w_px, h_px)
                    self._discover_tiles = tile_regions(
                        w_px, h_px, self.cfg.discover_tile_side, self.cfg.discover_overlap
                    )
                    self._discover_cursor = 0
                    self._discover_cooldown.clear()
                chosen_tiles, self._discover_cursor = select_discover_tiles(
                    self._discover_tiles,
                    self.cmc.motion_map if self.cmc is not None else None,
                    self.cmc.downscale if self.cmc is not None else 0.5,
                    self._discover_cursor,
                    budget,
                    self._discover_cooldown,
                    self.frame_idx,
                    use_motion=self.cfg.discover_motion_priority,
                )
                micro_cand: tuple[float, list[float]] | None = None
                for region in chosen_tiles:
                    _, _, side = region
                    for cid, s, b in self.detector.detect(
                        rgb,
                        region,
                        top_k=self.cfg.discover_top_k,
                        score_thr=self.cfg.discover_score_thr * 0.75
                        if self.cfg.discover_micro_zoom > 0
                        else self.cfg.discover_score_thr,
                    ):
                        if b[2] >= 0.85 * side or b[3] >= 0.85 * side:
                            continue
                        if s >= self.cfg.discover_score_thr:
                            dets.append((cid, s, b))
                        if (
                            self.cfg.discover_micro_zoom > 0
                            and self.cfg.discover_micro_zoom < side
                            and max(b[2], b[3]) <= 24.0
                            and s < self.cfg.high_thr
                            and (micro_cand is None or s > micro_cand[0])
                        ):
                            micro_cand = (s, b)
                    n_pass += 1
                if micro_cand is not None:
                    _, mb = micro_cand
                    mside = max(1, min(self.cfg.discover_micro_zoom, w_px, h_px))
                    mcx, mcy = mb[0] + mb[2] / 2.0, mb[1] + mb[3] / 2.0
                    mx0 = int(np.clip(round(mcx - mside / 2.0), 0, w_px - mside))
                    my0 = int(np.clip(round(mcy - mside / 2.0), 0, h_px - mside))
                    regions.append((mx0, my0, mside))
                    for cid, s, b in self.detector.detect(
                        rgb,
                        (mx0, my0, mside),
                        top_k=self.cfg.crop_top_k,
                        score_thr=self.cfg.crop_score_thr,
                    ):
                        if b[2] < 0.85 * mside and b[3] < 0.85 * mside:
                            dets.append((cid, s, b))
                    n_pass += 1

        dets = _nms_small_aware(dets, iou_thr=self.cfg.nms_iou)
        high = [d for d in dets if d[1] >= self.cfg.high_thr]
        low = [d for d in dets if d[1] < self.cfg.high_thr]

        m1, rest_t, rest_high = _match(self.tracks, high, self.cfg)
        rest_tracks = [self.tracks[i] for i in rest_t]
        m2, rest_t2, _ = _match(rest_tracks, low, self.cfg)

        occ_tracks = [rest_tracks[i] for i in rest_t2]
        unmatched_high_dets = [high[j] for j in rest_high]
        m3, rest_t3, rest_high_idx = _recover_occluded(
            occ_tracks, unmatched_high_dets, self.cfg
        )
        rest_high = [rest_high[j] for j in rest_high_idx]

        active_out: list[tuple[int, int, float, list[float]]] = []
        updates: list[tuple[Track, Det, bool]] = [(self.tracks[i], high[j], False) for i, j in m1]
        updates += [(rest_tracks[i], low[j], False) for i, j in m2]
        updates += [(occ_tracks[i], unmatched_high_dets[j], True) for i, j in m3]

        for trk, (cid, score, box), recovered in updates:
            gap_dt = self.frame_idx - trk.last_det_frame if trk.last_det_frame > 0 else 1
            if recovered or trk.misses >= 2:
                trk.kf.reanchor(box, dt=gap_dt, prev_cmc_box=trk.cmc_last_box)
            else:
                trk.kf.update(box)
            trk.cls, trk.score = cid, score
            trk.class_votes[cid] += score
            trk.avg_score += self.cfg.score_ema * (score - trk.avg_score)
            trk.hits += 1
            trk.misses = 0
            trk.last_det_frame = self.frame_idx
            trk.last_det_box = list(box)
            trk.cmc_last_box = list(box)
            if not trk.confirmed and trk.hits >= self.cfg.min_hits:
                trk.tid, self.next_tid = self.next_tid, self.next_tid + 1
            if trk.confirmed and trk.tid is not None and trk.avg_score >= self.cfg.out_score_thr:
                active_out.append(
                    (trk.tid, trk.voted_cls, score, _clamp_box(list(box), w_px, h_px))
                )

        for i in rest_t3:
            occ_tracks[i].misses += 1

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
                cmc_last_box=list(box),
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


class _MjpegServer:
    """Zero-dependency live MJPEG preview server (fallback when OpenCV is headless)."""

    def __init__(self, port: int = 8766) -> None:
        self.port = port
        self._cond = threading.Condition()
        self._jpeg: bytes | None = None
        self._stopped = False
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def do_GET(self) -> None:
                if self.path in ("/", "/index.html"):
                    html = (
                        b"<!doctype html><html><body style='margin:0;background:#111;"
                        b"display:flex;justify-content:center;align-items:center;height:100vh'>"
                        b"<img src='/stream.mjpg' style='max-width:100%;max-height:100vh'/>"
                        b"</body></html>"
                    )
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(html)))
                    self.end_headers()
                    self.wfile.write(html)
                    return
                if self.path != "/stream.mjpg":
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header(
                    "Content-Type", "multipart/x-mixed-replace; boundary=frame"
                )
                self.end_headers()
                try:
                    while not parent._stopped:
                        with parent._cond:
                            parent._cond.wait(timeout=0.5)
                            jpg = parent._jpeg
                        if jpg is None:
                            continue
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                            + str(len(jpg)).encode()
                            + b"\r\n\r\n"
                            + jpg
                            + b"\r\n"
                        )
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self._httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def push(self, bgr: np.ndarray) -> None:
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok:
            with self._cond:
                self._jpeg = buf.tobytes()
                self._cond.notify_all()

    def close(self) -> None:
        self._stopped = True
        with self._cond:
            self._cond.notify_all()
        self._httpd.shutdown()
        self._httpd.server_close()


def _has_cv2_gui() -> bool:
    try:
        cv2.namedWindow("__probe__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__probe__")
        return True
    except cv2.error:
        return False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--source", required=True, help="Video path (.mp4) or webcam index (e.g. 0)")
    p.add_argument("--weights", type=Path, default=None, help="PyTorch .pth checkpoint (runs on CUDA GPU)")
    p.add_argument("--onnx", type=Path, default=None, help="Static ONNX model (runs on CPU ORT)")
    p.add_argument("--save", type=Path, default=None, help="Optional output annotated MP4 path")
    p.add_argument("--no-display", action="store_true", help="Disable live preview completely")
    p.add_argument("--web", action="store_true", help="Stream live preview to browser (auto if OpenCV is headless)")
    p.add_argument("--port", type=int, default=8766, help="Port for --web live MJPEG preview")
    p.add_argument("--loop", action="store_true", help="Loop video file continuously for live viewing")
    p.add_argument("--no-crops", action="store_true", help="Disable zoomed ROI crops")
    p.add_argument("--min-crop-side", type=int, default=192, help="Min crop side in px (< 384 enables upscaling)")
    p.add_argument("--discover", action="store_true", help="Enable round-robin idle tile discovery for tiny drones")
    p.add_argument("--discover-tile-side", type=int, default=256, help="Tile side (px) for --discover")
    p.add_argument("--discover-budget-idle", type=int, default=6, help="Discovery tiles/frame while idle")
    p.add_argument("--discover-score-thr", type=float, default=0.45, help="Score threshold on discovery tiles")
    p.add_argument("--no-cmc", action="store_true", help="Disable camera motion compensation")
    p.add_argument("--high-thr", type=float, default=0.40)
    p.add_argument("--out-score-thr", type=float, default=0.40)
    p.add_argument("--max-age", type=int, default=45, help="Frames to keep track ID alive during occlusion")
    p.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = full video)")
    p.add_argument("--max-display-height", type=int, default=900, help="Scale preview window if taller")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        high_thr=args.high_thr,
        out_score_thr=args.out_score_thr,
        max_age=args.max_age,
        use_crops=not args.no_crops,
        min_crop_side=args.min_crop_side,
        use_cmc=not args.no_cmc,
        use_discover=args.discover,
        discover_tile_side=args.discover_tile_side,
        discover_budget_idle=args.discover_budget_idle,
        discover_score_thr=args.discover_score_thr,
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

    use_gui = False
    web_server: _MjpegServer | None = None
    if not args.no_display:
        if not args.web and _has_cv2_gui():
            use_gui = True
        else:
            web_server = _MjpegServer(port=args.port)
            print(
                f"OpenCV headless detected (or --web set) -> Live stream ready at "
                f"http://127.0.0.1:{args.port}  (Ctrl+C to stop)",
                flush=True,
            )

    tracker = OnlineTracker(detector, cfg)
    trails: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=30))
    fps_ema = 0.0
    frame_times: list[float] = []

    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                if args.loop and isinstance(src, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    tracker = OnlineTracker(detector, cfg)
                    trails.clear()
                    continue
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

            if use_gui:
                disp = vis
                if disp.shape[0] > args.max_display_height:
                    scale = args.max_display_height / disp.shape[0]
                    disp = cv2.resize(disp, None, fx=scale, fy=scale)
                cv2.imshow("C-UAS Live Tracking (q to quit)", disp)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            elif web_server is not None:
                web_server.push(vis)

            if args.max_frames > 0 and tracker.frame_idx >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if web_server is not None:
            web_server.close()
        if use_gui:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    if frame_times:
        med_ms = float(np.median(frame_times))
        print(
            f"Processed {len(frame_times)} frames | median={med_ms:.1f} ms/frame "
            f"({1000.0 / med_ms:.1f} FPS)"
        )


if __name__ == "__main__":
    main()
