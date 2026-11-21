"""Multi-scale SAHI / tiled COCO detection inference for static RF-DETR ONNX.

Runs a full-frame global pass plus optional overlapping square tiles at native
resolution (or smaller upscaled tiles) and merges detections per class via
Weighted Boxes Fusion (WBF) or class-aware NMS.

Improves ``small_AP@50`` and ``mAP@50:95`` on high-resolution scenes where
drones shrink below 16 px in a single 384x384 resize.

Usage (from starter-kit/):
    python predict_detect_tiled.py --onnx exports/rfdetr_20ep/full/inference_model.onnx \\
        --dataset-dir ../dataset --split valid --out preds_tiled.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image

from predict_onnx_baseline import _decode_rfdetr_topk, _iou_pairwise, _preprocess_imagenet


def _axis_starts(length: int, side: int, stride: int) -> list[int]:
    if length <= side:
        return [0]
    starts = list(range(0, length - side + 1, stride))
    if starts[-1] != length - side:
        starts.append(length - side)
    return starts


def _tile_boxes(w_px: int, h_px: int, side: int, overlap: float) -> list[tuple[int, int, int]]:
    s = max(1, min(side, w_px, h_px))
    if s >= w_px and s >= h_px:
        return []
    stride = max(1, int(round(s * (1.0 - overlap))))
    return [
        (x0, y0, s)
        for y0 in _axis_starts(h_px, s, stride)
        for x0 in _axis_starts(w_px, s, stride)
    ]


def _wbf_merge(
    dets: list[tuple[int, float, list[float]]],
    *,
    iou_thr: float = 0.5,
    max_dets: int = 100,
) -> list[tuple[int, float, list[float]]]:
    """Class-aware Weighted Boxes Fusion (preserving high-confidence consensus)."""
    by_cls: dict[int, list[tuple[float, np.ndarray]]] = defaultdict(list)
    for cid, score, box in dets:
        by_cls[cid].append((score, np.asarray(box, dtype=np.float64)))

    merged: list[tuple[int, float, list[float]]] = []
    for cid, items in by_cls.items():
        items.sort(key=lambda x: -x[0])
        clusters_boxes: list[list[np.ndarray]] = []
        clusters_scores: list[list[float]] = []
        rep_boxes: list[np.ndarray] = []

        for score, box in items:
            if not rep_boxes:
                clusters_boxes.append([box])
                clusters_scores.append([score])
                rep_boxes.append(box.copy())
                continue
            ious = _iou_pairwise(box[None, :], np.stack(rep_boxes))[0]
            best = int(np.argmax(ious))
            if ious[best] >= iou_thr:
                clusters_boxes[best].append(box)
                clusters_scores[best].append(score)
                ws = np.asarray(clusters_scores[best], dtype=np.float64)[:, None]
                bs = np.stack(clusters_boxes[best])
                rep_boxes[best] = (ws * bs).sum(axis=0) / ws.sum()
            else:
                clusters_boxes.append([box])
                clusters_scores.append([score])
                rep_boxes.append(box.copy())

        for rep, scores in zip(rep_boxes, clusters_scores, strict=True):
            # Take max score (slightly boosted only if confirmed by multiple passes,
            # capped at 1.0) so ranking matches COCO AP well without inflating FP.
            best_s = max(scores)
            fused_s = min(1.0, best_s + 0.03 * (len(scores) - 1) * (1.0 - best_s))
            merged.append((cid, float(fused_s), [float(v) for v in rep]))

    merged.sort(key=lambda x: -x[1])
    return merged[:max_dets]


def _make_runner(
    onnx_path: Path | None,
    backbone_onnx: Path | None,
    decoder_onnx: Path | None,
):
    """Return callable ``inp -> {'dets': ..., 'labels': ...}`` for full or split ONNX."""
    if backbone_onnx is not None and decoder_onnx is not None:
        bb = ort.InferenceSession(str(backbone_onnx), providers=["CPUExecutionProvider"])
        dec = ort.InferenceSession(str(decoder_onnx), providers=["CPUExecutionProvider"])
        bb_in = bb.get_inputs()[0].name
        bb_out = bb.get_outputs()[0].name
        dec_in = dec.get_inputs()[0].name
        dec_outs = [o.name for o in dec.get_outputs()]

        def _run_split(inp: np.ndarray) -> dict[str, np.ndarray]:
            feats = bb.run([bb_out], {bb_in: inp})[0]
            return dict(zip(dec_outs, dec.run(dec_outs, {dec_in: feats}), strict=True))

        return _run_split

    if onnx_path is None:
        raise SystemExit("Provide --onnx or both --backbone-onnx and --decoder-onnx")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]

    def _run_full(inp: np.ndarray) -> dict[str, np.ndarray]:
        return dict(zip(out_names, sess.run(out_names, {in_name: inp}), strict=True))

    return _run_full


def predict_detect_tiled(
    onnx_path: Path | None,
    dataset_dir: Path,
    split: str,
    out_json: Path,
    *,
    backbone_onnx: Path | None = None,
    decoder_onnx: Path | None = None,
    img_size: int = 384,
    tile_side: int = 512,
    tile_overlap: float = 0.2,
    tile_score_thr: float = 0.05,
    tile_score_weight: float = 0.9,
    min_dim_for_tiling: int = 640,
    merge_iou: float = 0.5,
    max_dets: int = 100,
) -> dict[str, Any]:
    coco_path = dataset_dir / split / "_annotations.coco.json"
    gt_doc = json.loads(coco_path.read_text(encoding="utf-8"))
    num_classes = len(gt_doc["categories"])
    run_model = _make_runner(onnx_path, backbone_onnx, decoder_onnx)

    preds: list[dict[str, Any]] = []
    latencies: list[float] = []
    passes_list: list[int] = []

    for im_rec in gt_doc["images"]:
        w_px = int(im_rec["width"])
        h_px = int(im_rec["height"])
        t0 = time.perf_counter()
        with Image.open(dataset_dir / split / im_rec["file_name"]) as im_raw:
            im = im_raw.convert("RGB")
            # 1. Global full-frame pass
            inp = _preprocess_imagenet(im, img_size)
            outs = run_model(inp)
            img_dets = _decode_rfdetr_topk(
                outs["dets"][0],
                outs["labels"][0],
                w_px,
                h_px,
                num_classes,
                top_k=100,
                score_thr=0.0,
            )
            n_pass = 1

            # 2. Tiled passes only when image is large enough that small targets shrink
            if tile_side > 0 and max(w_px, h_px) >= min_dim_for_tiling:
                tiles = _tile_boxes(w_px, h_px, tile_side, tile_overlap)
                for x0, y0, side in tiles:
                    crop = im.crop((x0, y0, x0 + side, y0 + side))
                    c_inp = _preprocess_imagenet(crop, img_size)
                    c_outs = run_model(c_inp)
                    n_pass += 1
                    for cid, score, b in _decode_rfdetr_topk(
                        c_outs["dets"][0],
                        c_outs["labels"][0],
                        side,
                        side,
                        num_classes,
                        top_k=15,
                        score_thr=tile_score_thr,
                    ):
                        # Ignore partial clipped huge boxes touching tile edges
                        if b[2] > 0.85 * side or b[3] > 0.85 * side:
                            continue
                        img_dets.append(
                            (
                                cid,
                                score * tile_score_weight,
                                [b[0] + x0, b[1] + y0, b[2], b[3]],
                            )
                        )

        merged = _wbf_merge(img_dets, iou_thr=merge_iou, max_dets=max_dets)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        passes_list.append(n_pass)

        for cid, score, bbox in merged:
            if bbox[2] <= 0.0 or bbox[3] <= 0.0:
                continue
            preds.append(
                {
                    "image_id": int(im_rec["id"]),
                    "category_id": cid,
                    "bbox": bbox,
                    "score": score,
                }
            )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(preds), encoding="utf-8")
    return {
        "predictions": len(preds),
        "images": len(gt_doc["images"]),
        "mean_passes": round(float(np.mean(passes_list)) if passes_list else 0.0, 2),
        "median_ms": round(float(np.median(latencies)) if latencies else 0.0, 2),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx", type=Path, default=None)
    p.add_argument("--backbone-onnx", type=Path, default=None)
    p.add_argument("--decoder-onnx", type=Path, default=None)
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--split", default="valid")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--tile-side", type=int, default=512)
    p.add_argument("--tile-overlap", type=float, default=0.2)
    p.add_argument("--tile-score-thr", type=float, default=0.05)
    p.add_argument("--tile-score-weight", type=float, default=0.9)
    p.add_argument("--min-dim-for-tiling", type=int, default=640)
    p.add_argument("--merge-iou", type=float, default=0.5)
    p.add_argument("--max-dets", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary = predict_detect_tiled(
        args.onnx,
        args.dataset_dir,
        args.split,
        args.out,
        backbone_onnx=args.backbone_onnx,
        decoder_onnx=args.decoder_onnx,
        img_size=args.img_size,
        tile_side=args.tile_side,
        tile_overlap=args.tile_overlap,
        tile_score_thr=args.tile_score_thr,
        tile_score_weight=args.tile_score_weight,
        min_dim_for_tiling=args.min_dim_for_tiling,
        merge_iou=args.merge_iou,
        max_dets=args.max_dets,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
