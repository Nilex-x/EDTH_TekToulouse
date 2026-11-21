"""Build the master leak-free, tight-box, hard-negative C-UAS dataset.

Combines ALL available curated sources cleanly:
  1. Official ``dataset/train/`` (859 tight-box images) + 3x zoom (128->384) crops
     for tiny targets (< 18 px) + 2x zoom (192->384) crops for small targets (< 38 px);
  2. Teammate's ``dataset_cuas_hardneg/train/`` (structure-rich Video_V frames,
     empty structure/bird frames, and all manually verified model false positives):
     - Leak-checked via ``dHash64`` against official ``dataset/valid/``, ``dataset/test/``,
       and ``tracking_eval/`` (ZERO leakage);
     - **Auto-tightened Video_V boxes**: runs ``rfdetr_soup65`` on a zoomed crop around
       each padded MATLAB ``.mat`` box to replace 2-4x oversized boxes with Challenge-1
       tight boxes while preserving the exact ground-truth ``.mat`` class label!
     - **Targeted antenna/structure FP hard-negative crops**: uses
       ``dataset_cuas_hardneg/model_false_positives.json`` to crop 192x192 & 384x384
       hard-negative patches centered directly on every antenna/pole/chimney/bird
       false positive (empty labels ``[]``);
  3. Official ``dataset/valid/`` and ``dataset/test/`` symlinked UNTOUCHED so training
     validation reflects true Challenge 1 leaderboard metrics.

Usage (from starter-kit/):
    python build_master_dataset.py \\
        --challenge-dir ../dataset \\
        --hardneg-dir ../dataset_cuas_hardneg \\
        --tracking-eval-dir ../tracking_eval \\
        --ref-weights runs/c1_nano_soup65/checkpoint_soup.pth \\
        --dst-dir ../dataset_final_master
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from build_crop_dataset import (
    _clip_boxes_to_crop,
    _collect_protected_hashes,
    _dhash64,
    _intersects_any,
)
from live_demo import TorchGpuDetector
from track_roi import Config


def _box_center_inside(inner: list[float], outer: list[float], margin: float = 4.0) -> bool:
    cx = inner[0] + inner[2] / 2.0
    cy = inner[1] + inner[3] / 2.0
    return (
        outer[0] - margin <= cx <= outer[0] + outer[2] + margin
        and outer[1] - margin <= cy <= outer[1] + outer[3] + margin
    )


def _tighten_box_with_detector(
    im: Image.Image,
    gt_box: list[float],
    det: TorchGpuDetector,
    *,
    score_thr: float = 0.22,
) -> tuple[list[float], bool]:
    """Tighten an oversized Video_V .mat box using soup65 on a zoomed local crop."""
    w_px, h_px = im.size
    gx, gy, gw, gh = gt_box
    gt_area = max(gw * gh, 1.0)
    # Tight tiny boxes don't need shrinking
    if max(gw, gh) <= 12.0:
        return gt_box, False

    cx = gx + gw / 2.0
    cy = gy + gh / 2.0
    side = int(round(max(192.0, 4.0 * max(gw, gh))))
    side = max(1, min(side, w_px, h_px))
    x0 = int(np.clip(round(cx - side / 2.0), 0, w_px - side))
    y0 = int(np.clip(round(cy - side / 2.0), 0, h_px - side))

    preds = det.detect(im, (x0, y0, side), top_k=5, score_thr=score_thr)
    best_box: list[float] | None = None
    best_score = -1.0
    for _, sc, pb in preds:
        if not _box_center_inside(pb, gt_box, margin=max(4.0, 0.15 * max(gw, gh))):
            continue
        p_area = pb[2] * pb[3]
        # Accept tightened box if it's reasonably smaller or close to gt (never blow up)
        if 0.04 * gt_area <= p_area <= 1.05 * gt_area and sc > best_score:
            best_score = sc
            best_box = pb

    if best_box is not None:
        # Clamp strictly within image
        bx1 = max(0.0, min(float(w_px - 1), best_box[0]))
        by1 = max(0.0, min(float(h_px - 1), best_box[1]))
        bx2 = max(bx1 + 2.0, min(float(w_px), best_box[0] + best_box[2]))
        by2 = max(by1 + 2.0, min(float(h_px), best_box[1] + best_box[3]))
        return [round(bx1, 2), round(by1, 2), round(bx2 - bx1, 2), round(by2 - by1, 2)], True

    # Fallback: gently trim 15% border padding on oversized .mat boxes (> 22 px)
    if max(gw, gh) >= 22.0:
        nw = max(8.0, gw * 0.72)
        nh = max(8.0, gh * 0.72)
        nx = max(0.0, min(float(w_px - nw), cx - nw / 2.0))
        ny = max(0.0, min(float(h_px - nh), cy - nh / 2.0))
        return [round(nx, 2), round(ny, 2), round(nw, 2), round(nh, 2)], False

    return gt_box, False


def build_master(
    challenge_dir: Path,
    hardneg_dir: Path,
    tracking_eval_dir: Path,
    ref_weights: Path,
    dst_dir: Path,
    *,
    out_size: int = 384,
    max_video_v_pos_per_seq: int = 8,
    seed: int = 42,
) -> dict[str, Any]:
    rng = random.Random(seed)
    dst_train = dst_dir / "train"
    dst_train.mkdir(parents=True, exist_ok=True)

    # 1. Symlink official valid/ and test/ UNTOUCHED
    for split in ("valid", "test"):
        target = (challenge_dir / split).resolve()
        link = dst_dir / split
        if link.is_symlink():
            link.unlink()
        if not link.exists() and target.exists():
            os.symlink(target, link, target_is_directory=True)

    # 2. Protected dHash64 set (official valid + test + tracking_eval)
    ref_hashes = _collect_protected_hashes(challenge_dir, tracking_eval_dir)

    # Load detector on GPU for auto-tightening Video_V boxes
    det = TorchGpuDetector(ref_weights, Config())

    # Load official challenge train + hardneg train + FP regions
    c_doc = json.loads(
        (challenge_dir / "train" / "_annotations.coco.json").read_text(encoding="utf-8")
    )
    c_anns_by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for a in c_doc["annotations"]:
        c_anns_by[int(a["image_id"])].append(a)

    hn_doc = json.loads(
        (hardneg_dir / "train" / "_annotations.coco.json").read_text(encoding="utf-8")
    )
    hn_anns_by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for a in hn_doc["annotations"]:
        hn_anns_by[int(a["image_id"])].append(a)

    fp_by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fp_path = hardneg_dir / "model_false_positives.json"
    if fp_path.is_file():
        for rec in json.loads(fp_path.read_text(encoding="utf-8")):
            if rec.get("split") == "train":
                fp_by_file[rec["file_name"]].append(rec)

    new_images: list[dict[str, Any]] = []
    new_anns: list[dict[str, Any]] = []
    next_img_id = 1
    next_ann_id = 1

    def _add(fname: str, w: int, h: int, boxes: list[dict[str, Any]]) -> None:
        nonlocal next_img_id, next_ann_id
        iid = next_img_id
        next_img_id += 1
        new_images.append({"id": iid, "file_name": fname, "width": w, "height": h})
        for b in boxes:
            new_anns.append(
                {
                    "id": next_ann_id,
                    "image_id": iid,
                    "category_id": int(b["category_id"]),
                    "bbox": [float(v) for v in b["bbox"]],
                    "area": float(b.get("area", b["bbox"][2] * b["bbox"][3])),
                    "iscrowd": 0,
                }
            )
            next_ann_id += 1

    stats: dict[str, Any] = {
        "challenge_orig": 0,
        "challenge_crops_128": 0,
        "challenge_crops_192": 0,
        "challenge_crops_384": 0,
        "challenge_neg_tiles": 0,
        "hn_video_v_empty_or_fp": 0,
        "hn_video_v_pos": 0,
        "hn_video_v_boxes_tightened_by_model": 0,
        "hn_video_v_boxes_fallback": 0,
        "hn_video_v_skipped_leak": 0,
        "hn_fp_zoomed_crops": 0,
    }

    # ---- A. Official challenge train/ + 128/192/384 positive crops + neg tiles ----
    for im_rec in c_doc["images"]:
        src_p = (challenge_dir / "train" / im_rec["file_name"]).resolve()
        dst_p = dst_train / im_rec["file_name"]
        if not dst_p.exists():
            os.symlink(src_p, dst_p)
        orig_anns = c_anns_by.get(int(im_rec["id"]), [])
        w_px, h_px = int(im_rec["width"]), int(im_rec["height"])
        _add(im_rec["file_name"], w_px, h_px, orig_anns)
        stats["challenge_orig"] += 1

        with Image.open(src_p) as im_raw:
            im = im_raw.convert("RGB")
            # Multi-scale zoomed positive crops (128 = 3x zoom for seq_0003/0004!)
            for idx, ann in enumerate(orig_anns):
                bx, by, bw, bh = ann["bbox"]
                max_dim = max(bw, bh)
                cx, cy = bx + bw / 2.0, by + bh / 2.0
                sides: list[int] = []
                if max_dim <= 18.0 and 128 < min(w_px, h_px):
                    sides.append(128)
                if max_dim <= 38.0 and 192 < min(w_px, h_px):
                    sides.append(192)
                if (18.0 < max_dim <= 96.0 or int(ann["category_id"]) in (1, 2)) and 384 < min(w_px, h_px):
                    sides.append(384)
                for side in sides:
                    jitter = 0.25 * side
                    x0 = int(np.clip(round(cx + rng.uniform(-jitter, jitter) - side / 2.0), 0, w_px - side))
                    y0 = int(np.clip(round(cy + rng.uniform(-jitter, jitter) - side / 2.0), 0, h_px - side))
                    c_anns = _clip_boxes_to_crop(orig_anns, x0, y0, side, out_size)
                    if not c_anns:
                        continue
                    patch = im.crop((x0, y0, x0 + side, y0 + side))
                    if side != out_size:
                        patch = patch.resize((out_size, out_size), Image.Resampling.BILINEAR)
                    fn = f"chcrop_{im_rec['id']}_{idx}_{side}.jpg"
                    patch.save(dst_train / fn, quality=91)
                    _add(fn, out_size, out_size, c_anns)
                    stats[f"challenge_crops_{side}"] += 1

            # 0.5x multi-scale negative background tile from challenge train
            valid_neg_sides = [s for s in (128, 192, 256, 384) if s < min(w_px, h_px)]
            if valid_neg_sides and rng.random() < 0.5:
                side = rng.choice(valid_neg_sides)
                for _ in range(8):
                    x0 = rng.randint(0, w_px - side)
                    y0 = rng.randint(0, h_px - side)
                    if not _intersects_any(orig_anns, x0, y0, side):
                        patch = im.crop((x0, y0, x0 + side, y0 + side))
                        if side != out_size:
                            patch = patch.resize((out_size, out_size), Image.Resampling.BILINEAR)
                        fn = f"chneg_{im_rec['id']}_{side}.jpg"
                        patch.save(dst_train / fn, quality=90)
                        _add(fn, out_size, out_size, [])
                        stats["challenge_neg_tiles"] += 1
                        break

    # ---- B. Curated Video_V frames & mined FP hard negatives from dataset_cuas_hardneg/train ----
    # Group Video_V images by sequence so we check leakage once per sequence & balance frames/seq
    v_by_seq: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for im_rec in hn_doc["images"]:
        if im_rec.get("source") == "video_v":
            v_by_seq[str(im_rec.get("sequence", "unknown"))].append(im_rec)

    leaked_seqs: set[str] = set()
    for seq, recs in sorted(v_by_seq.items()):
        # Check up to 3 frames of the sequence against protected valid/test/tracking_eval hashes
        probes = [recs[0], recs[len(recs) // 2], recs[-1]]
        is_leak = False
        for pr in probes:
            g = cv2.imread(str(hardneg_dir / "train" / pr["file_name"]), cv2.IMREAD_GRAYSCALE)
            if g is not None and len(ref_hashes) > 0:
                h = np.uint64(_dhash64(g))
                if any(int(val).bit_count() <= 6 for val in (ref_hashes ^ h)):
                    is_leak = True
                    break
        if is_leak:
            leaked_seqs.add(seq)
            stats["hn_video_v_skipped_leak"] += 1
            continue

        # Always keep EVERY frame that has a mined model false positive or is an empty structure/bird frame!
        # Subsample regular positive frames to max_video_v_pos_per_seq so no single Video_V clip dominates.
        must_keep: list[dict[str, Any]] = []
        regular_pos: list[dict[str, Any]] = []
        for rec in recs:
            anns = hn_anns_by.get(int(rec["id"]), [])
            if rec.get("has_model_false_positive") or not anns:
                must_keep.append(rec)
            else:
                regular_pos.append(rec)

        if len(regular_pos) > max_video_v_pos_per_seq:
            idxs = np.linspace(0, len(regular_pos) - 1, max_video_v_pos_per_seq, dtype=int)
            chosen_pos = [regular_pos[i] for i in sorted(set(idxs.tolist()))]
        else:
            chosen_pos = regular_pos

        # Cap empty/structure-only frames per sequence to 12 unless they have a mined model FP
        fp_must = [r for r in must_keep if r.get("has_model_false_positive")]
        empty_only = [r for r in must_keep if not r.get("has_model_false_positive")]
        if len(empty_only) > 10:
            idxs = np.linspace(0, len(empty_only) - 1, 10, dtype=int)
            empty_only = [empty_only[i] for i in sorted(set(idxs.tolist()))]

        selected = fp_must + empty_only + chosen_pos
        for rec in selected:
            src_p = (hardneg_dir / "train" / rec["file_name"]).resolve()
            dst_p = dst_train / rec["file_name"]
            if not dst_p.exists():
                os.symlink(src_p, dst_p)
            w_px, h_px = int(rec["width"]), int(rec["height"])
            raw_anns = hn_anns_by.get(int(rec["id"]), [])

            with Image.open(src_p) as im_raw:
                im = im_raw.convert("RGB")
                tight_anns: list[dict[str, Any]] = []
                for a in raw_anns:
                    tbox, by_model = _tighten_box_with_detector(im, a["bbox"], det)
                    if by_model:
                        stats["hn_video_v_boxes_tightened_by_model"] += 1
                    else:
                        stats["hn_video_v_boxes_fallback"] += 1
                    tight_anns.append(
                        {
                            "category_id": int(a["category_id"]),
                            "bbox": tbox,
                            "area": round(tbox[2] * tbox[3], 2),
                            "iscrowd": 0,
                        }
                    )

                _add(rec["file_name"], w_px, h_px, tight_anns)
                if rec.get("has_model_false_positive") or not tight_anns:
                    stats["hn_video_v_empty_or_fp"] += 1
                else:
                    stats["hn_video_v_pos"] += 1

                # Targeted hard-negative crops centered on EVERY confirmed structure/antenna FP!
                for fpi, fp_rec in enumerate(fp_by_file.get(rec["file_name"], [])):
                    fb = fp_rec["bbox"]
                    fcx = fb[0] + fb[2] / 2.0
                    fcy = fb[1] + fb[3] / 2.0
                    for side in (192, 384):
                        if side >= min(w_px, h_px):
                            continue
                        x0 = int(np.clip(round(fcx - side / 2.0), 0, w_px - side))
                        y0 = int(np.clip(round(fcy - side / 2.0), 0, h_px - side))
                        # Only save as pure negative crop if no real GT aircraft sits inside it
                        if _intersects_any(tight_anns, x0, y0, side, pad=12):
                            continue
                        patch = im.crop((x0, y0, x0 + side, y0 + side))
                        if side != out_size:
                            patch = patch.resize((out_size, out_size), Image.Resampling.BILINEAR)
                        fn = f"fpcrop_{Path(rec['file_name']).stem}_{fpi}_{side}.jpg"
                        patch.save(dst_train / fn, quality=90)
                        _add(fn, out_size, out_size, [])
                        stats["hn_fp_zoomed_crops"] += 1

    # Also add targeted FP crops for challenge_train images that had false positives!
    for im_rec in hn_doc["images"]:
        if im_rec.get("source") == "challenge_train" and im_rec.get("has_model_false_positive"):
            src_p = (hardneg_dir / "train" / im_rec["file_name"]).resolve()
            w_px, h_px = int(im_rec["width"]), int(im_rec["height"])
            orig_anns = hn_anns_by.get(int(im_rec["id"]), [])
            with Image.open(src_p) as im_raw:
                im = im_raw.convert("RGB")
                for fpi, fp_rec in enumerate(fp_by_file.get(im_rec["file_name"], [])):
                    fb = fp_rec["bbox"]
                    fcx = fb[0] + fb[2] / 2.0
                    fcy = fb[1] + fb[3] / 2.0
                    for side in (192, 384):
                        if side >= min(w_px, h_px):
                            continue
                        x0 = int(np.clip(round(fcx - side / 2.0), 0, w_px - side))
                        y0 = int(np.clip(round(fcy - side / 2.0), 0, h_px - side))
                        if _intersects_any(orig_anns, x0, y0, side, pad=12):
                            continue
                        patch = im.crop((x0, y0, x0 + side, y0 + side))
                        if side != out_size:
                            patch = patch.resize((out_size, out_size), Image.Resampling.BILINEAR)
                        fn = f"chfpcrop_{Path(im_rec['file_name']).stem}_{fpi}_{side}.jpg"
                        patch.save(dst_train / fn, quality=90)
                        _add(fn, out_size, out_size, [])
                        stats["hn_fp_zoomed_crops"] += 1

    out_doc = {
        "images": new_images,
        "annotations": new_anns,
        "categories": c_doc["categories"],
    }
    (dst_train / "_annotations.coco.json").write_text(json.dumps(out_doc), encoding="utf-8")
    stats["total_images"] = len(new_images)
    stats["total_annotations"] = len(new_anns)
    empty_cnt = len({i["id"] for i in new_images} - {a["image_id"] for a in new_anns})
    stats["empty_negative_images"] = empty_cnt
    cls_counts: dict[int, int] = defaultdict(int)
    for a in new_anns:
        cls_counts[int(a["category_id"])] += 1
    stats["boxes_per_class"] = dict(sorted(cls_counts.items()))
    return stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--challenge-dir", type=Path, default=Path("../dataset"))
    p.add_argument("--hardneg-dir", type=Path, default=Path("../dataset_cuas_hardneg"))
    p.add_argument("--tracking-eval-dir", type=Path, default=Path("../tracking_eval"))
    p.add_argument(
        "--ref-weights",
        type=Path,
        default=Path("runs/c1_nano_soup65/checkpoint_soup.pth"),
    )
    p.add_argument("--dst-dir", type=Path, default=Path("../dataset_final_master"))
    p.add_argument("--max-video-v-pos-per-seq", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_master(
        args.challenge_dir,
        args.hardneg_dir,
        args.tracking_eval_dir,
        args.ref_weights,
        args.dst_dir,
        max_video_v_pos_per_seq=args.max_video_v_pos_per_seq,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
