"""Build a leak-free crop/tile-augmented COCO training set from ``dataset/train/``.

Why:
  In Challenge 1, ~70% of validation targets are < 32x32 px (and drones in 1080p
  MOT videos shrink to 2-4 px when resized directly to 384x384). At inference time,
  ROI crops and tiles feed native-resolution or upscaled 192x192/384x384 patches
  to RF-DETR. Training only on full resized frames creates a domain mismatch and
  causes high false positives on background tiles.

What this script generates (strictly from ``train/``; ``valid/`` and ``test/``
are symlinked untouched):
  1. All original full-frame ``train/`` images and annotations;
  2. Positive jittered crops around small targets (384x384 native + 192x192
     upscaled to 384x384) to match ROI tracking crops and SAHI tiles;
  3. Hard-negative background tiles (no targets) sampled from ``train/`` frames
     so the detector suppresses sky/tree/horizon clutter when tiling;
  4. Optional copy-paste of small targets onto negative frames to boost rare
     classes (airplane, helicopter) and tiny drone recall.

Usage (from starter-kit/):
    python build_crop_dataset.py --src-dir ../dataset --dst-dir ../dataset_aug
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _clip_boxes_to_crop(
    anns: list[dict[str, Any]],
    x0: int,
    y0: int,
    side: int,
    out_size: int,
    *,
    min_vis_ratio: float = 0.4,
    min_box_px: float = 2.0,
) -> list[dict[str, Any]]:
    scale = out_size / float(side)
    kept: list[dict[str, Any]] = []
    for ann in anns:
        bx, by, bw, bh = ann["bbox"]
        orig_area = max(bw * bh, 1e-6)
        ix1 = max(bx, float(x0))
        iy1 = max(by, float(y0))
        ix2 = min(bx + bw, float(x0 + side))
        iy2 = min(by + bh, float(y0 + side))
        iw = ix2 - ix1
        ih = iy2 - iy1
        if iw <= 0 or ih <= 0:
            continue
        if (iw * ih) / orig_area < min_vis_ratio:
            continue
        nx = (ix1 - x0) * scale
        ny = (iy1 - y0) * scale
        nw = iw * scale
        nh = ih * scale
        if nw < min_box_px or nh < min_box_px:
            continue
        kept.append(
            {
                "category_id": int(ann["category_id"]),
                "bbox": [round(nx, 2), round(ny, 2), round(nw, 2), round(nh, 2)],
                "area": round(nw * nh, 2),
                "iscrowd": 0,
            }
        )
    return kept


def _intersects_any(
    anns: list[dict[str, Any]], x0: int, y0: int, side: int, pad: int = 8
) -> bool:
    for ann in anns:
        bx, by, bw, bh = ann["bbox"]
        if (
            bx + bw + pad > x0
            and bx - pad < x0 + side
            and by + bh + pad > y0
            and by - pad < y0 + side
        ):
            return True
    return False


def build_dataset(
    src_dir: Path,
    dst_dir: Path,
    *,
    out_size: int = 384,
    crop_sides: tuple[int, ...] = (384, 192),
    small_thresh_px: float = 48.0,
    neg_tiles_per_img: float = 0.45,
    copy_paste_count: int = 120,
    seed: int = 42,
) -> dict[str, Any]:
    rng = random.Random(seed)
    src_train = src_dir / "train"
    dst_train = dst_dir / "train"
    dst_train.mkdir(parents=True, exist_ok=True)

    # Symlink valid/ and test/ untouched so there is zero leakage or duplication.
    for split in ("valid", "test"):
        target = (src_dir / split).resolve()
        link = dst_dir / split
        if link.is_symlink() or link.exists():
            if link.is_symlink():
                link.unlink()
        if not link.exists() and target.exists():
            os.symlink(target, link, target_is_directory=True)

    doc = json.loads((src_train / "_annotations.coco.json").read_text(encoding="utf-8"))
    anns_by_img: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in doc["annotations"]:
        anns_by_img[int(ann["image_id"])].append(ann)

    new_images: list[dict[str, Any]] = []
    new_anns: list[dict[str, Any]] = []
    next_img_id = 1
    next_ann_id = 1

    def _add_record(
        file_name: str, w: int, h: int, boxes: list[dict[str, Any]]
    ) -> None:
        nonlocal next_img_id, next_ann_id
        img_id = next_img_id
        next_img_id += 1
        new_images.append({"id": img_id, "file_name": file_name, "width": w, "height": h})
        for b in boxes:
            new_anns.append(
                {
                    "id": next_ann_id,
                    "image_id": img_id,
                    "category_id": int(b["category_id"]),
                    "bbox": [float(v) for v in b["bbox"]],
                    "area": float(b.get("area", b["bbox"][2] * b["bbox"][3])),
                    "iscrowd": int(b.get("iscrowd", 0)),
                }
            )
            next_ann_id += 1

    # 1. Symlink all original train images + copy annotations
    stats = {"orig": 0, "pos_crops": 0, "neg_tiles": 0, "copy_paste": 0}
    small_patch_bank: list[tuple[int, Image.Image]] = []
    negative_imgs: list[Path] = []

    for im_rec in doc["images"]:
        src_img_path = (src_train / im_rec["file_name"]).resolve()
        dst_img_path = dst_train / im_rec["file_name"]
        if not dst_img_path.exists():
            os.symlink(src_img_path, dst_img_path)
        orig_anns = anns_by_img.get(int(im_rec["id"]), [])
        _add_record(
            im_rec["file_name"],
            int(im_rec["width"]),
            int(im_rec["height"]),
            orig_anns,
        )
        stats["orig"] += 1
        if not orig_anns:
            negative_imgs.append(src_img_path)

        w_px, h_px = int(im_rec["width"]), int(im_rec["height"])
        with Image.open(src_img_path) as im_raw:
            im = im_raw.convert("RGB")

            # Collect small target patches for copy-paste (especially rare classes)
            for ann in orig_anns:
                bx, by, bw, bh = ann["bbox"]
                if 5.0 <= max(bw, bh) <= 64.0:
                    x1 = max(0, int(round(bx)))
                    y1 = max(0, int(round(by)))
                    x2 = min(w_px, int(round(bx + bw)))
                    y2 = min(h_px, int(round(by + bh)))
                    if x2 - x1 >= 4 and y2 - y1 >= 4:
                        small_patch_bank.append(
                            (int(ann["category_id"]), im.crop((x1, y1, x2, y2)).copy())
                        )

            # 2. Positive crops around small/medium targets (matches ROI & tiles)
            for idx, ann in enumerate(orig_anns):
                bx, by, bw, bh = ann["bbox"]
                max_dim = max(bw, bh)
                cx = bx + bw / 2.0
                cy = by + bh / 2.0
                # Pick the crop scale that puts this target into the detector's sweet spot
                chosen_sides: list[int] = []
                if max_dim <= 36.0 and 192 < min(w_px, h_px):
                    chosen_sides.append(192)
                elif max_dim <= small_thresh_px * 2.0 and 384 < min(w_px, h_px):
                    chosen_sides.append(384)
                # Oversample rare classes (airplane=1, helicopter=2) with a second scale
                if int(ann["category_id"]) in (1, 2) and 384 < min(w_px, h_px) and 384 not in chosen_sides:
                    chosen_sides.append(384)
                for side in chosen_sides:
                    reps = 2 if int(ann["category_id"]) in (1, 2) else 1
                    for rep in range(reps):
                        jitter = 0.25 * side
                        jx = cx + rng.uniform(-jitter, jitter)
                        jy = cy + rng.uniform(-jitter, jitter)
                        x0 = int(np.clip(round(jx - side / 2.0), 0, w_px - side))
                        y0 = int(np.clip(round(jy - side / 2.0), 0, h_px - side))
                        crop_anns = _clip_boxes_to_crop(
                            orig_anns, x0, y0, side, out_size
                        )
                        if not crop_anns:
                            continue
                        patch = im.crop((x0, y0, x0 + side, y0 + side))
                        if side != out_size:
                            patch = patch.resize(
                                (out_size, out_size), Image.Resampling.BILINEAR
                            )
                        fname = f"crop_{im_rec['id']}_{idx}_{side}_{rep}.jpg"
                        patch.save(dst_train / fname, quality=92)
                        _add_record(fname, out_size, out_size, crop_anns)
                        stats["pos_crops"] += 1

            # 3. Hard-negative background tiles (teaches clutter rejection on tiles)
            if rng.random() < neg_tiles_per_img and min(w_px, h_px) > 220:
                side = rng.choice([s for s in crop_sides if s < min(w_px, h_px)])
                for attempt in range(8):
                    x0 = rng.randint(0, w_px - side)
                    y0 = rng.randint(0, h_px - side)
                    if not _intersects_any(orig_anns, x0, y0, side):
                        patch = im.crop((x0, y0, x0 + side, y0 + side))
                        if side != out_size:
                            patch = patch.resize(
                                (out_size, out_size), Image.Resampling.BILINEAR
                            )
                        fname = f"negtile_{im_rec['id']}_{side}_{attempt}.jpg"
                        patch.save(dst_train / fname, quality=92)
                        _add_record(fname, out_size, out_size, [])
                        stats["neg_tiles"] += 1
                        break

    # 4. Copy-paste small targets onto background crops/negatives with soft border blend
    if small_patch_bank and negative_imgs and copy_paste_count > 0:
        by_cls: dict[int, list[Image.Image]] = defaultdict(list)
        for cid, p in small_patch_bank:
            by_cls[cid].append(p)
        cls_ids = sorted(by_cls)
        for k in range(copy_paste_count):
            bg_path = rng.choice(negative_imgs)
            with Image.open(bg_path) as bg_raw:
                bg = bg_raw.convert("RGB")
                bw_px, bh_px = bg.size
                # Take a 384x384 crop if background is larger
                if min(bw_px, bh_px) >= out_size:
                    x0 = rng.randint(0, bw_px - out_size)
                    y0 = rng.randint(0, bh_px - out_size)
                    canvas = bg.crop((x0, y0, x0 + out_size, y0 + out_size))
                else:
                    canvas = bg.resize((out_size, out_size), Image.Resampling.BILINEAR)
            canvas_arr = np.asarray(canvas, dtype=np.float32).copy()
            cp_anns: list[dict[str, Any]] = []
            n_objs = rng.choice([1, 1, 2])
            for _ in range(n_objs):
                # Balance class sampling across drone/airplane/helicopter
                cid = rng.choice(cls_ids)
                patch = rng.choice(by_cls[cid])
                scale = rng.uniform(0.75, 1.5)
                pw = max(6, min(96, int(round(patch.width * scale))))
                ph = max(6, min(96, int(round(patch.height * scale))))
                p_arr = np.asarray(
                    patch.resize((pw, ph), Image.Resampling.BILINEAR), dtype=np.float32
                )
                px = rng.randint(4, out_size - pw - 4)
                py = rng.randint(4, out_size - ph - 4)
                # Soft cosine feathering on 1-2 px border to avoid hard rectangular seams
                yy = np.linspace(-1.0, 1.0, ph)[:, None]
                xx = np.linspace(-1.0, 1.0, pw)[None, :]
                alpha = np.clip(1.0 - 0.35 * np.maximum(np.abs(xx), np.abs(yy)) ** 4, 0.3, 1.0)
                alpha = alpha[:, :, None]
                roi = canvas_arr[py : py + ph, px : px + pw]
                canvas_arr[py : py + ph, px : px + pw] = alpha * p_arr + (1.0 - alpha) * roi
                cp_anns.append(
                    {
                        "category_id": cid,
                        "bbox": [float(px), float(py), float(pw), float(ph)],
                        "area": float(pw * ph),
                        "iscrowd": 0,
                    }
                )
            fname = f"copypaste_{k:04d}.jpg"
            Image.fromarray(np.clip(canvas_arr, 0, 255).astype(np.uint8)).save(
                dst_train / fname, quality=92
            )
            _add_record(fname, out_size, out_size, cp_anns)
            stats["copy_paste"] += 1

    out_doc = {
        "images": new_images,
        "annotations": new_anns,
        "categories": doc["categories"],
    }
    (dst_train / "_annotations.coco.json").write_text(
        json.dumps(out_doc), encoding="utf-8"
    )
    stats["total_images"] = len(new_images)
    stats["total_annotations"] = len(new_anns)
    cls_counts = defaultdict(int)
    for a in new_anns:
        cls_counts[int(a["category_id"])] += 1
    stats["boxes_per_class"] = dict(sorted(cls_counts.items()))
    return stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src-dir", type=Path, default=Path("../dataset"))
    p.add_argument("--dst-dir", type=Path, default=Path("../dataset_aug"))
    p.add_argument("--out-size", type=int, default=384)
    p.add_argument("--neg-tiles-per-img", type=float, default=0.45)
    p.add_argument("--copy-paste", type=int, default=120)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_dataset(
        args.src_dir,
        args.dst_dir,
        out_size=args.out_size,
        neg_tiles_per_img=args.neg_tiles_per_img,
        copy_paste_count=args.copy_paste,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
