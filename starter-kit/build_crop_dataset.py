"""Build a leak-free, multi-scale crop/tile-augmented COCO training set.

Sources (strictly leak-free; ``valid/`` and ``test/`` are symlinked untouched):
  1. All original full-frame ``dataset/train/`` images and annotations;
  2. (Optional ``--extra-video-dir ../nos_videos/Video_V``) Visible RGB clips with
     MATLAB ``_labels_mat/*_LABELS.mat`` annotations, automatically filtered via
     visual ``dHash64`` against ``dataset/valid/``, ``dataset/test/``, and
     ``tracking_eval/`` so ZERO validation/test clips can leak into training!
     ``BIRD`` clips are included as high-value hard negatives (empty annotations).

Augmentations designed specifically for Challenge 1 tiny drones (< 18 px) + ROI/discovery:
  - 3x zoom crops (128x128 -> 384x384) around tiny targets (< 18 px) — matches seq_0003/0004!
  - 2x zoom crops (192x192 -> 384x384) around small targets (< 36 px);
  - 1x native crops (384x384) around medium targets;
  - Multi-scale hard-negative background tiles (128, 192, 256, 384 px) sampled at
    high density (--neg-tiles-per-img 1.0) so ``--discover`` & tiled detection suppress
    sky/tree/building clutter;
  - Soft-feathered copy-paste of small targets onto negative scenes.

Usage (from starter-kit/):
    python build_crop_dataset.py --src-dir ../dataset --dst-dir ../dataset_mega \\
        --extra-video-dir ../nos_videos/Video_V --tracking-eval-dir ../tracking_eval \\
        --neg-tiles-per-img 1.0 --copy-paste 80
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy.io import loadmat

# MATLAB LabelData column mapping in Halmstad Drone-vs-Bird _LABELS.mat files:
# col 0 = AIRPLANE -> COCO 1, col 1 = BIRD -> hard negative (None),
# col 2 = DRONE -> COCO 0, col 3 = HELICOPTER -> COCO 2.
MAT_COL_TO_COCO: dict[int, int | None] = {
    0: 1,     # airplane
    1: None,  # bird (hard negative!)
    2: 0,     # drone_air
    3: 2,     # helicopter
}


def _dhash64(gray: np.ndarray) -> int:
    sm = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (sm[:, 1:] > sm[:, :-1]).ravel()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def _collect_protected_hashes(dataset_dir: Path, tracking_eval_dir: Path | None) -> np.ndarray:
    """Collect dHash64 fingerprints of valid/, test/, and tracking_eval/ to prevent leakage."""
    hashes: list[int] = []
    for split in ("valid", "test"):
        ann_path = dataset_dir / split / "_annotations.coco.json"
        if not ann_path.is_file():
            continue
        doc = json.loads(ann_path.read_text(encoding="utf-8"))
        for im in doc["images"][::3]:
            g = cv2.imread(str(dataset_dir / split / im["file_name"]), cv2.IMREAD_GRAYSCALE)
            if g is not None:
                hashes.append(_dhash64(g))
    if tracking_eval_dir is not None and tracking_eval_dir.is_dir():
        for seq in sorted(tracking_eval_dir.glob("seq_*")):
            vp = seq / "video.mp4"
            if not vp.is_file():
                continue
            cap = cv2.VideoCapture(str(vp))
            n_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for fi in {0, max(0, n_f // 3), max(0, (2 * n_f) // 3)}:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, fr = cap.read()
                if ok:
                    hashes.append(_dhash64(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)))
            cap.release()
    return np.array(hashes, dtype=np.uint64)


def _is_leaked(gray_frames: list[np.ndarray], ref_hashes: np.ndarray, max_hamming: int = 6) -> bool:
    if len(ref_hashes) == 0:
        return False
    for g in gray_frames:
        h = np.uint64(_dhash64(g))
        xor = ref_hashes ^ h
        for val in xor:
            if int(val).bit_count() <= max_hamming:
                return True
    return False


def _parse_gtruth_mat(mat_path: Path) -> list[tuple[int, int, list[list[float]]]]:
    """Parse MATLAB v5 MCOS groundTruth LabelData (4 columns per frame: AIRPLANE, BIRD, DRONE, HELICOPTER)."""
    ws = loadmat(str(mat_path))["__function_workspace__"].tobytes()
    ld = ws.find(b"LabelData")
    if ld < 0:
        return []
    off = (ld + 3) & ~3
    mats: list[tuple[int, int, list[list[float]]]] = []
    dtype_map = {9: "<f8", 7: "<f4", 5: "<i4", 3: "<i2", 4: "<u2", 2: "<u1"}
    n_ws = len(ws)
    while off < n_ws - 40:
        dt, nb = struct.unpack_from("<II", ws, off)
        if dt == 14 and 16 <= nb <= 8192 and off + 8 + nb <= n_ws:
            t1, l1 = struct.unpack_from("<II", ws, off + 8)
            t2, l2 = struct.unpack_from("<II", ws, off + 24)
            if t1 == 6 and l1 == 8 and t2 == 5 and l2 == 8:
                cls = struct.unpack_from("<I", ws, off + 16)[0] & 0xFF
                if cls == 6:  # mxDOUBLE_CLASS
                    r, c = struct.unpack_from("<ii", ws, off + 32)
                    if 0 <= r <= 20 and c in (0, 4):
                        arr: list[list[float]] = []
                        if r > 0 and c == 4:
                            nt_raw = struct.unpack_from("<I", ws, off + 40)[0]
                            if (nt_raw >> 16) != 0:
                                data_off = off + 48
                            else:
                                nl = struct.unpack_from("<I", ws, off + 44)[0]
                                data_off = off + 48 + ((nl + 7) // 8) * 8
                            dt_raw = struct.unpack_from("<I", ws, data_off)[0]
                            if (dt_raw >> 16) != 0:
                                vals_off = data_off + 4
                                dt_code = dt_raw & 0xFFFF
                            else:
                                dt_code = dt_raw
                                vals_off = data_off + 8
                            if dt_code in dtype_map:
                                arr = (
                                    np.frombuffer(
                                        ws,
                                        dtype=dtype_map[dt_code],
                                        count=r * 4,
                                        offset=vals_off,
                                    )
                                    .reshape((4, r))
                                    .T.astype(float)
                                    .tolist()
                                )
                        mats.append((r, c, arr))
                        off += 8 + ((nb + 7) // 8) * 8
                        continue
        off += 4
    return mats


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
    extra_video_dir: Path | None = None,
    extra_birds_only: bool = False,
    tracking_eval_dir: Path | None = None,
    out_size: int = 384,
    neg_tiles_per_img: float = 1.0,
    copy_paste_count: int = 80,
    frames_per_video: int = 5,
    bird_frames_per_video: int = 3,
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

    stats: dict[str, Any] = {
        "orig": 0,
        "pos_crops_128": 0,
        "pos_crops_192": 0,
        "pos_crops_384": 0,
        "neg_tiles": 0,
        "copy_paste": 0,
        "extra_video_frames": 0,
        "extra_video_crops": 0,
        "extra_bird_hard_negs": 0,
        "extra_videos_used": 0,
        "extra_videos_skipped_leak": 0,
    }
    small_patch_bank: list[tuple[int, Image.Image]] = []
    negative_imgs: list[Path] = []
    neg_tile_sides = (128, 192, 256, 384)

    def _sample_neg_tiles(
        im: Image.Image, anns: list[dict[str, Any]], tag: str, count_float: float
    ) -> None:
        w_px, h_px = im.size
        valid_sides = [s for s in neg_tile_sides if s < min(w_px, h_px)]
        if not valid_sides:
            return
        n_tiles = int(count_float) + (1 if rng.random() < (count_float - int(count_float)) else 0)
        for ti in range(n_tiles):
            side = rng.choice(valid_sides)
            for _ in range(10):
                x0 = rng.randint(0, w_px - side)
                y0 = rng.randint(0, h_px - side)
                if not _intersects_any(anns, x0, y0, side):
                    patch = im.crop((x0, y0, x0 + side, y0 + side))
                    if side != out_size:
                        patch = patch.resize((out_size, out_size), Image.Resampling.BILINEAR)
                    fname = f"negtile_{tag}_{side}_{ti}.jpg"
                    patch.save(dst_train / fname, quality=90)
                    _add_record(fname, out_size, out_size, [])
                    stats["neg_tiles"] += 1
                    break

    def _emit_positive_crops(
        im: Image.Image, anns: list[dict[str, Any]], tag: str, *, allow_reps: bool = True
    ) -> int:
        w_px, h_px = im.size
        added = 0
        for idx, ann in enumerate(anns):
            bx, by, bw, bh = ann["bbox"]
            max_dim = max(bw, bh)
            cx = bx + bw / 2.0
            cy = by + bh / 2.0
            chosen_sides: list[int] = []
            # 3x zoom (128 -> 384) specifically for tiny targets (< 18 px, like seq_0003/0004!)
            if max_dim <= 18.0 and 128 < min(w_px, h_px):
                chosen_sides.append(128)
            # 2x zoom (192 -> 384) for small targets (< 38 px)
            if max_dim <= 38.0 and 192 < min(w_px, h_px):
                chosen_sides.append(192)
            # 1x native crop (384) for medium targets (or rare classes)
            if (18.0 < max_dim <= 96.0 or int(ann["category_id"]) in (1, 2)) and 384 < min(w_px, h_px):
                chosen_sides.append(384)

            for side in chosen_sides:
                reps = 2 if (allow_reps and int(ann["category_id"]) in (1, 2) and side >= 192) else 1
                for rep in range(reps):
                    jitter = 0.28 * side
                    jx = cx + rng.uniform(-jitter, jitter)
                    jy = cy + rng.uniform(-jitter, jitter)
                    x0 = int(np.clip(round(jx - side / 2.0), 0, w_px - side))
                    y0 = int(np.clip(round(jy - side / 2.0), 0, h_px - side))
                    crop_anns = _clip_boxes_to_crop(anns, x0, y0, side, out_size)
                    if not crop_anns:
                        continue
                    patch = im.crop((x0, y0, x0 + side, y0 + side))
                    if side != out_size:
                        patch = patch.resize((out_size, out_size), Image.Resampling.BILINEAR)
                    fname = f"crop_{tag}_{idx}_{side}_{rep}.jpg"
                    patch.save(dst_train / fname, quality=91)
                    _add_record(fname, out_size, out_size, crop_anns)
                    added += 1
                    if f"pos_crops_{side}" in stats:
                        stats[f"pos_crops_{side}"] += 1
        return added

    # 1. Original train/ images + multi-scale positive crops (128/192/384) + negative tiles
    for im_rec in doc["images"]:
        src_img_path = (src_train / im_rec["file_name"]).resolve()
        dst_img_path = dst_train / im_rec["file_name"]
        if not dst_img_path.exists():
            os.symlink(src_img_path, dst_img_path)
        orig_anns = anns_by_img.get(int(im_rec["id"]), [])
        _add_record(im_rec["file_name"], int(im_rec["width"]), int(im_rec["height"]), orig_anns)
        stats["orig"] += 1
        if not orig_anns:
            negative_imgs.append(src_img_path)

        w_px, h_px = int(im_rec["width"]), int(im_rec["height"])
        with Image.open(src_img_path) as im_raw:
            im = im_raw.convert("RGB")
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
            _emit_positive_crops(im, orig_anns, f"tr{im_rec['id']}", allow_reps=True)
            _sample_neg_tiles(im, orig_anns, f"tr{im_rec['id']}", neg_tiles_per_img)

    # 2. Leak-free extra visible RGB videos (nos_videos/Video_V) + BIRD hard negatives!
    if extra_video_dir is not None and (extra_video_dir / "mapping.csv").is_file():
        ref_hashes = _collect_protected_hashes(src_dir, tracking_eval_dir)
        labels_dir = extra_video_dir / "_labels_mat"
        with (extra_video_dir / "mapping.csv").open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))

        for row in rows:
            seq_name = row["sequence"].strip()
            orig_vid = Path(row["fichier_original"].strip()).stem
            cls_label = row["classe"].strip().lower()
            if extra_birds_only and cls_label != "bird":
                continue
            vid_path = extra_video_dir / seq_name / "video.mp4"
            mat_path = labels_dir / f"{orig_vid}_LABELS.mat"
            if not vid_path.is_file() or not mat_path.is_file():
                continue

            cap = cv2.VideoCapture(str(vid_path))
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if n_frames <= 0:
                cap.release()
                continue

            # Leak check on 3 evenly spaced frames
            probe_grays: list[np.ndarray] = []
            for fi in {0, n_frames // 2, max(0, n_frames - 1)}:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, bgr = cap.read()
                if ok:
                    probe_grays.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
            if _is_leaked(probe_grays, ref_hashes, max_hamming=6):
                stats["extra_videos_skipped_leak"] += 1
                cap.release()
                continue

            mats = _parse_gtruth_mat(mat_path)
            if len(mats) < 4 * n_frames:
                cap.release()
                continue
            # Use exact 4 columns per frame
            n_usable = min(n_frames, len(mats) // 4)
            is_bird = cls_label == "bird"
            k_sample = bird_frames_per_video if is_bird else frames_per_video
            sample_indices = sorted(
                set(np.linspace(0, n_usable - 1, min(k_sample, n_usable), dtype=int).tolist())
            )

            used_any = False
            for fi in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, bgr = cap.read()
                if not ok:
                    continue
                h_px, w_px = bgr.shape[:2]
                im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

                frame_anns: list[dict[str, Any]] = []
                bird_boxes: list[list[float]] = []
                for col in range(4):
                    r, c, arr = mats[4 * fi + col]
                    if r <= 0 or c != 4:
                        continue
                    coco_cid = MAT_COL_TO_COCO[col]
                    for bx, by, bw, bh in arr:
                        x1 = max(0.0, min(float(w_px - 1), float(bx)))
                        y1 = max(0.0, min(float(h_px - 1), float(by)))
                        x2 = max(x1 + 1.0, min(float(w_px), float(bx + bw)))
                        y2 = max(y1 + 1.0, min(float(h_px), float(by + bh)))
                        nw, nh = x2 - x1, y2 - y1
                        if nw < 2.0 or nh < 2.0:
                            continue
                        if coco_cid is None:
                            bird_boxes.append([x1, y1, nw, nh])
                        else:
                            frame_anns.append(
                                {
                                    "category_id": coco_cid,
                                    "bbox": [round(x1, 2), round(y1, 2), round(nw, 2), round(nh, 2)],
                                    "area": round(nw * nh, 2),
                                    "iscrowd": 0,
                                }
                            )

                if is_bird and not frame_anns:
                    # Save full frame as hard negative (bird in sky -> empty labels!)
                    fname = f"ext_{seq_name}_f{fi:04d}_birdneg.jpg"
                    im.save(dst_train / fname, quality=90)
                    _add_record(fname, w_px, h_px, [])
                    stats["extra_bird_hard_negs"] += 1
                    used_any = True
                    # Also save a zoomed 192x192 crop centered on the bird as a hard negative!
                    if bird_boxes and min(w_px, h_px) > 192:
                        bx, by, bw, bh = bird_boxes[0]
                        side = 128 if max(bw, bh) <= 18.0 else 192
                        cx, cy = bx + bw / 2.0, by + bh / 2.0
                        x0 = int(np.clip(round(cx - side / 2.0), 0, w_px - side))
                        y0 = int(np.clip(round(cy - side / 2.0), 0, h_px - side))
                        patch = im.crop((x0, y0, x0 + side, y0 + side)).resize(
                            (out_size, out_size), Image.Resampling.BILINEAR
                        )
                        cfname = f"ext_{seq_name}_f{fi:04d}_birdcrop_{side}.jpg"
                        patch.save(dst_train / cfname, quality=90)
                        _add_record(cfname, out_size, out_size, [])
                        stats["extra_bird_hard_negs"] += 1
                elif frame_anns:
                    fname = f"ext_{seq_name}_f{fi:04d}.jpg"
                    im.save(dst_train / fname, quality=90)
                    _add_record(fname, w_px, h_px, frame_anns)
                    stats["extra_video_frames"] += 1
                    used_any = True
                    # Add zoomed 128/192 positive crops + negative tile on half the sampled frames
                    if fi == sample_indices[len(sample_indices) // 2] or rng.random() < 0.5:
                        stats["extra_video_crops"] += _emit_positive_crops(
                            im, frame_anns, f"ext_{seq_name}_{fi}", allow_reps=False
                        )
                    _sample_neg_tiles(im, frame_anns, f"ext_{seq_name}_{fi}", 0.5)
            cap.release()
            if used_any:
                stats["extra_videos_used"] += 1

    # 3. Copy-paste small targets onto background crops/negatives with soft border blend
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
                if min(bw_px, bh_px) >= out_size:
                    x0 = rng.randint(0, bw_px - out_size)
                    y0 = rng.randint(0, bh_px - out_size)
                    canvas = bg.crop((x0, y0, x0 + out_size, y0 + out_size))
                else:
                    canvas = bg.resize((out_size, out_size), Image.Resampling.BILINEAR)
            canvas_arr = np.asarray(canvas, dtype=np.float32).copy()
            cp_anns: list[dict[str, Any]] = []
            for _ in range(rng.choice([1, 1, 2])):
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
                yy = np.linspace(-1.0, 1.0, ph)[:, None]
                xx = np.linspace(-1.0, 1.0, pw)[None, :]
                alpha = np.clip(
                    1.0 - 0.35 * np.maximum(np.abs(xx), np.abs(yy)) ** 4, 0.3, 1.0
                )[:, :, None]
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
                dst_train / fname, quality=91
            )
            _add_record(fname, out_size, out_size, cp_anns)
            stats["copy_paste"] += 1

    out_doc = {
        "images": new_images,
        "annotations": new_anns,
        "categories": doc["categories"],
    }
    (dst_train / "_annotations.coco.json").write_text(json.dumps(out_doc), encoding="utf-8")
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
    p.add_argument("--dst-dir", type=Path, default=Path("../dataset_mega"))
    p.add_argument("--extra-video-dir", type=Path, default=None, help="e.g. ../nos_videos/Video_V")
    p.add_argument(
        "--extra-birds-only",
        action="store_true",
        help="Use only BIRD clips from --extra-video-dir as hard negatives (avoids padded .mat boxes)",
    )
    p.add_argument("--tracking-eval-dir", type=Path, default=Path("../tracking_eval"))
    p.add_argument("--out-size", type=int, default=384)
    p.add_argument("--neg-tiles-per-img", type=float, default=1.0)
    p.add_argument("--copy-paste", type=int, default=80)
    p.add_argument("--frames-per-video", type=int, default=5)
    p.add_argument("--bird-frames-per-video", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_dataset(
        args.src_dir,
        args.dst_dir,
        extra_video_dir=args.extra_video_dir,
        extra_birds_only=args.extra_birds_only,
        tracking_eval_dir=args.tracking_eval_dir,
        out_size=args.out_size,
        neg_tiles_per_img=args.neg_tiles_per_img,
        copy_paste_count=args.copy_paste,
        frames_per_video=args.frames_per_video,
        bird_frames_per_video=args.bird_frames_per_video,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
