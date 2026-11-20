# Challenge 1: Low-Altitude C-UAS Detection & Multi-Object Tracking

**Presentation:** [Two-slide PDF](presentation/challenge-slides.pdf) · [Editable PowerPoint](presentation/challenge-slides.pptx). Includes a challenge overview and real validation examples from this data pack.

> **Target Platform**: **Axelera Embedded 110m** — M.2 2280 Key-M (PCIe Gen3 x4), 1x Metis AIPU (214 TOPS INT8, 3.5–9 W, **1 GB LPDDR4x** on-card DRAM) connected to a host processor.
> **Recommended Detector**: **Roboflow RF-DETR** (`RFDETRNano` at `384x384`, ~30.2M params, Apache-2.0) + host tracker tail. YOLO / `ultralytics` are **not** used in this challenge.

---

## 1. Mission & Problem Statement

Low-altitude aerial threats (small multirotor/fixed-wing UAVs, manned rotorcraft, and low-flying aircraft) appear against clutter, tree lines, clouds, and urban horizons often at **< 32x32 px** apparent size. Build an edge deployable detection + multi-object tracking (MOT) pipeline that runs inside the **1 GB LPDDR4x** envelope of an Axelera Embedded 110m module without sacrificing small-target recall or track identity continuity.

---

## 2. Dataset Specification

- **Classes (0-indexed COCO IDs)**:
  - `0`: `drone_air`
  - `1`: `airplane`
  - `2`: `helicopter`
- **Leak-free split guarantee**: All frames from a source clip share one deduplicated content cluster (`dedup_root_key`) and never cross `train` / `valid` / `test` (verified across visual `dHash64` and SHA-256). All splits include negative (aircraft-free) background scenes; `test/` annotations are held out in public archives.

| Split | Images | Clips / Roots | `drone_air` boxes | `airplane` boxes | `helicopter` boxes | Empty Negative Images |
|---|---:|---:|---:|---:|---:|---:|
| `train/` | 859 | 14 | 592 | 144 | 106 | 48 |
| `valid/` | 422 | 8 | 276 | 36 | 75 | 48 |
| `test/` | 328 | 7 | — (held-out) | — (held-out) | — (held-out) | — (held-out) |

- **INT8 Calibration Set (`calibration/`)**: `200` stratified `384x384` RGB images sampled **strictly from `train/`** (zero validation/test leakage).
- **MOT Evaluation Sequences (`tracking_eval/`)**: `8` constant-frame-rate `30 fps` H.264 MP4 sequences (`seq_0001`–`seq_0008`, `72`–`150` frames each): `4` `valid` sequences (`522` total GT boxes in `seq_0001`–`seq_0004/gt/gt.txt`) and `4` `test` sequences (`seq_0005`–`seq_0008`, GT held out in public archives).

### Directory Layout

```text
challenge-1-cuas-tracking/
├── CHALLENGE.md
├── viewer.py                     # Zero-dependency local UI (COCO + MOT stream overlay)
├── dataset_card.json
├── calibration/                  # 200 train-only 384x384 JPEGs for Metis INT8 quantization
├── dataset/
│   ├── dataset_card.json
│   ├── train/_annotations.coco.json + *.jpg
│   ├── valid/_annotations.coco.json + *.jpg
│   └── test/_annotations.coco.json + *.jpg   # GT annotations held out in public archive
└── tracking_eval/
    ├── sequences_manifest.json
    └── seq_0001..seq_0008/
        ├── video.mp4
        └── gt/gt.txt             # present on valid sequences (test GT held out)
```

### Interactive Data Explorer (`viewer.py`)

Inspect detection boxes (including `< 32x32 px` small-target filters), negatives, calibration frames, and 30 fps MOT tracking streams offline using Python's standard library:

```bash
python3 viewer.py
# Or overlay your validation detections and MOT tracks:
python3 viewer.py --preds starter-kit/c1_preds.json --pred-dir starter-kit/pred_mot --port 8765
# Over SSH from a remote Axelera host:
ssh -L 8765:127.0.0.1:8765 user@edge-host
```

---

## 3. Baseline & Edge Deployment Architecture

1. **Train `RFDETRNano` (`384x384`)**:
   ```bash
   cd starter-kit
   uv run --with "rfdetr[onnx,train]" --with onnx --with onnxscript --with onnxruntime --with scipy \
     python train_rfdetr.py --dataset-dir ../dataset --model nano --epochs 20 --output-dir runs/c1_nano
   ```
2. **Export static ONNX (`opset=17`, `batch=1`, `384x384`)**:
   ```bash
   uv run --with "rfdetr[onnx,train]" --with onnx --with onnxscript --with onnxruntime \
     python export_rfdetr_axelera.py --model nano --num-classes 3 \
     --weights runs/c1_nano/checkpoint_best_total.pth \
     --calibration-dir ../calibration
   ```
   - `exports/rfdetr/full/inference_model.onnx` (`102.66 MB` FP32): inputs `input [1, 3, 384, 384]`; named outputs **`dets [1, 300, 4]`** and **`labels [1, 300, 4]`** (note: both share shape `[1, 300, 4]` for `num_classes=3` — **always index by tensor name, never by positional index**!).
   - `exports/rfdetr/backbone/inference_model.onnx` (`88.95 MB` FP32): outputs `features [1, 256, 24, 24]` cleanly without the 300-query `GridSample` deformable attention tail.
3. **Voyager SDK deployment exercise**: The shipped baseline runs the full detector on CPU ONNX Runtime and uses an IoU/Hungarian tracker. The backbone export is a starting point for Metis experiments, not a complete detector. Exporting and connecting a compatible host decoder is additional work; the kit does not implement automatic partitioning or ByteTrack. Follow the SDK guidance in `starter-kit/README.md`, calibrate with `train/` images using ImageNet-normalized inputs, and verify compiler support, INT8 accuracy, memory, and end-to-end latency on hardware. No compiled Metis baseline is supplied.

### Smoke-Test Reference (1-Epoch `RFDETRNano` on `valid/` Static ONNX Runtime CPU)

| Metric | 1-Epoch Smoke Result | Notes |
|---|---:|---|
| Val `mAP@50` (PyTorch Trainer) | **0.4490** | `mAP@50:95 = 0.2942`, `F1 = 0.5097` |
| Static ONNX `mAP@50` / `small_AP50` (`valid/`) | **0.4474 / 0.5058** | Top 100 query/class predictions per image on CPU ORT (`40.05 ms/frame`, `422` imgs) |
| Static ONNX + IoU Tracker (`valid` `tracking_eval/`) | **MOTA -0.0134 / IDF1 0.2481** | Track Continuity **0.4317**, `6` ID switches over `522` valid GT boxes |

---

## 4. Metrics and Proposed Judging Weights

The scorer returns individual metrics, not a combined leaderboard score. The percentages below are a proposed judging rubric. Organizers must publish the aggregation, normalization, tie-break rules, and hardware measurement protocol before using this dataset for a ranked event. Local scoring uses `valid/`; final scoring requires organizer-held `test/` labels.

Participants submit detection predictions (`preds.json`) and MOT tracking files (`pred_mot/seq_XXXX.txt`) evaluated on `valid/` locally with `starter-kit/evaluate_submissions.py` (default `--split valid` on `mot`):
- **Detection (`40%`)**: `mAP@50:95` & `small_AP@50` (`area < 32^2 px`)
- **Tracking (`40%`)**: `MOTA`, `IDF1`, and `track_continuity` across `tracking_eval/`
- **Edge Efficiency (`20%`)**: Compiled Metis model / host tail end-to-end throughput & memory fit (< 1 GB card DRAM)

This is a small curated pilot: many frames come from the same clip, so image count is not the number of independent scenes. The MOT scorer is class-agnostic and uses per-frame IoU matching for MOTA and identity-pair counts for IDF1; use the supplied implementation for comparisons rather than assuming exact TrackEval parity.
