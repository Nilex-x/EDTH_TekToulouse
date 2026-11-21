# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Hackathon entry (EDTH Toulouse) for **Challenge 1: Low-Altitude C-UAS Detection & Multi-Object Tracking** — see `CHALLENGE.md` for the full spec. Goal: detect and track small aerial targets (`0: drone_air`, `1: airplane`, `2: helicopter`, often < 32x32 px) with a pipeline deployable on an **Axelera Embedded 110m / Metis AIPU (1 GB on-card DRAM, INT8)**. Recommended detector is Roboflow **RF-DETR Nano @ 384x384**; YOLO/`ultralytics` is not allowed.

Judging (proposed): detection 40% (`mAP@50:95`, `small_AP@50`), tracking 40% (`MOTA`, `IDF1`, `track_continuity`), edge efficiency 20% (Metis throughput + memory fit).

## Data (not in git)

`dataset/`, `calibration/`, `tracking_eval/` and the root `dataset_card.json` are gitignored and must be present locally.
- `dataset/{train,valid,test}/_annotations.coco.json` + JPGs (COCO, 0-based category ids). `test/` labels are held out.
- `calibration/`: 200 train-only 384x384 JPEGs for INT8 quantization.
- `tracking_eval/seq_0001..0008/video.mp4` (30 fps). Only `seq_0001–0004` (split `valid`) have `gt/gt.txt`; split membership, resolution and frame counts come from `tracking_eval/sequences_manifest.json`.

Generated artifacts are gitignored too and must never be committed: `starter-kit/runs/` (checkpoints, >100 MB each), `starter-kit/exports/` (ONNX), `starter-kit/pred_mot*/` and `starter-kit/*preds*.json` (predictions), `.venv/`, `__pycache__/`, `*.egg-info/`. Write new outputs under these names/paths so they stay ignored.

## Commands

All pipeline commands run from `starter-kit/` (Python ≥ 3.11, `ffmpeg` on PATH for MOT). Setup: `uv venv && source .venv/bin/activate && uv pip install -e .` — or prefix commands with `uv run --with "rfdetr[onnx,train]" --with onnx --with onnxscript --with onnxruntime --with scipy`.

```bash
# Train (outputs runs/<dir>/checkpoint_best_total.pth)
python train_rfdetr.py --dataset-dir ../dataset --model nano --epochs 20 --output-dir runs/c1_nano

# Export static ONNX (opset 17, batch 1) -> exports/rfdetr/{full,backbone}/inference_model.onnx
python export_rfdetr_axelera.py --model nano --num-classes 3 \
  --weights runs/c1_nano/checkpoint_best_total.pth --calibration-dir ../calibration

# Detection predictions + score on valid/
python predict_onnx_baseline.py detect --onnx exports/rfdetr/full/inference_model.onnx \
  --dataset-dir ../dataset --split valid --out c1_preds.json
python evaluate_submissions.py detect --gt ../dataset/valid/_annotations.coco.json --preds c1_preds.json

# Tracking predictions + score on valid sequences
python predict_onnx_baseline.py mot --onnx exports/rfdetr/full/inference_model.onnx \
  --eval-dir ../tracking_eval --out-dir pred_mot/
python evaluate_submissions.py mot --gt-dir ../tracking_eval --pred-dir pred_mot/ --split valid

# Visual explorer (from repo root, stdlib only) with prediction overlays
python3 viewer.py --preds starter-kit/c1_preds.json --pred-dir starter-kit/pred_mot --port 8765
```

There is no test suite or linter configured; `evaluate_submissions.py` is the source of truth for metrics. Baseline reference (1-epoch smoke run, CPU ORT) is in `starter-kit/baseline_metrics.json`: mAP@50 0.447, small_AP50 0.506, MOTA −0.013, IDF1 0.248.

## Architecture

The starter kit is shared across several EDTH challenges, so `predict_onnx_baseline.py` and `evaluate_submissions.py` also contain `triage` / `acoustic` subcommands — ignore them; only `detect` and `mot` matter here.

Pipeline: `train_rfdetr.py` (thin wrapper over `rfdetr` `RFDETRNano.train`) → `export_rfdetr_axelera.py` (exports two ONNX graphs and verifies outputs) → `predict_onnx_baseline.py` (CPU ONNX Runtime inference + tracker) → `evaluate_submissions.py` (scoring).

Key details that span files:
- **Preprocessing**: RGB → bilinear resize to 384x384 (no letterbox) → `/255` → ImageNet mean/std → NCHW float32. Must match between export, inference, and any Metis `axcompile --transform preprocess.py` callback (not supplied; must be written).
- **ONNX outputs**: full model returns `dets [1,300,4]` (cxcywh normalized) and `labels [1,300,C+1]` (logits, last is background). For C=3 both are `[1,300,4]` — **always index outputs by name, never position**.
- **Decoding** (`_decode_rfdetr_topk`): sigmoid over the first C logits, top-K over flattened (query, class) pairs, rescale to original image px as `[x,y,w,h]`. Detection uses top-100 with no threshold; MOT uses top-10, `score_thr=0.25`, then greedy cross-query NMS at IoU 0.5.
- **Baseline tracker** (`predict_mot`): ffmpeg extracts frames to a temp dir; per frame, Hungarian matching on IoU (gate 0.25) against only the *previous frame's* tracks — no motion model, no track buffer, so any missed frame kills the track. This is the main place to improve tracking scores (e.g. Kalman/ByteTrack-style association, track persistence).
- **`track_roi.py`** (branch `feat/roi-crop-tracking`): alternative to `predict_mot` that imports the baseline's preprocess/decode/NMS helpers. Kalman + ByteTrack-style two-stage association, confirmed-only output gated by a running track score (`--out-score-thr`), plus native-resolution square crops around predicted positions of small tracks (`--no-crops` for ablation). Optional `--tiles` periodically sweeps the frame with native-res tiles to discover targets too small for the global pass (off by default: with the 1-epoch smoke model it added more false positives than hits). `track_sequence()` takes any object with a `detect(image, region, top_k=, score_thr=)` method, so the tracker can be tested without a model.
- **Backbone export** (`exports/rfdetr/backbone`, output `features [1,256,24,24]`) excludes the deformable-attention/GridSample decoder tail; it's a starting point for Metis partitioning, not a runnable detector. No compiled Metis baseline exists.
- **MOT scorer** is class-agnostic, IoU 0.5 per-frame matching; IDF1 uses identity-pair counts (not exact TrackEval parity). Unknown `seq_*.txt` files in the pred dir raise errors.

## Submission formats

- `preds.json`: list of `{"image_id": int, "category_id": 0-based int, "bbox": [x,y,w,h], "score": float∈[0,1]}`; finite values, w,h > 0; top 100 per image scored. Scorer rejects NaN/Inf strictly.
- `pred_mot/seq_XXXX.txt`: `frame_1based,track_id,x,y,w,h,conf,class_id_1based,1.0`; no duplicate `(frame, track_id)` rows.
