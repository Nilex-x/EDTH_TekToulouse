"""
Pipeline complet : détecteur ONNX (RF-DETR) + tracker Kalman sur les vidéos.

À utiliser dès que l'équipe a un modèle exporté (inference_model.onnx).
Les détections sont SAUVEGARDÉES dans un fichier JSON : ensuite on peut
régler le tracker autant qu'on veut sans relancer le modèle (beaucoup plus rapide).

Exemples :
    # 1re fois : lance le modèle, sauvegarde les détections, tracke, évalue
    python track_with_onnx.py --onnx ../starter-kit/exports/rfdetr/full/inference_model.onnx \
        --save-dets runs/dets.json

    # Ensuite : réutilise les détections, change juste les réglages du tracker
    python track_with_onnx.py --load-dets runs/dets.json --max-age 30 --min-hits 2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from kalman_tracker import KalmanTracker, TrackerConfig, write_mot_file

ROOT = Path(__file__).resolve().parent.parent  # dossier EDTH_TekToulouse
sys.path.insert(0, str(ROOT / "starter-kit"))


# =============================================================================
# 1. Lancer le détecteur sur chaque image de chaque vidéo
# =============================================================================


def run_detector(onnx_path: Path, eval_dir: Path, img_size: int, score_thr: float) -> dict:
    """Renvoie {"seq_0001": {"1": [[cls, score, [x,y,w,h]], ...], ...}, ...}."""
    import onnxruntime as ort
    from PIL import Image

    # On réutilise le décodage et le prétraitement du kit de départ
    from predict_onnx_baseline import _decode_rfdetr_topk, _nms_dets, _preprocess_imagenet

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    all_dets: dict = {}

    with tempfile.TemporaryDirectory() as td:
        for seq_dir in sorted(p for p in eval_dir.glob("seq_*") if (p / "video.mp4").exists()):
            # Découpe la vidéo en images avec ffmpeg
            frames_dir = Path(td) / seq_dir.name
            frames_dir.mkdir(parents=True)
            subprocess.run(
                ["ffmpeg", "-loglevel", "error", "-i", str(seq_dir / "video.mp4"),
                 "-fps_mode", "passthrough", str(frames_dir / "f_%06d.jpg")],
                check=True,
            )
            seq_dets = {}
            for f1, fpath in enumerate(sorted(frames_dir.glob("f_*.jpg")), 1):
                with Image.open(fpath) as im:
                    w_px, h_px = im.size
                    inp = _preprocess_imagenet(im, img_size)
                # Sorties lues PAR NOM ("dets", "labels"), jamais par position !
                outs = dict(zip(out_names, sess.run(out_names, {in_name: inp})))
                dets = _nms_dets(
                    _decode_rfdetr_topk(outs["dets"][0], outs["labels"][0], w_px, h_px, 3,
                                        top_k=10, score_thr=score_thr),
                    iou_thr=0.5,
                )
                seq_dets[str(f1)] = [[int(c), float(s), [float(v) for v in b]] for c, s, b in dets]
            all_dets[seq_dir.name] = seq_dets
            print(f"  {seq_dir.name} : {len(seq_dets)} images détectées")
    return all_dets


# =============================================================================
# 2. Programme principal
# =============================================================================


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx", type=Path, help="modèle ONNX exporté")
    p.add_argument("--load-dets", type=Path, help="réutiliser des détections déjà calculées")
    p.add_argument("--save-dets", type=Path, help="où sauvegarder les détections")
    p.add_argument("--eval-dir", type=Path, default=ROOT / "challenge-1-cuas-tracking" / "tracking_eval")
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "runs" / "pred_mot")
    p.add_argument("--img-size", type=int, default=384)
    # Seuil bas : le tracker Kalman filtre lui-même les fausses détections
    p.add_argument("--score-thr", type=float, default=0.15)
    p.add_argument("--min-score", type=float, default=0.15, help="filtre appliqué aux détections chargées")
    # Réglages du tracker
    p.add_argument("--max-age", type=int, default=15)
    p.add_argument("--min-hits", type=int, default=3)
    p.add_argument("--iou-threshold", type=float, default=0.2)
    p.add_argument("--center-threshold", type=float, default=1.5)
    p.add_argument("--no-interpolate", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # --- Étape 1 : obtenir les détections (modèle ou fichier) ---
    if args.load_dets:
        all_dets = json.loads(args.load_dets.read_text())
    elif args.onnx:
        print("Lancement du détecteur...")
        all_dets = run_detector(args.onnx, args.eval_dir, args.img_size, args.score_thr)
        if args.save_dets:
            args.save_dets.parent.mkdir(parents=True, exist_ok=True)
            args.save_dets.write_text(json.dumps(all_dets))
    else:
        sys.exit("Il faut --onnx ou --load-dets")

    # --- Étape 2 : tracker chaque vidéo ---
    cfg = TrackerConfig(
        max_age=args.max_age,
        min_hits=args.min_hits,
        iou_threshold=args.iou_threshold,
        center_threshold=args.center_threshold,
        interpolate_gaps=not args.no_interpolate,
    )
    for seq, seq_dets in all_dets.items():
        tracker = KalmanTracker(cfg)
        n_frames = max((int(f) for f in seq_dets), default=0)
        for f in range(1, n_frames + 1):
            dets = [(c, s, b) for c, s, b in seq_dets.get(str(f), []) if s >= args.min_score]
            tracker.step(f, dets)
        write_mot_file(args.out_dir / f"{seq}.txt", tracker.finish())
    print(f"Fichiers MOT écrits dans {args.out_dir}")

    # --- Étape 3 : note sur les séquences "valid" ---
    from evaluate_submissions import evaluate_mot_tracking

    m = evaluate_mot_tracking(args.eval_dir, args.out_dir, split="valid")
    print(f"MOTA={m['MOTA']:.4f}  IDF1={m['IDF1']:.4f}  continuité={m['track_continuity']:.4f}")


if __name__ == "__main__":
    main()
