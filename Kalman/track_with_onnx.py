"""Point d'entrée unifié (wrapper vers starter-kit/track_roi.py) + évaluation MOT.

Conserve la compatibilité CLI (--onnx, --load-dets, --save-dets, --max-age, --min-hits,
--no-interpolate) tout en utilisant le tracker unifié sans dépendance filterpy
(Kalman + ByteTrack + CMC + ROI crops upscalés + découverte round-robin + interpolation).

Exemples :
    python track_with_onnx.py --onnx ../starter-kit/exports/rfdetr_20ep/full/inference_model.onnx \\
        --save-dets runs/dets.json
    python track_with_onnx.py --load-dets runs/dets.json --no-crops --no-discover --max-age 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "starter-kit"))

from evaluate_submissions import evaluate_mot_tracking  # noqa: E402
from track_roi import main as track_roi_main  # noqa: E402


def main() -> None:
    # Fix default --eval-dir and --out-dir when invoked from Kalman/
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--eval-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--split", default="valid")
    known, remaining = parser.parse_known_args()

    eval_dir = known.eval_dir or (ROOT / "tracking_eval")
    out_dir = known.out_dir or (Path(__file__).resolve().parent / "runs" / "pred_mot")

    sys.argv = [
        sys.argv[0],
        "--eval-dir",
        str(eval_dir),
        "--out-dir",
        str(out_dir),
        "--split",
        known.split,
        *remaining,
    ]
    track_roi_main()

    if known.split in ("valid", "all"):
        m = evaluate_mot_tracking(eval_dir, out_dir, split="valid")
        print(
            f"\n[VALID MOT] MOTA={m['MOTA']:.4f}  IDF1={m['IDF1']:.4f}  "
            f"continuité={m['track_continuity']:.4f}  IDSW={m['id_switches']}"
        )


if __name__ == "__main__":
    main()
