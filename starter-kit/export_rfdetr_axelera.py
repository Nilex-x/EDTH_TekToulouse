"""Export RF-DETR to static-shape ONNX, validate with ONNX Runtime, and print axcompile CLI."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from rfdetr import RFDETRMedium, RFDETRNano, RFDETRSmall


class _RFDETRHostDecoder(torch.nn.Module):
    """Host-side RF-DETR transformer decoder + box/class heads taking Metis backbone features."""

    def __init__(self, lwdetr: torch.nn.Module, pos_embed: torch.Tensor) -> None:
        super().__init__()
        self.transformer = lwdetr.transformer
        self.refpoint_embed = lwdetr.refpoint_embed
        self.query_feat = lwdetr.query_feat
        self.bbox_embed = lwdetr.bbox_embed
        self.class_embed = lwdetr.class_embed
        self.num_queries = lwdetr.num_queries
        self.bbox_reparam = lwdetr.bbox_reparam
        self.register_buffer("pos_embed", pos_embed.detach().clone())

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        refpoint_embed_weight = self.refpoint_embed.weight[: self.num_queries]
        query_feat_weight = self.query_feat.weight[: self.num_queries]
        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(
            [features],
            None,
            [self.pos_embed],
            refpoint_embed_weight,
            query_feat_weight,
            cross_attn_srcs=None,
        )[:4]
        if hs is not None:
            if self.bbox_reparam:
                delta = self.bbox_embed(hs)
                cxcy = delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
                wh = delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
                outputs_coord = torch.cat([cxcy, wh], dim=-1)
            else:
                outputs_coord = (self.bbox_embed(hs) + ref_unsigmoid).sigmoid()
            outputs_class = self.class_embed(hs)
        else:
            outputs_class = self.transformer.enc_out_class_embed[0](hs_enc)
            outputs_coord = ref_enc
        return outputs_coord, outputs_class


def export_host_decoder(
    model: object, output_dir: Path, size: int
) -> tuple[Path, list[int]]:
    """Export the host-side decoder ONNX matching ``backbone/inference_model.onnx``."""
    import copy

    lw = copy.deepcopy(model.model.model).eval().cpu()  # type: ignore[attr-defined]
    lw.export()
    dummy_img = torch.zeros(1, 3, size, size, dtype=torch.float32)
    with torch.no_grad():
        srcs, _, poss, _ = lw.backbone(dummy_img)
    feat_sample = srcs[0]
    pos_sample = poss[0]
    host_dec = _RFDETRHostDecoder(lw, pos_sample).eval().cpu()

    dec_dir = output_dir / "host_decoder"
    dec_dir.mkdir(parents=True, exist_ok=True)
    dec_path = dec_dir / "inference_model.onnx"
    torch.onnx.export(
        host_dec,
        (feat_sample,),
        str(dec_path),
        input_names=["features"],
        output_names=["dets", "labels"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    return dec_path, list(feat_sample.shape)

MODELS = {
    "nano": (RFDETRNano, 384),
    "small": (RFDETRSmall, 512),
    "medium": (RFDETRMedium, 576),
}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def preprocess_image(image_path: Path, size: int) -> np.ndarray:
    """Resize and ImageNet-normalize an RGB image to static NCHW float32 tensor."""
    with Image.open(image_path) as im:
        rgb = im.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    nchw = np.transpose(arr, (2, 0, 1))[None, ...]
    return (nchw - IMAGENET_MEAN) / IMAGENET_STD


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def infer_num_classes(dataset_dir: Path) -> int:
    """Read foreground category count from ``train/_annotations.coco.json``."""
    coco_path = dataset_dir / "train" / "_annotations.coco.json"
    if not coco_path.exists():
        raise FileNotFoundError(f"Missing COCO annotations file: {coco_path}")
    cats = json.loads(coco_path.read_text(encoding="utf-8")).get("categories", [])
    if not cats:
        raise ValueError(f"Empty categories list in {coco_path}")
    return len(cats)


def verify_onnx_outputs(
    onnx_path: Path,
    sample_image: Path,
    size: int,
    backbone_only: bool,
    num_classes: int,
) -> dict[str, object]:
    """Run ONNX Runtime inference and verify output tensor names and shapes."""
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]
    inp = preprocess_image(sample_image, size)
    outputs = dict(
        zip(output_names, sess.run(output_names, {input_name: inp}), strict=True)
    )

    summary: dict[str, object] = {
        "onnx_path": str(onnx_path),
        "input_name": input_name,
        "input_shape": list(inp.shape),
        "outputs": {k: list(v.shape) for k, v in outputs.items()},
    }
    if not backbone_only:
        if "dets" not in outputs or "labels" not in outputs:
            raise RuntimeError(
                f"Expected named outputs 'dets' and 'labels', got {list(outputs)}. "
                "Always index RF-DETR ONNX outputs by name, never by positional index."
            )
        expected_logits = num_classes + 1
        if (
            outputs["labels"].ndim != 3
            or outputs["labels"].shape[-1] != expected_logits
        ):
            raise RuntimeError(
                f"Expected 'labels' shape [1, Q, {expected_logits}] for num_classes={num_classes}, "
                f"got {outputs['labels'].shape}"
            )
        probs = sigmoid(outputs["labels"][0][..., :num_classes])
        if len(probs) == 0:
            summary["top_detection"] = None
            return summary
        top_scores = probs.max(axis=-1)
        top_classes = probs.argmax(axis=-1)
        best_idx = int(np.argmax(top_scores))
        summary["top_detection"] = {
            "score": round(float(top_scores[best_idx]), 4),
            "class_id": int(top_classes[best_idx]),
            "bbox_cxcywh_norm": [
                round(float(v), 4) for v in outputs["dets"][0][best_idx]
            ],
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODELS), default="nano")
    parser.add_argument(
        "--weights", type=Path, default=None, help="Path to checkpoint_best_total.pth"
    )
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("exports/rfdetr"))
    parser.add_argument(
        "--mode",
        choices=("full", "backbone", "both"),
        default="both",
        help="'full' exports end-to-end ONNX for ORT; 'backbone' exports backbone+neck only. Metis deployment requires separate validation and integration.",
    )
    parser.add_argument("--sample-image", type=Path, default=None)
    parser.add_argument("--calibration-dir", type=Path, default=Path("../calibration"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset_dir is not None:
        num_classes = infer_num_classes(args.dataset_dir)
    elif args.num_classes is not None:
        num_classes = args.num_classes
    else:
        num_classes = 3

    cls, size = MODELS[args.model]
    kwargs: dict[str, object] = {"num_classes": num_classes}
    if args.weights is not None:
        kwargs["pretrain_weights"] = str(args.weights)
    model = cls(**kwargs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    exported: dict[str, Path] = {}

    if args.mode in ("full", "both"):
        full_dir = args.output_dir / "full"
        full_path = Path(
            model.export(
                output_dir=str(full_dir),
                output_name="inference_model",
                backbone_only=False,
                opset_version=17,
                shape=(size, size),
                batch_size=1,
                dynamic_batch=False,
                verbose=False,
            )
        )
        canonical_full = full_dir / "inference_model.onnx"
        if full_path.resolve() != canonical_full.resolve():
            shutil.copyfile(full_path, canonical_full)
        exported["full"] = canonical_full

    if args.mode in ("backbone", "both"):
        bb_dir = args.output_dir / "backbone"
        bb_path = Path(
            model.export(
                output_dir=str(bb_dir),
                output_name="inference_model",
                backbone_only=True,
                opset_version=17,
                shape=(size, size),
                batch_size=1,
                dynamic_batch=False,
                verbose=False,
            )
        )
        canonical_bb = bb_dir / "inference_model.onnx"
        if bb_path.resolve() != canonical_bb.resolve():
            shutil.copyfile(bb_path, canonical_bb)
        exported["backbone"] = canonical_bb
        dec_path, _ = export_host_decoder(model, args.output_dir, size)
        exported["host_decoder"] = dec_path

    reports: dict[str, dict[str, object]] = {}
    if args.sample_image is not None:
        if not args.sample_image.exists():
            raise FileNotFoundError(
                f"--sample-image does not exist: {args.sample_image}"
            )
        for mode_name, onnx_path in exported.items():
            if mode_name == "host_decoder":
                continue
            reports[mode_name] = verify_onnx_outputs(
                onnx_path,
                args.sample_image,
                size,
                backbone_only=(mode_name == "backbone"),
                num_classes=num_classes,
            )
        if "backbone" in exported and "host_decoder" in exported:
            inp = preprocess_image(args.sample_image, size)
            bb_sess = ort.InferenceSession(str(exported["backbone"]), providers=["CPUExecutionProvider"])
            dec_sess = ort.InferenceSession(str(exported["host_decoder"]), providers=["CPUExecutionProvider"])
            bb_in = bb_sess.get_inputs()[0].name
            bb_out = bb_sess.get_outputs()[0].name
            feats = bb_sess.run([bb_out], {bb_in: inp})[0]
            dec_in = dec_sess.get_inputs()[0].name
            dec_names = [o.name for o in dec_sess.get_outputs()]
            dec_outs = dict(zip(dec_names, dec_sess.run(dec_names, {dec_in: feats}), strict=True))
            split_rep: dict[str, object] = {
                "backbone_out_shape": list(feats.shape),
                "decoder_outputs": {k: list(v.shape) for k, v in dec_outs.items()},
            }
            if "full" in exported:
                f_sess = ort.InferenceSession(str(exported["full"]), providers=["CPUExecutionProvider"])
                f_in = f_sess.get_inputs()[0].name
                f_names = [o.name for o in f_sess.get_outputs()]
                f_outs = dict(zip(f_names, f_sess.run(f_names, {f_in: inp}), strict=True))
                split_rep["max_abs_diff_dets"] = round(
                    float(np.max(np.abs(dec_outs["dets"] - f_outs["dets"]))), 6
                )
                split_rep["max_abs_diff_labels"] = round(
                    float(np.max(np.abs(dec_outs["labels"] - f_outs["labels"]))), 6
                )
            reports["split_backbone_plus_host_decoder"] = split_rep

    print(
        json.dumps(
            {
                "exported": {k: str(v) for k, v in exported.items()},
                "verification": reports,
            },
            indent=2,
        )
    )
    target_onnx = exported.get("backbone") or exported["full"]
    print(
        "\n--- Axelera Voyager SDK Quantization / Compilation Guide ---\n"
        "RF-DETR ONNX expects ImageNet-normalized NCHW RGB inputs:\n"
        "  pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375]\n"
        "Pass your calibration folder as --imageset (or via model_quantization.yaml).\n"
        "Backbone export is not a detector; a compatible host decoder needs separate integration.\n"
        "Hardware compilation is unverified. Command template (supply preprocess.py with\n"
        "the SDK callback matching RGB resize, CHW scaling and normalization above):\n"
        f"axcompile -i {target_onnx} \\\n"
        f"  --imageset {args.calibration_dir} --transform preprocess.py \\\n"
        f"  --input-shape 1,3,{size},{size} \\\n"
        f"  -o {args.output_dir / 'metis_axm'}/\n"
    )


if __name__ == "__main__":
    main()
