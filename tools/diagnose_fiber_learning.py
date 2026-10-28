"""Audit training-label agreement and real gradients, without modifying labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader

from ureter_semis.data import UreterDataset, load_manifest
from ureter_semis.experiment import atomic_json
from ureter_semis.inference import SegmentationPredictor
from ureter_semis.losses import supervised_multiclass_loss
from ureter_semis.model import ensemble_logits
from ureter_semis.trainer import validate


def diagnose(manifest_path: str, checkpoint: str, output_path: str) -> None:
    output = Path(output_path).resolve()
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    root, manifest = load_manifest(manifest_path)
    rows = manifest["train"]["labeled"]
    dataset = UreterDataset(root, rows, 256, "val")
    predictor = SegmentationPredictor(checkpoint)
    model = predictor.models[0]
    agreement = validate((model,), DataLoader(dataset, batch_size=2, shuffle=False), predictor.device)
    positive = [i for i in range(len(rows)) if (dataset[i]["mask"] == 2).any()]
    batch = [dataset[i] for i in positive[:2]]
    images = torch.stack([r["image"] for r in batch]).to(predictor.device)
    targets = torch.stack([r["mask"] for r in batch]).to(predictor.device)
    logits = model(images)
    gradients = []
    for fiber_weight in (4., 32.):
        loss = supervised_multiclass_loss(logits, targets, torch.tensor([1., 2., fiber_weight], device=predictor.device))
        grad_logits, grad_head = torch.autograd.grad(loss, (logits, model.segmentation_head[0].weight), retain_graph=True)
        gradients.append({"fiber_weight": fiber_weight, "loss": float(loss.detach()),
            "fiber_output_weight_gradient_norm": float(grad_head[2].norm()),
            "fiber_logit_gradient_mean_on_fiber": float(grad_logits[:, 2][targets == 2].mean()),
            "fiber_logit_gradient_abs_mean_on_background": float(grad_logits[:, 2][targets == 0].abs().mean()),
            "fiber_probability_mean_on_annotated_fiber": float(logits.softmax(1)[:, 2][targets == 2].mean().detach()),
            "finite_nonzero_gradient": bool(torch.isfinite(grad_head).all() and grad_head[2].norm() > 0)})
    del logits, images, targets, grad_logits, grad_head, predictor, model
    torch.cuda.empty_cache()
    baseline = json.loads(Path("runs/baselines/a0_20261027/config.json").read_text(encoding="utf-8"))
    a0 = SegmentationPredictor(baseline["checkpoint"])
    disagreements = []
    with torch.inference_mode():
        for i in range(len(rows)):
            batch = dataset[i]
            probability = ensemble_logits(a0.models, batch["image"].unsqueeze(0).to(a0.device)).softmax(1)[0].cpu()
            target = batch["mask"]
            conflict = (probability[2] >= .95) & (target == 0)
            count = int(conflict.sum())
            if count >= 32:
                disagreements.append({"sample_id": rows[i]["sample_id"], "source_dataset": rows[i]["source_dataset"],
                    "annotated_fiber_pixels_256": int((target == 2).sum()),
                    "a0_confident_fiber_over_source_background_pixels": count, "index": i})
    disagreements.sort(key=lambda r: r["a0_confident_fiber_over_source_background_pixels"], reverse=True)
    examples = disagreements[:6]
    canvas = Image.new("RGB", (3 * 256, max(1, len(examples)) * 282), "white")
    for j, record in enumerate(examples):
        i = record["index"]; raw = Image.open(rows[i]["image"]).convert("RGB").resize((256, 256))
        target = dataset[i]["mask"].numpy()
        with torch.inference_mode():
            prediction = ensemble_logits(a0.models, dataset[i]["image"].unsqueeze(0).to(a0.device)).argmax(1)[0].cpu().numpy()
        for col, labels in enumerate((None, target, prediction)):
            array = np.asarray(raw).copy()
            if labels is not None:
                for label, color in ((1, [235, 40, 40]), (2, [20, 210, 235])):
                    region = labels == label
                    array[region] = (.5 * array[region] + .5 * np.array(color)).astype(np.uint8)
            canvas.paste(Image.fromarray(array), (col * 256, j * 282 + 26))
        ImageDraw.Draw(canvas).text((4, j * 282 + 5), f"{record['sample_id'][:38]} | Raw / Source mask / A0", fill="black")
    canvas.save(output / "source_label_disagreements.png")
    atomic_json(output / "diagnosis.json", {"training_label_agreement_provisional_not_truth": agreement,
        "gradient_batch_sample_ids": [rows[i]["sample_id"] for i in positive[:2]], "gradients": gradients,
        "a0_disagreement_rule": "A0 fiber probability>=0.95 over provided background, at least 32 pixels at 256 grid",
        "a0_disagreement_count": len(disagreements), "disagreements": disagreements,
        "interpretation": "Disagreements are annotation-error or model-error candidates, not confirmed labels. No automatic relabeling or video training."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True); parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(); diagnose(args.manifest, args.checkpoint, args.output)
