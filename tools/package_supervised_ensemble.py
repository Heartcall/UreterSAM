"""Package two independently trained supervised members for mean-logit inference."""
from __future__ import annotations

from pathlib import Path
import torch

from ureter_semis.data import load_manifest
from ureter_semis.experiment import atomic_checkpoint, input_identity, require_training_access, sha256
from ureter_semis.supervision import audit_supervision


def package(manifest_path: str, member_paths: list[str], output_path: str) -> Path:
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(output)
    paths = [Path(p).resolve() for p in member_paths]
    if len(paths) != 2 or len(set(paths)) != 2:
        raise ValueError("Need two distinct supervised members")
    root, manifest = load_manifest(manifest_path)
    require_training_access(manifest, True)
    audit_supervision(root, manifest)
    if manifest["train"].get("unlabeled"):
        raise ValueError("Independent supervised ensemble cannot use active unlabeled data")
    identity = input_identity(Path(manifest_path).resolve(), root, manifest)
    states = [torch.load(p, map_location="cpu", weights_only=True) for p in paths]
    for state in states:
        config = state["config"]
        if (state.get("model_count") != 1 or state["classes"] != ["background", "stone", "fiber"]
                or state["run_identity"]["inputs"] != identity
                or config.get("mode") != "supervised_single_network"
                or config.get("pseudo_labels") is not False or config.get("mutual_learning") is not False):
            raise ValueError("Only same-input, pure supervised single-network members are allowed")
    first, second = states
    configs = [{k: v for k, v in state["config"].items() if k != "seed"} for state in states]
    if (configs[0] != configs[1] or first["config"]["seed"] == second["config"]["seed"]
            or first["epoch"] != second["epoch"] or first["size"] != second["size"]
            or first.get("architecture", "unet_resnet34") != second.get("architecture", "unet_resnet34")
            or first["run_identity"]["source_hashes"] != second["run_identity"]["source_hashes"]):
        raise ValueError("Members must differ only in seed and have matching epochs, grids and training code")
    if first["model1"].keys() != second["model1"].keys() or any(
            first["model1"][key].shape != second["model1"][key].shape for key in first["model1"]):
        raise ValueError("Member architectures differ")
    members = [{"checkpoint": str(path), "sha256": sha256(path), "seed": state["config"]["seed"],
                "epoch": state["epoch"], "loader_initial_seed": state["config"]["seed"] + 1,
                "optimizer_updates": sum(r["optimizer_update_steps"] for r in state.get("history", []))}
               for path, state in zip(paths, states)]
    payload = {"schema_version": 4, "model_count": 2,
               "architecture": first.get("architecture", "unet_resnet34"),
               "classes": first["classes"], "size": first["size"],
               "model1": first["model1"], "model2": second["model1"], "ensemble_rule": "arithmetic mean logits",
               "ensemble_members": members, "inputs": identity,
               "source_hashes": first["run_identity"]["source_hashes"],
               "training_mode": "independent_supervised_members", "pseudo_labels": False,
               "mutual_learning": False, "role": "inference_only_development_candidate_not_accepted"}
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_checkpoint(output, payload)
    return output
