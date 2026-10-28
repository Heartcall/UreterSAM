"""Freeze a single changed training field before the matched short screen."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ureter_semis.experiment import atomic_json, sha256, source_identity


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    base = read("configs/train_a1_sup_kidney_only_20261027.json")
    candidate_path = ROOT / "configs/train_a1_sup_kidney_weight2_20261027.json"
    candidate = read(candidate_path)
    difference = {key: {"control": base["training"].get(key), "candidate": candidate["training"].get(key)}
                  for key in set(base["training"]) | set(candidate["training"])
                  if base["training"].get(key) != candidate["training"].get(key)}
    assert difference == {"class_weights": {"control": [1., 2., 4.], "candidate": [1., 2., 2.]}}
    control = read("runs/a1_sup_kidney_only_20261027/training/run_config.json")
    assert source_identity() == control["source_hashes"]
    assert not (ROOT / "runs/a1_sup_kidney_weight2_20261027").exists()
    frozen = {"training_difference": difference, "candidate_config_sha256": sha256(candidate_path),
              "control_inputs": control["inputs"], "control_source_hashes": control["source_hashes"],
              "control_runtime": control["runtime"], "primary_comparison": candidate["primary_comparison"],
              "diagnostic_sha256": sha256(ROOT / candidate["diagnostic_basis"]),
              "frozen_before_candidate_start": True}
    output = ROOT / "runs/a1_sup_kidney_weight2_20261027.preflight.json"
    assert not output.exists()
    atomic_json(output, frozen)
    print(json.dumps(difference))
