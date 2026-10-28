import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ureter_semis.experiment import atomic_json, sha256, source_identity


if __name__ == "__main__":
    output = ROOT / "runs/a1_sup_kidney_alt_20261028"
    def read(name): return json.loads((output / name).read_text(encoding="utf-8"))
    perf = read("performance_isolated.json")
    comparison = read("comparison.json")
    preflight = read("preflight.json")
    examples = read("failure_examples.json")
    checks = {
        "all_three_fresh_process_results_match_checkpoints": all(perf[name]["checkpoint_sha256"] == row["checkpoint_sha256"] == sha256(Path(row["checkpoint"])) for name, row in zip(("A0", "unet_control", "segformer_b0"), comparison["checkpoints"])),
        "same87_measurements12_warmup": all(r["samples"] == 87 and r["warmup_frames"] == 12 for r in perf.values()),
        "visualization_confusions_match_primary": examples["all_prediction_confusions_match_primary_comparison"],
        "small_stone_group_kept_undefined": all(r["small_stone_frames"] == 0 and r["small_stone_pixel_recall"] is None for r in read("spatial_detail.json")["models"].values()),
        "core_source_still_matches_training": source_identity() == preflight["source_hashes"]
    }
    atomic_json(ROOT / "runs/engineering/kidney_alt_supplemental_checks_20261028.json", {"checks": checks, "all_passed": all(checks.values())})
    print(json.dumps(checks)); assert all(checks.values())
