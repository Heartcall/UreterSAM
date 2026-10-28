"""Supplemental fresh-process timing to separate runtime allocation carryover."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    output = ROOT / "runs/a1_sup_kidney_alt_20261028"
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if len(sys.argv) == 3 and sys.argv[1] == "--member":
        from ureter_semis.data import load_manifest
        from ureter_semis.experiment import atomic_json, seed_training
        from tools.run_supervised_dual_screen import benchmark
        import torch
        name = sys.argv[2]
        row = next(r for r in summary["models"] if r["name"] == name)
        config = json.loads((ROOT / "configs/train_a1_sup_kidney_alt_20261028.json").read_text(encoding="utf-8"))
        root, manifest = load_manifest(ROOT / "datasets/ureter_semis_a1_sup_kidney_only_20261027/manifest.json")
        seed_training(42)
        result = benchmark(row["checkpoint"], root, manifest["val"], config["performance"])
        result["post_release_cuda_allocated_mib"] = torch.cuda.memory_allocated() / 2**20
        result["isolation"] = "Fresh sequential child process per model; same12 warmup and87 measured frames; supplementary timing, not a change to accuracy protocol"
        atomic_json(output / f"performance_isolated_{name}.json", result)
        print(name, result["timings"]["pipeline_ms"], result["cuda_peak_allocated_mib"], flush=True)
    else:
        all_rows = {}
        for row in summary["models"]:
            name = row["name"]
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--member", name], cwd=ROOT,
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0, check=True)
            all_rows[name] = json.loads((output / f"performance_isolated_{name}.json").read_text(encoding="utf-8"))
        (output / "performance_isolated.json").write_text(json.dumps(all_rows, indent=2) + "\n", encoding="utf-8")
