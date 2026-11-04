"""Reproducible epoch-boundary training state and input identity."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import random

import numpy as np
import torch


def require_training_access(manifest: dict, allow_unreviewed_experiment: bool = False) -> None:
    from .source_policy import require_training_sources
    require_training_sources(manifest)
    if manifest.get("training_ready") is not False:
        return
    waiver = manifest.get("review_waiver", {})
    if (allow_unreviewed_experiment and manifest.get("role") == "unreviewed_development_experiment"
            and manifest.get("experiment_ready") is True and waiver.get("user_authorized") is True):
        return
    raise ValueError("Dataset is an unreviewed draft. Review annotations and verified video/case groups, "
                     "then use tools.promote_review_dataset before training, or explicitly authorize a separate "
                     "unreviewed development experiment. " + "; ".join(manifest.get("training_blockers", [])))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_training(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def seed_worker(_worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def capture_rng() -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {"algorithm": numpy_state[0],
                  "keys": torch.tensor(numpy_state[1].astype(np.int64)),
                  "position": numpy_state[2], "has_gauss": numpy_state[3], "gauss": numpy_state[4]},
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n["algorithm"], n["keys"].cpu().numpy().astype(np.uint32),
                         n["position"], n["has_gauss"], n["gauss"]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume requires the recorded CUDA device topology")
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def input_identity(manifest_path: Path, root: Path, manifest: dict) -> dict:
    """Hash actual training/validation bytes; test pixels are never read."""
    files = {}
    for split, rows in (("labeled", manifest["train"]["labeled"]),
                        ("unlabeled", manifest["train"]["unlabeled"]), ("val", manifest["val"])):
        for row in rows:
            for key in (("image",) if split == "unlabeled" else ("image", "mask")):
                path = (root / row[key]).resolve()
                if str(path) not in files:
                    files[str(path)] = sha256(path)
                expected = row.get(f"{key}_sha256")
                if expected and expected != files[str(path)]:
                    raise ValueError(f"Reviewed {key} changed: {path}")
    return {"manifest_sha256": sha256(manifest_path), "files": files,
            "test_pixels_read": False}


def runtime_identity(device: torch.device) -> dict:
    import PIL
    import segmentation_models_pytorch as smp
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": np.__version__, "pillow": PIL.__version__, "smp": smp.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine(),
            "cpu_threads": torch.get_num_threads(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def source_identity() -> dict:
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "ureter_semis").glob("*.py"))
    paths += sorted((root / "third_party/Endo-SemiS/src/models").rglob("*.py"))
    return {str(p.relative_to(root)): sha256(p) for p in paths}


def atomic_json(path: Path, payload: dict | list) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def atomic_checkpoint(path: Path, checkpoint: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temp)
    os.replace(temp, path)
