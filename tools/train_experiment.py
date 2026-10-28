"""Launch a frozen training configuration without bypassing data-review checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ureter_semis.trainer import train_endosemis


def run(config_path: str | Path, output: str, resume: str | None = None,
        stop_after_epoch: int | None = None, seed: int | None = None) -> Path:
    path = Path(config_path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported experiment configuration")
    args = dict(config["training"])
    if seed is not None:
        if seed not in config.get("replication_seeds", [args["seed"]]):
            raise ValueError("Seed is outside the frozen replication set")
        args["seed"] = seed
    manifest = Path(args["manifest_path"])
    args["manifest_path"] = str((path.parent / manifest).resolve()) if not manifest.is_absolute() else str(manifest)
    return train_endosemis(**args, output_dir=output, resume=resume, stop_after_epoch=stop_after_epoch)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--seed", type=int, help="one of the frozen replication seeds")
    args = parser.parse_args()
    print(run(args.config, args.output, args.resume, args.stop_after_epoch, args.seed))


if __name__ == "__main__":
    main()
