"""Endo-SemiS demo for ureteroscopic stone and laser-fiber segmentation."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Endo-SemiS: stone + fiber segmentation")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="convert YOLO/COCO polygons to semantic masks")
    prepare.add_argument("--source", required=True, nargs="+", help="one or more extracted dataset directories")
    prepare.add_argument("--output", default="datasets/ureter_semis")
    prepare.add_argument("--format", choices=("auto", "yolo", "coco"), default="auto")
    prepare.add_argument("--labeled-fraction", type=float, default=0.2)
    prepare.add_argument("--seed", type=int, default=42)

    train = commands.add_parser("train", help="train the two-model Endo-SemiS adaptation")
    train.add_argument("--manifest", required=True)
    train.add_argument("--output", default="runs/endosemis")
    train.add_argument("--epochs", type=int, default=60)
    train.add_argument("--size", type=int, default=256)
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--workers", type=int, default=0)
    train.add_argument("--no-pretrained", action="store_true", help="skip ImageNet weights (mainly for offline checks)")
    train.add_argument("--class-weights", type=float, nargs=3, default=(1.0, 2.0, 4.0), metavar=("BG", "STONE", "FIBER"))
    train.add_argument("--confidence-threshold", type=float, default=0.85)
    train.add_argument("--uncertainty-quantile", type=float, default=0.8)
    train.add_argument("--dice-weight", type=float, default=1.0)

    demo = commands.add_parser("demo", help="render stone and fiber masks on a video")
    demo.add_argument("--checkpoint", required=True)
    demo.add_argument("--source", required=True, help="video path or camera index, such as 0")
    demo.add_argument("--output", default="runs/demo.mp4")
    demo.add_argument("--device", default="cuda:0")
    demo.add_argument("--show", action="store_true")
    demo.add_argument("--estimate-size", action="store_true", help="estimate stone size from the 0.4 mm fiber mask")
    demo.add_argument("--depth-model-size", choices=("small", "base", "large"), default="small")
    demo.add_argument("--fiber-diameter-mm", type=float, default=0.4)
    demo.add_argument("--fiber-protrusion-mm", type=float, default=4.0, help="visible fiber length used as a second scale reference")
    demo.add_argument("--depth-correction", action="store_true", help="use relative depth to correct scale between fiber and stone")
    demo.add_argument("--measurements", help="optional JSONL path for per-frame stone measurements")
    demo.add_argument("--no-camera-correction", action="store_true", help="disable camera intrinsics and distortion correction")

    measure = commands.add_parser("measure-images", help="estimate stone sizes in JPG/PNG images")
    measure.add_argument("--checkpoint", required=True)
    measure.add_argument("--source", required=True, help="one image or a directory; directories are read non-recursively")
    measure.add_argument("--output", default="runs/image_measurements")
    measure.add_argument("--device", default="cuda:0")
    measure.add_argument("--depth-model-size", choices=("small", "base", "large"), default="small")
    measure.add_argument("--fiber-diameter-mm", type=float, default=0.4)
    measure.add_argument("--measurements", help="optional JSONL output path")
    measure.add_argument("--no-camera-correction", action="store_true", help="disable camera intrinsics and distortion correction")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        from tools.convert_dataset import prepare_dataset

        manifest = prepare_dataset(args.source, args.output, args.format, args.labeled_fraction, args.seed)
        print(f"Manifest written to: {manifest}")
    elif args.command == "train":
        from ureter_semis.trainer import train_endosemis

        checkpoint = train_endosemis(
            manifest_path=args.manifest,
            output_dir=args.output,
            epochs=args.epochs,
            size=args.size,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device_name=args.device,
            workers=args.workers,
            pretrained=not args.no_pretrained,
            class_weights=tuple(args.class_weights),
            confidence_threshold=args.confidence_threshold,
            uncertainty_quantile=args.uncertainty_quantile,
            dice_weight=args.dice_weight,
        )
        print(f"Best checkpoint: {checkpoint}")
    elif args.command == "demo":
        from ureter_semis.demo import run_demo

        output = run_demo(
            args.checkpoint,
            args.source,
            args.output,
            args.show,
            args.device,
            estimate_size=args.estimate_size,
            depth_model_size=args.depth_model_size,
            fiber_diameter_mm=args.fiber_diameter_mm,
            fiber_protrusion_mm=args.fiber_protrusion_mm,
            depth_correction=args.depth_correction,
            measurements_path=args.measurements,
            camera_correction=not args.no_camera_correction,
        )
        print(f"Demo written to: {output}")
    else:
        from ureter_semis.demo import run_image_measurement

        output = run_image_measurement(
            args.checkpoint,
            args.source,
            args.output,
            args.device,
            args.depth_model_size,
            args.fiber_diameter_mm,
            args.measurements,
            camera_correction=not args.no_camera_correction,
        )
        print(f"Image measurements written to: {output}")


if __name__ == "__main__":
    main()
