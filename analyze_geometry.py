
"""Geometric analysis for UreterSAM prediction masks."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ureter_semis.data import load_manifest
from ureter_semis.demo import (
    _fiber_geometry,
    _fiber_ray,
    _laser_image_endpoint,
    _laser_target,
    _stone_components,
    _stone_measurement,
)

"""
可视化标记说明

红色：结石分割区域和轮廓

青色：光纤分割区域和轴线

黄色：估计光纤末端及方向延长线

绿色：二维方向与结石的候选相交位置
"""

STONE_COLOR = (40, 40, 235)   # BGR red
FIBER_COLOR = (235, 210, 20)  # BGR cyan


def to_point(p):
    return (
        int(round(float(p[0]))),
        int(round(float(p[1]))),
    )


def make_overlay(image_path, mask):
    """Overlay predicted mask onto the original image."""
    h, w = mask.shape

    image = Image.open(image_path).convert("RGB")
    image = image.resize(
        (w, h),
        Image.Resampling.BILINEAR,
    )

    canvas = cv2.cvtColor(
        np.asarray(image),
        cv2.COLOR_RGB2BGR,
    ).copy()

    for class_id, color in [
        (1, STONE_COLOR),
        (2, FIBER_COLOR),
    ]:
        region = mask == class_id

        if region.any():
            canvas[region] = (
                canvas[region].astype(np.float32) * 0.55
                + np.asarray(color) * 0.45
            ).astype(np.uint8)

    return canvas


def draw_geometry(canvas, mask, stones, fiber, ray, target):
    """Draw stone contours, centers, and fiber direction."""

    # Stone measurements
    for stone in stones:
        component = stone["_component"]

        contours, _ = cv2.findContours(
            component.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        cv2.drawContours(
            canvas, contours, -1, STONE_COLOR, 1
        )

        center = to_point((
            stone["center_x"],
            stone["center_y"],
        ))

        cv2.drawMarker(
            canvas,
            center,
            (255, 255, 255),
            cv2.MARKER_CROSS,
            10,
            1,
        )

        cv2.putText(
            canvas,
            f"S{stone['stone_id']}",
            (center[0] + 5, center[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Fiber geometry
    if fiber is not None:
        cv2.putText(
            canvas,
            f"Fiber width: {fiber['width_px']:.1f} px",
            (7, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            FIBER_COLOR,
            1,
            cv2.LINE_AA,
        )

    # Estimated fiber direction
    if ray is not None:
        proximal = to_point(ray["proximal"])
        tip = to_point(ray["tip"])

        cv2.line(
            canvas,
            proximal,
            tip,
            FIBER_COLOR,
            2,
            cv2.LINE_AA,
        )

        cv2.circle(
            canvas,
            tip,
            4,
            (0, 255, 255),
            -1,
        )

        # Extend the 2-D direction to the image boundary
        endpoint = _laser_image_endpoint(
            ray, mask.shape
        )

        if endpoint is not None:
            cv2.line(
                canvas,
                tip,
                to_point(endpoint),
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    # Mark only candidates actually aligned with the 2-D ray
    if target is not None and target["hit"]:
        hit = to_point(target["point"])

        cv2.drawMarker(
            canvas,
            hit,
            (0, 255, 0),
            cv2.MARKER_CROSS,
            14,
            2,
        )

        cv2.putText(
            canvas,
            "2D candidate",
            (hit[0] + 4, max(12, hit[1] - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return canvas


def main():
    parser = argparse.ArgumentParser(
        description="UreterSAM geometric analysis"
    )

    parser.add_argument(
        "--manifest",
        default="datasets/ureter_semis/manifest.json",
    )

    parser.add_argument(
        "--split",
        choices=["test", "val"],
        default="test",
    )

    parser.add_argument(
        "--pred-masks",
        default="runs/vis_test/pred_masks",
    )

    parser.add_argument(
        "--output",
        default="runs/geometry_test",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--min-stone-area",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--fiber-diameter-mm",
        type=float,
        default=None,
        help="Optional physical fiber diameter",
    )

    args = parser.parse_args()

    if (
        args.fiber_diameter_mm is not None
        and args.fiber_diameter_mm <= 0
    ):
        parser.error(
            "--fiber-diameter-mm must be positive"
        )

    if args.min_stone_area < 1:
        parser.error(
            "--min-stone-area must be >= 1"
        )

    # Load dataset paths
    root, manifest = load_manifest(args.manifest)
    samples = manifest[args.split]

    if args.limit > 0:
        samples = samples[:args.limit]

    mask_dir = Path(args.pred_masks)
    output = Path(args.output)
    overlay_dir = output / "overlays"

    if not mask_dir.is_dir():
        raise FileNotFoundError(
            f"Prediction masks not found: {mask_dir}"
        )

    overlay_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_rows = []
    image_records = []
    missing = []

    # Analyze every predicted mask
    for index, sample in enumerate(samples, 1):

        image_path = root / sample["image"]
        name = image_path.stem

        mask_path = mask_dir / f"{name}.png"

        if not mask_path.exists():
            missing.append(mask_path.name)
            continue

        mask = np.asarray(
            Image.open(mask_path).convert("L"),
            dtype=np.uint8,
        )

        values = set(np.unique(mask).tolist())

        if not values.issubset({0, 1, 2}):
            raise ValueError(
                f"Invalid mask classes: {mask_path}"
            )

        # 1. Stone connected components
        components = _stone_components(
            mask == 1,
            min_area=args.min_stone_area,
        )

        # 2. Fiber width and length
        fiber = _fiber_geometry(mask == 2)

        # 3. Fiber PCA direction
        ray = (
            _fiber_ray(fiber["component"])
            if fiber is not None
            else None
        )

        # 4. Candidate intersection with stone
        _, _, candidate = _laser_target(
            components,
            ray,
        )

        # This is only a 2-D candidate, not a physical impact.

        # 5. Optional approximate mm-per-pixel scale
        scale = None

        if (
            fiber is not None
            and args.fiber_diameter_mm is not None
            and fiber["width_px"] > 0
        ):
            scale = (
                args.fiber_diameter_mm
                / fiber["width_px"]
            )

        # 6. Analyze individual stones
        stones = []

        for stone_id, component in enumerate(
            components, start=1
        ):

            geometry = _stone_measurement(component)

            stone = {
                "stone_id": stone_id,
                **geometry,
                "_component": component,
            }

            if scale is not None:
                stone["width_mm_approx"] = (
                    stone["width_px"] * scale
                )

                stone["length_mm_approx"] = (
                    stone["length_px"] * scale
                )

                stone["area_mm2_approx"] = (
                    stone["area_px"] * scale ** 2
                )

                stone["equivalent_diameter_mm_approx"] = (
                    stone["equivalent_diameter_px"] * scale
                )

            stones.append(stone)

        # 7. Draw geometry on the image
        canvas = make_overlay(
            image_path,
            mask,
        )

        canvas = draw_geometry(
            canvas,
            mask,
            stones,
            fiber,
            ray,
            candidate,
        )

        output_image = (
            overlay_dir / f"{name}.png"
        )

        # Unicode-compatible PNG saving
        ok, buffer = cv2.imencode(
            ".png",
            canvas,
        )

        if not ok:
            raise RuntimeError(
                f"Cannot encode image: {output_image}"
            )

        buffer.tofile(str(output_image))

        # Remove temporary numpy masks for JSON storage
        for stone in stones:
            stone.pop("_component")

        fiber_record = None

        if fiber is not None:
            fiber_record = {
                "width_px": fiber["width_px"],
                "length_px": fiber["length_px"],
                "proximal_xy": (
                    list(ray["proximal"]) if ray else None
                ),
                "tip_xy": (
                    list(ray["tip"]) if ray else None
                ),
                "direction_xy": (
                    list(ray["direction"]) if ray else None
                ),
                "scale_mm_per_px_approx": scale,
            }

        target_record = None

        if candidate is not None:
            target_record = {
                "stone_id": candidate["stone_id"],
                "is_aligned_2d": bool(
                    candidate["hit"]
                ),
                "point_xy": list(
                    candidate["point"]
                ),
                "perpendicular_distance_px": (
                    candidate["perpendicular_distance"]
                ),
                "distance_along_ray_px": (
                    candidate["distance_along_ray"]
                ),
            }

        record = {
            "image": sample["image"],
            "mask_resolution": [
                int(mask.shape[1]),
                int(mask.shape[0]),
            ],
            "stone_count": len(stones),
            "stones": stones,
            "fiber": fiber_record,
            "laser_2d_candidate": target_record,
            "overlay": str(output_image),
        }

        image_records.append(record)

        # Each CSV row represents one stone.
        # Images without stones get an empty stone row.
        for stone in stones or [None]:
            csv_rows.append({
                "image": image_path.name,
                "stone_id": (
                    stone["stone_id"] if stone else None
                ),
                "area_px": (
                    stone["area_px"] if stone else None
                ),
                "width_px": (
                    stone["width_px"] if stone else None
                ),
                "length_px": (
                    stone["length_px"] if stone else None
                ),
                "center_x": (
                    stone["center_x"] if stone else None
                ),
                "center_y": (
                    stone["center_y"] if stone else None
                ),
                "fiber_width_px": (
                    fiber["width_px"] if fiber else None
                ),
                "fiber_length_px": (
                    fiber["length_px"] if fiber else None
                ),
                "width_mm_approx": (
                    stone.get("width_mm_approx")
                    if stone else None
                ),
                "length_mm_approx": (
                    stone.get("length_mm_approx")
                    if stone else None
                ),
                "area_mm2_approx": (
                    stone.get("area_mm2_approx")
                    if stone else None
                ),
                "aligned_2d_stone_id": (
                    candidate["stone_id"]
                    if candidate and candidate["hit"]
                    else None
                ),
            })

        print(
            f"[{index}/{len(samples)}] {name}: "
            f"stones={len(stones)}, "
            f"fiber={'yes' if fiber else 'no'}, "
            f"2D-alignment="
            f"{'yes' if candidate and candidate['hit'] else 'no'}"
        )

    if not image_records:
        raise RuntimeError(
            "No matching prediction masks found"
        )

    # 8. Save CSV
    with (
        output / "measurements.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(csv_rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(csv_rows)

    # 9. Save JSONL
    with (
        output / "measurements.jsonl"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        for record in image_records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                ) + "\n"
            )

    # 10. Generate summary
    summary = {
        "split": args.split,
        "images_processed": len(image_records),
        "masks_missing": len(missing),
        "images_with_stone": sum(
            r["stone_count"] > 0
            for r in image_records
        ),
        "images_with_fiber": sum(
            r["fiber"] is not None
            for r in image_records
        ),
        "stone_components_total": sum(
            r["stone_count"]
            for r in image_records
        ),
        "images_with_2d_alignment_candidate": sum(
            bool(
                r["laser_2d_candidate"]
                and r["laser_2d_candidate"][
                    "is_aligned_2d"
                ]
            )
            for r in image_records
        ),
        "fiber_diameter_mm_assumed": (
            args.fiber_diameter_mm
        ),
        "note": (
            "All pixel geometry uses prediction-mask "
            "resolution. Approximate mm measurements "
            "assume the stone and fiber are at similar "
            "depths. A 2-D ray candidate is NOT a "
            "validated physical laser impact."
        ),
        "missing_mask_names": missing[:50],
    }

    (
        output / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== Geometry Summary ===")

    for key, value in summary.items():
        if key != "missing_mask_names":
            print(f"{key}: {value}")

    print(f"Results: {output.resolve()}")


if __name__ == "__main__":
    main()
