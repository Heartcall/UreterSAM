from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .data import MEAN, STD
from .model import build_model


COLORS = {1: (40, 40, 235), 2: (235, 210, 20)}  # BGR: stone red, fiber cyan
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CAMERA_REFERENCE_SIZE = (1422, 1066)
CAMERA_K_REFERENCE = np.array(
    [[668.49, 0.0, 711.0], [0.0, 659.57, 533.0], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)
CAMERA_DISTORTION = np.array([-0.13106, 0.04229, 0.0, 0.0, -0.00778], dtype=np.float32)


def _camera_matrix_for_shape(height: int, width: int) -> np.ndarray:
    scale_x = width / CAMERA_REFERENCE_SIZE[0]
    scale_y = height / CAMERA_REFERENCE_SIZE[1]
    matrix = CAMERA_K_REFERENCE.copy()
    matrix[0, 0] *= scale_x
    matrix[0, 2] *= scale_x
    matrix[1, 1] *= scale_y
    matrix[1, 2] *= scale_y
    return matrix


def _geometry_prediction(prediction: np.ndarray, frame_shape: tuple[int, int], camera_correction: bool) -> np.ndarray:
    height, width = frame_shape
    resized = cv2.resize(prediction.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    if not camera_correction:
        return resized
    matrix = _camera_matrix_for_shape(height, width)
    map_x, map_y = cv2.initUndistortRectifyMap(
        matrix, CAMERA_DISTORTION, None, matrix, (width, height), cv2.CV_32FC1
    )
    return cv2.remap(resized, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)


def _geometry_depth(depth: np.ndarray, frame_shape: tuple[int, int], camera_correction: bool) -> np.ndarray:
    height, width = frame_shape
    resized = cv2.resize(depth.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    if not camera_correction:
        return resized
    matrix = _camera_matrix_for_shape(height, width)
    map_x, map_y = cv2.initUndistortRectifyMap(
        matrix, CAMERA_DISTORTION, None, matrix, (width, height), cv2.CV_32FC1
    )
    return cv2.remap(resized, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def _preprocess(frame: np.ndarray, size: int, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(cv2.resize(frame, (size, size)), cv2.COLOR_BGR2RGB)
    array = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
    return torch.from_numpy((array - MEAN) / STD).unsqueeze(0).to(device)


def _overlay(frame: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    prediction = cv2.resize(prediction.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    result = frame.copy()
    for class_id, color in COLORS.items():
        region = prediction == class_id
        if region.any():
            layer = np.zeros_like(frame)
            layer[:] = color
            result[region] = cv2.addWeighted(frame, 0.45, layer, 0.55, 0)[region]
    cv2.putText(result, "stone", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS[1], 2)
    cv2.putText(result, "fiber", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS[2], 2)
    return result


def _largest_component(mask: np.ndarray) -> np.ndarray | None:
    binary = mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return None
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == label


def _fiber_geometry(mask: np.ndarray) -> dict | None:
    component = _largest_component(mask)
    if component is None or int(component.sum()) < 5:
        return None
    distances = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 5)
    values = distances[distances > 0] * 2.0
    if values.size == 0:
        return None
    # The upper half avoids edge pixels while staying robust to small mask defects.
    width_px = float(np.median(values[values >= np.percentile(values, 50)]))
    ys, xs = np.where(component)
    rect_width, rect_height = cv2.minAreaRect(np.column_stack((xs, ys)).astype(np.float32))[1]
    return {
        "component": component,
        "width_px": width_px,
        "length_px": float(max(rect_width, rect_height)),
    }


def _stone_components(mask: np.ndarray, min_area: int = 5) -> list[np.ndarray]:
    binary = mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components = [labels == label for label in range(1, count) if stats[label, cv2.CC_STAT_AREA] >= min_area]
    return sorted(components, key=lambda component: int(component.sum()), reverse=True)


def _stone_measurement(component: np.ndarray) -> dict:
    area_px = float(component.sum())
    ys, xs = np.where(component)
    points = np.column_stack((xs, ys)).astype(np.float32)
    rect_w, rect_h = cv2.minAreaRect(points)[1]
    equivalent_diameter_px = float(2.0 * np.sqrt(area_px / np.pi))
    moments = cv2.moments(component.astype(np.uint8))
    center_x = float(moments["m10"] / moments["m00"]) if moments["m00"] else float(xs.mean())
    center_y = float(moments["m01"] / moments["m00"]) if moments["m00"] else float(ys.mean())
    return {
        "area_px": area_px,
        "width_px": float(min(rect_w, rect_h)),
        "length_px": float(max(rect_w, rect_h)),
        "equivalent_diameter_px": equivalent_diameter_px,
        "center_x": center_x,
        "center_y": center_y,
    }


def _depth_median(depth: np.ndarray, mask: np.ndarray) -> float | None:
    values = depth[mask]
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else None


def _depth_value(depth: np.ndarray, point: tuple[float, float], radius: int = 1) -> float | None:
    """Read a robust local depth value around an image point."""
    height, width = depth.shape[:2]
    x, y = int(round(point[0])), int(round(point[1]))
    if not (0 <= x < width and 0 <= y < height):
        return None
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    values = depth[y0:y1, x0:x1]
    values = values[np.isfinite(values) & (values > 1e-6)]
    return float(np.median(values)) if values.size else None


def _backproject_depth_point(
    point: tuple[float, float], depth_value: float, shape: tuple[int, int]
) -> np.ndarray:
    """Backproject a pixel using inverse-depth-like output into relative 3-D coordinates."""
    height, width = shape
    matrix = _camera_matrix_for_shape(height, width)
    fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
    cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
    z = 1.0 / max(float(depth_value), 1e-6)
    x = (point[0] - cx) * z / fx
    y = (point[1] - cy) * z / fy
    return np.array([x, y, z], dtype=np.float32)


def _fiber_ray(component: np.ndarray) -> dict | None:
    """Estimate the laser ray from the fiber shaft toward its image-inner tip."""
    ys, xs = np.where(component)
    if len(xs) < 5:
        return None
    points = np.column_stack((xs, ys)).astype(np.float32)
    mean, eigenvectors = cv2.PCACompute(points, mean=None, maxComponents=1)
    axis = eigenvectors[0]
    projections = ((points - mean) @ axis).ravel()
    low = np.percentile(projections, 5)
    high = np.percentile(projections, 95)
    ends = []
    for target in (low, high):
        selected = points[np.abs(projections - target) <= max(1.0, np.std(projections) * 0.05)]
        ends.append(selected.mean(axis=0))
    height, width = component.shape
    border_distance = [
        min(float(end[0]), float(end[1]), float(width - 1 - end[0]), float(height - 1 - end[1]))
        for end in ends
    ]
    proximal_index = int(np.argmin(border_distance))
    tip_index = 1 - proximal_index
    proximal = ends[proximal_index]
    tip = ends[tip_index]
    direction = tip - proximal
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        return None
    direction = direction / norm
    return {
        "proximal": tuple(float(value) for value in proximal),
        "tip": tuple(float(value) for value in tip),
        "direction": tuple(float(value) for value in direction),
    }


def _laser_target(
    components: list[np.ndarray],
    fiber_ray: dict | None,
) -> tuple[int | None, tuple[float, float] | None, dict | None]:
    """Find the first stone reached by the 2-D laser ray."""
    if fiber_ray is None or not components:
        return None, None, None
    tip = np.asarray(fiber_ray["tip"], dtype=np.float32)
    direction = np.asarray(fiber_ray["direction"], dtype=np.float32)
    candidates = []
    for stone_id, component in enumerate(components, start=1):
        ys, xs = np.where(component)
        points = np.column_stack((xs, ys)).astype(np.float32)
        vectors = points - tip
        projections = vectors @ direction
        forward = projections >= 0
        if not np.any(forward):
            continue
        points = points[forward]
        projections = projections[forward]
        vectors = vectors[forward]
        perpendicular = np.abs(vectors[:, 0] * direction[1] - vectors[:, 1] * direction[0])
        hit_indices = np.flatnonzero(perpendicular <= 4.0)
        if hit_indices.size:
            index = hit_indices[np.argmin(projections[hit_indices])]
            candidates.append({
                "stone_id": stone_id,
                "hit": True,
                "distance_along_ray": float(projections[index]),
                "perpendicular_distance": float(perpendicular[index]),
                "point": tuple(float(value) for value in points[index]),
            })
        else:
            index = int(np.argmin(perpendicular))
            candidates.append({
                "stone_id": stone_id,
                "hit": False,
                "distance_along_ray": float(projections[index]),
                "perpendicular_distance": float(perpendicular[index]),
                "point": tuple(float(value) for value in points[index]),
            })
    if not candidates:
        return None, None, None
    hits = [candidate for candidate in candidates if candidate["hit"]]
    selected = min(hits or candidates, key=lambda candidate: candidate["distance_along_ray"])
    return selected["stone_id"], selected["point"], selected


def _laser_aim_from_depth(
    depth: np.ndarray,
    fiber_ray: dict | None,
    fiber_component: np.ndarray | None,
) -> tuple[tuple[float, float], np.ndarray] | None:
    """Estimate the laser's scene intersection without using a stone mask."""
    if fiber_ray is None or depth.ndim != 2:
        return None
    height, width = depth.shape
    tip = tuple(float(value) for value in fiber_ray["tip"])
    proximal = tuple(float(value) for value in fiber_ray["proximal"])
    tip_xy = (int(round(tip[0])), int(round(tip[1])))
    proximal_xy = (int(round(proximal[0])), int(round(proximal[1])))
    if not (0 <= tip_xy[0] < width and 0 <= tip_xy[1] < height):
        return None
    if not (0 <= proximal_xy[0] < width and 0 <= proximal_xy[1] < height):
        return None
    tip_depth = _depth_value(depth, tip)
    proximal_depth = _depth_value(depth, proximal)
    if tip_depth is None or proximal_depth is None:
        return None
    tip_3d = _backproject_depth_point(tip, tip_depth, depth.shape)
    proximal_3d = _backproject_depth_point(proximal, proximal_depth, depth.shape)
    direction_3d = tip_3d - proximal_3d
    norm = float(np.linalg.norm(direction_3d))
    if norm <= 1e-6:
        return None
    direction_3d /= norm

    direction_2d = np.asarray(fiber_ray["direction"], dtype=np.float32)
    start_distance = max(12.0, float(np.linalg.norm(tip_3d - proximal_3d)) * 10.0)
    samples = np.arange(start_distance, 2.0 * max(width, height), 1.0, dtype=np.float32)
    candidates = []
    for distance in samples:
        point = np.asarray(tip, dtype=np.float32) + direction_2d * distance
        x, y = int(round(point[0])), int(round(point[1]))
        if x < 0 or x >= width or y < 0 or y >= height:
            break
        if fiber_component is not None and fiber_component[y, x]:
            continue
        value = _depth_value(depth, (float(x), float(y)))
        if value is None:
            continue
        point_3d = _backproject_depth_point((float(x), float(y)), value, depth.shape)
        forward_distance = float(np.dot(point_3d - tip_3d, direction_3d))
        if forward_distance <= 0:
            continue
        line_distance = float(np.linalg.norm(np.cross(point_3d - tip_3d, direction_3d)))
        candidates.append((line_distance, forward_distance, (float(x), float(y))))
    if not candidates:
        return None
    # The closest 3-D point to the laser ray is the scene intersection estimate.
    selected = min(candidates, key=lambda item: (item[0], item[1]))
    point_2d = selected[2]
    point_depth = _depth_value(depth, point_2d)
    if point_depth is None:
        return None
    return point_2d, _backproject_depth_point(point_2d, point_depth, depth.shape)


def _laser_image_endpoint(fiber_ray: dict | None, shape: tuple[int, int]) -> tuple[float, float] | None:
    if fiber_ray is None:
        return None
    height, width = shape
    point = np.asarray(fiber_ray["tip"], dtype=np.float32)
    direction = np.asarray(fiber_ray["direction"], dtype=np.float32)
    distances = []
    for coordinate, component_size, direction_value in (
        (point[0], width, direction[0]),
        (point[1], height, direction[1]),
    ):
        if direction_value > 1e-6:
            distances.append((component_size - 1 - coordinate) / direction_value)
        elif direction_value < -1e-6:
            distances.append((0 - coordinate) / direction_value)
    positive = [distance for distance in distances if distance > 0]
    if not positive:
        return None
    endpoint = point + direction * min(positive)
    return tuple(float(value) for value in endpoint)


def _estimate_stone_sizes(
    prediction: np.ndarray,
    depth: np.ndarray | None,
    fiber_diameter_mm: float,
    fiber_protrusion_mm: float | None,
    depth_correction: bool,
) -> tuple[list[dict], tuple[float, float] | None, dict | None, float | None]:
    if depth is not None and depth.shape != prediction.shape:
        depth = cv2.resize(
            depth.astype(np.float32),
            (prediction.shape[1], prediction.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    fiber = _fiber_geometry(prediction == 2)
    components = _stone_components(prediction == 1)
    fiber_ray = _fiber_ray(fiber["component"]) if fiber is not None else None
    target_id, _, ray_target = _laser_target(components, fiber_ray)
    laser_aim_3d = None
    laser_aim_result = (
        _laser_aim_from_depth(depth, fiber_ray, fiber["component"] if fiber is not None else None)
        if depth is not None
        else _laser_image_endpoint(fiber_ray, prediction.shape)
    )
    if isinstance(laser_aim_result, tuple) and len(laser_aim_result) == 2 and isinstance(laser_aim_result[1], np.ndarray):
        laser_aim_point = laser_aim_result[0]
        laser_aim_3d = laser_aim_result[1]
    else:
        laser_aim_point = laser_aim_result
    if fiber is None or not components or fiber["width_px"] <= 0 or fiber["length_px"] <= 0:
        return [], laser_aim_point, fiber_ray, None
    if target_id is None:
        return [], laser_aim_point, fiber_ray, None

    tip_to_laser_mm = None
    if depth is not None and laser_aim_3d is not None and fiber_ray is not None:
        tip_depth = _depth_value(depth, fiber_ray["tip"])
        if tip_depth is not None:
            tip_3d = _backproject_depth_point(fiber_ray["tip"], tip_depth, depth.shape)
            relative_distance = float(np.linalg.norm(laser_aim_3d - tip_3d))
            matrix = _camera_matrix_for_shape(depth.shape[0], depth.shape[1])
            # Convert relative inverse-depth coordinates to mm using the known fiber diameter.
            metric_scale = fiber_diameter_mm * float(matrix[0, 0]) * float(tip_depth) / fiber["width_px"]
            tip_to_laser_mm = relative_distance * metric_scale

    scale_from_diameter = fiber_diameter_mm / fiber["width_px"]
    scale_from_length = fiber_protrusion_mm / fiber["length_px"] if fiber_protrusion_mm is not None else None
    # When no protrusion length is supplied, use only the known tip diameter.
    scale_mm_per_px = (
        scale_from_diameter
        if scale_from_length is None
        else 0.7 * scale_from_diameter + 0.3 * scale_from_length
    )
    fiber_depth = _depth_median(depth, fiber["component"]) if depth is not None else None
    component = components[target_id - 1]
    stone = _stone_measurement(component)
    stone_depth = _depth_median(depth, component) if depth is not None else None
    correction = 1.0
    if depth_correction and fiber_depth is not None and stone_depth is not None and stone_depth > 1e-6:
        # Depth Anything's raw prediction is inverse-depth-like: larger means closer.
        correction = float(np.clip(fiber_depth / stone_depth, 0.5, 2.0))
    for key in ("width_px", "length_px", "equivalent_diameter_px"):
        stone[key.replace("_px", "_mm")] = stone[key] * scale_mm_per_px * correction
    # Approximate the unobserved third axis by the segmented short axis.
    stone["volume_mm3"] = float(
        np.pi / 6.0 * stone["length_mm"] * stone["width_mm"] * stone["width_mm"]
    )
    stone.update({
        "stone_id": target_id,
        "fiber_width_px": fiber["width_px"],
        "fiber_length_px": fiber["length_px"],
        "fiber_length_mm": fiber_protrusion_mm,
        "scale_from_diameter_mm_per_px": scale_from_diameter,
        "scale_from_length_mm_per_px": scale_from_length,
        "scale_mm_per_px": scale_mm_per_px,
        "depth_correction": correction,
        "fiber_depth": fiber_depth,
        "stone_depth": stone_depth,
        "laser_hit_point": laser_aim_point,
        "fiber_tip_to_laser_distance_mm": tip_to_laser_mm,
        "ray_hit": bool(ray_target and ray_target["hit"]),
    })
    return [stone], laser_aim_point, fiber_ray, tip_to_laser_mm


def _draw_measurements(
    frame: np.ndarray,
    prediction_shape: tuple[int, int],
    measurements: list[dict],
    laser_hit_point: tuple[float, float] | None,
    fiber_tip_to_laser_distance_mm: float | None = None,
    fiber_ray: dict | None = None,
) -> np.ndarray:
    result = frame.copy()
    pred_height, pred_width = prediction_shape
    height, width = frame.shape[:2]
    sx, sy = width / pred_width, height / pred_height
    for measurement in measurements:
        center = (int(round(measurement["center_x"] * sx)), int(round(measurement["center_y"] * sy)))
        cv2.circle(result, center, 5, (0, 255, 255), -1)
        #label = (
        #    f"#{measurement['stone_id']} "
        #    f"{measurement['width_mm']:.2f}x{measurement['length_mm']:.2f} mm"
        #)
        
        if "width_mm" in measurement and "length_mm" in measurement:
            label = (
                f"#{measurement['stone_id']} "
                f"{measurement['width_mm']:.2f}x"
                f"{measurement['length_mm']:.2f} mm"
            )
        else:
            label = (
                f"#{measurement['stone_id']} "
                f"{measurement['width_px']:.1f}x"
                f"{measurement['length_px']:.1f} px"
            )

        text_x = min(max(center[0] + 8, 5), max(5, width - 190))
        text_y = min(max(center[1] - 8, 18), height - 8)
        cv2.putText(result, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    if laser_hit_point is not None:
        hit = (int(round(laser_hit_point[0] * sx)), int(round(laser_hit_point[1] * sy)))
        tip = None
        if fiber_ray is not None and fiber_ray.get("tip") is not None:
            tip = (
                int(round(float(fiber_ray["tip"][0]) * sx)),
                int(round(float(fiber_ray["tip"][1]) * sy)),
            )
        if tip is not None:
            # BGR light yellow: visible over the endoscopic background without obscuring masks.
            line_color = (180, 220, 255)
            segment_count = max(1, int(np.linalg.norm(np.asarray(hit) - np.asarray(tip)) / 10.0))
            for index in range(segment_count):
                start_ratio = index / segment_count
                end_ratio = min(1.0, (index + 0.55) / segment_count)
                start = tuple(int(round(tip[axis] + (hit[axis] - tip[axis]) * start_ratio)) for axis in (0, 1))
                end = tuple(int(round(tip[axis] + (hit[axis] - tip[axis]) * end_ratio)) for axis in (0, 1))
                cv2.line(result, start, end, line_color, 2, cv2.LINE_AA)
            if fiber_tip_to_laser_distance_mm is not None:
                midpoint = (
                    int(round((tip[0] + hit[0]) / 2)),
                    int(round((tip[1] + hit[1]) / 2)),
                )
                distance_label = f"{fiber_tip_to_laser_distance_mm:.2f} mm"
                cv2.putText(
                    result,
                    distance_label,
                    (midpoint[0] + 6, max(18, midpoint[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    line_color,
                    1,
                    cv2.LINE_AA,
                )
        cv2.drawMarker(result, hit, (0, 255, 0), cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
        cv2.putText(result, "laser hit", (hit[0] + 8, max(18, hit[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    return result


def run_demo(
    checkpoint_path: str,
    source: str,
    output_path: str,
    show: bool = False,
    device_name: str = "cuda:0",
    estimate_size: bool = False,
    depth_model_size: str = "small",
    fiber_diameter_mm: float = 0.4,
    fiber_protrusion_mm: float = 4.0,
    depth_correction: bool = False,
    measurements_path: str | None = None,
    camera_correction: bool = True,
) -> Path:
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    size = int(checkpoint.get("size", 320))
    models = (build_model(pretrained=False).to(device), build_model(pretrained=False).to(device))
    models[0].load_state_dict(checkpoint["model1"])
    models[1].load_state_dict(checkpoint["model2"])
    for model in models:
        model.eval()

    depth_estimator = None
    if estimate_size:
        from .depth_anything_v2_infer import DepthAnythingV2

        depth_estimator = DepthAnythingV2(model_size=depth_model_size, device=str(device))

    source_value: str | int = int(source) if source.isdecimal() else source
    capture = cv2.VideoCapture(source_value)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    measurement_file = None
    if measurements_path:
        measurement_file = Path(measurements_path).expanduser().resolve()
        measurement_file.parent.mkdir(parents=True, exist_ok=True)
        measurement_file = measurement_file.open("w", encoding="utf-8")

    with torch.no_grad():
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            tensor = _preprocess(frame, size, device)
            logits = (models[0](tensor) + models[1](tensor)) / 2
            probabilities = torch.softmax(logits, dim=1)
            _, prediction = probabilities.max(dim=1)
            prediction = prediction.squeeze(0).cpu().numpy()
            rendered = _overlay(frame, prediction)
            geometry_prediction = _geometry_prediction(prediction, frame.shape[:2], camera_correction)
            measurements = []
            fiber_geometry = _fiber_geometry(geometry_prediction == 2)
            fiber_ray = _fiber_ray(fiber_geometry["component"]) if fiber_geometry is not None else None
            components = _stone_components(geometry_prediction == 1)
            # target_id, _, _ = _laser_target(components, fiber_ray)
            # laser_hit_point = _laser_image_endpoint(fiber_ray, geometry_prediction.shape)
            target_id, target_point, ray_target = _laser_target(
                components, fiber_ray
            )

            laser_hit_point = (
                target_point
                if ray_target is not None and ray_target.get("hit", False)
            else None
            )
            fiber_tip_to_laser_distance_mm = None
            if depth_estimator is not None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                depth = _geometry_depth(depth_estimator.infer(rgb_frame), frame.shape[:2], camera_correction)
                measurements, laser_hit_point, fiber_ray, fiber_tip_to_laser_distance_mm = _estimate_stone_sizes(
                    geometry_prediction, depth, fiber_diameter_mm, fiber_protrusion_mm, depth_correction
                )
                
            # elif target_id is not None:
            #    measurements = [dict(_stone_measurement(components[target_id - 1]), stone_id=target_id)]

            # else:
            #    measurements, laser_hit_point, fiber_ray, fiber_tip_to_laser_distance_mm = _estimate_stone_sizes(
            #        geometry_prediction,
            #        None,
            #        fiber_diameter_mm,
            #        fiber_protrusion_mm,
            #        False,
            #    )

            elif (
                target_id is not None
                and ray_target is not None
                and ray_target.get("hit", False)
            ):
                measurements = [
                    dict(
                    _stone_measurement(components[target_id - 1]),
                    stone_id=target_id,
                    ray_hit=True,
                    )
                ]
            
            rendered = _draw_measurements(
                rendered,
                geometry_prediction.shape,
                measurements,
                laser_hit_point,
                fiber_tip_to_laser_distance_mm,
                fiber_ray,
            )
            if measurement_file is not None:
                measurement_file.write(json.dumps({
                    "frame": frame_index,
                    "measurements": measurements,
                    "laser_hit_point": laser_hit_point,
                    "fiber_tip_to_laser_distance_mm": fiber_tip_to_laser_distance_mm,
                    "fiber_ray": fiber_ray,
                }, ensure_ascii=False) + "\n")
            writer.write(rendered)
            if show:
                cv2.imshow("Ureter Endo-SemiS", rendered)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            frame_index += 1
    capture.release()
    writer.release()
    if measurement_file is not None:
        measurement_file.close()
    if show:
        cv2.destroyAllWindows()
    return output


def run_image_measurement(
    checkpoint_path: str,
    source: str,
    output_dir: str,
    device_name: str = "cuda:0",
    depth_model_size: str = "small",
    fiber_diameter_mm: float = 0.4,
    measurements_path: str | None = None,
    camera_correction: bool = True,
) -> Path:
    """Measure JPG/PNG images using only the known fiber tip diameter."""
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    size = int(checkpoint.get("size", 320))
    models = (build_model(pretrained=False).to(device), build_model(pretrained=False).to(device))
    models[0].load_state_dict(checkpoint["model1"])
    models[1].load_state_dict(checkpoint["model2"])
    for model in models:
        model.eval()

    from .depth_anything_v2_infer import DepthAnythingV2

    depth_estimator = DepthAnythingV2(model_size=depth_model_size, device=str(device))
    source_path = Path(source).expanduser()
    if source_path.is_file():
        inputs = [source_path]
    else:
        inputs = sorted(
            path for path in source_path.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        )
    if not inputs:
        raise RuntimeError(f"No JPG/PNG images found in: {source_path}")

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    measurements_file = Path(measurements_path).expanduser().resolve() if measurements_path else output / "measurements.jsonl"
    measurements_file.parent.mkdir(parents=True, exist_ok=True)

    with measurements_file.open("w", encoding="utf-8") as handle, torch.no_grad():
        for image_path in inputs:
            frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Cannot read image: {image_path}")
            tensor = _preprocess(frame, size, device)
            logits = (models[0](tensor) + models[1](tensor)) / 2
            _, prediction = torch.softmax(logits, dim=1).max(dim=1)
            prediction = prediction.squeeze(0).cpu().numpy()
            depth = _geometry_depth(
                depth_estimator.infer(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                frame.shape[:2],
                camera_correction,
            )
            geometry_prediction = _geometry_prediction(prediction, frame.shape[:2], camera_correction)
            measurements, laser_hit_point, fiber_ray, fiber_tip_to_laser_distance_mm = _estimate_stone_sizes(
                geometry_prediction,
                depth,
                fiber_diameter_mm,
                fiber_protrusion_mm=None,
                depth_correction=True,
            )
            rendered = _overlay(frame, prediction)
            rendered = _draw_measurements(
                rendered,
                geometry_prediction.shape,
                measurements,
                laser_hit_point,
                fiber_tip_to_laser_distance_mm,
                fiber_ray,
            )
            output_image = output / image_path.name
            cv2.imwrite(str(output_image), rendered)
            handle.write(json.dumps({
                "image": str(image_path),
                "output": str(output_image),
                "fiber_diameter_mm": fiber_diameter_mm,
                "fiber_length_used": False,
                "measurements": measurements,
                "laser_hit_point": laser_hit_point,
                "fiber_tip_to_laser_distance_mm": fiber_tip_to_laser_distance_mm,
                "fiber_ray": fiber_ray,
            }, ensure_ascii=False) + "\n")
            print(f"{image_path} -> {output_image}")
    return output
