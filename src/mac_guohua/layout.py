from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon

from .schemas import LayoutPlan, Region


@dataclass
class RasterizedLayout:
    semantic_image: Image.Image
    class_map: np.ndarray
    region_map: np.ndarray
    region_masks: dict[str, Image.Image]
    blank_ratio: float
    overlap_penalty: float


def _pixel_polygon(
    region: Region,
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for x, y in region.polygon:
        px = int(round(x * (width - 1)))
        py = int(round(y * (height - 1)))
        points.append((px, py))
    return points


def _normalized_polygon(region: Region) -> Polygon:
    poly = Polygon([(float(x), float(y)) for x, y in region.polygon])
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def calculate_overlap_penalty(
    regions: Iterable[Region],
    blank_labels: set[str],
) -> float:
    object_regions = [
        r for r in regions if r.label.lower() not in blank_labels
    ]

    penalty = 0.0

    for i in range(len(object_regions)):
        p1 = _normalized_polygon(object_regions[i])
        if p1.is_empty:
            continue

        for j in range(i + 1, len(object_regions)):
            p2 = _normalized_polygon(object_regions[j])
            if p2.is_empty:
                continue

            intersection = p1.intersection(p2).area
            if intersection > 0:
                penalty += float(intersection)

    return min(1.0, penalty)


def rasterize_layout(
    layout: LayoutPlan,
    width: int,
    height: int,
    palette: dict[str, list[int]],
    blank_labels: set[str],
) -> RasterizedLayout:
    labels = list(palette.keys())
    if "unknown" not in labels:
        labels.append("unknown")
        palette = dict(palette)
        palette["unknown"] = [128, 128, 128]

    label_to_id = {label: i for i, label in enumerate(labels)}

    default_label = "blank" if "blank" in label_to_id else "unknown"
    class_map = np.full(
        (height, width),
        label_to_id[default_label],
        dtype=np.uint8,
    )
    region_map = np.full((height, width), -1, dtype=np.int32)

    sorted_regions = sorted(
        enumerate(layout.regions),
        key=lambda item: (item[1].depth_order, item[0]),
    )

    for original_index, region in sorted_regions:
        polygon = _pixel_polygon(region, width, height)
        mask_image = Image.new("L", (width, height), 0)
        ImageDraw.Draw(mask_image).polygon(polygon, fill=255)

        mask = np.asarray(mask_image) > 0
        label = region.label.lower()
        class_id = label_to_id.get(label, label_to_id["unknown"])

        class_map[mask] = class_id
        region_map[mask] = original_index

    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for label, class_id in label_to_id.items():
        rgb[class_map == class_id] = np.asarray(
            palette.get(label, palette["unknown"]),
            dtype=np.uint8,
        )

    region_masks: dict[str, Image.Image] = {}
    for index, region in enumerate(layout.regions):
        mask = (region_map == index).astype(np.uint8) * 255
        region_masks[region.id] = Image.fromarray(mask, mode="L")

    blank_ids = {
        label_to_id[label]
        for label in blank_labels
        if label in label_to_id
    }
    blank_mask = np.isin(class_map, list(blank_ids))
    blank_ratio = float(blank_mask.mean())

    overlap_penalty = calculate_overlap_penalty(
        layout.regions,
        blank_labels,
    )

    return RasterizedLayout(
        semantic_image=Image.fromarray(rgb, mode="RGB"),
        class_map=class_map,
        region_map=region_map,
        region_masks=region_masks,
        blank_ratio=blank_ratio,
        overlap_penalty=overlap_penalty,
    )


def layout_score(
    blank_ratio: float,
    target: float,
    overlap_penalty: float,
    overlap_lambda: float,
) -> float:
    return (
        -abs(blank_ratio - target)
        - overlap_lambda * overlap_penalty
    )


def image_negative_space_ratio(
    image: Image.Image,
    threshold: float = 0.90,
) -> float:
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return float((gray > threshold).mean())