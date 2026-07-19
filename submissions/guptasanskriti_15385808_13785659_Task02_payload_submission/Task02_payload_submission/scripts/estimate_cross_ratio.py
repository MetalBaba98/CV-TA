#!/usr/bin/env python3
"""Estimate distance along a road line using single-view cross-ratio.

The config JSON contains image points A', B'/box, C', V':
  A' = reference start, here the red MB line on the chosen blue road line
  B' = object point, usually bottom-center of the detection box
  C' = known reference end, here the arch on the same road line
  V' = vanishing point of the road direction

Real-world setup:
  A = 0 m
  C = --reference-distance-m metres from A, measured in Google Maps
  V = point at infinity
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="JSON file with points")
    parser.add_argument("--reference-distance-m", type=float, default=None)
    parser.add_argument("--annotated-out", type=Path, default=Path("outputs/cross_ratio_annotation.png"))
    return parser.parse_args()


def pt(value: Iterable[float]) -> Point:
    x, y = value
    return float(x), float(y)


def scalar_on_line(p: Point, a: Point, v: Point) -> float:
    """Project p onto the directed A'->V' image line and return a 1D coordinate."""
    p_arr = np.array(p, dtype=float)
    a_arr = np.array(a, dtype=float)
    v_arr = np.array(v, dtype=float)
    direction = v_arr - a_arr
    length = np.linalg.norm(direction)
    if length == 0:
        raise ValueError("A' and V' cannot be the same point")
    unit = direction / length
    return float(np.dot(p_arr - a_arr, unit))


def cross_ratio(a: float, b: float, c: float, v: float) -> float:
    """CR(A,B;C,V) using signed 1D coordinates."""
    return ((c - a) / (c - b)) / ((v - a) / (v - b))


def estimate_ab_m(a: Point, b: Point, c: Point, v: Point, ac_m: float) -> dict[str, float]:
    a1 = scalar_on_line(a, a, v)
    b1 = scalar_on_line(b, a, v)
    c1 = scalar_on_line(c, a, v)
    v1 = scalar_on_line(v, a, v)
    cr = cross_ratio(a1, b1, c1, v1)
    ab_m = ac_m * (1.0 - 1.0 / cr)
    bc_m = ac_m - ab_m
    return {
        "a_scalar": a1,
        "b_scalar": b1,
        "c_scalar": c1,
        "v_scalar": v1,
        "cross_ratio": cr,
        "distance_A_to_B_m": ab_m,
        "distance_B_to_C_m": bc_m,
    }


def bbox_points(xyxy: list[float]) -> dict[str, Point]:
    x1, y1, x2, y2 = xyxy
    return {
        "bottom_center": ((x1 + x2) / 2.0, y2),
        "bottom_left": (x1, y2),
        "bottom_right": (x2, y2),
        "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
        "top_center": ((x1 + x2) / 2.0, y1),
    }


def draw_annotation(config: dict, candidates: dict[str, Point], chosen_name: str, out_path: Path) -> None:
    image = cv2.imread(config["image"])
    if image is None:
        raise FileNotFoundError(config["image"])

    a = pt(config["points"]["A"])
    c = pt(config["points"]["C"])
    v = pt(config["points"]["V"])
    b = candidates[chosen_name]

    dark_blue = (120, 30, 0)
    green = (0, 180, 0)
    red = (0, 0, 230)
    white = (255, 255, 255)

    cv2.line(image, tuple(map(round, a)), tuple(map(round, v)), dark_blue, 8)
    for label, point in [("A'", a), ("B'", b), ("C'", c), ("V'", v)]:
        point_i = tuple(map(round, point))
        cv2.circle(image, point_i, 16, red if label == "B'" else green, -1)
        cv2.circle(image, point_i, 18, white, 3)
        cv2.putText(
            image,
            label,
            (point_i[0] + 18, point_i[1] - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            white,
            6,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            label,
            (point_i[0] + 18, point_i[1] - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            dark_blue,
            2,
            cv2.LINE_AA,
        )

    if "bbox_xyxy" in config:
        x1, y1, x2, y2 = [round(v) for v in config["bbox_xyxy"]]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    ac_m = args.reference_distance_m or float(config["reference_distance_m"])

    a = pt(config["points"]["A"])
    c = pt(config["points"]["C"])
    v = pt(config["points"]["V"])
    chosen_name = config.get("chosen_object_point", "bottom_center")

    if "bbox_xyxy" in config:
        candidates = bbox_points([float(x) for x in config["bbox_xyxy"]])
    else:
        candidates = {"manual_B": pt(config["points"]["B"])}
        chosen_name = "manual_B"

    print(f"Reference A-to-C distance from Google Maps: {ac_m:.1f} m")
    print("Candidate object reference points:")

    results = {}
    for name, b in candidates.items():
        try:
            result = estimate_ab_m(a, b, c, v, ac_m)
        except ZeroDivisionError:
            continue
        if not all(math.isfinite(value) for value in result.values()):
            continue
        results[name] = result
        print(
            f"  {name:>13}: A->B = {result['distance_A_to_B_m']:7.2f} m, "
            f"B->C = {result['distance_B_to_C_m']:7.2f} m, "
            f"CR = {result['cross_ratio']:.4f}"
        )

    if chosen_name not in results:
        raise SystemExit(f"chosen_object_point={chosen_name!r} was not computed")

    chosen = results[chosen_name]
    print()
    print(f"Chosen B' point: {chosen_name}")
    print(f"Estimated bicycle/person distance from A: {chosen['distance_A_to_B_m']:.2f} m")
    print(f"Estimated distance from object to C/arch: {chosen['distance_B_to_C_m']:.2f} m")

    if "bottom_center" in results and "top_center" in results:
        span = abs(
            results["bottom_center"]["distance_B_to_C_m"]
            - results["top_center"]["distance_B_to_C_m"]
        )
        print(f"Bottom-center vs top-center changes object-to-arch by {span:.2f} m")

    draw_annotation(config, candidates, chosen_name, args.annotated_out)
    print(f"saved annotation: {args.annotated_out}")


if __name__ == "__main__":
    main()
