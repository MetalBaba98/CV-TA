#!/usr/bin/env python3
"""Detect COCO objects with a YOLO model and save boxes for Task 02.

COCO class ids used in this assignment:
  person  = 0
  bicycle = 1

Example:
  python scripts/detect_yolo.py images/bicycle_bb.png --class-id 1
  python scripts/detect_yolo.py images/firstPersonView.png --class-id 0
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, help="Input image path")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO weights")
    parser.add_argument("--class-id", type=int, default=1, help="COCO class id")
    parser.add_argument("--all-classes", action="store_true", help="Do not filter by class")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument(
        "--device",
        default="auto",
        help="YOLO/PyTorch device: auto, cpu, mps on Apple Silicon, or 0 for CUDA GPU",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(args.out_dir / "ultralytics_config"))
    os.environ.setdefault("MPLCONFIGDIR", str(args.out_dir / "matplotlib_config"))

    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Ultralytics is not installed in this Python environment.\n"
            "Install it in your OpenCV conda env, then rerun:\n"
            "  pip install ultralytics\n"
            "or use the environment where ultralytics is already available."
        ) from exc

    model = YOLO(args.model)
    device = args.device
    if device == "auto":
        import torch

        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "0"
        else:
            device = "cpu"
    print(f"using device: {device}")
    classes = None if args.all_classes else [args.class_id]
    results = model.predict(
        source=str(args.image),
        classes=classes,
        conf=args.conf,
        device=device,
        verbose=False,
    )

    detections = []
    for result in results:
        names = result.names
        for box in result.boxes:
            cls_id = int(box.cls.item())
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": names.get(cls_id, str(cls_id)),
                    "confidence": float(box.conf.item()),
                    "xyxy": [x1, y1, x2, y2],
                    "bottom_center": [(x1 + x2) / 2.0, y2],
                    "center": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
                }
            )

    stem = args.image.stem
    suffix = "all" if args.all_classes else f"class{args.class_id}"
    json_path = args.out_dir / f"{stem}_detections_{suffix}.json"
    json_path.write_text(json.dumps({"image": str(args.image), "detections": detections}, indent=2))

    if results:
        annotated = results[0].plot()
        image_path = args.out_dir / f"{stem}_yolo_{suffix}.png"
        cv2.imwrite(str(image_path), annotated)
    else:
        image_path = None

    print(f"saved detections: {json_path}")
    if image_path:
        print(f"saved annotation: {image_path}")
    print(f"found {len(detections)} object(s)")
    for i, det in enumerate(detections, start=1):
        print(
            f"{i}. {det['class_name']} conf={det['confidence']:.3f} "
            f"xyxy={tuple(round(v, 1) for v in det['xyxy'])}"
        )


if __name__ == "__main__":
    main()
