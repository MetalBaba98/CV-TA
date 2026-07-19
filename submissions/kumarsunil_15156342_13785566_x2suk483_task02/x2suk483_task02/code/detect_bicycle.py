#!/usr/bin/env python3
"""
Task 02 -- object detection with a pretrained YOLO model (Ultralytics).

Detects the bicycle (and people) in an image, prints each bounding box, and
marks the ground-contact reference point (bottom-centre of the box), which is
the point fed into the cross-ratio distance estimate.

COCO class ids used here:  person = 0,  bicycle = 1
(see https://docs.ultralytics.com/tasks/detect and cocodataset.org).

Install:  pip install ultralytics
Run:      python3 detect_bicycle.py images/sample.JPG
The first run downloads the yolov8x.pt weights automatically. Do NOT commit
the .pt weights to your submission zip -- they are fetched from the internet.
"""
import sys
from ultralytics import YOLO

WANT = {0: "person", 1: "bicycle"}


def main(img_path, weights="yolov8x.pt", conf=0.25):
    model = YOLO(weights)                      # pretrained on COCO
    res = model(img_path, conf=conf, verbose=False)[0]

    print(f"image: {img_path}")
    for b in res.boxes:
        cls = int(b.cls[0])
        if cls not in WANT:
            continue
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
        gx, gy = (x1 + x2) / 2.0, y2          # ground-contact reference
        print(f"  {WANT[cls]:8s} conf={float(b.conf[0]):.2f}  "
              f"box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})  "
              f"ground-contact=({gx:.0f},{gy:.0f})")

    out = res.plot()                           # annotated image (numpy BGR)
    import cv2
    cv2.imwrite("detection_out.png", out)
    print("annotated image saved -> detection_out.png")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "images/sample.JPG"
    main(path)
