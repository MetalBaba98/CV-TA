"""
detect_bicycle.py
-----------------
Runs YOLOv8 object detection on the provided bicycle image (sample.JPG)
and the personal lane photo (firstPersonView image).

Classes of interest (COCO dataset):
  class 0  = person
  class 1  = bicycle

Usage:
  python3 detect_bicycle.py
"""

from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import os

MODEL = 'yolov8n.pt'          # nano model; swap to yolov8s.pt for better accuracy
IMG_SIZE = 4032               # use full resolution so small objects are found


def detect(image_path, classes, conf=0.25):
    model = YOLO(MODEL)
    results = model(image_path, classes=classes, conf=conf, imgsz=IMG_SIZE, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = [v.item() for v in box.xyxy[0]]
            cls  = int(box.cls[0].item())
            conf = box.conf[0].item()
            name = r.names[cls]
            bx   = (x1 + x2) / 2          # bottom-center x
            by   = y2                       # bottom-center y (ground contact)
            cx   = (x1 + x2) / 2          # center x
            cy   = (y1 + y2) / 2          # center y
            detections.append({
                'name': name, 'cls': cls, 'conf': conf,
                'box': (x1, y1, x2, y2),
                'bottom_center': (bx, by),
                'center': (cx, cy),
                'top_center': (cx, y1),
            })
    return detections


def save_annotated(image_path, detections, out_path):
    img  = Image.open(image_path).copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 56)
    except Exception:
        font = ImageFont.load_default()

    for d in detections:
        x1, y1, x2, y2 = d['box']
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=6)
        label = f"{d['name']}  {d['conf']:.2f}"
        draw.text((x1, max(0, y1 - 60)), label, fill=(0, 255, 0), font=font)
        bx, by = d['bottom_center']
        r = 20
        draw.ellipse([bx-r, by-r, bx+r, by+r], fill=(255, 0, 0))

    img.save(out_path)
    print(f"  Annotated image saved → {out_path}")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # ── Task 3: bicycle in sample.JPG ──────────────────────────────────────
    print("=" * 60)
    print("Task 3 — Bicycle detection on sample.JPG")
    print("=" * 60)
    bike_dets = detect('images/sample.JPG', classes=[1], conf=0.1)
    if not bike_dets:
        print("  No bicycle found with class filter — trying all classes...")
        bike_dets = detect('images/sample.JPG', classes=None, conf=0.1)

    for d in bike_dets:
        if d['name'] == 'bicycle':
            print(f"\n  Detected : {d['name']}  (confidence = {d['conf']:.3f})")
            print(f"  Bounding box   : ({d['box'][0]:.0f}, {d['box'][1]:.0f}) "
                  f"to ({d['box'][2]:.0f}, {d['box'][3]:.0f})")
            print(f"  bottom-center  : ({d['bottom_center'][0]:.1f}, {d['bottom_center'][1]:.1f})  ← USE THIS for cross-ratio")
            print(f"  center         : ({d['center'][0]:.1f}, {d['center'][1]:.1f})")
            print(f"  top-center     : ({d['top_center'][0]:.1f}, {d['top_center'][1]:.1f})")

    save_annotated('images/sample.JPG', bike_dets, 'images/bicycle_yolo_output.png')

    # ── Task 4: person in firstPersonView image (ThirdPersonView.jpg) ──────
    print("\n" + "=" * 60)
    print("Task 4 — Person detection on firstPersonView image")
    print("=" * 60)
    # NOTE: The file used here is ThirdPersonView.jpg because the user's
    # naming convention was swapped — see explanation in answers.tex.
    person_dets = detect('images/ThirdPersonView.jpg', classes=[0], conf=0.4)
    for d in person_dets:
        print(f"\n  Detected : {d['name']}  (confidence = {d['conf']:.3f})")
        print(f"  Bounding box   : ({d['box'][0]:.0f}, {d['box'][1]:.0f}) "
              f"to ({d['box'][2]:.0f}, {d['box'][3]:.0f})")
        print(f"  bottom-center  : ({d['bottom_center'][0]:.1f}, {d['bottom_center'][1]:.1f})")

    save_annotated('images/ThirdPersonView.jpg', person_dets, 'images/person_yolo_output.png')
