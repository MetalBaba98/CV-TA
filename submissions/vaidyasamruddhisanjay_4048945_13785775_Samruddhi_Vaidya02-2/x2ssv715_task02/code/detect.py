from ultralytics import YOLO
import cv2

model = YOLO("yolov8x.pt")

results = model.predict(
    source="images/sample.jpg",
    imgsz=1920,
    conf=0.15
)

img = cv2.imread("images/sample.jpg")

for r in results:
    for box in r.boxes:

        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]

        # COCO class 1 = bicycle
        if class_name == "bicycle":

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                img,
                f"bicycle {float(box.conf[0]):.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            print("Bicycle found")
            print(f"Box: ({x1}, {y1}) ({x2}, {y2})")

cv2.imwrite("images/bicycle_only.jpg", img)
print("Saved: results/bicycle_only.jpg")