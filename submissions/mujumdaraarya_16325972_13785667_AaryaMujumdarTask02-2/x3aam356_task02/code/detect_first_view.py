from ultralytics import YOLO
import cv2

# Larger model for small distant objects
model = YOLO("yolov8x.pt")

# Run detection
results = model.predict(
    source="images/first_view.jpeg",
    imgsz=1920,
    conf=0.10,
    verbose=True
)

# Read original image
img = cv2.imread("images/first_view.jpeg")

person_count = 0

for r in results:
    for box in r.boxes:

        cls_id = int(box.cls[0])

        # COCO class 0 = person
        if cls_id == 0:

            person_count += 1

            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            print("\nPERSON DETECTED")
            print(f"Confidence = {conf:.3f}")

            # Bounding box
            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                img,
                f"person {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # Bottom-center point
            ground_x = int((x1 + x2) / 2)
            ground_y = int(y2)

            print(f"Ground point = ({ground_x}, {ground_y})")

            cv2.circle(
                img,
                (ground_x, ground_y),
                6,
                (0, 0, 255),
                -1
            )

# Save image
output_path = "images/person_detection.jpg"
cv2.imwrite(output_path, img)

print(f"\nDetected {person_count} persons")
print(f"Saved output to: {output_path}")