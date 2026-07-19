from ultralytics import YOLO

# Load the model
model = YOLO("yolo11l.pt")

results = model(
    "../data/sample.JPG", 
    save=True, 
    conf=0.02, 
    iou=0.45, 
    classes=[1],
    project="../../../results",
    name=".",
    exist_ok=True
)

for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0]

        # Extract the confidence score (comes as a 1-element tensor)
        conf_score = box.conf[0].item()

        print("Bicycle Found")
        print(f"Confidence: {conf_score:.2f}")
        print(
            f"Coordinates: [{x1.item():.1f}, {y1.item():.1f}, {x2.item():.1f}, {y2.item():.1f}]"
        )
