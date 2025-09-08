from ultralytics import YOLO

# Load model
model = YOLO("runs/detect/plant_disease_yolov8/weights/best.pt")

# Predict on new images
results = model.predict(
    source="dataset/images/val",  # change to your test folder
    save=True,
    imgsz=640,
    conf=0.25
)

for r in results:
    print(r.boxes)
