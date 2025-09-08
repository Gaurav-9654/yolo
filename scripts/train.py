from ultralytics import YOLO

# Load pretrained YOLOv8n
model = YOLO("yolov8n.pt")

# Train on dataset
model.train(
    data="configs/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="plant_disease_yolov8"
)
