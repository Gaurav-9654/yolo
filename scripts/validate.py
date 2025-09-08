from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/plant_disease_yolov8/weights/best.pt")

# Validate model
metrics = model.val()
print(metrics)
