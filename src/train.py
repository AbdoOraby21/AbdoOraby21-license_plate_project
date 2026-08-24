from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_YAML = BASE_DIR / "data.yaml"

# Load pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

# Train the model
results = model.train(
    data=str(DATA_YAML),
    epochs=50,
    imgsz=640,
    batch=16,
    patience=10,
    project=str(BASE_DIR / "runs"),
    name="license_plate_detection",
    pretrained=True,
    verbose=True
)

print("\nTraining completed!")

print("Best model:")
print(
    BASE_DIR
    / "runs"
    / "license_plate_detection"
    / "weights"
    / "best.pt"
)