from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "runs"
    / "license_plate_detection"
    / "weights"
    / "best.pt"
)

DATA_YAML = BASE_DIR / "data.yaml"

model = YOLO(str(MODEL_PATH))

print("=" * 50)
print("Evaluating License Plate Detection Model")
print("=" * 50)

metrics = model.val(
    data=str(DATA_YAML),
    split="test",
    imgsz=640,
    batch=16,
    plots=True
)

print("\nTest Results")
print("=" * 50)

print(f"Precision : {metrics.box.mp:.4f}")
print(f"Recall    : {metrics.box.mr:.4f}")
print(f"mAP@50    : {metrics.box.map50:.4f}")
print(f"mAP@50-95 : {metrics.box.map:.4f}")

print("\nEvaluation completed.")
print(f"Results saved in: {BASE_DIR / 'runs'}")