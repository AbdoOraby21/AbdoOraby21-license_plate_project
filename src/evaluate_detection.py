from pathlib import Path
from ultralytics import YOLO


# =========================
# Paths
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "runs"
    / "license_plate_detection"
    / "weights"
    / "best.pt"
)

DATA_YAML = BASE_DIR / "data.yaml"

# =========================
# Check Files
# =========================

if not MODEL_PATH.exists():
    print("ERROR: Model not found:")
    print(MODEL_PATH)
    raise SystemExit(1)

if not DATA_YAML.exists():
    print("ERROR: data.yaml not found:")
    print(DATA_YAML)
    raise SystemExit(1)


# =========================
# Load Model
# =========================

print("=" * 70)
print("YOLOv8 LICENSE PLATE DETECTION EVALUATION")
print("=" * 70)

print("\nLoading model...")

model = YOLO(
    str(MODEL_PATH)
)

print("Model loaded successfully.")


# =========================
# Validation
# =========================

print("\nRunning evaluation...")

metrics = model.val(
    data=str(DATA_YAML),
    split="test",
    imgsz=640,
    conf=0.25,
    iou=0.50,
    verbose=True
)


# =========================
# Extract Metrics
# =========================

precision = float(
    metrics.box.mp
)

recall = float(
    metrics.box.mr
)

map50 = float(
    metrics.box.map50
)

map50_95 = float(
    metrics.box.map
)


# =========================
# Print Results
# =========================

print("\n")
print("=" * 70)
print("FINAL DETECTION RESULTS")
print("=" * 70)

print(
    f"\nPrecision     : "
    f"{precision * 100:.2f}%"
)

print(
    f"Recall        : "
    f"{recall * 100:.2f}%"
)

print(
    f"mAP@50        : "
    f"{map50 * 100:.2f}%"
)

print(
    f"mAP@50-95     : "
    f"{map50_95 * 100:.2f}%"
)

print("=" * 70)


# =========================
# Save Results
# =========================

results_file = (
    BASE_DIR
    / "runs"
    / "license_plate_detection"
    / "detection_metrics.txt"
)

with open(
    results_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "YOLOv8 License Plate Detection Results\n"
    )

    file.write(
        "=" * 50 + "\n\n"
    )

    file.write(
        f"Precision: {precision * 100:.2f}%\n"
    )

    file.write(
        f"Recall: {recall * 100:.2f}%\n"
    )

    file.write(
        f"mAP@50: {map50 * 100:.2f}%\n"
    )

    file.write(
        f"mAP@50-95: {map50_95 * 100:.2f}%\n"
    )


print(
    "\nMetrics saved to:"
)

print(results_file)