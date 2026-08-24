from ocr import baseline_ocr
from plate_database import init_database, save_plate
from pathlib import Path

import cv2
from ultralytics import YOLO

from ocr import baseline_ocr


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

TEST_IMAGES_DIR = (
    BASE_DIR
    / "dataset"
    / "yolo"
    / "images"
    / "test"
)

OUTPUT_DIR = (
    BASE_DIR
    / "runs"
    / "final_demo"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================
# Settings
# =========================

CONFIDENCE = 0.40


# =========================
# Load Model
# =========================

print("=" * 70)
print("LICENSE PLATE DETECTION & RECOGNITION")
print("=" * 70)

print("\nLoading YOLO model...")

model = YOLO(
    str(MODEL_PATH)
)

init_database()

print("Model loaded successfully.")


# =========================
# Process Image
# =========================

def process_image(image_path):

    print(
        f"\nProcessing: "
        f"{image_path.name}"
    )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print(
            "ERROR: Could not read image."
        )

        return

    # -------------------------
    # YOLO Detection
    # -------------------------

    results = model(
        image,
        conf=CONFIDENCE,
        verbose=False
    )

    best_box = None
    best_conf = 0.0

    # -------------------------
    # Find Best Detection
    # -------------------------

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            confidence = float(
                box.conf[0]
                .cpu()
                .numpy()
            )

            if confidence > best_conf:

                best_conf = confidence

                best_box = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .astype(int)
                )

    # -------------------------
    # No Detection
    # -------------------------

    if best_box is None:

        print(
            "No license plate detected."
        )

        output = image.copy()

        cv2.putText(
            output,
            "Plate: NOT DETECTED",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2
        )

        output_path = (
            OUTPUT_DIR
            / image_path.name
        )

        cv2.imwrite(
            str(output_path),
            output
        )

        print(
            f"Saved: {output_path}"
        )

        return

    # -------------------------
    # Coordinates
    # -------------------------

    x1, y1, x2, y2 = best_box

    h, w = image.shape[:2]

    x1 = max(
        0,
        min(x1, w - 1)
    )

    x2 = max(
        0,
        min(x2, w - 1)
    )

    y1 = max(
        0,
        min(y1, h - 1)
    )

    y2 = max(
        0,
        min(y2, h - 1)
    )

    # -------------------------
    # Crop Plate
    # -------------------------

    plate_crop = image[
        y1:y2,
        x1:x2
    ]

    if plate_crop.size == 0:

        print(
            "Invalid plate crop."
        )

        return

    # -------------------------
    # OCR
    # -------------------------

    plate_text = baseline_ocr(
        plate_crop
    )

    if not plate_text:

        plate_text = "NOT RECOGNIZED"

    if plate_text != "NOT RECOGNIZED":
        save_plate(
           plate_number=plate_text,
            confidence=best_conf,
            image_name=image_path.name
        )
    
    # -------------------------
    # Draw Bounding Box
    # -------------------------

    output = image.copy()

    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3
    )

    # -------------------------
    # Detection Label
    # -------------------------

    detection_label = (
        f"Plate Detection: "
        f"{best_conf:.2f}"
    )

    cv2.putText(
        output,
        detection_label,
        (x1, max(30, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )

    # -------------------------
    # OCR Result
    # -------------------------

    ocr_label = (
        f"Plate: {plate_text}"
    )

    label_y = min(
        h - 20,
        y2 + 35
    )

    cv2.rectangle(
        output,
        (
            x1,
            max(0, y2)
        ),
        (
            min(
                w - 1,
                x1 + 300
            ),
            min(
                h - 1,
                label_y + 10
            )
        ),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        output,
        ocr_label,
        (x1 + 5, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # -------------------------
    # Save Output
    # -------------------------

    output_path = (
        OUTPUT_DIR
        / image_path.name
    )

    cv2.imwrite(
        str(output_path),
        output
    )

    # -------------------------
    # Print Results
    # -------------------------

    print(
        f"Detection confidence: "
        f"{best_conf:.3f}"
    )

    print(
        f"Recognized plate: "
        f"{plate_text}"
    )

    print(
        f"Saved result to:"
    )

    print(output_path)


# =========================
# Main
# =========================

if __name__ == "__main__":

    # -------------------------
    # Select Images
    # -------------------------

    images = sorted([
        p
        for p in TEST_IMAGES_DIR.iterdir()
        if p.suffix.lower()
        in {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp"
        }
    ])

    print(
        f"\nTest images available: "
        f"{len(images)}"
    )

    # -------------------------
    # Process first 10 images
    # -------------------------

    print(
        "\nProcessing first 10 test images..."
    )

    for image_path in images[:10]:

        process_image(
            image_path
        )

    print("\n")
    print("=" * 70)
    print("FINAL DEMO COMPLETED")
    print("=" * 70)

    print(
        "\nResults saved in:"
    )

    print(OUTPUT_DIR)