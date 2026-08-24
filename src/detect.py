from pathlib import Path
import cv2
from ultralytics import YOLO

from ocr import baseline_ocr, improved_ocr


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

OUTPUT_DIR = BASE_DIR / "runs" / "ocr_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Load YOLO model
# =========================

model = YOLO(str(MODEL_PATH))


# =========================
# Detection + OCR
# =========================

def process_image(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read: {image_path}")
        return

    results = model(
        image,
        conf=0.40,
        verbose=False
    )

    output = image.copy()

    detected = False

    for result in results:

        boxes = result.boxes

        if boxes is None:
            continue

        for box in boxes:

            detected = True

            # Bounding box coordinates
            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )

            confidence = float(
                box.conf[0].cpu().numpy()
            )

            # Make sure coordinates are valid
            h, w = image.shape[:2]

            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))

            # Crop license plate
            plate_crop = image[y1:y2, x1:x2]

            if plate_crop.size == 0:
                continue

            # =========================
            # OCR
            # =========================

            baseline_text = baseline_ocr(
                plate_crop
            )

            improved_text = improved_ocr(
                plate_crop
            )

            # =========================
            # Draw bounding box
            # =========================

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Detection confidence
            cv2.putText(
                output,
                f"Plate {confidence:.2f}",
                (x1, max(20, y1 - 35)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # Baseline OCR
            cv2.putText(
                output,
                f"Baseline: {baseline_text or 'N/A'}",
                (x1, max(20, y1 - 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2
            )

            # Improved OCR
            cv2.putText(
                output,
                f"Improved: {improved_text or 'N/A'}",
                (x1, min(h - 10, y2 + 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

            print(
                f"{image_path.name} | "
                f"Confidence: {confidence:.3f} | "
                f"Baseline: {baseline_text or 'N/A'} | "
                f"Improved: {improved_text or 'N/A'}"
            )

    if not detected:
        print(
            f"{image_path.name} | "
            "No license plate detected"
        )

    # Save result
    output_path = OUTPUT_DIR / image_path.name

    cv2.imwrite(
        str(output_path),
        output
    )


# =========================
# Main
# =========================

if __name__ == "__main__":

    images = [
        p
        for p in TEST_IMAGES_DIR.iterdir()
        if p.suffix.lower()
        in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]

    print("=" * 60)
    print("License Plate Detection + OCR")
    print("=" * 60)

    print(f"Test images: {len(images)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    for image_path in images:
        process_image(image_path)

    print("\nProcessing completed!")

