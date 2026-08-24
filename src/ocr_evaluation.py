from pathlib import Path
import cv2
import csv

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

RESULTS_DIR = BASE_DIR / "runs" / "ocr_evaluation"
CROPS_DIR = RESULTS_DIR / "plate_crops"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE = RESULTS_DIR / "ocr_results.csv"


# =========================
# Load YOLO
# =========================

model = YOLO(str(MODEL_PATH))


# =========================
# Accuracy Functions
# =========================

def character_accuracy(predicted, ground_truth):

    predicted = predicted.upper()
    ground_truth = ground_truth.upper()

    if not ground_truth:
        return 0.0

    correct = 0

    for i in range(min(len(predicted), len(ground_truth))):

        if predicted[i] == ground_truth[i]:
            correct += 1

    return correct / len(ground_truth)


def exact_match(predicted, ground_truth):

    return predicted.upper() == ground_truth.upper()


# =========================
# Process Image
# =========================

def process_image(image_path):

    image = cv2.imread(str(image_path))

    if image is None:

        print(f"Could not read: {image_path}")
        return None

    results = model(
        image,
        conf=0.40,
        verbose=False
    )

    best_box = None
    best_conf = 0

    # Find highest-confidence plate
    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            confidence = float(
                box.conf[0].cpu().numpy()
            )

            if confidence > best_conf:

                best_conf = confidence

                best_box = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .astype(int)
                )

    if best_box is None:

        print("No license plate detected.")
        return None

    x1, y1, x2, y2 = best_box

    h, w = image.shape[:2]

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))

    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    # Crop
    plate_crop = image[y1:y2, x1:x2]

    if plate_crop.size == 0:

        print("Invalid crop.")
        return None

    # =========================
    # Save crop
    # =========================

    crop_path = CROPS_DIR / image_path.name

    cv2.imwrite(
        str(crop_path),
        plate_crop
    )

    # =========================
    # OCR
    # =========================

    baseline = baseline_ocr(plate_crop)

    improved = improved_ocr(plate_crop)

    # =========================
    # Display information
    # =========================

    print("\n" + "=" * 60)

    print(f"Image: {image_path.name}")

    print(
        f"Detection confidence: "
        f"{best_conf:.3f}"
    )

    print(
        f"Baseline OCR: "
        f"{baseline or 'N/A'}"
    )

    print(
        f"Improved OCR: "
        f"{improved or 'N/A'}"
    )

    print(
        f"\nPlate crop saved here:"
    )

    print(crop_path)

    print("=" * 60)

    print(
        "\nOpen the crop image above "
        "and enter the REAL plate number."
    )

    ground_truth = input(
        "Ground Truth (or SKIP): "
    ).strip().upper()

    if ground_truth == "SKIP" or not ground_truth:

        return None

    # =========================
    # Calculate accuracy
    # =========================

    baseline_char_acc = character_accuracy(
        baseline,
        ground_truth
    )

    improved_char_acc = character_accuracy(
        improved,
        ground_truth
    )

    baseline_exact = exact_match(
        baseline,
        ground_truth
    )

    improved_exact = exact_match(
        improved,
        ground_truth
    )

    print("\nResults:")

    print(
        f"Baseline character accuracy: "
        f"{baseline_char_acc * 100:.2f}%"
    )

    print(
        f"Improved character accuracy: "
        f"{improved_char_acc * 100:.2f}%"
    )

    print(
        f"Baseline exact match: "
        f"{'YES' if baseline_exact else 'NO'}"
    )

    print(
        f"Improved exact match: "
        f"{'YES' if improved_exact else 'NO'}"
    )

    return {
        "image": image_path.name,
        "ground_truth": ground_truth,
        "baseline": baseline,
        "improved": improved,
        "baseline_char_accuracy": baseline_char_acc,
        "improved_char_accuracy": improved_char_acc,
        "baseline_exact": int(baseline_exact),
        "improved_exact": int(improved_exact),
    }


# =========================
# Main
# =========================

if __name__ == "__main__":

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

    print("=" * 60)
    print("MANUAL OCR EVALUATION")
    print("=" * 60)

    print(
        f"Images available: {len(images)}"
    )

    results = []

    for index, image_path in enumerate(
        images,
        start=1
    ):

        print(
            f"\n[{index}/{len(images)}]"
        )

        result = process_image(
            image_path
        )

        if result is not None:

            results.append(result)

    # =========================
    # Save CSV
    # =========================

    if results:

        fieldnames = [
            "image",
            "ground_truth",
            "baseline",
            "improved",
            "baseline_char_accuracy",
            "improved_char_accuracy",
            "baseline_exact",
            "improved_exact",
        ]

        with open(
            CSV_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerows(results)

        # =========================
        # Overall statistics
        # =========================

        baseline_accuracy = sum(
            r["baseline_char_accuracy"]
            for r in results
        ) / len(results)

        improved_accuracy = sum(
            r["improved_char_accuracy"]
            for r in results
        ) / len(results)

        baseline_exact_accuracy = sum(
            r["baseline_exact"]
            for r in results
        ) / len(results)

        improved_exact_accuracy = sum(
            r["improved_exact"]
            for r in results
        ) / len(results)

        print("\n")
        print("=" * 60)
        print("FINAL OCR EVALUATION")
        print("=" * 60)

        print(
            f"Images evaluated: "
            f"{len(results)}"
        )

        print(
            f"\nBaseline character accuracy: "
            f"{baseline_accuracy * 100:.2f}%"
        )

        print(
            f"Improved character accuracy: "
            f"{improved_accuracy * 100:.2f}%"
        )

        print(
            f"\nBaseline exact-match accuracy: "
            f"{baseline_exact_accuracy * 100:.2f}%"
        )

        print(
            f"Improved exact-match accuracy: "
            f"{improved_exact_accuracy * 100:.2f}%"
        )

        print(
            f"\nCSV saved to:"
        )

        print(CSV_FILE)

    else:

        print("\nNo images were evaluated.")

