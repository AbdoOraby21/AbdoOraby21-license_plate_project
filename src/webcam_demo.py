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


# =========================
# Settings
# =========================

CONFIDENCE = 0.40
CAMERA_INDEX = 0


# =========================
# Load Model
# =========================

print("=" * 70)
print("REAL-TIME LICENSE PLATE DETECTION")
print("=" * 70)

print("\nLoading YOLO model...")

model = YOLO(
    str(MODEL_PATH)
)

print("Model loaded successfully.")


# =========================
# Open Webcam
# =========================

cap = cv2.VideoCapture(
    CAMERA_INDEX
)

if not cap.isOpened():

    print("\nERROR: Could not open webcam.")

    raise SystemExit(1)


print("\nWebcam started.")
print("Press Q to quit.")


# =========================
# Main Loop
# =========================

while True:

    ret, frame = cap.read()

    if not ret:

        print(
            "ERROR: Could not read frame."
        )

        break

    # -------------------------
    # YOLO Detection
    # -------------------------

    results = model(
        frame,
        conf=CONFIDENCE,
        verbose=False
    )

    best_box = None
    best_conf = 0.0

    # -------------------------
    # Find Best Plate
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
    # Draw Detection
    # -------------------------

    if best_box is not None:

        x1, y1, x2, y2 = best_box

        h, w = frame.shape[:2]

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

        plate_crop = frame[
            y1:y2,
            x1:x2
        ]

        plate_text = ""

        if plate_crop.size > 0:

            plate_text = baseline_ocr(
                plate_crop
            )

        if not plate_text:

            plate_text = "NOT RECOGNIZED"

        # -------------------------
        # Bounding Box
        # -------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        # -------------------------
        # Detection Confidence
        # -------------------------

        detection_text = (
            f"Detection: "
            f"{best_conf:.2f}"
        )

        cv2.putText(
            frame,
            detection_text,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        # -------------------------
        # OCR Text
        # -------------------------

        text_y = min(
            h - 20,
            y2 + 35
        )

        label = (
            f"Plate: {plate_text}"
        )

        # Background for text

        text_width = 300

        cv2.rectangle(
            frame,
            (x1, y2),
            (
                min(
                    w - 1,
                    x1 + text_width
                ),
                min(
                    h - 1,
                    text_y + 10
                )
            ),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            label,
            (x1 + 5, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

    else:

        # -------------------------
        # No Detection
        # -------------------------

        cv2.putText(
            frame,
            "No License Plate Detected",
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    # -------------------------
    # Window
    # -------------------------

    cv2.imshow(
        "License Plate Detection & Recognition",
        frame
    )

    # -------------------------
    # Quit
    # -------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# =========================
# Cleanup
# =========================

cap.release()

cv2.destroyAllWindows()

print("\nWebcam demo stopped.")