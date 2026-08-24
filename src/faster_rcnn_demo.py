from pathlib import Path

import cv2
import torch
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights
)
from torchvision.transforms import functional as F


# =========================
# Paths
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

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
    / "faster_rcnn"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================
# Device
# =========================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("FASTER R-CNN LICENSE PLATE DEMO")
print("=" * 70)

print(f"\nDevice: {DEVICE}")


# =========================
# Load Faster R-CNN
# =========================

print("\nLoading pretrained Faster R-CNN...")

weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT

model = fasterrcnn_resnet50_fpn(
    weights=weights
)

model.to(DEVICE)

model.eval()

print("Model loaded successfully.")


# =========================
# Process Image
# =========================

def process_image(image_path):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        return

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    tensor = F.to_tensor(
        rgb
    ).to(DEVICE)

    with torch.no_grad():

        prediction = model(
            [tensor]
        )[0]

    boxes = prediction["boxes"]
    scores = prediction["scores"]
    labels = prediction["labels"]

    output = image.copy()

    detected = False

    for box, score, label in zip(
        boxes,
        scores,
        labels
    ):

        confidence = float(
            score.cpu()
        )

        # Faster R-CNN COCO model
        # only detects COCO classes.
        #
        # Therefore this is a
        # pretrained comparison/demo,
        # not a license-plate-trained model.

        if confidence < 0.50:
            continue

        x1, y1, x2, y2 = (
            box
            .cpu()
            .numpy()
            .astype(int)
        )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        cv2.putText(
            output,
            f"Object {confidence:.2f}",
            (x1, max(25, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

        detected = True

    if not detected:

        cv2.putText(
            output,
            "No object detected",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
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
        f"{image_path.name} -> "
        f"saved"
    )


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

    print(
        f"\nTest images: {len(images)}"
    )

    print(
        "\nProcessing first 10 images..."
    )

    for image_path in images[:10]:

        process_image(
            image_path
        )

    print("\n")
    print("=" * 70)
    print("FASTER R-CNN DEMO COMPLETED")
    print("=" * 70)

    print(
        f"\nResults saved to:\n"
        f"{OUTPUT_DIR}"
    )