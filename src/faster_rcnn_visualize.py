import os
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection import fasterrcnn_resnet50_fpn


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "runs" / "faster_rcnn" / "best.pt"

TEST_IMAGES = ROOT / "dataset" / "yolo" / "images" / "test"

OUTPUT_DIR = ROOT / "runs" / "faster_rcnn" / "predictions"

CONF_THRESHOLD = 0.50

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("Faster R-CNN Visualization")
print("=" * 70)

print(f"Device : {DEVICE}")
print(f"Model  : {MODEL_PATH}")
print(f"Images : {TEST_IMAGES}")
print(f"Output : {OUTPUT_DIR}")
print("=" * 70)

print("\nLoading Faster R-CNN model...")

model = fasterrcnn_resnet50_fpn(
    weights=None,
    weights_backbone=None,
    num_classes=2
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    elif (
        "model" in checkpoint
        and isinstance(checkpoint["model"], dict)
    ):
        state_dict = checkpoint["model"]

    else:
        state_dict = checkpoint

else:
    raise RuntimeError(
        "Unsupported checkpoint format."
    )


# Remove module. prefix if present
clean_state_dict = {}

for key, value in state_dict.items():

    if key.startswith("module."):
        key = key[7:]

    clean_state_dict[key] = value


model.load_state_dict(
    clean_state_dict,
    strict=False
)

model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# ============================================================
# FIND IMAGES
# ============================================================

extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

image_files = sorted(
    [
        p
        for p in TEST_IMAGES.rglob("*")
        if p.suffix.lower() in extensions
    ]
)

if not image_files:

    raise RuntimeError(
        f"No images found in:\n{TEST_IMAGES}"
    )

print(
    f"\nFound {len(image_files)} test images."
)

print("\nRunning detection...\n")


# ============================================================
# FONT
# ============================================================

try:

    font = ImageFont.truetype(
        "arial.ttf",
        18
    )

except:

    font = ImageFont.load_default()


# ============================================================
# PROCESS IMAGES
# ============================================================

with torch.no_grad():

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        original_image = image.copy()

        width, height = image.size

        # ----------------------------------------------------
        # Convert to tensor
        # ----------------------------------------------------

        image_tensor = torch.from_numpy(
            np.array(image)
        ).permute(
            2, 0, 1
        ).float() / 255.0

        image_tensor = image_tensor.to(
            DEVICE
        )

        # ----------------------------------------------------
        # Inference
        # ----------------------------------------------------

        output = model(
            [image_tensor]
        )[0]

        boxes = output["boxes"].detach().cpu()
        scores = output["scores"].detach().cpu()
        labels = output["labels"].detach().cpu()

        # ----------------------------------------------------
        # Draw detections
        # ----------------------------------------------------

        draw = ImageDraw.Draw(
            original_image
        )

        detection_count = 0

        for box, score, label in zip(
            boxes,
            scores,
            labels
        ):

            confidence = float(score)

            if confidence < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(
                int,
                box.tolist()
            )

            detection_count += 1

            # Bounding box
            draw.rectangle(
                [
                    x1,
                    y1,
                    x2,
                    y2
                ],
                outline="red",
                width=3
            )

            # Label
            text = (
                f"License Plate "
                f"{confidence:.2f}"
            )

            # Get text bounding box
            try:

                text_box = draw.textbbox(
                    (x1, y1),
                    text,
                    font=font
                )

                text_width = (
                    text_box[2] -
                    text_box[0]
                )

                text_height = (
                    text_box[3] -
                    text_box[1]
                )

            except:

                text_width = len(text) * 8
                text_height = 15

            # Text background
            text_y = max(
                0,
                y1 - text_height - 4
            )

            draw.rectangle(
                [
                    x1,
                    text_y,
                    x1 + text_width + 6,
                    text_y + text_height + 4
                ],
                fill="red"
            )

            # Text
            draw.text(
                (
                    x1 + 3,
                    text_y + 2
                ),
                text,
                fill="white",
                font=font
            )

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        output_path = (
            OUTPUT_DIR /
            image_path.name
        )

        original_image.save(
            output_path
        )

        print(
            f"[{index:02d}/{len(image_files)}] "
            f"{image_path.name} "
            f"-> {detection_count} detections"
        )


# ============================================================
# FINISHED
# ============================================================

print("\n")
print("=" * 70)
print("VISUALIZATION COMPLETED")
print("=" * 70)

print(
    f"Saved {len(image_files)} images."
)

print(
    f"\nResults saved to:\n{OUTPUT_DIR}"
)

print("=" * 70)