import os
from pathlib import Path

import torch
import numpy as np
import cv2
import pytesseract

from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection import fasterrcnn_resnet50_fpn


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "runs" / "faster_rcnn" / "best.pt"

TEST_IMAGES = ROOT / "dataset" / "yolo" / "images" / "test"

OUTPUT_DIR = (
    ROOT
    / "runs"
    / "faster_rcnn"
    / "ocr_predictions"
)

CROPS_DIR = (
    ROOT
    / "runs"
    / "faster_rcnn"
    / "ocr_crops"
)

CONF_THRESHOLD = 0.50

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# TESSERACT
# ============================================================

TESSERACT_PATH = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

if not os.path.exists(TESSERACT_PATH):

    raise RuntimeError(
        "Tesseract was not found at:\n"
        f"{TESSERACT_PATH}\n\n"
        "Please check the Tesseract installation path."
    )

pytesseract.pytesseract.tesseract_cmd = (
    TESSERACT_PATH
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CROPS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("Faster R-CNN + OCR")
print("=" * 70)

print(f"Device : {DEVICE}")
print(f"Model  : {MODEL_PATH}")
print(f"Images : {TEST_IMAGES}")
print(f"Output : {OUTPUT_DIR}")
print(f"OCR    : {TESSERACT_PATH}")

print("=" * 70)


# ============================================================
# LOAD MODEL
# ============================================================

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
# CHECKPOINT
# ============================================================

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    elif (
        "model" in checkpoint
        and isinstance(
            checkpoint["model"],
            dict
        )
    ):

        state_dict = checkpoint[
            "model"
        ]

    else:

        state_dict = checkpoint

else:

    raise RuntimeError(
        "Unsupported checkpoint format."
    )


# Remove module. prefix
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
        if p.suffix.lower()
        in extensions
    ]
)

if not image_files:

    raise RuntimeError(
        f"No images found in:\n{TEST_IMAGES}"
    )


print(
    f"\nFound {len(image_files)} test images."
)

print("\nRunning detection + OCR...\n")


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
# OCR FUNCTION
# ============================================================

def run_ocr(crop):

    """
    Preprocess license plate crop
    and run Tesseract OCR.
    """

    if crop is None or crop.size == 0:

        return ""


    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    scale = 4

    crop = cv2.resize(
        crop,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )


    # --------------------------------------------------------
    # Grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------------
    # Noise reduction
    # --------------------------------------------------------

    gray = cv2.bilateralFilter(
        gray,
        9,
        75,
        75
    )


    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU
    )


    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    config = (
        "--oem 3 "
        "--psm 7 "
        "-c "
        "tessedit_char_whitelist="
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789"
    )

    text = pytesseract.image_to_string(
        thresh,
        config=config
    )


    # --------------------------------------------------------
    # Clean text
    # --------------------------------------------------------

    text = text.strip()

    text = " ".join(
        text.split()
    )

    return text


# ============================================================
# PROCESS
# ============================================================

with torch.no_grad():

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        image_np = np.array(image)

        height, width = image_np.shape[:2]


        # ----------------------------------------------------
        # Tensor
        # ----------------------------------------------------

        image_tensor = torch.from_numpy(
            image_np
        ).permute(
            2,
            0,
            1
        ).float() / 255.0

        image_tensor = image_tensor.to(
            DEVICE
        )


        # ----------------------------------------------------
        # Detection
        # ----------------------------------------------------

        output = model(
            [image_tensor]
        )[0]


        boxes = (
            output["boxes"]
            .detach()
            .cpu()
        )

        scores = (
            output["scores"]
            .detach()
            .cpu()
        )


        # ----------------------------------------------------
        # Create output image
        # ----------------------------------------------------

        result_image = image.copy()

        draw = ImageDraw.Draw(
            result_image
        )


        detection_count = 0


        # ----------------------------------------------------
        # Process detections
        # ----------------------------------------------------

        for det_idx, (
            box,
            score
        ) in enumerate(
            zip(
                boxes,
                scores
            )
        ):

            confidence = float(score)


            if confidence < CONF_THRESHOLD:

                continue


            x1, y1, x2, y2 = map(
                int,
                box.tolist()
            )


            # Clamp
            x1 = max(
                0,
                min(x1, width - 1)
            )

            y1 = max(
                0,
                min(y1, height - 1)
            )

            x2 = max(
                0,
                min(x2, width)
            )

            y2 = max(
                0,
                min(y2, height)
            )


            if x2 <= x1 or y2 <= y1:

                continue


            detection_count += 1


            # ------------------------------------------------
            # Crop license plate
            # ------------------------------------------------

            crop = image_np[
                y1:y2,
                x1:x2
            ]


            # ------------------------------------------------
            # OCR
            # ------------------------------------------------

            text = run_ocr(
                crop
            )


            # ------------------------------------------------
            # Save crop
            # ------------------------------------------------

            crop_name = (
                f"{image_path.stem}"
                f"_plate_{detection_count}"
                f".png"
            )

            crop_path = (
                CROPS_DIR
                / crop_name
            )


            cv2.imwrite(
                str(crop_path),
                crop
            )


            # ------------------------------------------------
            # Draw bounding box
            # ------------------------------------------------

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


            # ------------------------------------------------
            # Label
            # ------------------------------------------------

            if text:

                label_text = (
                    f"Plate: {text} "
                    f"({confidence:.2f})"
                )

            else:

                label_text = (
                    f"Plate: OCR failed "
                    f"({confidence:.2f})"
                )


            # ------------------------------------------------
            # Text size
            # ------------------------------------------------

            try:

                text_box = draw.textbbox(
                    (x1, y1),
                    label_text,
                    font=font
                )

                text_width = (
                    text_box[2]
                    - text_box[0]
                )

                text_height = (
                    text_box[3]
                    - text_box[1]
                )

            except:

                text_width = (
                    len(label_text) * 8
                )

                text_height = 15


            text_y = max(
                0,
                y1
                - text_height
                - 6
            )


            # ------------------------------------------------
            # Background
            # ------------------------------------------------

            draw.rectangle(
                [
                    x1,
                    text_y,
                    x1
                    + text_width
                    + 8,
                    text_y
                    + text_height
                    + 6
                ],
                fill="red"
            )


            # ------------------------------------------------
            # Text
            # ------------------------------------------------

            draw.text(
                (
                    x1 + 4,
                    text_y + 3
                ),
                label_text,
                fill="white",
                font=font
            )


            # ------------------------------------------------
            # Console
            # ------------------------------------------------

            print(
                f"    Plate {detection_count}: "
                f"confidence={confidence:.2f} | "
                f"OCR='{text}'"
            )


        # ----------------------------------------------------
        # Save final image
        # ----------------------------------------------------

        output_path = (
            OUTPUT_DIR
            / image_path.name
        )


        result_image.save(
            output_path
        )


        print(
            f"[{index:02d}/{len(image_files)}] "
            f"{image_path.name} "
            f"-> {detection_count} plates"
        )


# ============================================================
# DONE
# ============================================================

print("\n")
print("=" * 70)
print("Faster R-CNN + OCR COMPLETED")
print("=" * 70)

print(
    f"\nAnnotated images:\n"
    f"{OUTPUT_DIR}"
)

print(
    f"\nLicense plate crops:\n"
    f"{CROPS_DIR}"
)

print("=" * 70)