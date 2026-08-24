from pathlib import Path
import random
import shutil
import xml.etree.ElementTree as ET

# =========================
# Configuration
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

IMAGES_DIR = BASE_DIR / "dataset" / "images"
ANNOTATIONS_DIR = BASE_DIR / "dataset" / "annotations"
OUTPUT_DIR = BASE_DIR / "dataset" / "yolo"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

SEED = 42

# License plate is the only class
CLASS_ID = 0


# =========================
# Create directories
# =========================

for split in ["train", "val", "test"]:
    (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


# =========================
# XML → YOLO
# =========================

def convert_xml_to_yolo(xml_file, label_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find("size")

    if size is None:
        raise ValueError(f"Missing <size> in {xml_file}")

    width = int(size.find("width").text)
    height = int(size.find("height").text)

    yolo_lines = []

    for obj in root.findall("object"):

        name = obj.find("name")

        if name is None:
            continue

        # Dataset should contain license plates.
        # We treat every annotated object as a license plate.
        bbox = obj.find("bndbox")

        if bbox is None:
            continue

        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # Convert Pascal VOC → YOLO
        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height

        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height

        # Keep values inside valid YOLO range
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        box_width = max(0, min(1, box_width))
        box_height = max(0, min(1, box_height))

        yolo_lines.append(
            f"{CLASS_ID} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{box_width:.6f} "
            f"{box_height:.6f}"
        )

    label_file.write_text("\n".join(yolo_lines), encoding="utf-8")


# =========================
# Find images
# =========================

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

images = [
    p for p in IMAGES_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in image_extensions
]

if not images:
    raise RuntimeError(f"No images found in {IMAGES_DIR}")


# =========================
# Match XML files
# =========================

pairs = []

for image in images:

    xml_file = ANNOTATIONS_DIR / f"{image.stem}.xml"

    if not xml_file.exists():
        print(f"Warning: annotation not found for {image.name}")
        continue

    pairs.append((image, xml_file))


if not pairs:
    raise RuntimeError("No image/XML pairs found.")


# =========================
# Shuffle dataset
# =========================

random.seed(SEED)
random.shuffle(pairs)

total = len(pairs)

train_end = int(total * TRAIN_RATIO)
val_end = train_end + int(total * VAL_RATIO)

train_data = pairs[:train_end]
val_data = pairs[train_end:val_end]
test_data = pairs[val_end:]


# =========================
# Copy files
# =========================

def process_split(data, split):

    for image_file, xml_file in data:

        destination_image = (
            OUTPUT_DIR / "images" / split / image_file.name
        )

        destination_label = (
            OUTPUT_DIR / "labels" / split / f"{image_file.stem}.txt"
        )

        shutil.copy2(image_file, destination_image)

        convert_xml_to_yolo(
            xml_file,
            destination_label
        )


process_split(train_data, "train")
process_split(val_data, "val")
process_split(test_data, "test")


# =========================
# Create data.yaml
# =========================

yaml_content = f"""path: {OUTPUT_DIR.as_posix()}
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: license_plate
"""

yaml_file = BASE_DIR / "data.yaml"
yaml_file.write_text(yaml_content, encoding="utf-8")


# =========================
# Summary
# =========================

print("\nDataset preparation completed!")
print("=" * 40)

print(f"Total images : {total}")
print(f"Train        : {len(train_data)}")
print(f"Validation   : {len(val_data)}")
print(f"Test         : {len(test_data)}")

print("\nYOLO dataset:")
print(OUTPUT_DIR)

print("\ndata.yaml:")
print(yaml_file)