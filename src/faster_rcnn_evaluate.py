import os
import yaml
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import box_iou


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "runs" / "faster_rcnn" / "best.pt"
DATA_YAML = ROOT / "data.yaml"

CONF_THRESHOLD = 0.05
IOU_THRESHOLD = 0.50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# LOAD DATASET CONFIG
# ============================================================

with open(DATA_YAML, "r", encoding="utf-8") as f:
    data_config = yaml.safe_load(f)

# Find test image directory
test_path = data_config.get("test")

if test_path is None:
    raise RuntimeError("Could not find 'test:' inside data.yaml")

# Use "path:" from data.yaml as the dataset root
dataset_root = data_config.get("path")

if dataset_root:
    dataset_root = Path(dataset_root)

    if not dataset_root.is_absolute():
        dataset_root = ROOT / dataset_root
else:
    dataset_root = ROOT

test_path = Path(test_path)

if not test_path.is_absolute():
    test_path = dataset_root / test_path

test_path = test_path.resolve()

# Labels directory
labels_path = dataset_root / "labels" / "test"
labels_path = labels_path.resolve()

test_path = test_path.resolve()

# YOLO labels are normally:
# dataset/yolo/images/test
# dataset/yolo/labels/test

if "images" in test_path.parts:
    labels_path = Path(
        str(test_path).replace(
            os.sep + "images" + os.sep,
            os.sep + "labels" + os.sep
        )
    )
else:
    labels_path = ROOT / "dataset" / "yolo" / "labels" / "test"


# ============================================================
# FIND TEST IMAGES
# ============================================================

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

image_files = sorted(
    [
        p for p in test_path.rglob("*")
        if p.suffix.lower() in image_extensions
    ]
)

if len(image_files) == 0:
    raise RuntimeError(
        f"No test images found in:\n{test_path}"
    )

print("=" * 70)
print("Faster R-CNN Evaluation")
print("=" * 70)

print(f"Device       : {DEVICE}")
print(f"Model        : {MODEL_PATH}")
print(f"Test images  : {test_path}")
print(f"Test labels  : {labels_path}")
print(f"Images count : {len(image_files)}")
print("=" * 70)


# ============================================================
# CREATE MODEL
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
# HANDLE DIFFERENT CHECKPOINT FORMATS
# ============================================================

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    elif "model" in checkpoint and isinstance(
        checkpoint["model"], dict
    ):
        state_dict = checkpoint["model"]

    else:
        # Assume checkpoint itself is state_dict
        state_dict = checkpoint

else:
    raise RuntimeError(
        "Unsupported checkpoint format."
    )


# Remove possible DataParallel prefix
clean_state_dict = {}

for key, value in state_dict.items():
    new_key = key

    if new_key.startswith("module."):
        new_key = new_key[7:]

    clean_state_dict[new_key] = value


missing, unexpected = model.load_state_dict(
    clean_state_dict,
    strict=False
)

if missing:
    print("\nWarning: Missing keys:")
    print(missing[:10])

if unexpected:
    print("\nWarning: Unexpected keys:")
    print(unexpected[:10])

model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# ============================================================
# YOLO LABEL LOADER
# ============================================================

def load_yolo_labels(label_file, image_width, image_height):
    """
    Load YOLO format labels.

    Format:
    class_id x_center y_center width height

    All coordinates are normalized [0,1].
    """

    boxes = []
    labels = []

    if not label_file.exists():
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.int64)
        )

    with open(label_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) < 5:
            continue

        class_id = int(float(parts[0]))

        xc = float(parts[1])
        yc = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])

        # YOLO normalized -> pixel coordinates

        x1 = (xc - w / 2) * image_width
        y1 = (yc - h / 2) * image_height
        x2 = (xc + w / 2) * image_width
        y2 = (yc + h / 2) * image_height

        # Clamp coordinates
        x1 = max(0, min(x1, image_width))
        y1 = max(0, min(y1, image_height))
        x2 = max(0, min(x2, image_width))
        y2 = max(0, min(y2, image_height))

        boxes.append([x1, y1, x2, y2])

        # Faster R-CNN:
        # 0 = background
        # 1 = license plate
        labels.append(class_id + 1)

    if len(boxes) == 0:
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.int64)
        )

    return (
        torch.tensor(boxes, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.int64)
    )


# ============================================================
# AP CALCULATION
# ============================================================

def compute_ap(recalls, precisions):
    """
    Compute AP using the 101-point interpolation method.
    """

    recalls = np.asarray(recalls)
    precisions = np.asarray(precisions)

    mrec = np.concatenate(
        ([0.0], recalls, [1.0])
    )

    mpre = np.concatenate(
        ([0.0], precisions, [0.0])
    )

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(
            mpre[i - 1],
            mpre[i]
        )

    recall_points = np.linspace(0, 1, 101)

    ap = 0.0

    for r in recall_points:
        indices = np.where(mrec >= r)[0]

        if len(indices) > 0:
            ap += np.max(mpre[indices])

    ap /= 101.0

    return ap


# ============================================================
# COLLECT DETECTIONS
# ============================================================

all_predictions = []
all_ground_truths = []

total_gt = 0
total_predictions = 0

print("\nRunning inference...\n")

with torch.no_grad():

    for index, image_path in enumerate(image_files, start=1):

        image = Image.open(image_path).convert("RGB")

        width, height = image.size

        image_tensor = torch.from_numpy(
            np.array(image)
        ).permute(2, 0, 1).float() / 255.0

        image_tensor = image_tensor.to(DEVICE)

        output = model(
            [image_tensor]
        )[0]

        pred_boxes = output["boxes"].detach().cpu()
        pred_scores = output["scores"].detach().cpu()
        pred_labels = output["labels"].detach().cpu()

        # Confidence filtering
        keep = pred_scores >= CONF_THRESHOLD

        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]
        pred_labels = pred_labels[keep]

        # Ground truth label file
        label_file = (
            labels_path /
            f"{image_path.stem}.txt"
        )

        gt_boxes, gt_labels = load_yolo_labels(
            label_file,
            width,
            height
        )

        all_predictions.append({
            "boxes": pred_boxes,
            "scores": pred_scores,
            "labels": pred_labels
        })

        all_ground_truths.append({
            "boxes": gt_boxes,
            "labels": gt_labels
        })

        total_gt += len(gt_boxes)
        total_predictions += len(pred_boxes)

        print(
            f"[{index:02d}/{len(image_files)}] "
            f"{image_path.name} | "
            f"GT: {len(gt_boxes)} | "
            f"Pred: {len(pred_boxes)}"
        )


# ============================================================
# METRICS
# ============================================================

def evaluate_at_iou(iou_threshold):
    """
    Calculate Precision, Recall and AP
    for one IoU threshold.
    """

    detections = []

    total_gt = 0

    # Prepare detections
    for image_idx in range(len(image_files)):

        preds = all_predictions[image_idx]
        gts = all_ground_truths[image_idx]

        gt_boxes = gts["boxes"]
        gt_labels = gts["labels"]

        total_gt += len(gt_boxes)

        for i in range(len(preds["boxes"])):

            detections.append({
                "image_idx": image_idx,
                "box": preds["boxes"][i],
                "score": float(preds["scores"][i]),
                "label": int(preds["labels"][i])
            })

    # Sort by confidence
    detections.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    matched = {}

    for image_idx in range(len(image_files)):
        matched[image_idx] = set()

    tp = []
    fp = []

    for detection in detections:

        image_idx = detection["image_idx"]

        pred_box = detection["box"]
        pred_label = detection["label"]

        gt_boxes = all_ground_truths[
            image_idx
        ]["boxes"]

        gt_labels = all_ground_truths[
            image_idx
        ]["labels"]

        best_iou = 0.0
        best_gt_idx = -1

        if len(gt_boxes) > 0:

            ious = box_iou(
                pred_box.unsqueeze(0),
                gt_boxes
            )[0]

            for gt_idx in range(len(gt_boxes)):

                # Class must match
                if pred_label != int(
                    gt_labels[gt_idx]
                ):
                    continue

                if gt_idx in matched[image_idx]:
                    continue

                current_iou = float(
                    ious[gt_idx]
                )

                if current_iou > best_iou:
                    best_iou = current_iou
                    best_gt_idx = gt_idx

        if (
            best_gt_idx >= 0
            and best_iou >= iou_threshold
        ):
            tp.append(1)
            fp.append(0)

            matched[image_idx].add(
                best_gt_idx
            )

        else:
            tp.append(0)
            fp.append(1)

    if len(tp) == 0:
        return 0.0, 0.0, 0.0

    tp = np.array(tp)
    fp = np.array(fp)

    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)

    recalls = cumulative_tp / max(
        total_gt, 1
    )

    precisions = cumulative_tp / np.maximum(
        cumulative_tp + cumulative_fp,
        1e-9
    )

    ap = compute_ap(
        recalls,
        precisions
    )

    final_precision = (
        cumulative_tp[-1] /
        max(
            cumulative_tp[-1] +
            cumulative_fp[-1],
            1
        )
    )

    final_recall = (
        cumulative_tp[-1] /
        max(total_gt, 1)
    )

    return (
        final_precision,
        final_recall,
        ap
    )


# ============================================================
# mAP@50
# ============================================================

precision50, recall50, map50 = evaluate_at_iou(
    0.50
)


# ============================================================
# mAP@50-95
# ============================================================

aps = []

for iou_threshold in np.arange(
    0.50,
    0.96,
    0.05
):

    _, _, ap = evaluate_at_iou(
        float(iou_threshold)
    )

    aps.append(ap)

map5095 = float(
    np.mean(aps)
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL FASTER R-CNN RESULTS")
print("=" * 70)

print(f"Test Images       : {len(image_files)}")
print(f"Ground Truth      : {total_gt}")
print(f"Predictions       : {total_predictions}")

print()
print(f"Precision          : {precision50 * 100:.2f}%")
print(f"Recall             : {recall50 * 100:.2f}%")
print(f"mAP@50             : {map50 * 100:.2f}%")
print(f"mAP@50-95          : {map5095 * 100:.2f}%")

print("=" * 70)

print("\nComparison with YOLOv8:")
print("-" * 70)

print(
    f"{'Metric':<20}"
    f"{'YOLOv8':>15}"
    f"{'Faster R-CNN':>20}"
)

print("-" * 70)

print(
    f"{'Precision':<20}"
    f"{95.26:>14.2f}%"
    f"{precision50 * 100:>19.2f}%"
)

print(
    f"{'Recall':<20}"
    f"{89.58:>14.2f}%"
    f"{recall50 * 100:>19.2f}%"
)

print(
    f"{'mAP@50':<20}"
    f"{92.15:>14.2f}%"
    f"{map50 * 100:>19.2f}%"
)

print(
    f"{'mAP@50-95':<20}"
    f"{51.70:>14.2f}%"
    f"{map5095 * 100:>19.2f}%"
)

print("-" * 70)