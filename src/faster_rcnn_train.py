from pathlib import Path

import cv2
import torch

from torch.utils.data import Dataset, DataLoader

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights
)

from torchvision.transforms import functional as F

from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor
)


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    BASE_DIR
    / "dataset"
    / "yolo"
)

TRAIN_IMAGES = (
    DATASET_DIR
    / "images"
    / "train"
)

TRAIN_LABELS = (
    DATASET_DIR
    / "labels"
    / "train"
)

VAL_IMAGES = (
    DATASET_DIR
    / "images"
    / "val"
)

VAL_LABELS = (
    DATASET_DIR
    / "labels"
    / "val"
)

TEST_IMAGES = (
    DATASET_DIR
    / "images"
    / "test"
)

TEST_LABELS = (
    DATASET_DIR
    / "labels"
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

MODEL_PATH = (
    OUTPUT_DIR
    / "best.pt"
)


# ============================================================
# Settings
# ============================================================

# 0 = background
# 1 = license plate

NUM_CLASSES = 2

BATCH_SIZE = 2

NUM_EPOCHS = 5

LEARNING_RATE = 0.005

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Dataset
# ============================================================

class LicensePlateDataset(Dataset):

    def __init__(
        self,
        images_dir,
        labels_dir
    ):

        self.images_dir = Path(
            images_dir
        )

        self.labels_dir = Path(
            labels_dir
        )

        self.images = []

        # ----------------------------------------------------
        # Find valid images
        # ----------------------------------------------------

        for image_path in sorted(
            self.images_dir.iterdir()
        ):

            if image_path.suffix.lower() not in {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp"
            }:

                continue

            try:

                image = cv2.imread(
                    str(image_path)
                )

                if image is None:

                    print(
                        "WARNING: "
                        f"Skipping unreadable image: "
                        f"{image_path.name}"
                    )

                    continue

                self.images.append(
                    image_path
                )

            except Exception as error:

                print(
                    "WARNING: "
                    f"Skipping {image_path.name}: "
                    f"{error}"
                )

        print(
            f"Valid images loaded: "
            f"{len(self.images)}"
        )


    # ========================================================
    # Length
    # ========================================================

    def __len__(self):

        return len(
            self.images
        )


    # ========================================================
    # Get Item
    # ========================================================

    def __getitem__(
        self,
        index
    ):

        image_path = (
            self.images[index]
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            raise RuntimeError(
                "Image became unreadable: "
                f"{image_path}"
            )

        height, width = (
            image.shape[:2]
        )

        # ----------------------------------------------------
        # BGR -> RGB
        # ----------------------------------------------------

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # Convert image to tensor
        # ----------------------------------------------------

        image = F.to_tensor(
            image
        )

        # ----------------------------------------------------
        # Label path
        # ----------------------------------------------------

        label_path = (
            self.labels_dir
            / f"{image_path.stem}.txt"
        )

        boxes = []

        labels = []

        # ----------------------------------------------------
        # Read YOLO annotations
        # ----------------------------------------------------

        if label_path.exists():

            with open(
                label_path,
                "r",
                encoding="utf-8"
            ) as file:

                for line in file:

                    parts = (
                        line.strip()
                        .split()
                    )

                    if len(parts) != 5:

                        continue

                    try:

                        class_id = int(
                            parts[0]
                        )

                        x_center = float(
                            parts[1]
                        )

                        y_center = float(
                            parts[2]
                        )

                        box_width = float(
                            parts[3]
                        )

                        box_height = float(
                            parts[4]
                        )

                    except ValueError:

                        continue

                    # ------------------------------------------------
                    # YOLO normalized coordinates
                    # -> Faster R-CNN pixel coordinates
                    # ------------------------------------------------

                    x1 = (
                        x_center
                        - box_width / 2
                    ) * width

                    y1 = (
                        y_center
                        - box_height / 2
                    ) * height

                    x2 = (
                        x_center
                        + box_width / 2
                    ) * width

                    y2 = (
                        y_center
                        + box_height / 2
                    ) * height

                    # ------------------------------------------------
                    # Clamp coordinates
                    # ------------------------------------------------

                    x1 = max(
                        0,
                        min(
                            x1,
                            width - 1
                        )
                    )

                    y1 = max(
                        0,
                        min(
                            y1,
                            height - 1
                        )
                    )

                    x2 = max(
                        0,
                        min(
                            x2,
                            width - 1
                        )
                    )

                    y2 = max(
                        0,
                        min(
                            y2,
                            height - 1
                        )
                    )

                    # ------------------------------------------------
                    # Ignore invalid boxes
                    # ------------------------------------------------

                    if (
                        x2 <= x1
                        or
                        y2 <= y1
                    ):

                        continue

                    boxes.append(
                        [
                            x1,
                            y1,
                            x2,
                            y2
                        ]
                    )

                    # Faster R-CNN:
                    # 0 = background
                    # 1 = license plate

                    labels.append(1)

        # ----------------------------------------------------
        # Convert boxes to tensors
        # ----------------------------------------------------

        boxes = torch.as_tensor(
            boxes,
            dtype=torch.float32
        )

        labels = torch.as_tensor(
            labels,
            dtype=torch.int64
        )

        # ----------------------------------------------------
        # Empty annotations
        # ----------------------------------------------------

        if len(boxes) == 0:

            boxes = torch.zeros(
                (0, 4),
                dtype=torch.float32
            )

            labels = torch.zeros(
                (0,),
                dtype=torch.int64
            )

        # ----------------------------------------------------
        # Area
        # ----------------------------------------------------

        area = (
            (
                boxes[:, 2]
                -
                boxes[:, 0]
            )
            *
            (
                boxes[:, 3]
                -
                boxes[:, 1]
            )
        )

        # ----------------------------------------------------
        # Crowd flag
        # ----------------------------------------------------

        iscrowd = torch.zeros(
            (
                len(boxes),
            ),
            dtype=torch.int64
        )

        # ----------------------------------------------------
        # Target dictionary
        # ----------------------------------------------------

        target = {

            "boxes": boxes,

            "labels": labels,

            "image_id": torch.tensor(
                [index]
            ),

            "area": area,

            "iscrowd": iscrowd
        }

        return (
            image,
            target
        )


# ============================================================
# Collate Function
# ============================================================

def collate_fn(batch):

    return tuple(
        zip(*batch)
    )


# ============================================================
# Create Faster R-CNN Model
# ============================================================

def create_model():

    print(
        "\nLoading pretrained "
        "Faster R-CNN..."
    )

    weights = (
        FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )

    model = (
        fasterrcnn_resnet50_fpn(
            weights=weights
        )
    )

    # --------------------------------------------------------
    # Replace COCO classifier
    # --------------------------------------------------------

    in_features = (
        model
        .roi_heads
        .box_predictor
        .cls_score
        .in_features
    )

    model.roi_heads.box_predictor = (
        FastRCNNPredictor(
            in_features,
            NUM_CLASSES
        )
    )

    return model


# ============================================================
# Training
# ============================================================

def train():

    print("=" * 70)

    print(
        "FASTER R-CNN "
        "LICENSE PLATE TRAINING"
    )

    print("=" * 70)

    print(
        f"\nDevice: {DEVICE}"
    )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print(
        "\nLoading dataset..."
    )

    train_dataset = (
        LicensePlateDataset(
            TRAIN_IMAGES,
            TRAIN_LABELS
        )
    )

    val_dataset = (
        LicensePlateDataset(
            VAL_IMAGES,
            VAL_LABELS
        )
    )

    print(
        f"\nTraining images: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation images: "
        f"{len(val_dataset)}"
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_model()

    model.to(
        DEVICE
    )

    # --------------------------------------------------------
    # Trainable parameters
    # --------------------------------------------------------

    params = [

        parameter

        for parameter
        in model.parameters()

        if parameter.requires_grad
    ]

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.SGD(

        params,

        lr=LEARNING_RATE,

        momentum=0.9,

        weight_decay=0.0005
    )

    print(
        "\nStarting training..."
    )

    print(
        f"Epochs: {NUM_EPOCHS}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Learning rate: "
        f"{LEARNING_RATE}"
    )

    # --------------------------------------------------------
    # Best loss
    # --------------------------------------------------------

    best_loss = float(
        "inf"
    )

    # ========================================================
    # Epoch Loop
    # ========================================================

    for epoch in range(
        NUM_EPOCHS
    ):

        model.train()

        epoch_loss = 0.0

        batch_count = 0

        print(
            "\n"
            + "-" * 70
        )

        print(
            f"Epoch "
            f"{epoch + 1}/{NUM_EPOCHS}"
        )

        print(
            "-" * 70
        )

        # ----------------------------------------------------
        # Batch Loop
        # ----------------------------------------------------

        for images, targets in train_loader:

            images = [

                image.to(
                    DEVICE
                )

                for image
                in images
            ]

            targets = [

                {

                    key: value.to(
                        DEVICE
                    )

                    for key, value
                    in target.items()

                }

                for target
                in targets
            ]

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            loss_dict = model(
                images,
                targets
            )

            losses = sum(
                loss

                for loss
                in loss_dict.values()
            )

            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            optimizer.zero_grad()

            losses.backward()

            optimizer.step()

            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            epoch_loss += (
                losses.item()
            )

            batch_count += 1

            if batch_count % 20 == 0:

                print(
                    f"Batch "
                    f"{batch_count}/"
                    f"{len(train_loader)} "
                    f"- Loss: "
                    f"{losses.item():.4f}"
                )

        # ----------------------------------------------------
        # Average loss
        # ----------------------------------------------------

        average_loss = (
            epoch_loss
            /
            max(
                1,
                len(train_loader)
            )
        )

        print(
            "\nEpoch result:"
        )

        print(
            f"Average Loss: "
            f"{average_loss:.4f}"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if average_loss < best_loss:

            best_loss = (
                average_loss
            )

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print(
                "\nBest model saved!"
            )

            print(
                MODEL_PATH
            )

    # ========================================================
    # Training Finished
    # ========================================================

    print("\n")

    print("=" * 70)

    print(
        "TRAINING COMPLETED"
    )

    print("=" * 70)

    print(
        f"\nBest training loss: "
        f"{best_loss:.4f}"
    )

    print(
        "\nModel saved to:"
    )

    print(
        MODEL_PATH
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    train()