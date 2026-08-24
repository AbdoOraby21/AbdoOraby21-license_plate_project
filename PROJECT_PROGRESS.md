# License Plate Detection & Recognition System

## Pattern Recognition Final Assignment

**University:** Badr University in Assiut
**Project Type:** Object Detection + OCR
**Main Technologies:** Python, YOLOv8, OpenCV, Tesseract OCR

---

# 1. Project Description

The project is a License Plate Detection and Recognition System.

The main objective is to build a complete computer vision pipeline:

```text
Input Image
    ↓
YOLOv8 License Plate Detection
    ↓
Bounding Box
    ↓
License Plate Cropping
    ↓
OCR
    ↓
Plate Number
```

The system is designed for real-world applications such as:

* Traffic monitoring
* Parking systems
* Vehicle security
* License plate recognition

---

# 2. Assignment Requirements

The assignment requires:

### Dataset

Use either:

* A public license plate dataset
* OR a custom dataset with at least 100 images

Annotations must contain license plate bounding boxes in YOLO format.

### Model

* Train a YOLO model for license plate detection.
* Use pretrained weights / transfer learning.

### Pipeline

The system must implement:

1. License plate detection
2. License plate cropping
3. OCR text extraction

### Output

The final output should contain:

* Bounding box around the license plate
* Extracted license plate number

Example:

```text
Plate: ABC123
```

### Experiments

At least two OCR experiments are required:

1. Baseline OCR without preprocessing
2. Improved OCR with image preprocessing

### Evaluation

The assignment evaluates:

* Detection Performance — 25%
* OCR Accuracy — 20%
* Pipeline Integration — 20%
* Experiments & Improvements — 10%
* Visualization — 10%
* Code Quality — 5%
* Explanation & Understanding — 10%

---

# 3. Dataset

The project uses a license plate dataset containing vehicle images and license plate annotations.

The annotations were converted/prepared in YOLO format.

Expected structure:

```text
dataset/
└── yolo/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    │
    └── labels/
        ├── train/
        ├── val/
        └── test/
```

Each YOLO annotation follows:

```text
class_id x_center y_center width height
```

The license plate is represented by a bounding box.

---

# 4. YOLOv8 Detection

A YOLOv8 model was trained for license plate detection.

The trained model is stored at:

```text
runs/
└── license_plate_detection/
    └── weights/
        └── best.pt
```

The trained model can detect license plates and return:

* Bounding box coordinates
* Detection confidence
* License plate location

The detection pipeline is working.

---

# 5. Detection Pipeline

The implemented pipeline follows:

```text
Vehicle Image
      ↓
YOLOv8
      ↓
License Plate Detection
      ↓
Highest Confidence Bounding Box
      ↓
Crop License Plate
      ↓
OCR
```

The system selects the highest-confidence detected license plate.

The detected plate is cropped from the original image.

Cropped plates are saved in:

```text
runs/
└── ocr_evaluation/
    └── plate_crops/
```

---

# 6. OCR Implementation

Tesseract OCR is used for text recognition.

Tesseract executable:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

The OCR module contains:

```python
baseline_ocr()
improved_ocr()
```

---

# 7. Baseline OCR

The baseline performs OCR directly on the detected license plate crop.

No image preprocessing is applied.

Pipeline:

```text
Plate Crop
    ↓
Tesseract OCR
    ↓
Clean Text
```

Tesseract uses:

```text
--psm 7
```

and a whitelist containing:

```text
ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789
```

The purpose of the baseline is to provide a reference for comparing the improved OCR.

---

# 8. Initial Improved OCR

The first improved OCR version used:

```text
Grayscale
    ↓
Upscaling
    ↓
Gaussian Blur
    ↓
Otsu Threshold
    ↓
Tesseract OCR
```

The purpose was to improve:

* Character visibility
* Image size
* Noise reduction
* Contrast between characters and plate background

---

# 9. First OCR Evaluation

Manual OCR evaluation was performed.

For each image, the system displayed:

* Image name
* Detection confidence
* Baseline OCR result
* Improved OCR result
* Cropped plate location

The real plate number was then entered manually as Ground Truth.

The results were saved to:

```text
runs/
└── ocr_evaluation/
    └── ocr_results.csv
```

A total of:

```text
42 images
```

were evaluated.

---

# 10. Initial OCR Results

The first complete evaluation produced:

```text
Total evaluated images: 42

Baseline Character Accuracy : 42.20%
Improved Character Accuracy : 36.74%

Baseline Exact Match : 19.05%
Improved Exact Match : 16.67%
```

Comparison:

```text
Improved OCR better : 9 images
Baseline OCR better : 11 images
Same result         : 22 images
```

OCR failures:

```text
Baseline OCR failures : 9
Improved OCR failures : 11
```

Overall:

```text
Improved OCR decreased character accuracy
by 5.45 percentage points.
```

---

# 11. Example OCR Results

Some examples from the evaluation:

| Image       | Ground Truth | Baseline   | Improved   |
| ----------- | ------------ | ---------- | ---------- |
| Cars109.png | CZI7KOD      | CZI7KOD    | ACZI7KOD   |
| Cars11.png  | WOR5I6K      | WORSIGK    | WORSIGK    |
| Cars111.png | MH20EE7598   | MH20EE7598 | MH20EE7598 |
| Cars112.png | SHAKNBK      | Empty      | Empty      |
| Cars140.png | CH01AN0001   | Empty      | Empty      |
| Cars145.png | CH01AN0001   | Empty      | Empty      |
| Cars15.png  | TM21BY0166   | NM         | Empty      |
| Cars162.png | MAHINDRA     | TAHIR      | TATINES    |

These results demonstrate that OCR performance varies significantly depending on:

* Image quality
* Plate size
* Character clarity
* Lighting
* Blur
* Plate orientation
* Detection crop quality

---

# 12. Problems Encountered

## Problem 1 — OpenCV Window Freezing

The original OCR evaluation script used:

```python
cv2.imshow()
```

The OpenCV window sometimes became:

```text
Not Responding
```

This happened because the program was waiting for user input while also handling the OpenCV GUI window.

### Solution

The evaluation script was changed so that it no longer depends on `cv2.imshow()`.

Instead, the detected plate crop is saved to:

```text
runs/ocr_evaluation/plate_crops/
```

The user can open the crop manually and enter the Ground Truth in the terminal.

This made the evaluation more stable.

---

# 13. Problem 2 — OCR Results Were Inconsistent

The initial improved OCR did not always improve recognition.

For example:

```text
Ground Truth:
CZI7KOD

Baseline:
CZI7KOD

Improved:
ACZI7KOD
```

The improved version introduced an incorrect extra character.

Another example:

```text
Ground Truth:
TM21BY0166

Baseline:
NM

Improved:
Empty
```

This demonstrates that preprocessing can sometimes remove or distort characters instead of improving them.

---

# 14. Problem 3 — Initial OCR Accuracy Calculation

The original character accuracy function compared characters at the same positions.

For example:

```text
Ground Truth: CZI7KOD
Prediction:   ACZI7KOD
```

This type of calculation does not properly account for:

* Insertions
* Deletions
* Character shifts

Therefore, the current character accuracy metric is useful as an initial measurement, but it is not the strongest possible OCR evaluation metric.

### Planned Solution

Implement a Levenshtein-distance-based metric.

Possible metrics:

* Character Error Rate (CER)
* Edit Distance
* Normalized OCR Accuracy
* Exact Match Accuracy

---

# 15. Current Improved OCR

The Improved OCR has now been upgraded to a stronger preprocessing pipeline.

Current pipeline:

```text
Plate Crop
    ↓
Grayscale
    ↓
Upscaling ×4
    ↓
CLAHE Contrast Enhancement
    ↓
Bilateral Filtering
    ↓
Adaptive Threshold
    ↓
Morphological Cleaning
    ↓
Tesseract OCR
```

The improved version is designed to handle:

* Uneven lighting
* Low contrast
* Noise
* Small characters
* Different plate conditions

The new version needs to be evaluated against the same 42 images.

---

# 16. Important Experimental Rule

The old results must not be deleted.

The old experiment represents:

```text
Baseline
vs
Initial Improved OCR
```

The new experiment represents:

```text
Baseline
vs
Improved OCR v2
```

This allows the project to demonstrate iterative improvement.

The old CSV should be preserved:

```text
runs/ocr_evaluation/ocr_results.csv
```

If the new evaluation overwrites it, the old results should first be backed up.

---

# 17. Current Project Status

## Completed

### Dataset

```text
DONE
```

### YOLO annotations

```text
DONE
```

### YOLOv8 training

```text
DONE
```

### License plate detection

```text
DONE
```

### License plate cropping

```text
DONE
```

### Tesseract installation

```text
DONE
```

### Baseline OCR

```text
DONE
```

### Improved OCR

```text
IMPLEMENTED
```

### OCR evaluation

```text
DONE - 42 images
```

### CSV results

```text
DONE
```

---

# 18. Remaining Tasks

The following tasks still need to be completed.

## Task 1 — Re-run Improved OCR

Run the new preprocessing pipeline on the same 42 images.

Command:

```powershell
python src/ocr_evaluation.py
```

Use the same Ground Truth values.

---

## Task 2 — Analyze New OCR Results

Run:

```powershell
python src/analyze_ocr_results.py
```

Compare:

* Baseline Character Accuracy
* Improved Character Accuracy
* Baseline Exact Match
* Improved Exact Match
* OCR failures
* Number of images improved

---

## Task 3 — Improve OCR Evaluation Metric

Implement Levenshtein Distance / Character Error Rate.

The evaluation should ideally report:

```text
Character Accuracy
Character Error Rate
Exact Match Accuracy
```

---

## Task 4 — Detection Evaluation

YOLO detection metrics still need to be collected.

Required metrics:

```text
Precision
Recall
mAP50
mAP50-95
```

These should be generated using the YOLO validation/test dataset.

---

## Task 5 — Final Visualization

Create a final detection + recognition script.

Expected output:

```text
┌─────────────────────────────┐
│                             │
│       Vehicle Image         │
│                             │
│       ┌──────────────┐      │
│       │ License Plate│      │
│       └──────────────┘      │
│                             │
│       Plate: ABC123         │
│                             │
└─────────────────────────────┘
```

The final image should contain:

* Bounding box
* Detection confidence
* OCR result

Example:

```text
Plate: CZI7KOD
Confidence: 0.91
```

---

# 19. Final Pipeline

The final system should be:

```text
                INPUT
                  │
                  ▼
           Vehicle Image
                  │
                  ▼
              YOLOv8
                  │
                  ▼
        License Plate Detection
                  │
                  ▼
          Bounding Box
                  │
                  ▼
          Plate Cropping
                  │
                  ▼
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
   Baseline OCR       Improved OCR
        │                   │
        ▼                   ▼
    Tesseract           Preprocessing
        │                   │
        └─────────┬─────────┘
                  ▼
           OCR Evaluation
                  │
                  ▼
          Final Plate Text
```

---

# 20. Final Demonstration

The final project should support:

```text
Input:
car.jpg

Output:

License Plate Detected
Confidence: 0.93

Plate:
ABC123
```

The system should work on:

* Images

And optionally:

* Video
* Webcam

---

# 21. Optional Bonus Features

The assignment provides optional bonus features.

Possible additions:

### Faster R-CNN Comparison

Compare YOLOv8 with Faster R-CNN.

### Real-Time Webcam

Run detection and OCR from a webcam.

### Database

Store:

```text
Plate Number
Date
Time
Confidence
Image
```

These features are optional and should only be implemented after completing the required tasks.

---

# 22. Final Report Structure

The final report should contain:

## 1. Introduction

Explain the problem and why license plate recognition is useful.

## 2. Dataset

Explain:

* Dataset source
* Number of images
* Annotation format
* Train/Validation/Test split

## 3. Methodology

Explain:

* YOLOv8
* Transfer learning
* Detection
* Cropping
* Tesseract OCR
* Preprocessing

## 4. System Pipeline

Show:

```text
Detection → Cropping → OCR
```

## 5. Detection Results

Report:

* Precision
* Recall
* mAP50
* mAP50-95

## 6. OCR Results

Compare:

| Metric             | Baseline | Improved |
| ------------------ | -------: | -------: |
| Character Accuracy |      TBD |      TBD |
| Exact Match        |      TBD |      TBD |
| CER                |      TBD |      TBD |

## 7. Experiments

Explain the difference between:

* Raw OCR
* Preprocessed OCR

## 8. Error Analysis

Discuss why OCR fails.

Examples:

* Small plate
* Blur
* Poor lighting
* Low contrast
* Incorrect detection crop
* Similar characters such as O/0 and S/5

## 9. Visualization

Include example outputs showing:

* Bounding box
* Plate number

## 10. Conclusion

Summarize the final results and limitations.

---

# 23. Current Priority

The next steps should be completed in this exact order:

```text
1. Re-run OCR with Improved OCR v2
          ↓
2. Analyze OCR results
          ↓
3. Implement better OCR evaluation metric
          ↓
4. Calculate YOLO Precision / Recall / mAP
          ↓
5. Create final visualization
          ↓
6. Test complete pipeline
          ↓
7. Prepare final report
          ↓
8. Prepare GitHub repository
```

---

# 24. Current Overall Status

```text
Dataset                  ████████████████████ 100%
YOLOv8 Detection         ████████████████████ 100%
Cropping                 ████████████████████ 100%
Baseline OCR             ████████████████████ 100%
Improved OCR             ███████████████░░░░░  75%
OCR Evaluation           █████████████████░░░  85%
Detection Metrics        ░░░░░░░░░░░░░░░░░░░░   0%
Final Visualization      ░░░░░░░░░░░░░░░░░░░░   0%
Final Report             ░░░░░░░░░░░░░░░░░░░░   0%
GitHub Finalization      ░░░░░░░░░░░░░░░░░░░░   0%
```

## Overall

The core License Plate Detection → Cropping → OCR pipeline is already implemented.

The main remaining work is **evaluation, final visualization, improving OCR reliability, and documenting the results**.
