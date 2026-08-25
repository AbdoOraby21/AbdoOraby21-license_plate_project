# License Plate Detection & Recognition System

## Pattern Recognition Final Assignment

**Project:** License Plate Detection & Recognition System

**Category:** Object Detection + OCR

---

# 1. Project Overview

This project implements a complete license plate detection and recognition pipeline.

The system uses:

1. YOLOv8 for license plate detection.
2. Image cropping to extract the detected license plate.
3. Tesseract OCR for license plate text recognition.
4. Image preprocessing as an improved OCR approach.
5. Evaluation metrics for both detection and OCR.
6. Visualization of detection and recognition results.
7. SQLite database storage as an optional bonus feature.
8. Faster R-CNN as a second detector, trained and compared against YOLOv8.
9. Real-time webcam detection as an optional bonus feature.

The complete pipeline is:

```text
Input Image
     |
     v
YOLOv8 Detection
     |
     v
License Plate Bounding Box
     |
     v
Crop License Plate
     |
     +----------------------+
     |                      |
     v                      v
Baseline OCR          Improved OCR
     |                      |
     v                      v
Tesseract              Preprocessing
     |                      |
     +----------+-----------+
                |
                v
        Recognized Plate
                |
                v
        Display / Database
```

---

# 2. Objectives

The main objectives of this project are:

* Detect vehicle license plates in images using YOLOv8.
* Crop detected license plates.
* Extract plate text using OCR.
* Compare baseline OCR with an improved preprocessing-based OCR.
* Evaluate license plate detection performance.
* Evaluate OCR recognition accuracy.
* Visualize the final results.
* Integrate the complete detection-to-recognition pipeline.
* Compare YOLOv8 with a second detector (Faster R-CNN).
* Demonstrate real-time detection through a live webcam feed.
* Demonstrate an optional database integration.

---

# 3. Dataset

The project uses a license plate image dataset converted into YOLO format.

The dataset contains:

* Training images
* Validation images
* Test images
* YOLO-format annotation files

The annotations contain bounding boxes around license plates.

The YOLO annotation format is:

```text
class_id x_center y_center width height
```

All coordinates are normalized between 0 and 1.

Dataset split used in the project:

```text
Training images: approximately 303
Validation images: approximately 86
Test images: 44
```

The test set contains 44 images and 48 license plate instances.

---

# 4. Project Structure

```text
license_plate_project/
│
├── dataset/
│   └── yolo/
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       │
│       ├── labels/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       │
│       └── data.yaml
│
├── src/
│   ├── ocr.py
│   ├── ocr_evaluation.py
│   ├── analyze_ocr_results.py
│   ├── evaluate_detection.py
│   ├── final_demo.py
│   ├── plate_database.py
│   ├── view_database.py
│   ├── faster_rcnn_train.py
│   ├── faster_rcnn_visualize.py
│   ├── faster_rcnn_ocr.py
│   ├── faster_rcnn_evaluate.py
│   └── webcam_detect.py
│
├── runs/
│   ├── license_plate_detection/
│   ├── ocr_evaluation/
│   ├── final_demo/
│   └── faster_rcnn/
│       ├── best.pt
│       ├── predictions/
│       ├── ocr_predictions/
│       └── ocr_crops/
│
├── requirements.txt
├── README.md
└── ...
```

---

# 5. YOLOv8 Detection

## How YOLO Works

YOLO stands for:

```text
You Only Look Once
```

YOLO is a one-stage object detection algorithm.

Instead of running a separate process for finding regions and then classifying those regions, YOLO processes the image in a single neural network pipeline.

The input image is passed through the network and the model predicts:

* Bounding box coordinates
* Object confidence
* Object class

For this project, the model has one object class:

```text
license plate
```

The basic process is:

```text
Image
  |
  v
YOLOv8 Neural Network
  |
  +-------------------+
  |                   |
  v                   v
Bounding Box       Confidence
  |
  v
License Plate
```

YOLOv8 uses a convolutional neural network architecture to extract visual features from the image.

The network learns patterns such as:

* Plate shapes
* Plate edges
* Plate position
* Visual characteristics of license plates

During inference, YOLO predicts bounding boxes and confidence scores.

A confidence threshold is then used to determine which detections should be accepted.

In this project, the detection confidence threshold used in the final pipeline is approximately:

```text
0.40
```

The detection with the highest confidence is selected when processing an image containing a license plate.

---

# 6. Transfer Learning

The project uses pretrained YOLOv8 weights.

Instead of training a detection model completely from scratch, pretrained weights are used and adapted to the license plate detection task.

This approach is called:

```text
Transfer Learning
```

The pretrained model already contains useful visual features learned from a large dataset.

The model is then fine-tuned using the license plate dataset.

The same transfer learning approach is used for the Faster R-CNN bonus model (Section 16), starting from COCO-pretrained ResNet-50 FPN weights.

---

# 7. Detection Pipeline

The detection stage works as follows:

```text
Input Image
     |
     v
YOLOv8
     |
     v
Detect License Plate
     |
     v
Bounding Box
     |
     v
Crop Plate
```

The detected bounding box is converted into image coordinates and used to crop the license plate.

The crop is then passed to the OCR stage.

---

# 8. OCR System

Tesseract OCR is used for text recognition.

The OCR implementation contains two approaches.

## 8.1 Baseline OCR

The baseline approach sends the cropped plate directly to Tesseract.

```text
Detected Plate
      |
      v
Tesseract OCR
      |
      v
Recognized Text
```

The configuration used is:

```text
--psm 7
```

The OCR output is cleaned by:

* Converting characters to uppercase.
* Removing spaces.
* Removing special characters.
* Keeping only letters and numbers.

---

# 9. Improved OCR

The improved OCR version applies image preprocessing before sending the image to Tesseract.

The preprocessing pipeline is:

```text
License Plate Crop
        |
        v
Grayscale
        |
        v
Upscaling
        |
        v
Gaussian Blur
        |
        v
Otsu Threshold
        |
        v
Tesseract OCR
```

The image is upscaled by:

```text
3x
```

Gaussian blur is then applied to reduce noise.

Finally, Otsu thresholding converts the image into a binary image.

---

# 10. OCR Evaluation (YOLOv8 Detections)

The OCR system was evaluated using manually entered ground-truth plate numbers, on plates cropped from YOLOv8 detections.

The evaluation contains:

* Character accuracy
* Exact-match accuracy
* Comparison between baseline and improved OCR
* OCR failure count

The final V3 OCR evaluation produced:

```text
Total images: 42
```

### Character Accuracy

```text
Baseline : 41.84%
Improved : 30.27%
```

### Exact Match

```text
Baseline : 28.57%
Improved : 19.05%
```

### Comparison

```text
Improved OCR better : 6
Baseline OCR better : 9
Same result         : 27
```

### OCR Failures

```text
Baseline failures : 10
Improved failures : 17
```

### Overall Difference

```text
Improved OCR decreased accuracy
by 11.57 percentage points.
```

These results show that the implemented preprocessing pipeline did not improve OCR performance on this dataset.

This is an important experimental result because preprocessing does not always improve OCR.

---

# 11. Example OCR Results (YOLOv8 Detections)

Examples from the evaluation include:

```text
Ground Truth : CZI7KOD
Baseline     : CZI7KOD
Improved     : ACZI7KOD
```

Baseline:

```text
100% character accuracy
Exact match: YES
```

Improved:

```text
0% exact match
```

Another example:

```text
Ground Truth : MH20EE7598
Baseline     : MH20EE7598
Improved     : MH20EE7598
```

Both methods correctly recognized the plate.

Another example:

```text
Ground Truth : CH01AN0001
Baseline     : empty
Improved     : empty
```

This represents an OCR failure.

---

# 12. Detection Evaluation

The YOLOv8 model was evaluated on the test dataset.

Test set:

```text
44 images
48 license plate instances
```

The final detection results were:

| Metric    | Result |
| --------- | -----: |
| Precision | 95.26% |
| Recall    | 89.58% |
| mAP@50    | 88.82% |
| mAP@50-95 | 50.28% |

These results show that the YOLOv8 detector performs well at locating license plates.

> **Note:** Section 16 reports a second YOLOv8 evaluation run (92.15% / 51.70% for mAP@50 / mAP@50-95), produced by the comparison script used to benchmark YOLOv8 against Faster R-CNN under identical conditions. The small differences between the two runs come from that script's evaluation settings and do not change the overall conclusion: YOLOv8 detects plates with high precision and strong overlap accuracy.

---

# 13. Detection Metrics

## Precision

Precision measures how many of the detected plates were actually correct.

```text
Precision = TP / (TP + FP)
```

The project achieved:

```text
95.26%
```

---

## Recall

Recall measures how many of the actual license plates were successfully detected.

```text
Recall = TP / (TP + FN)
```

The project achieved:

```text
89.58%
```

---

## mAP@50

mAP@50 measures mean Average Precision using an IoU threshold of 0.50.

The project achieved:

```text
88.82%
```

---

## mAP@50-95

This metric evaluates detection performance over multiple IoU thresholds from 0.50 to 0.95.

The project achieved:

```text
50.28%
```

---

# 14. Final End-to-End Demo

A final demo was implemented to combine detection and OCR.

The pipeline is:

```text
Input Image
     |
     v
YOLOv8
     |
     v
License Plate Detection
     |
     v
Crop
     |
     v
OCR
     |
     v
Plate Number
     |
     v
Output Image
```

The demo was tested on the first 10 test images.

Example results:

```text
Cars109.png
Detection confidence: 0.895
Recognized plate: CZI7KOD
```

```text
Cars11.png
Detection confidence: 0.863
Recognized plate: WORSIGK
```

```text
Cars111.png
Detection confidence: 0.829
Recognized plate: MH20EE7598
```

Some images produced:

```text
NOT RECOGNIZED
```

while some images had no detected license plate.

All demo results were saved under:

```text
runs/final_demo/
```

---

# 15. Optional Bonus: Database Integration

A SQLite database was added as an optional bonus.

The database stores recognized license plates.

The stored information includes:

* ID
* Plate number
* Detection confidence
* Image name
* Detection date/time

The database allows detected plates to be stored instead of only displaying them on screen.

The database files are located under the project:

```text
runs/
```

The database functionality is implemented using:

```text
src/plate_database.py
```

and can be viewed using:

```text
src/view_database.py
```

This provides a basic real-world application scenario such as:

```text
Parking System
Traffic Monitoring
Security System
```

---

# 16. Optional Bonus: Faster R-CNN

A second object detection approach was implemented and trained as an optional bonus, to compare a two-stage detector against YOLOv8's one-stage approach.

The model uses:

```text
Faster R-CNN
+
ResNet-50 FPN
```

with pretrained COCO weights, fine-tuned on the license plate dataset (transfer learning, same principle as Section 6).

## 16.1 Resolved: Path-Reading Issue

The earlier blocker — Faster R-CNN training producing `Valid images loaded: 0` — was caused by two bugs in the dataset loader, not the Arabic project path itself:

1. `numpy` was used (`np.fromfile`) without being imported.
2. The decoded image was checked under the wrong variable name (`image` instead of `img`), so every image was silently skipped even when it decoded correctly.

Both were fixed by importing `numpy` and reading every image through a single helper function using `np.fromfile()` + `cv2.imdecode()` in place of `cv2.imread()`, in both the dataset scan and `__getitem__`. This resolved the Unicode-path incompatibility with `cv2.imread()` on Windows. With the fix in place, training loaded the full dataset correctly and completed successfully.

## 16.2 Faster R-CNN Detection Results

Evaluated on the same 44-image / 48-instance test set as YOLOv8:

| Metric    |  Result |
| --------- | ------: |
| Precision |  67.65% |
| Recall    |  95.83% |
| mAP@50    |  93.88% |
| mAP@50-95 |  50.96% |

```text
Test Images  : 44
Ground Truth : 48
Predictions  : 68
```

## 16.3 YOLOv8 vs. Faster R-CNN

| Metric    |     YOLOv8 | Faster R-CNN |
| --------- | ---------: | -----------: |
| Precision |     95.26% |       67.65% |
| Recall    |     89.58% |       95.83% |
| mAP@50    |     92.15% |       93.88% |
| mAP@50-95 |     51.70% |       50.96% |

### Interpretation

* **YOLOv8 has much higher precision.** It rarely produces a false-positive box — when YOLOv8 says "license plate," it almost always is one.
* **Faster R-CNN has much higher recall.** It misses fewer real plates, but at the cost of predicting **68 boxes for only 48 ground-truth plates** — it over-detects, generating extra false-positive boxes (e.g. text elsewhere on the vehicle or background regions it mistakes for plates).
* **mAP@50 is close for both models** (92.15% vs. 93.88%), showing both localize plates well when they do fire correctly.
* **mAP@50-95 is nearly identical** (51.70% vs. 50.96%), meaning neither model has a clear edge in tight bounding-box precision — both lose accuracy at stricter IoU thresholds by a similar amount.
* **Practical takeaway:** YOLOv8 is the better fit for this project's pipeline, since a high false-positive rate from Faster R-CNN means more junk crops reach the OCR stage (confirmed in Section 16.4). YOLOv8's precision-first behavior keeps the pipeline cleaner end-to-end.

## 16.4 Faster R-CNN + OCR

The trained Faster R-CNN model was also connected to the same OCR pipeline (baseline Tesseract), to see how detector choice affects the full pipeline, not just detection metrics.

```text
Model  : runs/faster_rcnn/best.pt
Images : 44 test images
OCR    : Tesseract (baseline config)
```

Selected results:

```text
Cars109.png -> confidence=0.99 | OCR='ACZI7KOD'
Cars11.png  -> confidence=1.00 | OCR='WORSI6GK'
Cars198.png -> confidence=1.00 | OCR='MHO1AV8866'
Cars199.png -> confidence=1.00 | OCR='VMHO1RE8017'
Cars226.png -> confidence=1.00 | OCR='fTN19S4523'
Cars255.png -> confidence=1.00 | OCR='dWH20EJ0364'
```

Several images also show the over-detection problem from Section 16.3 directly affecting OCR — extra low-value boxes get OCR'd along with the real plate:

```text
Cars112.png -> Plate 1: confidence=1.00 | OCR=''
               Plate 2: confidence=0.97 | OCR='DLIVIALLY'
Cars19.png  -> Plate 1: confidence=0.99 | OCR='AER2011'
               Plate 2: confidence=0.87 | OCR='7HWESTERNMOTOR'
Cars224.png -> Plate 1: confidence=0.97 | OCR='rAe8AN777'
               Plate 2: confidence=0.93 | OCR='LEFTTODIE'
```

`DLIVIALLY`, `7HWESTERNMOTOR`, and `LEFTTODIE` are not license plates — they are text read from the extra, spurious boxes Faster R-CNN produced. A number of detections (mostly the lower-confidence second and third boxes per image) also returned empty OCR output, consistent with them not being real, legible plates.

### Interpretation

* Faster R-CNN's higher recall does not translate into cleaner OCR output — its extra false-positive boxes generate noisy, non-plate text that a downstream system would need to filter out.
* This is a concrete illustration of why **Pipeline Integration** matters, not just individual detector metrics: a detector's precision/recall trade-off has direct, visible consequences once it feeds into OCR.
* All annotated images are saved under `runs/faster_rcnn/ocr_predictions/`, and cropped plates under `runs/faster_rcnn/ocr_crops/`.

---

# 17. Optional Bonus: Real-Time Webcam Detection

Real-time detection was implemented using a live webcam feed with the trained YOLOv8 model.

```text
Webcam Frame
     |
     v
YOLOv8 Detection
     |
     v
Bounding Box + Confidence
     |
     v
Live Overlay on Video Feed
```

The script captures frames continuously via OpenCV (`cv2.VideoCapture`), runs YOLOv8 inference on each frame, and draws the bounding box and confidence score directly on the live video window.

In testing, the webcam feed detects license plates correctly in real time, consistent with the model's strong precision on the static test set.

Implemented in:

```text
src/webcam_detect.py
```

This demonstrates the pipeline's applicability to live scenarios such as:

```text
Parking Entry/Exit Gates
Live Traffic Monitoring
Real-Time Security Checkpoints
```

---

# 18. Problems Encountered

Several problems were encountered during development.

## 18.1 OpenCV GUI Problem

The initial OCR evaluation used OpenCV GUI functionality.

The program could become unresponsive on Windows because of:

```python
cv2.imshow()
```

and waiting for terminal input simultaneously.

### Solution

The evaluation was changed to save cropped plates to disk instead of opening OpenCV windows.

---

## 18.2 OCR Accuracy Problem

The improved OCR preprocessing did not improve recognition.

The final results showed:

```text
Baseline : 41.84%
Improved : 30.27%
```

The improved method decreased performance by:

```text
11.57 percentage points
```

This demonstrates that simple preprocessing is not guaranteed to improve OCR.

---

## 18.3 OCR CSV Permission Error

During evaluation, the program encountered:

```text
PermissionError: [Errno 13] Permission denied
```

while trying to write:

```text
ocr_results.csv
```

The issue was related to access to the CSV file.

The evaluation process was then adjusted and the final OCR summary was successfully generated.

---

## 18.4 Faster R-CNN Dataset Loading Problem (Resolved)

Faster R-CNN training initially reported:

```text
Valid images loaded: 0
Training images: 0
Validation images: 0
```

This was first suspected to be a Windows/Arabic-path issue with `cv2.imread()`. Closer inspection found two actual causes in the dataset loader:

1. Missing `import numpy as np`.
2. A variable-name mismatch (`image` vs. `img`) in the image-validity check, which caused every successfully decoded image to be discarded anyway.

### Solution

* Added the missing `numpy` import.
* Introduced a single `read_image_unicode()` helper using `np.fromfile()` + `cv2.imdecode()`, used consistently in both the dataset scan and `__getitem__`, replacing direct `cv2.imread()` calls.
* Added an early, descriptive error if zero valid images are found, instead of letting `DataLoader`/`RandomSampler` fail with an opaque `num_samples` error.

After these fixes, all training and validation images loaded correctly and training completed (Section 16).

---

# 19. Technologies Used

```text
Python
YOLOv8
Ultralytics
OpenCV
Tesseract OCR
PyTesseract
PyTorch
Torchvision
NumPy
SQLite
```

---

# 20. Main Files

## YOLO Detection

```text
src/evaluate_detection.py
```

Used to evaluate the YOLOv8 detector.

---

## OCR

```text
src/ocr.py
```

Contains:

```text
baseline_ocr()
improved_ocr()
```

---

## OCR Evaluation

```text
src/ocr_evaluation.py
```

Used for manual ground-truth evaluation.

---

## OCR Analysis

```text
src/analyze_ocr_results.py
```

Generates the final OCR statistics.

---

## Final Demo

```text
src/final_demo.py
```

Runs the complete:

```text
Detection → Cropping → OCR
```

pipeline.

---

## Database

```text
src/plate_database.py
src/view_database.py
```

Used for the optional database bonus.

---

## Faster R-CNN

```text
src/faster_rcnn_train.py       # training
src/faster_rcnn_evaluate.py    # detection metrics + YOLOv8 comparison
src/faster_rcnn_visualize.py   # bounding-box visualization
src/faster_rcnn_ocr.py         # detection + OCR pipeline
```

---

## Webcam

```text
src/webcam_detect.py
```

Used for the optional real-time webcam bonus.

---

# 21. How to Run

## Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run Detection Evaluation

```bash
python src/evaluate_detection.py
```

---

## Run OCR Evaluation

```bash
python src/ocr_evaluation.py
```

---

## Analyze OCR Results

```bash
python src/analyze_ocr_results.py
```

---

## Run Final Demo

```bash
python src/final_demo.py
```

---

## View Database

```bash
python src/view_database.py
```

---

## Faster R-CNN

```bash
python src/faster_rcnn_train.py
python src/faster_rcnn_evaluate.py
python src/faster_rcnn_visualize.py
python src/faster_rcnn_ocr.py
```

---

## Real-Time Webcam

```bash
python src/webcam_detect.py
```

---

# 22. Final Results

## YOLOv8 Detection

```text
Precision     : 95.26%
Recall        : 89.58%
mAP@50        : 88.82%
mAP@50-95     : 50.28%
```

## Faster R-CNN Detection

```text
Precision     : 67.65%
Recall        : 95.83%
mAP@50        : 93.88%
mAP@50-95     : 50.96%
```

## OCR (YOLOv8 detections)

```text
Baseline Character Accuracy : 41.84%
Improved Character Accuracy : 30.27%

Baseline Exact Match        : 28.57%
Improved Exact Match        : 19.05%
```

The baseline OCR performed better than the implemented preprocessing-based OCR on the evaluated test images.

Faster R-CNN's OCR output (Section 16.4) was noisier overall than YOLOv8's, driven mainly by its extra false-positive detections being sent through OCR unfiltered.

---

# 23. Assignment Requirements Status

| Requirement             | Status                                          |
| ------------------------ | ------------------------------------------------ |
| Public/custom dataset   | Completed                                       |
| YOLO-format annotations | Completed                                       |
| YOLOv8 detection        | Completed                                       |
| Transfer learning       | Completed                                       |
| License plate cropping  | Completed                                       |
| Tesseract OCR           | Completed                                       |
| Baseline OCR            | Completed                                       |
| Improved OCR            | Completed                                       |
| OCR comparison          | Completed                                       |
| Detection evaluation    | Completed                                       |
| Visualization           | Completed                                       |
| End-to-end pipeline     | Completed                                       |
| Database bonus          | Completed                                       |
| Faster R-CNN bonus      | Completed — trained, evaluated, compared, OCR'd |
| Webcam bonus            | Completed — real-time detection working         |

---

# 24. Conclusion

The project successfully implements a complete license plate detection and recognition pipeline.

YOLOv8 achieved strong license plate detection performance with:

```text
95.26% Precision
89.58% Recall
88.82% mAP@50
```

The OCR experiment showed that the baseline Tesseract approach performed better than the implemented preprocessing pipeline — an important, honestly-reported negative result rather than an assumed improvement.

Both optional bonuses beyond the database were completed. Faster R-CNN was trained successfully after resolving a dataset-loading bug, and directly compared against YOLOv8: it achieved higher recall (95.83% vs. 89.58%) but noticeably lower precision (67.65% vs. 95.26%), producing 68 predicted boxes against only 48 real plates. Feeding those detections through the same OCR pipeline showed the practical cost of that trade-off — extra, non-plate text getting OCR'd alongside genuine plates — making YOLOv8 the stronger choice for this project's end-to-end pipeline. Real-time webcam detection was also implemented and verified working, extending the system from static images to live video.

The project also includes an optional SQLite database integration for storing recognized license plates.

The project demonstrates the integration of:

```text
Computer Vision
+
Object Detection
+
Image Processing
+
OCR
+
Database Integration
+
Real-Time Inference
```

for a real-world license plate recognition application.