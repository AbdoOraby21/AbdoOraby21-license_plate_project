**University:** Badr University in Assiut  
**School:** Artificial Intelligence and Data Management  
**Project Category:** Object Detection + OCR  
**Model:** YOLOv8  
**OCR Engine:** Tesseract OCR  

---

# 1. Project Overview

This project implements a License Plate Detection and Recognition System.

The system is designed to detect vehicle license plates in images and then extract the license plate text using OCR.

The complete pipeline is:

```text
Input Image
     ↓
YOLOv8 License Plate Detection
     ↓
Bounding Box
     ↓
License Plate Cropping
     ↓
Tesseract OCR
     ↓
Recognized Plate Number
     ↓
Final Visualization
````

The project demonstrates how computer vision object detection can be integrated with OCR for a real-world application.

Possible applications include:

* Traffic monitoring
* Parking systems
* Vehicle identification
* Security systems
* Automated license plate recognition

---

# 2. Project Objectives

The main objectives of the project are:

1. Detect vehicle license plates using YOLOv8.
2. Train a YOLO model using transfer learning.
3. Crop the detected license plate.
4. Extract the plate number using Tesseract OCR.
5. Compare baseline OCR with an improved OCR approach.
6. Evaluate detection performance.
7. Evaluate OCR performance.
8. Produce a final visualization showing the detected plate and recognized text.

---

# 3. Dataset

The project uses a license plate dataset containing vehicle images with annotated license plate bounding boxes.

The dataset was converted into YOLO format.

The annotations contain:

```text
class_id
x_center
y_center
width
height
```

All bounding box coordinates are normalized according to the YOLO annotation format.

The dataset was divided into training, validation, and testing sets.

The test set contains:

* 44 images
* 48 license plate instances
* 0 background-only images
* 0 corrupted images

---

# 4. Dataset Structure

The dataset follows a YOLO-style structure:

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

The dataset configuration is defined in:

```text
data.yaml
```

---

# 5. YOLOv8 Model

YOLOv8 was selected because it provides fast and effective object detection and is suitable for real-time computer vision applications.

A pretrained YOLOv8 model was used as the starting point.

Transfer learning was applied by training the pretrained model on the license plate dataset.

The trained model is stored as:

```text
runs/
└── license_plate_detection/
    └── weights/
        └── best.pt
```

---

# 6. Detection Stage

The detection stage uses YOLOv8 to locate license plates.

For every input image, the model produces:

* Bounding box coordinates
* Confidence score
* Detected class

The system selects the highest-confidence license plate detection.

The detected bounding box is then used to crop the license plate.

---

# 7. Detection Evaluation

The trained YOLOv8 model was evaluated on the test set.

Test set:

```text
Images:    44
Instances: 48
```

Final detection results:

| Metric    | Result |
| --------- | -----: |
| Precision | 95.26% |
| Recall    | 89.58% |
| mAP@50    | 88.82% |
| mAP@50-95 | 50.28% |

---

## 7.1 Precision

The model achieved:

```text
Precision = 95.26%
```

This indicates that most of the detections classified as license plates were correct.

---

## 7.2 Recall

The model achieved:

```text
Recall = 89.58%
```

This means that the model detected most of the license plates present in the test images.

---

## 7.3 mAP@50

The model achieved:

```text
mAP@50 = 88.82%
```

This indicates strong detection performance when using an IoU threshold of 0.50.

---

## 7.4 mAP@50-95

The model achieved:

```text
mAP@50-95 = 50.28%
```

This metric is more strict because it evaluates the detector over multiple IoU thresholds.

The lower value compared with mAP@50 indicates that some bounding boxes could be localized more precisely.

---

# 8. Plate Cropping

After detecting the license plate, the system extracts the corresponding region from the original image.

Example pipeline:

```text
Original Image
       ↓
YOLO Bounding Box
       ↓
(x1, y1, x2, y2)
       ↓
Crop Plate Region
       ↓
OCR
```

The cropped plates are stored in:

```text
runs/
└── ocr_evaluation/
    └── plate_crops/
```

---

# 9. OCR Stage

Tesseract OCR was selected for license plate text recognition.

Two OCR configurations were evaluated:

1. Baseline OCR
2. Improved OCR

---

# 10. Baseline OCR

The baseline OCR applies Tesseract directly to the cropped license plate.

The main configuration used was:

```text
--psm 7
```

A character whitelist was also used:

```text
ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789
```

The purpose of the baseline was to provide a reference point for evaluating the effect of image preprocessing.

---

# 11. Improved OCR

Several preprocessing approaches were tested.

The experiments included image enhancement techniques such as:

* Grayscale conversion
* Image upscaling
* Gaussian filtering
* Thresholding
* CLAHE
* Sharpening

The goal was to determine whether image preprocessing could improve OCR performance.

---

# 12. OCR Experiments

Multiple preprocessing versions were tested.

## Experiment 1

Results:

| Metric             | Baseline | Improved |
| ------------------ | -------: | -------: |
| Character Accuracy |   42.20% |   36.74% |

The improved version decreased character accuracy by:

```text
5.45 percentage points
```

---

## Experiment 2

Results:

| Metric             | Baseline | Improved |
| ------------------ | -------: | -------: |
| Character Accuracy |   41.84% |   22.60% |
| Exact Match        |   28.57% |    9.52% |

The preprocessing pipeline used aggressive thresholding, which negatively affected OCR performance.

The improved OCR decreased character accuracy by:

```text
19.25 percentage points
```

---

## Experiment 3

A more conservative preprocessing pipeline was tested.

The pipeline included:

```text
Grayscale
    ↓
Upscaling
    ↓
CLAHE
    ↓
Mild Sharpening
    ↓
Tesseract
```

Final results:

| Metric             | Baseline | Improved |
| ------------------ | -------: | -------: |
| Character Accuracy |   41.84% |   30.27% |
| Exact Match        |   28.57% |   19.05% |

The improved version decreased character accuracy by:

```text
11.57 percentage points
```

---

# 13. Final OCR Results

The final selected OCR evaluation produced:

## Baseline

```text
Character Accuracy: 41.84%
Exact Match:        28.57%
```

## Improved

```text
Character Accuracy: 30.27%
Exact Match:        19.05%
```

Comparison:

```text
Improved OCR better : 6 images
Baseline OCR better : 9 images
Same result         : 27 images
```

OCR failures:

```text
Baseline failures : 10
Improved failures : 17
```

---

# 14. OCR Error Analysis

The preprocessing approaches did not improve the OCR results.

This is an important experimental finding rather than a failure of the entire project.

The baseline OCR performed better because some license plate images already contained sufficiently clear character information.

Aggressive preprocessing can remove useful information from characters.

For example:

* Thresholding can remove thin character strokes.
* Incorrect contrast enhancement can distort characters.
* Different plates have different lighting conditions.
* Some plates contain reflections.
* Some plates are too small after detection.
* Some crops contain insufficient character resolution.
* Tesseract may confuse visually similar characters.

Examples of possible OCR confusion include:

```text
O ↔ 0
I ↔ 1
S ↔ 5
B ↔ 8
G ↔ 6
```

Therefore, preprocessing should not automatically be considered an improvement for every image.

---

# 15. Final Detection + Recognition Pipeline

A final demonstration script was implemented.

The final pipeline performs:

```text
Input Image
     ↓
YOLOv8 Detection
     ↓
Select Best License Plate
     ↓
Crop License Plate
     ↓
Baseline Tesseract OCR
     ↓
Draw Bounding Box
     ↓
Display Plate Text
     ↓
Save Final Result
```

The final output is saved in:

```text
runs/
└── final_demo/
```

---

# 16. Final Demo Results

The final demonstration was tested on 10 test images.

Examples:

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

Some images were successfully detected but OCR failed:

```text
Cars112.png → NOT RECOGNIZED
Cars140.png → NOT RECOGNIZED
Cars145.png → NOT RECOGNIZED
Cars15.png  → NOT RECOGNIZED
```

Some images also produced no license plate detection:

```text
Cars100.png → No license plate detected
Cars138.png → No license plate detected
```

These examples are useful for demonstrating the limitations of the complete pipeline.

---

# 17. Problems Encountered During Development

Several technical problems were encountered during development.

## 17.1 OpenCV Display Problem

Initially, the OCR evaluation used:

```python
cv2.imshow()
```

This caused the OpenCV window to become unresponsive on Windows while the program was waiting for terminal input.

### Solution

The system was changed to save plate crops to disk instead of opening OpenCV windows.

---

## 17.2 CSV Permission Error

During OCR evaluation, the program produced:

```text
PermissionError: [Errno 13] Permission denied
```

The problem occurred while trying to write:

```text
ocr_results.csv
```

The file was likely locked by another application such as Excel.

### Solution

The CSV was closed before running the evaluation again.

A separate automatic re-evaluation script was also created to avoid entering Ground Truth values again.

---

## 17.3 Missing data.yaml Path

The detection evaluation initially expected:

```text
dataset/yolo/data.yaml
```

However, the actual file was located in the project root:

```text
data.yaml
```

### Solution

The evaluation script was updated to use:

```text
BASE_DIR / "data.yaml"
```

---

## 17.4 OCR Preprocessing Performance

The first preprocessing approaches reduced OCR accuracy instead of improving it.

Several versions were tested before reaching the final experimental configuration.

This demonstrated that image preprocessing must be selected based on the characteristics of the dataset.

---

# 18. Final Project Structure

The main project structure is:

```text
license_plate_project/
│
├── data.yaml
├── README.md
├── PROJECT_SUMMARY.md
│
├── dataset/
│   └── yolo/
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       │
│       └── labels/
│           ├── train/
│           ├── val/
│           └── test/
│
├── src/
│   ├── ocr.py
│   ├── ocr_evaluation.py
│   ├── analyze_ocr_results.py
│   ├── re_evaluate_ocr.py
│   ├── evaluate_detection.py
│   └── final_demo.py
│
└── runs/
    ├── license_plate_detection/
    │   ├── weights/
    │   │   └── best.pt
    │   └── detection_metrics.txt
    │
    ├── ocr_evaluation/
    │   ├── plate_crops/
    │   ├── plate_crops_v3/
    │   ├── ocr_results.csv
    │   └── ocr_summary.csv
    │
    └── final_demo/
        ├── Cars100.png
        ├── Cars109.png
        ├── Cars11.png
        └── ...
```

---

# 19. Technologies Used

The project uses:

* Python
* OpenCV
* YOLOv8
* Ultralytics
* Tesseract OCR
* Pytesseract
* NumPy
* Regular Expressions
* CSV
* PowerShell
* Git/GitHub

---

# 20. Evaluation Summary

## Detection

```text
Precision     : 95.26%
Recall        : 89.58%
mAP@50        : 88.82%
mAP@50-95     : 50.28%
```

## OCR

```text
Baseline Character Accuracy : 41.84%
Baseline Exact Match        : 28.57%

Improved Character Accuracy : 30.27%
Improved Exact Match        : 19.05%
```

---

# 21. Interpretation of Results

The detection component achieved strong performance.

The high precision of 95.26% indicates that the detector produces relatively few false positive detections.

The recall of 89.58% indicates that most license plates were successfully detected.

The mAP@50 of 88.82% demonstrates good overall detection performance.

However, the OCR stage is considerably more challenging.

The OCR accuracy is affected by:

* Image resolution
* Plate size
* Lighting
* Reflections
* Blur
* Character style
* Background noise
* Plate orientation
* Detection crop quality

Therefore, the overall system performance is limited more by the recognition stage than by the license plate detector.

---

# 22. Limitations

The current system has several limitations.

1. OCR performance is relatively low.
2. The dataset contains different plate styles and image conditions.
3. Tesseract is a general-purpose OCR engine and is not specifically trained for license plates.
4. Some license plates are too small or unclear.
5. The system currently selects the highest-confidence detection.
6. The final demo processes static images.
7. Real-time video/webcam processing was not implemented.

---

# 23. Future Improvements

Possible improvements include:

### Better OCR

A license-plate-specific OCR model could replace Tesseract.

Possible approaches include:

* EasyOCR
* PaddleOCR
* CRNN
* Transformer-based OCR

### Better preprocessing

An adaptive preprocessing system could select different preprocessing methods depending on image quality.

### More training data

Increasing the dataset size could improve detection performance.

### Data augmentation

Possible augmentations include:

* Rotation
* Brightness changes
* Contrast changes
* Blur
* Noise
* Scaling

### Real-time detection

The system could be extended to work with:

* Webcam
* CCTV cameras
* Video files

### Database integration

Detected plate numbers could be stored in a database with:

* Plate number
* Date
* Time
* Image
* Confidence score

---

# 24. Conclusion

This project successfully implemented a complete License Plate Detection and Recognition pipeline.

YOLOv8 was trained to detect license plates and achieved:

```text
Precision     = 95.26%
Recall        = 89.58%
mAP@50        = 88.82%
mAP@50-95     = 50.28%
```

The detected license plates were successfully cropped and passed to Tesseract OCR.

The OCR experiments showed that the baseline approach performed better than the tested preprocessing approaches:

```text
Baseline Character Accuracy = 41.84%
Improved Character Accuracy = 30.27%
```

Although preprocessing did not improve OCR performance, the experiment demonstrated an important computer vision principle: preprocessing techniques do not always improve recognition and must be selected according to the characteristics of the input data.

The final system successfully integrates:

```text
Object Detection
       +
Image Cropping
       +
OCR
       +
Visualization
```

This fulfills the main requirements of the License Plate Detection & Recognition System assignment.

````

### وبعد ما تعمله

شغّل:

```powershell
git status
````

وبعدين:

```powershell
git add PROJECT_SUMMARY.md src/
git commit -m "complete license plate detection and OCR pipeline"
git push
```

