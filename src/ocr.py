from pathlib import Path
import cv2
import pytesseract
import re


# =========================
# Tesseract Configuration
# =========================

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# =========================
# Clean OCR Text
# =========================

def clean_text(text):
    """
    Remove spaces and special characters
    from OCR output.
    """

    text = text.upper()

    # Keep only letters and numbers
    text = re.sub(r"[^A-Z0-9]", "", text)

    return text


# =========================
# Baseline OCR
# =========================

def baseline_ocr(image):
    """
    OCR directly on the cropped license plate.
    No preprocessing.
    """

    config = "--psm 7"

    text = pytesseract.image_to_string(
        image,
        config=config
    )

    return clean_text(text)


# =========================
# Improved OCR
# =========================

def improved_ocr(image):
    """
    Apply image preprocessing before OCR.

    Pipeline:
        Grayscale
        ↓
        Upscaling
        ↓
        Gaussian Blur
        ↓
        Otsu Threshold
        ↓
        Tesseract OCR
    """

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Upscale image
    gray = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    # Reduce noise
    blurred = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    # Binary threshold
    threshold = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    config = "--psm 7"

    text = pytesseract.image_to_string(
        threshold,
        config=config
    )

    return clean_text(text)


# =========================
# Test OCR
# =========================

if __name__ == "__main__":

    print("OCR module loaded successfully.")

    print("\nAvailable functions:")
    print("- baseline_ocr()")
    print("- improved_ocr()")
