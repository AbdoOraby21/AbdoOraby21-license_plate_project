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
    Convert OCR output to uppercase
    and keep only English letters and numbers.
    """

    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)

    return text


# =========================
# Baseline OCR
# =========================

def baseline_ocr(image):
    """
    OCR directly on the original license plate crop.
    No preprocessing.
    """

    config = (
        "--psm 7 "
        "-c tessedit_char_whitelist="
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

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
    Conservative image enhancement before OCR.

    Pipeline:

        Original Crop
             ↓
        Grayscale
             ↓
        Upscaling
             ↓
        CLAHE
             ↓
        Mild Sharpening
             ↓
        Tesseract OCR
    """

    # -------------------------
    # Grayscale
    # -------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # -------------------------
    # Upscaling
    # -------------------------

    gray = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    # -------------------------
    # Contrast Enhancement
    # -------------------------

    clahe = cv2.createCLAHE(
        clipLimit=1.5,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    # -------------------------
    # Mild Sharpening
    # -------------------------

    gaussian = cv2.GaussianBlur(
        enhanced,
        (0, 0),
        1
    )

    sharpened = cv2.addWeighted(
        enhanced,
        1.4,
        gaussian,
        -0.4,
        0
    )

    # -------------------------
    # OCR
    # -------------------------

    config = (
        "--psm 7 "
        "-c tessedit_char_whitelist="
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    text = pytesseract.image_to_string(
        sharpened,
        config=config
    )

    return clean_text(text)


# =========================
# Test
# =========================

if __name__ == "__main__":

    print("OCR module loaded successfully.")

    print("\nAvailable functions:")

    print("- baseline_ocr()")
    print("- improved_ocr()")