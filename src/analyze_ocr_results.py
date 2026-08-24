from pathlib import Path
import pandas as pd


# =========================
# Paths
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

CSV_FILE = (
    BASE_DIR
    / "runs"
    / "ocr_evaluation"
    / "ocr_results.csv"
)


# =========================
# Load CSV
# =========================

if not CSV_FILE.exists():
    print("ERROR: CSV file not found!")
    print(CSV_FILE)
    exit()

df = pd.read_csv(CSV_FILE)


# =========================
# Basic Information
# =========================

total_images = len(df)

print("\n" + "=" * 70)
print("OCR EVALUATION ANALYSIS")
print("=" * 70)

print(f"Total evaluated images: {total_images}")


# =========================
# Character Accuracy
# =========================

baseline_char = df["baseline_char_accuracy"].mean()
improved_char = df["improved_char_accuracy"].mean()


# =========================
# Exact Match Accuracy
# =========================

baseline_exact = df["baseline_exact"].mean()
improved_exact = df["improved_exact"].mean()


# =========================
# Count Improvements
# =========================

improved_better = (
    df["improved_char_accuracy"]
    > df["baseline_char_accuracy"]
).sum()

baseline_better = (
    df["baseline_char_accuracy"]
    > df["improved_char_accuracy"]
).sum()

same_result = (
    df["improved_char_accuracy"]
    == df["baseline_char_accuracy"]
).sum()


# =========================
# OCR Failures
# =========================

baseline_failures = (
    df["baseline"]
    .fillna("")
    .astype(str)
    .str.strip()
    .eq("")
).sum()

improved_failures = (
    df["improved"]
    .fillna("")
    .astype(str)
    .str.strip()
    .eq("")
).sum()


# =========================
# Print Results
# =========================

print("\n" + "-" * 70)
print("CHARACTER ACCURACY")
print("-" * 70)

print(
    f"Baseline Character Accuracy : "
    f"{baseline_char * 100:.2f}%"
)

print(
    f"Improved Character Accuracy : "
    f"{improved_char * 100:.2f}%"
)


print("\n" + "-" * 70)
print("EXACT MATCH ACCURACY")
print("-" * 70)

print(
    f"Baseline Exact Match : "
    f"{baseline_exact * 100:.2f}%"
)

print(
    f"Improved Exact Match : "
    f"{improved_exact * 100:.2f}%"
)


print("\n" + "-" * 70)
print("COMPARISON")
print("-" * 70)

print(
    f"Improved OCR better : "
    f"{improved_better} images"
)

print(
    f"Baseline OCR better : "
    f"{baseline_better} images"
)

print(
    f"Same result         : "
    f"{same_result} images"
)


print("\n" + "-" * 70)
print("OCR FAILURES")
print("-" * 70)

print(
    f"Baseline OCR failures : "
    f"{baseline_failures}"
)

print(
    f"Improved OCR failures : "
    f"{improved_failures}"
)


# =========================
# Improvement Percentage
# =========================

difference = improved_char - baseline_char

print("\n" + "-" * 70)
print("OVERALL IMPROVEMENT")
print("-" * 70)

if difference > 0:

    print(
        f"Improved OCR increased character accuracy by "
        f"{difference * 100:.2f} percentage points."
    )

elif difference < 0:

    print(
        f"Improved OCR decreased character accuracy by "
        f"{abs(difference) * 100:.2f} percentage points."
    )

else:

    print("No difference between Baseline and Improved OCR.")


# =========================
# Save Summary
# =========================

summary = {
    "Total Images": total_images,

    "Baseline Character Accuracy":
        baseline_char * 100,

    "Improved Character Accuracy":
        improved_char * 100,

    "Baseline Exact Match":
        baseline_exact * 100,

    "Improved Exact Match":
        improved_exact * 100,

    "Improved Better Images":
        improved_better,

    "Baseline Better Images":
        baseline_better,

    "Same Result Images":
        same_result,

    "Baseline OCR Failures":
        baseline_failures,

    "Improved OCR Failures":
        improved_failures,

    "Accuracy Difference":
        difference * 100,
}


summary_df = pd.DataFrame(
    [summary]
)

summary_file = (
    BASE_DIR
    / "runs"
    / "ocr_evaluation"
    / "ocr_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)


print("\n" + "=" * 70)

print("Summary saved to:")

print(summary_file)

print("=" * 70)