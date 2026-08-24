from plate_database import init_database, get_all_plates


init_database()

rows = get_all_plates()

print("=" * 80)
print("DETECTED LICENSE PLATES")
print("=" * 80)

if not rows:

    print("\nNo plates stored yet.")

else:

    print(
        f"\n{'ID':<5}"
        f"{'PLATE':<20}"
        f"{'CONFIDENCE':<15}"
        f"{'IMAGE':<20}"
        f"{'DATE'}"
    )

    print("-" * 80)

    for row in rows:

        record_id = row[0]
        plate = row[1]
        confidence = row[2]
        image = row[3]
        detected_at = row[4]

        print(
            f"{record_id:<5}"
            f"{plate:<20}"
            f"{confidence:<15.3f}"
            f"{image:<20}"
            f"{detected_at}"
        )