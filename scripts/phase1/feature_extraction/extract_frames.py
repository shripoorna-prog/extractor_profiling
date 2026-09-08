import csv
import ast
import numpy as np
import cv2
import os
import sys

# Allow very large CSV fields
csv.field_size_limit(10_000_000)

csv_path = r"C:\Users\Shripoorna\Downloads\desk2-circle\camera-image_raw.csv"
output_dir = r"C:\Users\Shripoorna\OneDrive\Desktop\extractor_profiling\frames"

os.makedirs(output_dir, exist_ok=True)

FRAME_SKIP = 10
MAX_FRAMES = 100

count = 0
saved = 0

with open(csv_path, "r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:

        if count % FRAME_SKIP != 0:
            count += 1
            continue

        if saved >= MAX_FRAMES:
            break

        height = int(row["height"])
        width = int(row["width"])

        # Convert the stored byte-string back to image bytes
        data = ast.literal_eval(row["data"])
        img = np.frombuffer(data, dtype=np.uint8)

        # BGR8 image
        img = img.reshape((height, width, 3))

        filename = os.path.join(
            output_dir,
            f"frame_{saved:04d}.png"
        )

        cv2.imwrite(filename, img)

        saved += 1
        count += 1

        print(f"Saved frame {saved}/{MAX_FRAMES}")

print()
print("Done!")
print(f"Frames saved: {saved}")
print(f"Output folder: {output_dir}")