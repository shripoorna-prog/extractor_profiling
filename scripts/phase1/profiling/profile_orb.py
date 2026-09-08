import cv2
import glob
import time
import numpy as np

# Load extracted frames
image_paths = sorted(glob.glob(r".\frames\*.png"))

print(f"Images found: {len(image_paths)}")

# ORB detector
orb = cv2.ORB_create(nfeatures=1000)

times = []
keypoints_counts = []

# Warm-up
for path in image_paths[:5]:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    orb.detectAndCompute(img, None)

# Benchmark
for path in image_paths:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    start = time.perf_counter()

    keypoints, descriptors = orb.detectAndCompute(img, None)

    end = time.perf_counter()

    elapsed_ms = (end - start) * 1000

    times.append(elapsed_ms)
    keypoints_counts.append(len(keypoints))

print("\n===== ORB BASELINE =====")
print(f"Frames processed : {len(times)}")
print(f"Average time     : {np.mean(times):.2f} ms/frame")
print(f"Median time      : {np.median(times):.2f} ms/frame")
print(f"Min time         : {np.min(times):.2f} ms/frame")
print(f"Max time         : {np.max(times):.2f} ms/frame")
print(f"Average keypoints: {np.mean(keypoints_counts):.1f}")
print(f"Max keypoints    : {np.max(keypoints_counts)}")