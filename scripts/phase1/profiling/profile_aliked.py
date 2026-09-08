import os
import time
import cv2
import torch
import numpy as np
from kornia.feature import ALIKED


# ============================================================
# SETTINGS
# ============================================================

FRAME_DIR = r"C:\Users\Shripoorna\OneDrive\Desktop\extractor_profiling\frames"

MAX_KEYPOINTS = 1000

# CPU because your PyTorch installation is CPU-only
DEVICE = torch.device("cpu")


# ============================================================
# FIND IMAGES
# ============================================================

image_files = sorted([
    os.path.join(FRAME_DIR, f)
    for f in os.listdir(FRAME_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

print("Images found:", len(image_files))

if len(image_files) == 0:
    raise RuntimeError("No images found in the frames folder.")


# ============================================================
# LOAD ALIKED
# ============================================================

print("\nLoading pretrained ALIKED model...")

model = ALIKED.from_pretrained(
    "aliked-n16",
    max_num_keypoints=MAX_KEYPOINTS,
    device=DEVICE
)

model = model.to(DEVICE)
model.eval()

print("ALIKED loaded successfully.")
print("Device:", DEVICE)


# ============================================================
# PROFILE
# ============================================================

times = []
keypoint_counts = []

print("\n===== ALIKED BASELINE =====")

with torch.inference_mode():

    for i, image_path in enumerate(image_files):

        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------
        image = cv2.imread(image_path)

        if image is None:
            print(f"WARNING: Could not load {image_path}")
            continue

        # OpenCV BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # H x W x 3 -> 1 x 3 x H x W
        image_tensor = torch.from_numpy(
            image
        ).permute(2, 0, 1).float() / 255.0

        image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

        # ----------------------------------------------------
        # Warm-up first frame
        # ----------------------------------------------------
        if i == 0:
            _ = model(image_tensor)
            continue

        # ----------------------------------------------------
        # Timing
        # ----------------------------------------------------
        start = time.perf_counter()

        features = model(image_tensor)

        end = time.perf_counter()

        elapsed_ms = (end - start) * 1000.0

        # ----------------------------------------------------
        # ALIKED returns a list of ALIKEDFeatures
        # ----------------------------------------------------
        feature = features[0]

        keypoints = feature.keypoints

        num_keypoints = keypoints.shape[0]

        times.append(elapsed_ms)
        keypoint_counts.append(num_keypoints)

        print(
            f"Frame {i:03d}: "
            f"{elapsed_ms:.2f} ms, "
            f"{num_keypoints} keypoints"
        )


# ============================================================
# RESULTS
# ============================================================

if len(times) == 0:
    raise RuntimeError("No frames were successfully processed.")


times_np = np.array(times)
keypoints_np = np.array(keypoint_counts)


print("\n========================================")
print("       ALIKED BASELINE RESULTS")
print("========================================")

print(f"Frames processed : {len(times)}")
print(f"Average time     : {np.mean(times_np):.2f} ms/frame")
print(f"Median time      : {np.median(times_np):.2f} ms/frame")
print(f"Min time         : {np.min(times_np):.2f} ms/frame")
print(f"Max time         : {np.max(times_np):.2f} ms/frame")

print(f"Average keypoints: {np.mean(keypoints_np):.1f}")
print(f"Max keypoints    : {np.max(keypoints_np)}")
print(f"Min keypoints    : {np.min(keypoints_np)}")

print("========================================")