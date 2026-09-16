import cv2
import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from adapters.aliked import ALIKEDExtractor


image = cv2.imread(r".\frames\frame_0000.png")

if image is None:
    raise FileNotFoundError("Could not load .\frames\frame_0000.png")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

extractor = ALIKEDExtractor(max_num_keypoints=1000)

keypoints, descriptors, scores = extractor.extract(image_tensor)

print("\n===== ALIKED ADAPTER TEST =====")
print("Keypoints shape      :", keypoints.shape)
print("Descriptors shape    :", descriptors.shape)
print("Scores shape         :", scores.shape)

print("Keypoints dtype      :", keypoints.dtype)
print("Descriptors dtype    :", descriptors.dtype)
print("Scores dtype         :", scores.dtype)

print("\nALIKED adapter working successfully!")