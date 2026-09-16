import cv2
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from adapters.akaze import AKAZEExtractor


image_path = ROOT / "frames" / "frame_0000.png"

image = cv2.imread(str(image_path))

if image is None:
    raise RuntimeError(f"Could not load image: {image_path}")

extractor = AKAZEExtractor()

keypoints, descriptors = extractor.extract(image)

print()
print("===== AKAZE ADAPTER TEST =====")
print("Keypoints count      :", len(keypoints))
print("Descriptors shape    :", None if descriptors is None else descriptors.shape)
print("Keypoint dtype       :", keypoints[0].pt.__class__.__name__ if keypoints else "N/A")
print("Descriptor dtype     :", None if descriptors is None else descriptors.dtype)

print()
print("AKAZE adapter working successfully!")
