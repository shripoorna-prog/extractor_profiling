import sys
import cv2
import torch

# Allow importing the original SuperPoint implementation
sys.path.insert(0, ".\\SuperPointPretrainedNetwork")

from demo_superpoint import SuperPointNet


# --------------------------------------------------
# 1. Load model
# --------------------------------------------------

model = SuperPointNet()

weights_path = ".\\SuperPointPretrainedNetwork\\superpoint_v1.pth"

model.load_state_dict(
    torch.load(
        weights_path,
        map_location="cpu"
    )
)

model.eval()

print("SuperPoint model loaded successfully!")


# --------------------------------------------------
# 2. Load one image
# --------------------------------------------------

image_path = ".\\frames\\frame_0001.png"
image = cv2.imread(
    image_path,
    cv2.IMREAD_GRAYSCALE
)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {image_path}"
    )

print("Image shape:", image.shape)


# --------------------------------------------------
# 3. Convert image to PyTorch tensor
# --------------------------------------------------

image_tensor = torch.from_numpy(image).float() / 255.0

image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)

print("Tensor shape:", image_tensor.shape)


# --------------------------------------------------
# 4. Run SuperPoint
# --------------------------------------------------

with torch.no_grad():

    semi, desc = model(image_tensor)


# --------------------------------------------------
# 5. Inspect outputs
# --------------------------------------------------

print("\n===== SUPERPOINT RAW OUTPUT =====")

print("Semi shape       :", semi.shape)
print("Descriptor shape :", desc.shape)
print("Descriptor dtype :", desc.dtype)

print("\nSuperPoint raw network inference successful!")