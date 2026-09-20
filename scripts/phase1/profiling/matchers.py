import cv2
import numpy as np
import torch

from lightglue import LightGlue


# ============================================================
# CLASSICAL MATCHERS
# ============================================================

def match_orb(desc1, desc2, ratio=0.75):
    """
    ORB descriptor matching using Hamming distance
    with Lowe's ratio test.
    """

    if desc1 is None or desc2 is None:
        return []

    if len(desc1) == 0 or len(desc2) == 0:
        return []

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    knn_matches = matcher.knnMatch(
        desc1,
        desc2,
        k=2
    )

    good = []

    for pair in knn_matches:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < ratio * n.distance:
            good.append((m.queryIdx, m.trainIdx))

    return good


def match_akaze(desc1, desc2, ratio=0.75):
    """
    AKAZE descriptor matching using Hamming distance
    with Lowe's ratio test.
    """

    if desc1 is None or desc2 is None:
        return []

    if len(desc1) == 0 or len(desc2) == 0:
        return []

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    knn_matches = matcher.knnMatch(
        desc1,
        desc2,
        k=2
    )

    good = []

    for pair in knn_matches:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < ratio * n.distance:
            good.append((m.queryIdx, m.trainIdx))

    return good


def match_l2(desc1, desc2, ratio=0.8):
    """
    L2 descriptor matching.

    Used for XFeat descriptors.
    """

    if desc1 is None or desc2 is None:
        return []

    if len(desc1) == 0 or len(desc2) == 0:
        return []

    desc1 = np.asarray(desc1, dtype=np.float32)
    desc2 = np.asarray(desc2, dtype=np.float32)

    matcher = cv2.BFMatcher(cv2.NORM_L2)

    knn_matches = matcher.knnMatch(
        desc1,
        desc2,
        k=2
    )

    good = []

    for pair in knn_matches:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < ratio * n.distance:
            good.append((m.queryIdx, m.trainIdx))

    return good


# ============================================================
# LIGHTGLUE MATCHER
# ============================================================

_LIGHTGLUE_CACHE = {}


def get_lightglue(feature_type):
    """
    Create and cache the official LightGlue matcher.

    Supported feature types:
        - aliked
        - superpoint
    """

    feature_type = feature_type.lower()

    if feature_type not in _LIGHTGLUE_CACHE:

        matcher = LightGlue(
            features=feature_type
        )

        matcher = matcher.eval()

        _LIGHTGLUE_CACHE[feature_type] = matcher

    return _LIGHTGLUE_CACHE[feature_type]


def _to_tensor(value):
    if torch.is_tensor(value):
        return value

    return torch.from_numpy(
        np.asarray(value)
    )


def match_lightglue(
    kp1,
    desc1,
    scores1,
    kp2,
    desc2,
    scores2,
    image_size,
    feature_type="aliked"
):
    """
    LightGlue matching for learned features.

    Default feature type is ALIKED.

    For SuperPoint, call with:
        feature_type="superpoint"
    """

    if kp1 is None or kp2 is None:
        return []

    if desc1 is None or desc2 is None:
        return []

    if len(kp1) == 0 or len(kp2) == 0:
        return []

    device = torch.device("cpu")

    kpts0 = _to_tensor(kp1).float().to(device)
    kpts1 = _to_tensor(kp2).float().to(device)

    descriptors0 = _to_tensor(desc1).float().to(device)
    descriptors1 = _to_tensor(desc2).float().to(device)

    if kpts0.ndim == 2:
        kpts0 = kpts0.unsqueeze(0)

    if kpts1.ndim == 2:
        kpts1 = kpts1.unsqueeze(0)

    if descriptors0.ndim == 2:
        descriptors0 = descriptors0.unsqueeze(0)

    if descriptors1.ndim == 2:
        descriptors1 = descriptors1.unsqueeze(0)

    if scores1 is None:
        scores0 = torch.ones(
            kpts0.shape[1],
            dtype=torch.float32,
            device=device
        )
    else:
        scores0 = _to_tensor(scores1).float().to(device).reshape(-1)

    if scores2 is None:
        scores1_tensor = torch.ones(
            kpts1.shape[1],
            dtype=torch.float32,
            device=device
        )
    else:
        scores1_tensor = _to_tensor(scores2).float().to(device).reshape(-1)

    scores0 = scores0.unsqueeze(0)
    scores1_tensor = scores1_tensor.unsqueeze(0)

    h, w = image_size[1], image_size[0]

    image_size_tensor = torch.tensor(
        [[w, h]],
        dtype=torch.float32,
        device=device
    )

    data = {
        "image0": {
            "keypoints": kpts0,
            "descriptors": descriptors0,
            "keypoint_scores": scores0,
            "image_size": image_size_tensor,
        },
        "image1": {
            "keypoints": kpts1,
            "descriptors": descriptors1,
            "keypoint_scores": scores1_tensor,
            "image_size": image_size_tensor,
        }
    }

    matcher = get_lightglue(feature_type)

    with torch.inference_mode():
        result = matcher(data)

    matches = result["matches"]

    if torch.is_tensor(matches):
        matches = matches.detach().cpu().numpy()

    matches = np.asarray(matches)

    if matches.ndim == 3:
        matches = matches[0]

    output = []

    for m in matches:

        if len(m) < 2:
            continue

        i = int(m[0])
        j = int(m[1])

        if i < 0 or j < 0:
            continue

        output.append((i, j))

    return output