import cv2
import numpy as np


def calculate_match_ratio(
    num_matches,
    num_keypoints1,
    num_keypoints2
):
    """
    Match ratio = number of matches relative
    to the smaller keypoint count.
    """

    denominator = min(
        num_keypoints1,
        num_keypoints2
    )

    if denominator == 0:
        return 0.0

    return num_matches / denominator


def calculate_inlier_ratio(
    num_inliers,
    num_matches
):
    """
    Fraction of matches that survive RANSAC.
    """

    if num_matches == 0:
        return 0.0

    return num_inliers / num_matches


def calculate_repeatability(
    keypoints1,
    keypoints2,
    homography=None,
    distance_threshold=3.0
):
    """
    Estimate repeatability between two frames.

    If a homography is available, keypoints from
    frame 1 are projected into frame 2.

    A keypoint is considered repeated if a keypoint
    exists within distance_threshold pixels.
    """

    if homography is None:
        return 0.0

    if len(keypoints1) == 0 or len(keypoints2) == 0:
        return 0.0

    keypoints1 = np.asarray(
        keypoints1,
        dtype=np.float32
    )

    keypoints2 = np.asarray(
        keypoints2,
        dtype=np.float32
    )

    points = keypoints1.reshape(
        -1, 1, 2
    )

    projected = cv2.perspectiveTransform(
        points,
        homography
    ).reshape(-1, 2)

    repeated = 0

    for point in projected:

        distances = np.linalg.norm(
            keypoints2 - point,
            axis=1
        )

        if np.min(distances) <= distance_threshold:
            repeated += 1

    return repeated / len(keypoints1)