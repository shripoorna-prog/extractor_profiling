import cv2
import numpy as np


def run_ransac(
    keypoints1,
    keypoints2,
    matches,
    threshold=1.0,
    confidence=0.999
):
    """
    Run RANSAC using the fundamental matrix.

    Parameters
    ----------
    keypoints1 : numpy.ndarray
        Keypoints from image 1, shape (N, 2)

    keypoints2 : numpy.ndarray
        Keypoints from image 2, shape (N, 2)

    matches : numpy.ndarray
        Match pairs, shape (M, 2)
        matches[i] = [index_in_image1, index_in_image2]

    threshold : float
        RANSAC reprojection threshold in pixels.

    confidence : float
        RANSAC confidence.

    Returns
    -------
    result : dict
        Fundamental matrix, inlier mask,
        number of inliers and inlier ratio.
    """

    # Not enough matches for fundamental matrix estimation
    if matches is None or len(matches) < 8:
        return {
            "fundamental_matrix": None,
            "inlier_mask": None,
            "num_matches": 0 if matches is None else len(matches),
            "num_inliers": 0,
            "inlier_ratio": 0.0,
        }

    # Get corresponding coordinates
    points1 = keypoints1[
        matches[:, 0].astype(int)
    ]

    points2 = keypoints2[
        matches[:, 1].astype(int)
    ]

    # Run fundamental matrix RANSAC
    fundamental_matrix, mask = cv2.findFundamentalMat(
        points1,
        points2,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=threshold,
        confidence=confidence
    )

    # RANSAC failed
    if mask is None:
        return {
            "fundamental_matrix": fundamental_matrix,
            "inlier_mask": None,
            "num_matches": len(matches),
            "num_inliers": 0,
            "inlier_ratio": 0.0,
        }

    # Convert mask to 1-D boolean array
    mask = mask.ravel().astype(bool)

    num_inliers = int(np.sum(mask))
    num_matches = len(matches)

    inlier_ratio = (
        num_inliers / num_matches
        if num_matches > 0
        else 0.0
    )

    return {
        "fundamental_matrix": fundamental_matrix,
        "inlier_mask": mask,
        "num_matches": num_matches,
        "num_inliers": num_inliers,
        "inlier_ratio": inlier_ratio,
    }


def get_inlier_matches(matches, inlier_mask):
    """
    Return only the matches that survived RANSAC.
    """

    if inlier_mask is None:
        return np.empty(
            (0, 2),
            dtype=np.int32
        )

    return matches[inlier_mask]


if __name__ == "__main__":
    print("RANSAC module loaded successfully.")