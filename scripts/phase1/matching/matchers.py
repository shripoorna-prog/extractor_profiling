import numpy as np
import cv2


def match_binary(
    descriptors1,
    descriptors2,
    ratio_threshold=0.8
):
    """
    Match binary descriptors using Hamming distance.

    Used for:
        ORB
        AKAZE
    """

    if descriptors1 is None or descriptors2 is None:
        return np.empty((0, 2), dtype=np.int32)

    if len(descriptors1) == 0 or len(descriptors2) < 2:
        return np.empty((0, 2), dtype=np.int32)

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=False
    )

    knn_matches = matcher.knnMatch(
        descriptors1,
        descriptors2,
        k=2
    )

    good_matches = []

    for pair in knn_matches:

        if len(pair) < 2:
            continue

        best = pair[0]
        second = pair[1]

        if best.distance < ratio_threshold * second.distance:
            good_matches.append([
                best.queryIdx,
                best.trainIdx
            ])

    return np.asarray(
        good_matches,
        dtype=np.int32
    )


def match_l2(
    descriptors1,
    descriptors2,
    ratio_threshold=0.8
):
    """
    Match floating-point descriptors using L2 distance.

    Used for:
        XFeat
    """

    if descriptors1 is None or descriptors2 is None:
        return np.empty((0, 2), dtype=np.int32)

    if len(descriptors1) == 0 or len(descriptors2) < 2:
        return np.empty((0, 2), dtype=np.int32)

    descriptors1 = np.asarray(
        descriptors1,
        dtype=np.float32
    )

    descriptors2 = np.asarray(
        descriptors2,
        dtype=np.float32
    )

    matches = []

    for i, descriptor in enumerate(descriptors1):

        distances = np.linalg.norm(
            descriptors2 - descriptor,
            axis=1
        )

        nearest = np.argsort(
            distances
        )[:2]

        best_idx = nearest[0]
        second_idx = nearest[1]

        best_distance = distances[best_idx]
        second_distance = distances[second_idx]

        if best_distance < ratio_threshold * second_distance:

            matches.append([
                i,
                best_idx
            ])

    return np.asarray(
        matches,
        dtype=np.int32
    )


def get_matcher_type(extractor_name):
    """
    Return the appropriate matcher for an extractor.
    """

    if extractor_name in ["ORB", "AKAZE"]:
        return "hamming"

    if extractor_name == "XFeat":
        return "l2"

    if extractor_name in ["ALIKED", "SuperPoint"]:
        return "lightglue"

    raise ValueError(
        f"Unknown extractor: {extractor_name}"
    )