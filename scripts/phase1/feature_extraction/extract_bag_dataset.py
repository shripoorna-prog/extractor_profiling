import csv
import json
from pathlib import Path

import cv2
import numpy as np
import rosbag


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

BAG_PATH = PROJECT_ROOT / "data" / "desk2-circle.bag"
OUTPUT_DIR = PROJECT_ROOT / "data" / "desk2-circle"

RGB_DIR = OUTPUT_DIR / "rgb"
THERMAL_DIR = OUTPUT_DIR / "thermal"
DEPTH_DIR = OUTPUT_DIR / "depth"


# ============================================================
# ROS TOPICS
# ============================================================

RGB_TOPIC = "/kinect2/qhd/image_color_rect"
DEPTH_TOPIC = "/kinect2/qhd/image_depth_rect"
THERMAL_TOPIC = "/optris/thermal_image"
CAMERA_INFO_TOPIC = "/kinect2/qhd/camera_info"
POSE_TOPIC = "/vrpn_client_node/RigidBody/pose"


# ============================================================
# SETUP
# ============================================================

for directory in [RGB_DIR, THERMAL_DIR, DEPTH_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# ROS IMAGE -> NUMPY
# ============================================================

def ros_image_to_numpy(msg):
    """
    Convert a ROS sensor_msgs/Image message to a NumPy array
    without requiring cv_bridge.

    Supported encodings in this dataset:
        bgr8
        mono16
        16UC1
    """

    encoding = msg.encoding.lower()

    if encoding == "bgr8":
        dtype = np.uint8
        channels = 3

    elif encoding in ("mono16", "16uc1"):
        dtype = np.uint16
        channels = 1

    else:
        raise ValueError(
            f"Unsupported image encoding: {msg.encoding}"
        )

    bytes_per_pixel = np.dtype(dtype).itemsize * channels

    expected_row_bytes = msg.width * bytes_per_pixel

    # ROS Image messages can contain row padding.
    if msg.step < expected_row_bytes:
        raise ValueError(
            f"Invalid step={msg.step} for "
            f"{msg.width}x{msg.height} {msg.encoding}"
        )

    raw = np.frombuffer(msg.data, dtype=dtype)

    if channels == 3:
        row_elements = msg.step // np.dtype(dtype).itemsize

        image = raw.reshape(
            msg.height,
            row_elements
        )

        image = image[:, :msg.width * 3]

        image = image.reshape(
            msg.height,
            msg.width,
            3
        )

    else:
        row_elements = msg.step // np.dtype(dtype).itemsize

        image = raw.reshape(
            msg.height,
            row_elements
        )

        image = image[:, :msg.width]

    # ROS image data may be stored big-endian.
    if msg.is_bigendian and dtype == np.uint16:
        image = image.byteswap().newbyteorder()

    return image


# ============================================================
# CALIBRATION
# ============================================================

def extract_calibration(bag):

    messages = bag.read_messages(
        topics=[CAMERA_INFO_TOPIC]
    )

    try:
        _, msg, timestamp = next(messages)

    except StopIteration:
        raise RuntimeError(
            "No CameraInfo messages found."
        )

    calibration = {
        "topic": CAMERA_INFO_TOPIC,
        "timestamp": timestamp.to_sec(),

        "width": msg.width,
        "height": msg.height,

        "K": list(msg.K),
        "D": list(msg.D),
        "R": list(msg.R),
        "P": list(msg.P),

        "fx": msg.K[0],
        "fy": msg.K[4],
        "cx": msg.K[2],
        "cy": msg.K[5],

        "distortion_model": msg.distortion_model,
    }

    output_file = OUTPUT_DIR / "calibration.json"

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            calibration,
            f,
            indent=4
        )

    print("\nCamera calibration saved:")
    print(output_file)

    print("\nIntrinsic matrix K:")

    print(
        np.array(msg.K).reshape(3, 3)
    )

    print("\nDistortion coefficients:")

    print(
        np.array(msg.D)
    )


# ============================================================
# RGB EXTRACTION
# ============================================================

def extract_rgb(bag):

    output_csv = OUTPUT_DIR / "rgb_timestamps.csv"

    count = 0

    with open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "frame",
            "timestamp"
        ])

        for _, msg, timestamp in bag.read_messages(
            topics=[RGB_TOPIC]
        ):

            image = ros_image_to_numpy(msg)

            filename = f"frame_{count:05d}.png"

            output_path = RGB_DIR / filename

            success = cv2.imwrite(
                str(output_path),
                image
            )

            if not success:
                raise RuntimeError(
                    f"Failed to write {output_path}"
                )

            writer.writerow([
                filename,
                timestamp.to_sec()
            ])

            count += 1

            if count % 100 == 0:
                print(
                    f"  RGB frames: {count}"
                )

    print(
        f"\nRGB frames extracted: {count}"
    )


# ============================================================
# DEPTH EXTRACTION
# ============================================================

def extract_depth(bag):

    output_csv = OUTPUT_DIR / "depth_timestamps.csv"

    count = 0

    with open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "frame",
            "timestamp"
        ])

        for _, msg, timestamp in bag.read_messages(
            topics=[DEPTH_TOPIC]
        ):

            image = ros_image_to_numpy(msg)

            if image.dtype != np.uint16:
                image = image.astype(
                    np.uint16
                )

            filename = f"depth_{count:05d}.png"

            output_path = DEPTH_DIR / filename

            success = cv2.imwrite(
                str(output_path),
                image
            )

            if not success:
                raise RuntimeError(
                    f"Failed to write {output_path}"
                )

            writer.writerow([
                filename,
                timestamp.to_sec()
            ])

            count += 1

            if count % 100 == 0:
                print(
                    f"  Depth frames: {count}"
                )

    print(
        f"\nDepth frames extracted: {count}"
    )


# ============================================================
# THERMAL EXTRACTION
# ============================================================

def extract_thermal(bag):

    output_csv = OUTPUT_DIR / "thermal_timestamps.csv"

    count = 0

    with open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "frame",
            "timestamp"
        ])

        for _, msg, timestamp in bag.read_messages(
            topics=[THERMAL_TOPIC]
        ):

            image = ros_image_to_numpy(msg)

            if image.dtype != np.uint16:
                image = image.astype(
                    np.uint16
                )

            filename = f"thermal_{count:05d}.png"

            output_path = THERMAL_DIR / filename

            success = cv2.imwrite(
                str(output_path),
                image
            )

            if not success:
                raise RuntimeError(
                    f"Failed to write {output_path}"
                )

            writer.writerow([
                filename,
                timestamp.to_sec()
            ])

            count += 1

            if count % 100 == 0:
                print(
                    f"  Thermal frames: {count}"
                )

    print(
        f"\nThermal frames extracted: {count}"
    )


# ============================================================
# VRPN POSE EXTRACTION
# ============================================================

def extract_poses(bag):

    output_csv = OUTPUT_DIR / "poses.csv"

    count = 0

    with open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "timestamp",
            "frame_id",

            "position_x",
            "position_y",
            "position_z",

            "orientation_x",
            "orientation_y",
            "orientation_z",
            "orientation_w",
        ])

        for _, msg, timestamp in bag.read_messages(
            topics=[POSE_TOPIC]
        ):

            writer.writerow([
                timestamp.to_sec(),
                msg.header.frame_id,

                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,

                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ])

            count += 1

    print(
        f"\nVRPN poses extracted: {count}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if not BAG_PATH.exists():

        raise FileNotFoundError(
            f"Bag file not found:\n{BAG_PATH}"
        )

    print("=" * 60)
    print("DESK2-CIRCLE ROS BAG DATASET EXTRACTION")
    print("=" * 60)

    print(
        f"\nBag:\n{BAG_PATH}"
    )

    print(
        f"\nOutput:\n{OUTPUT_DIR}"
    )

    with rosbag.Bag(
        str(BAG_PATH),
        "r"
    ) as bag:

        print(
            "\n[1/5] Extracting camera calibration..."
        )

        extract_calibration(bag)

        print(
            "\n[2/5] Extracting Kinect RGB..."
        )

        extract_rgb(bag)

        print(
            "\n[3/5] Extracting depth..."
        )

        extract_depth(bag)

        print(
            "\n[4/5] Extracting thermal..."
        )

        extract_thermal(bag)

        print(
            "\n[5/5] Extracting VRPN poses..."
        )

        extract_poses(bag)

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)

    print(
        f"""
{OUTPUT_DIR.name}/
├── rgb/
├── depth/
├── thermal/
├── calibration.json
├── rgb_timestamps.csv
├── depth_timestamps.csv
├── thermal_timestamps.csv
└── poses.csv
"""
    )


if __name__ == "__main__":
    main()