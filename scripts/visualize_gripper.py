import os
import re

import cv2 as cv
import imageio
import numpy as np
from scipy.spatial.transform import Rotation as R

# ───────────────────────── user-tweakable constants ────────────────────────── #
AXIS_PERM = np.array([0, 2, 1])  # X -> dx (right/left), Z -> dy (forward/backward), Y -> dz (down/up)
AXIS_SIGN = np.array([1, 1, 1])  # start with no flips

# Threshold (in cm) below which a motion component is considered negligible.
THRESH_CM = 0.3

ARROW_SCALE_PX_PER_CM = 30  # 1 cm motion = 5 pixels arrow length
FONT = cv.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.4
FONT_THICKNESS = 1
TEXT_COLOR = (255, 255, 255)  # White text
TEXT_OUTLINE_COLOR = (0, 0, 0)  # Black outline for better contrast


def _to_str_list(x):
    if isinstance(x, (list, tuple)):
        seq = x
    elif isinstance(x, np.ndarray):
        seq = x.tolist()
    else:
        return None
    out = []
    for item in seq:
        if isinstance(item, (bytes, np.bytes_)):
            out.append(item.decode("utf-8"))
        else:
            out.append(str(item))
    return out


# ───────────────────────── helper functions ──────────────────────────────── #


def load_camera_calibration(episode_id: str, calibration_dir: str = "."):
    """
    Load camera calibration data for a specific episode.

    Args:
        episode_id: The episode ID to load calibration for
        calibration_dir: Directory containing calibration JSON files

    Returns:
        tuple: (camera_intrinsics, camera_extrinsics) or (None, None) if not found
    """
    import json
    import os

    # Load the extrinsics
    cam2base_extrinsics_path = os.path.join(calibration_dir, "cam2base_extrinsics.json")
    if not os.path.exists(cam2base_extrinsics_path):
        print(f"Warning: {cam2base_extrinsics_path} not found")
        return None, None

    with open(cam2base_extrinsics_path) as f:
        cam2base_extrinsics = json.load(f)

    if episode_id not in cam2base_extrinsics:
        print(f"Warning: Episode {episode_id} not found in extrinsics")
        return None, None

    # Load the intrinsics
    intrinsics_path = os.path.join(calibration_dir, "intrinsics.json")
    if not os.path.exists(intrinsics_path):
        print(f"Warning: {intrinsics_path} not found")
        return None, None

    with open(intrinsics_path) as f:
        intrinsics = json.load(f)

    if episode_id not in intrinsics:
        print(f"Warning: Episode {episode_id} not found in intrinsics")
        return None, None

    # Find camera serial number (first numeric key)
    episode_extrinsics = cam2base_extrinsics[episode_id]
    camera_serial = None
    for k in episode_extrinsics.keys():
        if k.isdigit():
            camera_serial = k
            break

    if camera_serial is None:
        print(f"Warning: No camera serial found for episode {episode_id}")
        return None, None

    # Get extrinsics for this camera
    extracted_extrinsics = episode_extrinsics[camera_serial]
    extracted_intrinsics = intrinsics[episode_id][camera_serial]

    # Convert extrinsics to transformation matrix

    pos = extracted_extrinsics[0:3]  # translation
    rot_mat = R.from_euler("xyz", extracted_extrinsics[3:6]).as_matrix()  # rotation

    # Make homogeneous transformation matrix
    cam_to_base_extrinsics_matrix = np.eye(4)
    cam_to_base_extrinsics_matrix[:3, :3] = rot_mat
    cam_to_base_extrinsics_matrix[:3, 3] = pos

    # Extract intrinsics
    fx, cx, fy, cy = extracted_intrinsics["cameraMatrix"]
    camera_intrinsics = [fx, fy, cx, cy]

    return camera_intrinsics, cam_to_base_extrinsics_matrix


def _sum_language_actions(actions_list):
    import re

    # Accumulate per-direction totals
    totals = {
        "left": 0.0,
        "right": 0.0,
        "forward": 0.0,
        "backward": 0.0,
        "up": 0.0,
        "down": 0.0,
    }
    units = dict.fromkeys(totals.keys(), "cm")
    if actions_list is None:
        return None
    for action in actions_list:
        if not action:
            continue
        parts = action.split(" and ")
        for mv in parts:
            m = re.match(r"move\s+(\w+)\s+([\d.]+)\s*(\w+)", mv.strip())
            if not m:
                continue
            direction = m.group(1)
            value = float(m.group(2))
            unit = m.group(3)
            if direction in totals:
                totals[direction] += value
                units[direction] = unit
    # Compute axis-wise nets
    result = []
    # X axis: right/left
    net = totals["right"] - totals["left"]
    if net > 0:
        result.append(f"move right {net:.2f} {units['right']}")
    elif net < 0:
        result.append(f"move left {abs(net):.2f} {units['left']}")
    # Y axis: forward/backward
    net = totals["forward"] - totals["backward"]
    if net > 0:
        result.append(f"move forward {net:.2f} {units['forward']}")
    elif net < 0:
        result.append(f"move backward {abs(net):.2f} {units['backward']}")
    # Z axis: up/down
    net = totals["up"] - totals["down"]
    if net > 0:
        result.append(f"move up {net:.2f} {units['up']}")
    elif net < 0:
        result.append(f"move down {abs(net):.2f} {units['down']}")
    if len(result) == 0:
        return "(no motion)"
    return " and ".join(result)


def describe_movement(gripper_poses: np.ndarray, gripper_actions: np.ndarray) -> list[str]:
    """Return list of NL phrases for a 3-D translation *in metres*."""

    sentences = []

    for i in range(len(gripper_poses) - 1):
        p_i = gripper_poses[i][:3, 3]
        p_ip1 = gripper_poses[i + 1][:3, 3]
        t_cam = p_ip1 - p_i  # camera-frame delta
        v = (AXIS_SIGN * t_cam[AXIS_PERM]) * 100.0
        # dT = np.linalg.inv(gripper_poses[i]) @ gripper_poses[i + 1]
        # t = dT[:3, 3]
        # dR = dT[:3, :3]

        # v = (AXIS_SIGN * t[AXIS_PERM]) * 100.0
        dx, dy, dz = v

        # each timestep is a sentence

        parts = []
        parts.append(f"move {'right' if dx > 0 else 'left'} {abs(dx):.2f} cm")
        parts.append(f"move {'forward' if dy > 0 else 'backward'} {abs(dy):.2f} cm")
        parts.append(f"move {'down' if dz > 0 else 'up'} {abs(dz):.2f} cm")

        # droll, dpitch, dyaw = R.from_matrix(dR).as_euler("xyz")

        # droll = gripper_rotations[i, 0] - gripper_rotations[i - 1, 0]
        # dpitch = gripper_rotations[i, 1] - gripper_rotations[i - 1, 1]
        # dyaw = gripper_rotations[i, 2] - gripper_rotations[i - 1, 2]

        # parts.append(f"rotate roll {droll / np.pi * 180:.2f}°")
        # parts.append(f"rotate pitch {dpitch / np.pi * 180:.2f}°")
        # parts.append(f"rotate yaw {dyaw / np.pi * 180:.2f}°")
        parts.append(f"set gripper to {gripper_actions[i]:.2f}")
        sentence = " and ".join(parts) if parts else "(negligible motion)"
        sentences.append(sentence)
    return sentences


def reasoning_to_gripper_poses_and_actions(
    sentences: list[str],
    start_pose: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Invert `describe_movement`.

    Given a list of reasoning sentences (one per transition), reconstruct the
    sequence of gripper poses (4x4) and the per-step gripper actions.

    - Each sentence is expected to contain phrases like
      "move right 1.23 cm and move forward 4.56 cm and move down 7.89 cm and set gripper to 0.50".
    - Rotation is assumed constant (no rotation changes), matching describe_movement.
    - The first pose is `start_pose` (defaults to identity), then each sentence
      adds a translation delta to produce the next pose.

    Returns:
        (gripper_poses, gripper_actions)
        - gripper_poses: array of shape (len(sentences)+1, 4, 4)
        - gripper_actions: array of shape (len(sentences),)
    """
    if start_pose is None:
        start_pose = np.eye(4, dtype=float)
    else:
        start_pose = np.array(start_pose, dtype=float)

    num_steps = len(sentences)
    poses = np.zeros((num_steps + 1, 4, 4), dtype=float)
    poses[0] = start_pose
    actions = np.zeros((num_steps,), dtype=float)

    # Regex patterns
    move_pat = re.compile(r"move\s+(right|left|forward|backward|up|down)\s+([\-\d\.]+)\s*cm", re.IGNORECASE)
    grip_pat = re.compile(r"set\s+gripper\s+to\s+([\-\d\.]+)", re.IGNORECASE)

    for i, sentence in enumerate(sentences):
        # Parse movements; accumulate in centimeters along (dx, dy, dz)
        dx_cm = dy_cm = dz_cm = 0.0
        for m in move_pat.finditer(sentence):
            direction = m.group(1).lower()
            value_cm = float(m.group(2))
            if direction == "right":
                dx_cm += value_cm
            elif direction == "left":
                dx_cm -= value_cm
            elif direction == "forward":
                dy_cm += value_cm
            elif direction == "backward":
                dy_cm -= value_cm
            elif direction == "down":
                dz_cm += value_cm
            elif direction == "up":
                dz_cm -= value_cm

        # Parse gripper action (defaults to previous if missing, else 0.0 for first)
        grip_match = grip_pat.search(sentence)
        if grip_match:
            actions[i] = float(grip_match.group(1))
        else:
            actions[i] = actions[i - 1] if i > 0 else 0.0

        # Convert from language (dx,dy,dz) in cm back to camera-frame translation (meters)
        v_cm = np.array([dx_cm, dy_cm, dz_cm], dtype=float)
        v_m = v_cm / 100.0

        # Recall: v = (AXIS_SIGN * t_cam[AXIS_PERM]) * 100
        # Therefore: t_cam[AXIS_PERM] = v_m / AXIS_SIGN
        t_cam = np.zeros(3, dtype=float)
        # Avoid division-by-zero if AXIS_SIGN contains zeros (shouldn't, but safe)
        sign_safe = np.where(AXIS_SIGN == 0, 1.0, AXIS_SIGN.astype(float))
        t_mapped = v_m / sign_safe
        t_cam[AXIS_PERM] = t_mapped

        # Integrate translation to form next pose; keep rotation unchanged
        next_pose = poses[i].copy()
        next_pose[:3, 3] = poses[i][:3, 3] + t_cam
        poses[i + 1] = next_pose

    return poses, actions


def _draw_arrow(frame: np.ndarray, t_cam: np.ndarray, text: str, scale_px_cm: float) -> np.ndarray:
    """Overlay arrow and NL text onto *frame* and return a new image."""
    img = frame.copy()
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    # 2-D arrow tip ----------------------------------------------------------
    v = (AXIS_SIGN * t_cam[AXIS_PERM]) * 100.0  # cm (dx,dy,dz)
    dx_cm, _, dz_cm = v  # arrow only uses right/left & up/down

    end_x = int(cx + dx_cm * scale_px_cm)
    end_y = int(cy + dz_cm * scale_px_cm)  # +dz (down) increases y

    cv.arrowedLine(img, (cx, cy), (end_x, end_y), (0, 0, 255), 2, tipLength=0.25)

    # put text at top-left ---------------------------------------------------
    # cv.putText(img, text, (10, 30), FONT, 0.7, (0, 255, 0), 2, cv.LINE_AA)

    # draw a circle
    radius = 5
    cv.circle(img, (cx, cy), radius, (255, 0, 0), -1)

    # Add text below the video frame
    # Create a larger image with space below for text
    text_height = 80  # pixels for text area below video
    new_h = h + text_height
    new_img = np.zeros((new_h, w, 3), dtype=np.uint8)

    # Copy the original frame to the top
    new_img[:h, :, :] = img

    # Split text into two lines
    words = text.split()
    if len(words) <= 4:
        # If text is short, keep it on one line
        line1 = text
        line2 = ""
    else:
        # Split roughly in the middle
        mid = len(words) // 2
        line1 = " ".join(words[:mid])
        line2 = " ".join(words[mid:])

    # Add text below the video frame
    text_y1 = h + 25  # First line
    text_y2 = h + 55  # Second line

    # Draw text with black outline for better contrast
    cv.putText(
        new_img,
        line1,
        (10, text_y1),
        FONT,
        FONT_SCALE,
        TEXT_OUTLINE_COLOR,
        FONT_THICKNESS + 1,
        cv.LINE_AA,
    )
    cv.putText(
        new_img,
        line1,
        (10, text_y1),
        FONT,
        FONT_SCALE,
        TEXT_COLOR,
        FONT_THICKNESS,
        cv.LINE_AA,
    )
    if line2:
        cv.putText(
            new_img,
            line2,
            (10, text_y2),
            FONT,
            FONT_SCALE,
            TEXT_OUTLINE_COLOR,
            FONT_THICKNESS + 1,
            cv.LINE_AA,
        )
        cv.putText(
            new_img,
            line2,
            (10, text_y2),
            FONT,
            FONT_SCALE,
            TEXT_COLOR,
            FONT_THICKNESS,
            cv.LINE_AA,
        )

    return new_img


def visualize_movement(
    frames,
    T_cam_ee,
    sentences,
    out_path: str = "motion_vis.mp4",
    fps: int = 3,
) -> None:
    """Save *out_path* video with arrow + text overlays for each transition."""
    txt_path = os.path.splitext(out_path)[0] + ".txt"

    with (
        imageio.get_writer(out_path, fps=fps, codec="libx264", quality=4) as writer,
        open(txt_path, "a") as f,
    ):
        # For the first frame, just add text below without arrow
        first_frame = frames[0].copy()
        text_height = 80
        h, w = first_frame.shape[:2]
        new_h = h + text_height
        new_first_frame = np.zeros((new_h, w, 3), dtype=np.uint8)
        new_first_frame[:h, :, :] = first_frame

        # Split "Starting position" into two lines
        line1 = "Starting"
        line2 = "position"

        # Draw text with black outline for better contrast
        cv.putText(
            new_first_frame,
            line1,
            (10, h + 25),
            FONT,
            FONT_SCALE,
            TEXT_OUTLINE_COLOR,
            FONT_THICKNESS + 1,
            cv.LINE_AA,
        )
        cv.putText(
            new_first_frame,
            line1,
            (10, h + 25),
            FONT,
            FONT_SCALE,
            TEXT_COLOR,
            FONT_THICKNESS,
            cv.LINE_AA,
        )

        cv.putText(
            new_first_frame,
            line2,
            (10, h + 55),
            FONT,
            FONT_SCALE,
            TEXT_OUTLINE_COLOR,
            FONT_THICKNESS + 1,
            cv.LINE_AA,
        )
        cv.putText(
            new_first_frame,
            line2,
            (10, h + 55),
            FONT,
            FONT_SCALE,
            TEXT_COLOR,
            FONT_THICKNESS,
            cv.LINE_AA,
        )

        writer.append_data(new_first_frame[..., ::-1])  # BGR to RGB

        for i in range(len(T_cam_ee) - 1):
            dT = np.linalg.inv(T_cam_ee[i]) @ T_cam_ee[i + 1]
            t = dT[:3, 3]
            # dR = dT[:3, :3]
            sentence = sentences[i]
            frame_vis = _draw_arrow(frames[i + 1], t, sentence, ARROW_SCALE_PX_PER_CM)

            writer.append_data(frame_vis[..., ::-1])  # BGR to RGB
            f.write(f"frame {i:03d} → {i + 1:03d}: {sentence}\n")


def visualize_movement_no_arrow(
    frames,
    sentences,
    gripper_poses=None,  # 4x4 transformation matrices for gripper positions
    camera_intrinsics=None,  # camera intrinsics matrix [fx, fy, cx, cy]
    camera_extrinsics=None,  # camera to base transformation matrix
    gripper_scalars=None,  # optional per-frame scalar (e.g., gripper open/close)
    out_path: str = "motion_vis.mp4",
    fps: int = 3,
    subsample_factor: int = 1,
) -> None:
    """
    Save *out_path* video with gripper point visualization and text overlays for each transition.

    Args:
        frames: List of video frames as numpy arrays
        sentences: List of language descriptions for each movement
        gripper_poses: Optional list of 4x4 transformation matrices for gripper positions in base frame
        camera_intrinsics: Optional camera intrinsics [fx, fy, cx, cy] for gripper projection
        camera_extrinsics: Optional 4x4 camera-to-base transformation matrix
        out_path: Output video file path
        fps: Output video frame rate
        subsample_factor: Factor to subsample frames (1 = no subsampling)

    Note:
        If gripper_poses, camera_intrinsics, and camera_extrinsics are all provided,
        the function will project the gripper position to 2D pixel coordinates and
        draw a red circle with white outline at the gripper location on each frame.
        Pixel coordinates are scaled from the calibration resolution (inferred from
        principal point) to the actual frame size, following the approach in
        calibration_example.py.
    """

    # Check if we have the necessary calibration data for gripper visualization
    draw_gripper = gripper_poses is not None and camera_intrinsics is not None and camera_extrinsics is not None

    if draw_gripper:
        # Extract camera intrinsics
        fx, fy, cx, cy = camera_intrinsics

        # Create intrinsics matrix
        intrinsics_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        # Get camera to base transformation
        cam_to_base = camera_extrinsics
        base_to_cam = np.linalg.inv(cam_to_base)

        # Infer calibration resolution from principal point assuming near-center
        calib_w = max(1, int(round(2.0 * cx)))
        calib_h = max(1, int(round(2.0 * cy)))

    with (
        imageio.get_writer(out_path, fps=fps, codec="libx264", quality=4) as writer,
    ):
        # For the first frame, just add text below without arrow
        first_frame = frames[0].copy()
        draw_scalar = gripper_scalars is not None and len(gripper_scalars) > 0
        text_height = 110 if draw_scalar else 80
        h, w = first_frame.shape[:2]
        new_h = h + text_height
        new_first_frame = np.zeros((new_h, w, 3), dtype=np.uint8)
        new_first_frame[:h, :, :] = first_frame

        # Draw gripper point on first frame if calibration data available
        if draw_gripper:
            # Get gripper position in base frame and transform to camera frame
            gripper_pos_base = gripper_poses[0][:3, 3]  # 3D position
            gripper_pos_homogeneous = np.append(gripper_pos_base, 1.0)  # Make homogeneous

            # Transform from base to camera frame
            gripper_pos_cam = base_to_cam @ gripper_pos_homogeneous
            gripper_pos_cam = gripper_pos_cam[:3]  # Remove homogeneous component

            # Project to pixel coordinates
            pixel_pos = intrinsics_matrix @ gripper_pos_cam
            z = pixel_pos[2]
            if z > 1e-6:
                uv = pixel_pos[:2] / z  # Perspective division
                # Scale from calibration resolution to frame resolution
                scale_x = w / float(calib_w)
                scale_y = h / float(calib_h)
                x = int(round(uv[0] * scale_x))
                y = int(round(uv[1] * scale_y))
                # Clip coordinates to image bounds (with small margin)
                x = int(np.clip(x, 0, w))
                y = int(np.clip(y, 0, h))
                cv.circle(new_first_frame, (x, y), 6, (0, 0, 255), -1)  # Red filled circle
                # cv.circle(
                #     new_first_frame, (x, y), 10, (255, 255, 255), 2
                # )  # White outline

        # Split "Starting position" into two lines
        line1 = "Starting"
        line2 = "position"

        # Draw text with black outline for better contrast
        cv.putText(
            new_first_frame,
            line1,
            (10, h + 25),
            FONT,
            FONT_SCALE,
            TEXT_OUTLINE_COLOR,
            FONT_THICKNESS + 1,
            cv.LINE_AA,
        )
        cv.putText(
            new_first_frame,
            line1,
            (10, h + 25),
            FONT,
            FONT_SCALE,
            TEXT_COLOR,
            FONT_THICKNESS,
            cv.LINE_AA,
        )

        cv.putText(
            new_first_frame,
            line2,
            (10, h + 55),
            FONT,
            FONT_SCALE,
            TEXT_OUTLINE_COLOR,
            FONT_THICKNESS + 1,
            cv.LINE_AA,
        )
        cv.putText(
            new_first_frame,
            line2,
            (10, h + 55),
            FONT,
            FONT_SCALE,
            TEXT_COLOR,
            FONT_THICKNESS,
            cv.LINE_AA,
        )

        # Optional third line: per-frame gripper scalar value
        if draw_scalar:
            try:
                scalar_val = float(gripper_scalars[0])
                scalar_text = f"gripper: {scalar_val:.3f}"
            except Exception:
                scalar_text = "gripper: (n/a)"
            cv.putText(
                new_first_frame,
                scalar_text,
                (10, h + 85),
                FONT,
                FONT_SCALE,
                TEXT_OUTLINE_COLOR,
                FONT_THICKNESS + 1,
                cv.LINE_AA,
            )
            cv.putText(
                new_first_frame,
                scalar_text,
                (10, h + 85),
                FONT,
                FONT_SCALE,
                TEXT_COLOR,
                FONT_THICKNESS,
                cv.LINE_AA,
            )

        writer.append_data(new_first_frame[..., ::-1])  # BGR to RGB

        # Ensure that frame i+1 shows motion delta from i -> i+1
        # Iterate up to the last available sentence respecting subsampling
        for i in range(0, len(sentences) - subsample_factor + 1, subsample_factor):
            sentence = _sum_language_actions(sentences[i : i + subsample_factor])

            # Add text below the video frame
            # Create a larger image with space below for text
            text_height = 110 if draw_scalar else 80  # pixels for text area below video
            new_h = h + text_height
            new_img = np.zeros((new_h, w, 3), dtype=np.uint8)

            # Copy the original frame to the top
            new_img[:h, :, :] = frames[i + subsample_factor]

            # Draw gripper point on current frame if calibration data available
            if draw_gripper:
                frame_idx = i + subsample_factor
                if frame_idx < len(gripper_poses):
                    # Get gripper position in base frame and transform to camera frame
                    gripper_pos_base = gripper_poses[frame_idx][:3, 3]  # 3D position
                    gripper_pos_homogeneous = np.append(gripper_pos_base, 1.0)  # Make homogeneous

                    # Transform from base to camera frame
                    gripper_pos_cam = base_to_cam @ gripper_pos_homogeneous
                    gripper_pos_cam = gripper_pos_cam[:3]  # Remove homogeneous component

                    # Project to pixel coordinates
                    pixel_pos = intrinsics_matrix @ gripper_pos_cam
                    z = pixel_pos[2]
                    if z > 1e-6:
                        uv = pixel_pos[:2] / z  # Perspective division
                        # Scale from calibration resolution to frame resolution
                        scale_x = w / float(calib_w)
                        scale_y = h / float(calib_h)
                        x = int(round(uv[0] * scale_x))
                        y = int(round(uv[1] * scale_y))
                        # Clip coordinates to image bounds
                        x = int(np.clip(x, 0, w))
                        y = int(np.clip(y, 0, h))
                        cv.circle(new_img, (x, y), 6, (0, 0, 255), -1)  # Red filled circle
                        # cv.circle(
                        #     new_img, (x, y), 10, (255, 255, 255), 2
                        # )  # White outline

            # Split text into two lines
            words = sentence.split()
            if len(words) <= 4:
                # If text is short, keep it on one line
                line1 = sentence
                line2 = ""
            else:
                # Split roughly in the middle
                mid = len(words) // 2
                line1 = " ".join(words[:mid])
                line2 = " ".join(words[mid:])

            # Add text below the video frame
            text_y1 = h + 25  # First line
            text_y2 = h + 55  # Second line

            # Draw text with black outline for better contrast
            cv.putText(
                new_img,
                line1,
                (10, text_y1),
                FONT,
                FONT_SCALE,
                TEXT_OUTLINE_COLOR,
                FONT_THICKNESS + 1,
                cv.LINE_AA,
            )
            cv.putText(
                new_img,
                line1,
                (10, text_y1),
                FONT,
                FONT_SCALE,
                TEXT_COLOR,
                FONT_THICKNESS,
                cv.LINE_AA,
            )
            if line2:
                cv.putText(
                    new_img,
                    line2,
                    (10, text_y2),
                    FONT,
                    FONT_SCALE,
                    TEXT_OUTLINE_COLOR,
                    FONT_THICKNESS + 1,
                    cv.LINE_AA,
                )
                cv.putText(
                    new_img,
                    line2,
                    (10, text_y2),
                    FONT,
                    FONT_SCALE,
                    TEXT_COLOR,
                    FONT_THICKNESS,
                    cv.LINE_AA,
                )

            # Optional third line: per-frame gripper scalar value
            if draw_scalar:
                frame_idx = i + subsample_factor
                if frame_idx < len(frames) and gripper_scalars is not None and frame_idx < len(gripper_scalars):
                    try:
                        scalar_val = float(gripper_scalars[frame_idx])
                        scalar_text = f"gripper: {scalar_val:.3f}"
                    except Exception:
                        scalar_text = "gripper: (n/a)"
                else:
                    scalar_text = "gripper: (n/a)"
                cv.putText(
                    new_img,
                    scalar_text,
                    (10, h + 85),
                    FONT,
                    FONT_SCALE,
                    TEXT_OUTLINE_COLOR,
                    FONT_THICKNESS + 1,
                    cv.LINE_AA,
                )
                cv.putText(
                    new_img,
                    scalar_text,
                    (10, h + 85),
                    FONT,
                    FONT_SCALE,
                    TEXT_COLOR,
                    FONT_THICKNESS,
                    cv.LINE_AA,
                )

            writer.append_data(new_img[..., ::-1])  # BGR to RGB
