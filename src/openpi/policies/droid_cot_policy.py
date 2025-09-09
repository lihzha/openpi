import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_droid_example() -> dict:
    """Creates a random input example for the Droid policy."""
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        # "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/cartesian_position": np.random.rand(6),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


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


def _sum_language_actions(actions_list, sum_decimal):
    import re

    # Determine rounding/formatting behavior from sum_decimal
    decimals = 0
    no_number = False
    if isinstance(sum_decimal, str):
        if sum_decimal == "no_number":
            no_number = True
        else:
            m = re.fullmatch(r"(\d+)f", sum_decimal)
            if m:
                try:
                    decimals = int(m.group(1))
                except Exception:
                    decimals = 0

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
    last_gripper_value_str = None
    if actions_list is None:
        return None
    for action in actions_list:
        if not action:
            continue
        parts = action.split(" and ")
        for mv in parts:
            mv = mv.strip()
            # Capture the last gripper command in the chunk
            g = re.match(r"set\s+gripper\s+to\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))", mv)
            if g:
                last_gripper_value_str = g.group(1)
                continue
            # Sum directional move commands
            m = re.match(r"move\s+(\w+)\s+([\d.]+)\s*(\w+)", mv)
            if not m:
                continue
            direction = m.group(1)
            value = float(m.group(2))
            unit = m.group(3)
            if direction in totals:
                totals[direction] += value
                units[direction] = unit

    # Helper to format a magnitude with configured decimals
    def _fmt_mag(x: float) -> str:
        return f"{x:.{decimals}f}"

    # Compute axis-wise nets
    result = []
    # X axis: right/left
    net = totals["right"] - totals["left"]
    if no_number:
        if net > 0:
            result.append("move right")
        elif net < 0:
            result.append("move left")
    else:
        mag = round(abs(net), decimals)
        if net > 0 and mag > 0:
            result.append(f"move right {_fmt_mag(mag)} {units['right']}")
        elif net < 0 and mag > 0:
            result.append(f"move left {_fmt_mag(mag)} {units['left']}")
    # Y axis: forward/backward
    net = totals["forward"] - totals["backward"]
    if no_number:
        if net > 0:
            result.append("move forward")
        elif net < 0:
            result.append("move backward")
    else:
        mag = round(abs(net), decimals)
        if net > 0 and mag > 0:
            result.append(f"move forward {_fmt_mag(mag)} {units['forward']}")
        elif net < 0 and mag > 0:
            result.append(f"move backward {_fmt_mag(mag)} {units['backward']}")
    # Z axis: up/down
    net = totals["up"] - totals["down"]
    if no_number:
        if net > 0:
            result.append("move up")
        elif net < 0:
            result.append("move down")
    else:
        mag = round(abs(net), decimals)
        if net > 0 and mag > 0:
            result.append(f"move up {_fmt_mag(mag)} {units['up']}")
        elif net < 0 and mag > 0:
            result.append(f"move down {_fmt_mag(mag)} {units['down']}")

    # Append the final gripper setting if present (rounded to 1 decimal place)
    if last_gripper_value_str is not None:
        try:
            gv = float(last_gripper_value_str)
            result.append(f"set gripper to {gv:.2f}")
        except Exception:
            # Fallback to raw string if parsing fails
            result.append(f"set gripper to {last_gripper_value_str}")

    return " and ".join(result)


@dataclasses.dataclass(frozen=True)
class DroidCoTInputs(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int
    sum_decimal: str = "1f"
    # Train-time dropout probs (set to 0.0 for val/inference)
    wrist_image_dropout_prob: float = 0.0
    text_state_dropout_prob: float = 0.0

    # Optional history support (when enabled in config)
    use_history: bool = False
    history_steps: int = 8

    # Optional global stats for state; used to produce a compact, binned summary in the prompt.
    # Expect stats for the key "state" from normalization assets.
    state_norm_stats: transforms.NormStats | None = None
    # Number of uniform bins to discretize each state dimension into when augmenting the prompt.
    num_state_bins: int = 16
    use_text_state: bool = True

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0CoT

    def __call__(self, data: dict) -> dict:
        state = np.concatenate([data["observation/cartesian_position"], data["observation/gripper_position"]])
        state = transforms.pad_to_dim(state, self.action_dim)

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        # lihan: always name base image as "exterior_image_1_left", though it should come from the camera which language action is annotated.
        base_image = None
        wrist_image = None
        base_image_mask = np.False_
        wrist_image_mask = np.False_
        if "observation/exterior_image_1_left" in data:
            base_image = _parse_image(data["observation/exterior_image_1_left"])
            base_image_mask = np.True_
        if "observation/wrist_image_left" in data:
            wrist_image = _parse_image(data["observation/wrist_image_left"])
            wrist_image_mask = np.True_

        if base_image is None:
            assert wrist_image is not None
            base_image = np.zeros_like(wrist_image)

        if wrist_image is None:
            assert base_image is not None
            wrist_image = np.zeros_like(base_image)

        # Optional dropout: randomly mask out wrist image
        if self.wrist_image_dropout_prob > 0.0:
            if np.random.rand() < float(self.wrist_image_dropout_prob):
                wrist_image_mask = np.False_

        if self.model_type == _model.ModelType.PI0CoT:
            names_list: list[str] = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
            images_list: list[np.ndarray] = [base_image, wrist_image, np.zeros_like(base_image)]
            image_masks_list: list[np.bool_] = [base_image_mask, wrist_image_mask, np.False_]

            # History images (optional)
            if self.use_history:
                # If wrist dropout is applied, drop all wrist histories too
                wrist_hist_mask = wrist_image_mask

                # Try to fetch explicit history arrays if present; otherwise, repeat the current image
                def _maybe_get_history(key: str, fallback_image: np.ndarray) -> list[np.ndarray]:
                    hist = data.get(key)
                    if hist is None:
                        # Fallback: repeat the current frame
                        return [np.array(fallback_image) for _ in range(int(self.history_steps))]
                    # hist could be a list/tuple/np.ndarray of images
                    if isinstance(hist, np.ndarray):
                        # Accept shapes [T,H,W,C] or [T,C,H,W]
                        if hist.ndim == 4 and hist.shape[-1] in (1, 3):
                            seq = [hist[i] for i in range(hist.shape[0])]
                        elif hist.ndim == 4 and hist.shape[1] in (1, 3):
                            seq = [einops.rearrange(hist[i], "c h w -> h w c") for i in range(hist.shape[0])]
                        else:
                            seq = [hist]
                    elif isinstance(hist, (list, tuple)):
                        seq = list(hist)
                    else:
                        seq = [hist]
                    # Normalize and cap length
                    seq = [_parse_image(img) for img in seq]
                    if len(seq) > int(self.history_steps):
                        seq = seq[-int(self.history_steps) :]
                    return seq

                # Wrist history
                wrist_hist_seq = _maybe_get_history("observation/wrist_image_left_history", wrist_image)
                for idx, img in enumerate(wrist_hist_seq, start=1):
                    names_list.append(f"left_wrist_history_{idx}_rgb")
                    images_list.append(img)
                    image_masks_list.append(wrist_hist_mask)

                # Base history (optional if provided)
                base_hist = data.get("observation/exterior_image_1_left_history")
                if base_hist is not None:
                    base_hist_seq = _maybe_get_history("observation/exterior_image_1_left_history", base_image)
                    for idx, img in enumerate(base_hist_seq, start=1):
                        names_list.append(f"base_history_{idx}_rgb")
                        images_list.append(img)
                        image_masks_list.append(base_image_mask)

            names = tuple(names_list)
            images = tuple(images_list)
            image_masks = tuple(image_masks_list)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = np.array(actions)

        if "prompt" in data:
            # Normalize prompt to python str
            prompt_val = data["prompt"]
            if isinstance(prompt_val, bytes):
                prompt_str = prompt_val.decode("utf-8")
            elif isinstance(prompt_val, str):
                prompt_str = prompt_val
            else:
                prompt_item = np.asarray(prompt_val).item()
                prompt_str = (
                    prompt_item.decode("utf-8") if isinstance(prompt_item, (bytes, np.bytes_)) else str(prompt_item)
                )

            if self.use_text_state:
                # Compute per-dimension uniform-bin indices based on global stats (q01/q99 preferred)
                def _bin_with_bounds(x: np.ndarray, low: np.ndarray, high: np.ndarray, bins: int) -> np.ndarray:
                    # Avoid degenerate ranges
                    rng = np.maximum(high - low, 1e-6)
                    pos = np.clip((x - low) / rng, 0.0, 1.0)
                    idx = np.floor(pos * bins).astype(int)
                    return np.clip(idx, 0, bins - 1)

                if self.state_norm_stats is not None and (
                    getattr(self.state_norm_stats, "q01", None) is not None
                    and getattr(self.state_norm_stats, "q99", None) is not None
                ):
                    lows = np.asarray(self.state_norm_stats.q01)
                    highs = np.asarray(self.state_norm_stats.q99)
                elif self.state_norm_stats is not None:
                    # Fall back to mean ± 3σ if quantiles are unavailable
                    mean = np.asarray(self.state_norm_stats.mean)
                    std = np.asarray(self.state_norm_stats.std)
                    lows = mean - 3.0 * std
                    highs = mean + 3.0 * std
                else:
                    # If no stats available, use per-dimension symmetric bounds around current value (degenerate but safe)
                    # Choose a small range to avoid division by zero
                    widths = np.maximum(np.abs(state), 1.0)
                    lows = state - widths
                    highs = state + widths

                binned = _bin_with_bounds(state, lows, highs, self.num_state_bins)
                tail = f" Current robot state: [{','.join(map(str, binned.tolist()))}]"
                # Optional dropout: randomly drop the text-state tail from the prompt
                if not (self.text_state_dropout_prob > 0.0 and np.random.rand() < float(self.text_state_dropout_prob)):
                    prompt_str = f"{prompt_str}{tail}"
            inputs["prompt"] = prompt_str

        if "language_actions" in data:
            seq = _to_str_list(data["language_actions"])
            if seq is not None:
                summed = _sum_language_actions(seq, self.sum_decimal)
                if summed is not None and len(summed) > 0:
                    inputs["language_actions"] = summed
            else:
                # Scalar/bytes case
                la = data["language_actions"]
                if isinstance(la, bytes):
                    la = la.decode("utf-8")
                else:
                    raise ValueError(f"Language actions is not a bytes string: {la}")
                inputs["language_actions"] = la

        # Optional calibration/context passthroughs for visualization
        if "camera_intrinsics" in data:
            inputs["camera_intrinsics"] = np.asarray(data["camera_intrinsics"], dtype=np.float32)
        if "camera_extrinsics" in data:
            inputs["camera_extrinsics"] = np.asarray(data["camera_extrinsics"], dtype=np.float32)
        if "observation/cartesian_position_window" in data:
            inputs["cartesian_position_window"] = np.asarray(
                data["observation/cartesian_position_window"], dtype=np.float32
            )
        return inputs


@dataclasses.dataclass(frozen=True)
class DroidCoTOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 8 dims.
        actions = data.get("actions")
        if actions is not None:
            actions = np.asarray(actions[:, :7])
        return {"actions": actions, "reasoning": data.get("reasoning")}
