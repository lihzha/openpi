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
            result.append(f"set gripper to {gv:.1f}")
        except Exception:
            # Fallback to raw string if parsing fails
            result.append(f"set gripper to {last_gripper_value_str}")

    return " and ".join(result)


@dataclasses.dataclass(frozen=True)
class DroidCoTInputs(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int
    sum_decimal: str = "1f"

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0CoT

    def __call__(self, data: dict) -> dict:
        state = np.concatenate([data["observation/cartesian_position"], data["observation/gripper_position"]])
        state = transforms.pad_to_dim(state, self.action_dim)

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        # lihan: always name base image as "exterior_image_1_left", though it should come from the camera which language action is annotated.
        assert "observation/exterior_image_1_left" in data
        base_image = _parse_image(data["observation/exterior_image_1_left"])
        # wrist_image = _parse_image(data["observation/wrist_image_left"])

        if self.model_type == _model.ModelType.PI0CoT:
            names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
            images = (base_image, np.zeros_like(base_image), np.zeros_like(base_image))
            image_masks = (np.True_, np.False_, np.False_)
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
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

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
        return inputs


@dataclasses.dataclass(frozen=True)
class DroidCoTOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 8 dims.
        actions = data.get("actions")
        if actions is not None:
            actions = np.asarray(actions[:, :7])
        return {"actions": actions, "reasoning": data.get("reasoning")}
