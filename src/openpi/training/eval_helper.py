import logging
import re

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
import openpi.models.tokenizer as _tokenizer
from openpi.training import utils as _utils

AXIS_PERM = np.array([0, 2, 1], dtype=np.int32)
AXIS_SIGN = np.array([1.0, 1.0, 1.0], dtype=np.float32)


def _decode_reasoning_strings(obs: _model.Observation, tokenizer) -> list[str]:
    """Extract and decode the reasoning (language action) tokens per example.

    Returns one decoded string per example. If reasoning fields are absent, returns [].
    """
    tokens = _utils.to_local_array(obs.tokenized_prompt)
    rmask = _utils.to_local_array(obs.tokenized_reasoning_mask)
    out: list[str] = []
    for i in range(tokens.shape[0]):
        sel = tokens[i][rmask[i].astype(bool)]
        try:
            text = tokenizer.decode(sel.astype(np.int32))
        except Exception:
            text = ""
        out.append(text)
    return out


def _parse_language_delta_cm(text: str) -> np.ndarray | None:
    """Parse summed language action text -> net [right, forward, down] in cm.

    Accepts parts joined by " and ", like: "move right 10.3cm and move up 1.2cm and move forward 1.35cm".
    Recognized directions: left/right, forward/backward, up/down. Units: mm, cm, m.
    Returns None if no valid movements found.
    """
    totals = {"left": 0.0, "right": 0.0, "forward": 0.0, "backward": 0.0, "up": 0.0, "down": 0.0}
    any_valid = False
    for part in filter(None, [p.strip() for p in text.split(" and ")]):
        m = re.match(r"move\s+(\w+)\s+([-+]?\d*\.?\d+)\s*(\w+)", part, flags=re.IGNORECASE)
        if not m:
            continue
        direction = m.group(1).lower()
        try:
            value = float(m.group(2))
        except Exception:
            continue
        unit = m.group(3).lower()
        # Normalize to cm
        if unit.startswith("mm"):
            value = value / 10.0
        elif unit == "m" or (unit.startswith("m") and not unit.startswith("mm")):
            value = value * 100.0
        totals[direction] = totals.get(direction, 0.0) + value
        any_valid = True
    if "set gripper to" in text:
        any_valid = True
    if not any_valid:
        return None
    right = totals["right"] - totals["left"]
    forward = totals["forward"] - totals["backward"]
    down = totals["down"] - totals["up"]
    return np.array([right, forward, down], dtype=np.float32)


def _invert_camera_axis_map(v_cm: np.ndarray) -> np.ndarray:
    """Invert AXIS_PERM mapping to camera-frame delta in metres.

    Mirrors scripts/visualization/train_vis_gripper.py logic.
    """

    t_cam = np.zeros(3, dtype=np.float32)
    t_cam[AXIS_PERM] = (v_cm / 100.0) / AXIS_SIGN
    return t_cam


def _project_point(
    base_xyz: np.ndarray, cam_T_base: np.ndarray, intr: np.ndarray, out_hw: tuple[int, int]
) -> tuple[int, int] | None:
    """Project base-frame 3D point to pixel coordinates respecting resize_with_pad letterboxing.

    intr: [fx, fy, cx, cy] measured at calibration resolution (Wc≈2*cx, Hc≈2*cy).
    """
    if base_xyz is None or intr is None or cam_T_base is None:
        return None
    if intr.shape[-1] != 4 or cam_T_base.shape[-2:] != (4, 4):
        return None
    fx, fy, cx, cy = intr.tolist()
    if fx == 0 or fy == 0:
        return None
    base_to_cam = np.linalg.inv(cam_T_base)
    p_base_h = np.array([base_xyz[0], base_xyz[1], base_xyz[2], 1.0], dtype=np.float32)
    p_cam = base_to_cam @ p_base_h
    z = float(p_cam[2])
    if z <= 1e-6:
        return None
    # Calibration pixel coordinates (before resize/pad)
    u = fx * (p_cam[0] / z) + cx
    v = fy * (p_cam[1] / z) + cy
    # Derive calibration resolution from principal point
    Ht, Wt = int(out_hw[0]), int(out_hw[1])
    Wc = max(1.0, 2.0 * cx)
    Hc = max(1.0, 2.0 * cy)
    # Compute resized (letterboxed) dimensions identical to resize_with_pad
    ratio = max(Wc / Wt, Hc / Ht)
    resized_w = int(Wc / ratio)
    resized_h = int(Hc / ratio)
    pad_w0 = (Wt - resized_w) // 2
    pad_h0 = (Ht - resized_h) // 2
    # Scale and offset
    x = int(np.round(u * (resized_w / Wc) + pad_w0))
    y = int(np.round(v * (resized_h / Hc) + pad_h0))
    x = int(np.clip(x, 0, Wt - 1))
    y = int(np.clip(y, 0, Ht - 1))
    return x, y


def _draw_dot(img_u8: np.ndarray, xy: tuple[int, int] | None, color: tuple[int, int, int]) -> np.ndarray:
    out = img_u8.copy()
    if xy is None:
        return out
    x, y = xy
    H, W = out.shape[:2]
    rr = 4
    y0, y1 = max(0, y - rr), min(H, y + rr + 1)
    x0, x1 = max(0, x - rr), min(W, x + rr + 1)
    for yy in range(y0, y1):
        for xx in range(x0, x1):
            if (yy - y) ** 2 + (xx - x) ** 2 <= rr * rr:
                out[yy, xx] = color
    return out


def _draw_line(
    img_u8: np.ndarray, xy0: tuple[int, int] | None, xy1: tuple[int, int] | None, color: tuple[int, int, int]
) -> np.ndarray:
    """Draw a simple line between two points. Uses OpenCV if available, otherwise a fallback.

    Args:
        img_u8: Image array uint8 [H,W,3]
        xy0: Start point (x, y) or None
        xy1: End point (x, y) or None
        color: BGR color tuple
    """
    out = img_u8.copy()
    if xy0 is None or xy1 is None:
        return out
    try:
        import cv2

        cv2.line(out, xy0, xy1, color=color, thickness=2, lineType=cv2.LINE_AA)
        return out
    except Exception:
        pass

    # Fallback: simple DDA interpolation
    x0, y0 = xy0
    x1, y1 = xy1
    H, W = out.shape[:2]
    dx = x1 - x0
    dy = y1 - y0
    steps = int(max(abs(dx), abs(dy)))
    if steps <= 0:
        if 0 <= y0 < H and 0 <= x0 < W:
            out[y0, x0] = color
        return out
    for i in range(steps + 1):
        t = i / steps
        x = int(round(x0 + t * dx))
        y = int(round(y0 + t * dy))
        if 0 <= y < H and 0 <= x < W:
            out[y, x] = color
    return out


def prepare_eval_batch(batch):
    # Process the batch to remove reasoning and update masks
    obs, actions = batch

    # Find position 108 (start of reasoning) in each batch item
    batch_size = obs.tokenized_prompt.shape[0]
    new_tokenized_prompts = []
    new_tokenized_prompt_masks = []
    new_tokenized_reasoning_masks = []

    for i in range(batch_size):
        prompt_tokens = obs.tokenized_prompt[i]

        # Find position of token 108 (start of reasoning)
        # Ensure prompt_tokens is int32 for the comparison
        prompt_tokens_int32 = prompt_tokens.astype(jnp.int32)
        pos_108 = jnp.where(prompt_tokens_int32 == 108, size=1, fill_value=-1)[0]

        # Log tensor types for debugging
        logging.info(
            f"Batch {i}: prompt_tokens dtype: {prompt_tokens.dtype}, prompt_tokens_int32 dtype: {prompt_tokens_int32.dtype}, pos_108: {pos_108}"
        )

        if pos_108[0] >= 0:
            # Remove everything after token 108 (inclusive)
            prompt_without_reasoning = prompt_tokens[: pos_108[0] + 1]
            original_length = prompt_tokens.shape[0]

            # Left pad to maintain the same length
            padding_length = original_length - prompt_without_reasoning.shape[0]
            # Ensure consistent dtype for concatenation
            padding_zeros = jnp.zeros(padding_length, dtype=prompt_tokens.dtype)
            prompt_without_reasoning = prompt_without_reasoning.astype(prompt_tokens.dtype)
            padded_prompt = jnp.concatenate([padding_zeros, prompt_without_reasoning])

            # Create new mask: True for non-zero tokens, False for padding
            new_mask = (padded_prompt != 0).astype(jnp.bool_)

            # Create reasoning mask: all False - ensure consistent dtype
            reasoning_mask = jnp.zeros(original_length, dtype=jnp.bool_)

        else:
            # No token 108 found, keep original
            padded_prompt = prompt_tokens
            # Ensure consistent dtype for the mask
            if obs.tokenized_prompt_mask is not None:
                new_mask = obs.tokenized_prompt_mask[i].astype(jnp.bool_)
            else:
                # Create a boolean mask of the same length as prompt_tokens
                new_mask = jnp.ones(prompt_tokens.shape[0], dtype=jnp.bool_)
            # Create reasoning mask with consistent dtype - use original length instead of zeros_like
            reasoning_mask = jnp.zeros(prompt_tokens.shape[0], dtype=jnp.bool_)

            logging.info(f"Batch {i}: No token 108 found, keeping original prompt")

        new_tokenized_prompts.append(padded_prompt)
        new_tokenized_prompt_masks.append(new_mask)
        new_tokenized_reasoning_masks.append(reasoning_mask)

    # Ensure all tensors have consistent types before stacking
    # All masks should be boolean, all prompts should be int32
    new_tokenized_prompts = [p.astype(jnp.int32) for p in new_tokenized_prompts]
    new_tokenized_prompt_masks = [m.astype(jnp.bool_) for m in new_tokenized_prompt_masks]
    new_tokenized_reasoning_masks = [r.astype(jnp.bool_) for r in new_tokenized_reasoning_masks]

    # Stack the processed tensors
    new_tokenized_prompt = jnp.stack(new_tokenized_prompts)
    new_tokenized_prompt_mask = jnp.stack(new_tokenized_prompt_masks)
    new_tokenized_reasoning_mask = jnp.stack(new_tokenized_reasoning_masks)

    # Create new observation with modified prompts and masks
    new_obs = _model.Observation(
        images=obs.images,
        image_masks=obs.image_masks,
        state=obs.state,
        tokenized_prompt=new_tokenized_prompt,
        tokenized_prompt_mask=new_tokenized_prompt_mask,
        tokenized_reasoning_mask=new_tokenized_reasoning_mask,
        token_ar_mask=obs.token_ar_mask,
        token_loss_mask=obs.token_loss_mask,
        example_mask=obs.example_mask,
    )

    # Create new batch with modified observation
    new_batch = (new_obs, actions)
    return new_batch


def subsample_batch(
    batch: tuple[_model.Observation, _model.Actions],
    idx: jax.Array,
) -> tuple[_model.Observation, _model.Actions]:
    obs, acts = batch

    def take0(x):
        return jnp.take(x, idx, axis=0)

    obs_k = jax.tree.map(take0, obs)
    acts_k = jax.tree.map(take0, acts)
    return obs_k, acts_k


def eval_step(
    gt_batch: tuple[_model.Observation, _model.Actions],
    id_buf: jax.Array,
    t_final: jax.Array,
    tok: _tokenizer.PaligemmaTokenizer,
    k_local: int,
):
    l2_cm_values: list[float] = []
    to_log: list[np.ndarray] = []
    # Always run reasoning sampling across all processes; restrict decoding/logging to process 0.
    # Bound to local batch size to avoid indexing errors
    if jax.process_index() == 0:
        gt_texts = _decode_reasoning_strings(gt_batch[0], tok)
        # Decode sampled reasoning tokens
        ids = _utils.to_local_array(id_buf)
        # Be robust to bounds: clamp final index
        t_host = int(np.clip(_utils.to_local_scalar(t_final), 0, ids.shape[1] - 1))
        # Prepare images now to compute consistent local count
        first_cam_key = next(iter(gt_batch[0].images))
        imgs = _utils.to_local_array(gt_batch[0].images[first_cam_key])
        imgs_u8 = ((np.asarray(imgs) + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
        # Derive safe local loop bound across all sources
        k_decode = int(min(k_local, ids.shape[0], imgs_u8.shape[0], len(gt_texts)))
        pred_texts: list[str] = []
        for bi in range(k_decode):
            seq = ids[bi, : t_host + 1, 0].astype(np.int32)
            pred_texts.append(tok.decode(seq))

        # Compute L2 metric over parsed movement vectors (in cm)
        for bi in range(k_decode):
            logging.info(f"GT text: {gt_texts[bi]}")
            logging.info(f"Pred text: {pred_texts[bi]}")
            gt_vec = _parse_language_delta_cm(gt_texts[bi])
            pred_vec = _parse_language_delta_cm(pred_texts[bi])
            if pred_vec is None:
                continue
            l2_cm = float(np.linalg.norm(gt_vec - pred_vec))
            l2_cm_values.append(l2_cm)

        if not l2_cm_values:
            return None, None

        # Prepare annotated images for a subset
        # Choose a camera to display
        # Optional 3D->2D projection inputs
        cart = gt_batch[0].cartesian_position_window
        intr_all = gt_batch[0].camera_intrinsics
        extr_all = gt_batch[0].camera_extrinsics
        cart_np = _utils.to_local_array(cart)
        intr_np = _utils.to_local_array(intr_all)
        extr_np = _utils.to_local_array(extr_all)
        if cart_np is None or intr_np is None or extr_np is None:
            logging.info("No extrinsics/intrinsics/cartesian position available. Try vis_dataset=True.")
            return None, None
        for bi in range(k_decode):
            vis = imgs_u8[bi]
            H, W = vis.shape[:2]
            if cart_np.shape[1] >= 1:
                # [T,6]
                seq = np.asarray(cart_np[bi])
                if seq.ndim == 2 and seq.shape[-1] >= 3:
                    start_xyz = seq[0, :3]
                    end_xyz = seq[-1, :3]
            ci = np.asarray(intr_np[bi])
            intr = ci[0] if ci.ndim == 2 else ci
            ce = np.asarray(extr_np[bi])
            extr = ce[0] if ce.ndim == 3 else ce
            start_xy = _project_point(start_xyz, extr, intr, (H, W))
            end_true_xy = _project_point(end_xyz, extr, intr, (H, W))
            # Predicted end via language delta
            v_cm = _parse_language_delta_cm(pred_texts[bi])
            if v_cm is None:
                continue
            t_cam = _invert_camera_axis_map(v_cm)
            R_cb = extr[:3, :3]
            t_base = R_cb @ t_cam
            pred_xyz = start_xyz + t_base
            pred_end_xy = _project_point(pred_xyz, extr, intr, (H, W))
            # Draw dots
            vis2 = vis
            vis2 = _draw_dot(vis2, start_xy, (0, 255, 255))  # GT start (yellow)
            vis2 = _draw_dot(vis2, pred_end_xy, (0, 0, 255))  # Pred end (red)
            vis2 = _draw_dot(vis2, end_true_xy, (0, 255, 0))  # GT end (green)
            vis2 = _draw_line(vis2, start_xy, end_true_xy, (0, 255, 0))
            vis2 = _draw_line(vis2, start_xy, pred_end_xy, (0, 0, 255))
            to_log.append(vis2)

    return l2_cm_values, to_log
