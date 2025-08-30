import logging
import os
import platform

import etils.epath as epath
import jax
import numpy as np
from rail_tpu_utils import prevent_cross_region
import cv2

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils

def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def _format_sharding(shard) -> str:
    try:
        import jax
    except Exception:
        return "<no-jax>"
    if isinstance(shard, jax.sharding.NamedSharding):
        mesh = shard.mesh
        mesh_desc = ", ".join(f"{k}={v}" for k, v in mesh.shape.items())
        return f"NamedSharding(mesh=[{mesh_desc}], spec={shard.spec})"
    if hasattr(shard, "devices"):
        # PositionalSharding and others expose .devices()
        try:
            ndev = len(shard.devices())
        except Exception:
            ndev = "?"
        return f"{type(shard).__name__}(devices={ndev})"
    return str(shard)


def _get_array(obj):
    # nnx.Param-like leaves store the array in .value
    if hasattr(obj, "value") and hasattr(obj.value, "sharding"):
        return obj.value
    return obj


def _pytree_array_leaves(tree):
    leaves = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        arr = _get_array(leaf)
        if hasattr(arr, "shape") and hasattr(arr, "sharding"):
            leaves.append((path, arr))
    return leaves


def log_mesh_and_sharding_header(mesh: jax.sharding.Mesh, *, title: str):
    mesh_desc = ", ".join(f"{k}={v}" for k, v in mesh.shape.items())
    try:
        import numpy as _np

        total = int(_np.prod(list(mesh.shape.values())))
    except Exception:
        total = "?"
    logging.info(f"{title}: mesh axes [{mesh_desc}] total_devices={total}")


def log_batch_sharding(batch):
    def fmt_path(path):
        return jax.tree_util.keystr(path)

    lines = []
    for path, arr in _pytree_array_leaves(batch):
        try:
            ex_shape = None
            # Example addressable shard shape on this host (if available)
            if hasattr(arr, "addressable_shards") and arr.addressable_shards:
                ex_shape = arr.addressable_shards[0].data.shape
            shard_str = _format_sharding(arr.sharding)
            line = f"{fmt_path(path)}: global={tuple(arr.shape)} dtype={arr.dtype} | {shard_str}"
            if ex_shape is not None:
                line += f" | local_shard={tuple(ex_shape)}"
            lines.append(line)
        except Exception as e:
            lines.append(f"{fmt_path(path)}: <error formatting sharding: {e}>")
    if lines:
        logging.info("Batch sharding summary:\n" + "\n".join(lines))


def _decode_reasoning_strings(obs: _model.Observation, tokenizer) -> list[str]:
    """Extract and decode the reasoning (language action) tokens per example."""
    if obs.tokenized_prompt is None or obs.tokenized_reasoning_mask is None:
        return []
    tokens = jax.device_get(obs.tokenized_prompt)
    rmask = jax.device_get(obs.tokenized_reasoning_mask)
    out: list[str] = []
    for i in range(tokens.shape[0]):
        sel = tokens[i][rmask[i].astype(bool)]
        try:
            text = tokenizer.decode(sel.astype(np.int32))
        except Exception:
            text = ""
        out.append(text)
    return out


def _parse_language_delta_cm(text: str) -> np.ndarray:
    """Parse summed language action text -> net [right, forward, down] in cm."""
    import re

    totals = {"left": 0.0, "right": 0.0, "forward": 0.0, "backward": 0.0, "up": 0.0, "down": 0.0}
    for part in filter(None, [p.strip() for p in text.split(" and ")]):
        m = re.match(r"move\s+(\w+)\s+([-+]?\d*\.?\d+)\s*(\w+)", part)
        if not m:
            continue
        direction = m.group(1).lower()
        try:
            value = float(m.group(2))
        except Exception:
            continue
        unit = m.group(3).lower()
        # Convert to cm
        if unit.startswith("mm"):
            value = value / 10.0
        elif unit.startswith("m") and not unit.startswith("mm"):
            value = value * 100.0
        totals[direction] = totals.get(direction, 0.0) + value
    right = totals["right"] - totals["left"]
    forward = totals["forward"] - totals["backward"]
    down = totals["down"] - totals["up"]
    return np.array([right, forward, down], dtype=np.float32)


def _invert_camera_axis_map(v_cm: np.ndarray) -> np.ndarray:
    """Invert AXIS_PERM logic from scripts/visualize_gripper.py to camera-frame delta in metres.

    visualize_gripper uses: v_cm = (AXIS_SIGN * t_cam[AXIS_PERM]) * 100
    with AXIS_PERM = [0, 2, 1] and AXIS_SIGN = [1, 1, 1].
    Here we invert to get t_cam in metres from v_cm in cm.
    """
    AXIS_PERM = np.array([0, 2, 1], dtype=np.int32)
    AXIS_SIGN = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    t_cam = np.zeros(3, dtype=np.float32)
    t_cam[AXIS_PERM] = (v_cm / 100.0) / AXIS_SIGN
    return t_cam


def _project_point(base_xyz: np.ndarray, cam_T_base: np.ndarray, intr: np.ndarray, out_hw: tuple[int, int]) -> tuple[int, int] | None:
    """Project base-frame 3D point to pixel coordinates respecting resize_with_pad letterboxing.

    intr: [fx, fy, cx, cy] measured at calibration resolution (Wc≈2*cx, Hc≈2*cy).
    We compute resized size and padding exactly like image_tools.resize_with_pad and then map (u,v).
    """
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
    Wt = int(out_hw[1])
    Ht = int(out_hw[0])
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


def _draw_dot(img_u8: np.ndarray, xy: tuple[int, int], color: tuple[int, int, int]) -> np.ndarray:
    """Draw a small filled circle (radius 4) at xy on RGB image."""
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


def _wrap_text_to_lines(text: str, max_chars_per_line: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for w in words:
        add = len(w) + (1 if cur else 0)
        if cur_len + add > max_chars_per_line:
            if cur:
                lines.append(" ".join(cur))
            cur = [w]
            cur_len = len(w)
        else:
            cur.append(w)
            cur_len += add
    if cur:
        lines.append(" ".join(cur))
    return lines[:4]  # cap lines to avoid huge overlays


def _draw_text_block(img: np.ndarray, text: str, area: tuple[int, int, int, int]) -> np.ndarray:
    """Draw wrapped text with outline over a semi-transparent dark box.

    area: (x0, y0, x1, y1) in image coordinates.
    """
    try:
        import cv2
    except Exception:
        return img
    x0, y0, x1, y1 = area
    x0 = max(0, x0); y0 = max(0, y0); x1 = min(img.shape[1], x1); y1 = min(img.shape[0], y1)
    overlay = img.copy()
    # Semi-transparent background (lighter to reduce apparent black area)
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), thickness=-1)
    alpha = 0.5
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    # Text parameters scaled by height
    block_h = max(1, y1 - y0)
    base_scale = 1.2
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.4, min(1.2, block_h / 110.0)) * base_scale
    thickness = 2
    color = (255, 255, 255)
    outline = (0, 0, 0)
    max_chars = 75
    lines = _wrap_text_to_lines(text, max_chars)
    line_h = max(10, int(10 * scale))
    y = y0 - 10
    for line in lines:
        # Outline
        cv2.putText(img, line, (x0 + 8, y), font, scale, outline, thickness + 3, cv2.LINE_AA)
        # Text
        cv2.putText(img, line, (x0 + 8, y), font, scale, color, thickness, cv2.LINE_AA)
        y += line_h + 10
    return img


def _make_legend_bar(width: int, height: int = 28) -> np.ndarray:
    try:
        import cv2
    except Exception:
        cv2 = None
    bar = np.zeros((height, width, 3), dtype=np.uint8)
    bar[:] = 32  # dark gray
    cx = 12
    items = [((0, 255, 255), "GT start"), ((0, 0, 255), "Pred end"), ((0, 255, 0), "GT end")]
    try:
        if cv2 is not None:
            for color, label in items:
                cv2.circle(bar, (cx, height // 2), 6, color, -1)
                cv2.putText(bar, label, (cx + 12, height // 2 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cx += 110
    except Exception:
        pass
    return bar


def _compose_pages(rows: list[np.ndarray], target_max_height: int = 1600) -> list[np.ndarray]:
    pages: list[np.ndarray] = []
    if not rows:
        return pages
    row_h = rows[0].shape[0]
    per_page = max(1, (target_max_height - 28) // row_h)
    for i in range(0, len(rows), per_page):
        chunk = rows[i : i + per_page]
        grid = np.concatenate(chunk, axis=0)
        legend = _make_legend_bar(grid.shape[1], height=40)
        page = np.concatenate([legend, grid], axis=0)
        pages.append(page)
    return pages


def _is_tpu_runtime() -> bool:
    try:
        return any(d.platform == "tpu" for d in jax.devices())
    except Exception:
        return False


def main(config: _config.TrainConfig):
    if ("v6" in config.name and config.fsdp_devices > 8) or ("v4" in config.name and config.fsdp_devices > 4):
        jax.distributed.initialize()
    data_dir = save_dir = config.data.rlds_data_dir
    cache_dir = os.environ.get("OPENPI_DATA_HOME", None)
    if _is_tpu_runtime() and (str(data_dir).startswith("gs://") or str(save_dir).startswith("gs://")):
        prevent_cross_region(data_dir, save_dir)
        if cache_dir is not None:
            prevent_cross_region(cache_dir, save_dir)
    # Determine effective FSDP devices for single-process GPU/CPU runs.
    process_count = getattr(jax, "process_count", lambda: 1)()
    local_devices = getattr(jax, "local_device_count", lambda: 1)()
    global_devices = getattr(jax, "device_count", lambda: local_devices)()
    init_logging()
    logging.info(f"Local devices: {local_devices}, Global devices: {global_devices}, Process count: {process_count}")
    if process_count == 1:
        # Choose the largest divisor of available devices not exceeding configured fsdp_devices
        target = min(config.fsdp_devices, max(1, local_devices))
        effective_fsdp_devices = 1
        for d in range(target, 0, -1):
            if global_devices % d == 0:
                effective_fsdp_devices = d
                break
        if effective_fsdp_devices != config.fsdp_devices:
            logging.info(
                "Using fsdp_devices=%d for single-process run (available devices=%d)",
                effective_fsdp_devices,
                global_devices,
            )
    else:
        effective_fsdp_devices = config.fsdp_devices
        assert global_devices % effective_fsdp_devices == 0

    logging.info(
        f"Summation steps: {config.data.summation_steps}, left_pad: {config.data.left_pad}, sum_decimal: {config.data.sum_decimal}, ema_decay: {config.ema_decay}"
    )
    logging.info(f"Running on: {platform.node()}")

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))


    mesh = sharding.make_mesh(effective_fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    # Human-readable mesh overview
    log_mesh_and_sharding_header(mesh, title="Device mesh")
    logging.info("Data sharding spec: %s", _format_sharding(data_sharding))

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
        seed=config.seed,
    )
    tok = data_loader._data_loader._dataset._transform.transforms[-1].tokenizer

    data_iter = iter(data_loader)
    logging.info("Before getting batch")
    batch = next(data_iter)
    logging.info("After getting batch")
    logging.info(f"Initialized data loader (shapes):\n{training_utils.array_tree_to_info(batch)}")
    # Sharding details for the first batch
    log_batch_sharding(batch)

    
    for j in range(10):
        # Visualize language-action projection per example
        obs = batch[0]
        # Decode reasoning strings
        reasoning_texts = _decode_reasoning_strings(obs, tok)
        # Prepare start/end images for the first camera view
        first_cam_key = next(iter(obs.images.keys()))
        imgs = obs.images[first_cam_key]
        # imgs shape: [B, T, H, W, C] after grouping; pick t0 and t_end
        start_imgs = np.array(imgs[:, 0])
        end_imgs = np.array(imgs[:, -1])
        B = start_imgs.shape[0]
        vis_rows = []
        for i in range(B):
            start_u8 = ((start_imgs[i] + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
            end_u8 = ((end_imgs[i] + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
            # Project true start/end gripper points if cartesian window and calibs available
            start_xyz = None
            end_xyz = None
            if getattr(obs, "cartesian_position_window", None) is not None:
                cart = np.array(obs.cartesian_position_window[i])  # [T,6]
                if cart.ndim == 2 and cart.shape[-1] >= 3:
                    start_xyz = cart[0, :3]
                    end_xyz = cart[-1, :3]
            intr = None
            extr = None
            if getattr(obs, "camera_intrinsics", None) is not None:
                # camera_intrinsics shape may be [B,T,4]; take first along T
                ci = np.array(obs.camera_intrinsics[i])
                intr = ci[0] if ci.ndim == 2 else ci
            if getattr(obs, "camera_extrinsics", None) is not None:
                # camera_extrinsics shape may be [B,T,4,4]; take first along T
                ce = np.array(obs.camera_extrinsics[i])
                extr = ce[0] if ce.ndim == 3 else ce
            H, W = start_u8.shape[:2]
            start_xy = _project_point(start_xyz, extr, intr, (H, W)) if (start_xyz is not None and intr is not None and extr is not None) else None
            end_true_xy = _project_point(end_xyz, extr, intr, (H, W)) if (end_xyz is not None and intr is not None and extr is not None) else None
            # Predicted end via language action delta in camera frame
            pred_end_xy = None
            if reasoning_texts:
                v_cm = _parse_language_delta_cm(reasoning_texts[i] if i < len(reasoning_texts) else "")
                t_cam = _invert_camera_axis_map(v_cm)
                # Approximate base-frame delta by inverting camera->base rotation
                if extr is not None and start_xyz is not None:
                    R_cb = extr[:3, :3]
                    t_base = R_cb @ t_cam  # camera -> base
                    pred_xyz = start_xyz + t_base
                    pred_end_xy = _project_point(pred_xyz, extr, intr, (H, W))
            # Build three-column row and annotate text overlay
            la_text = reasoning_texts[i] if i < len(reasoning_texts) else ""
            col1 = _draw_dot(start_u8, start_xy, (0, 255, 255))  # GT start
            if pred_end_xy is not None:
                col1 = _draw_dot(col1, pred_end_xy, (0, 0, 255))  # Pred end on start frame for side-by-side comparison
            col2 = _draw_dot(end_u8, pred_end_xy, (0, 0, 255)) if pred_end_xy is not None else end_u8  # Pred end
            col3 = _draw_dot(end_u8, end_true_xy, (0, 255, 0)) if end_true_xy is not None else end_u8  # GT end
            row = np.concatenate([col1, col2, col3], axis=1)
            # Single bottom overlay spanning the entire row
            band_h_row = max(16, row.shape[0] // 14)
            row = _draw_text_block(row, la_text, (4, row.shape[0] - band_h_row - 2, row.shape[1] - 4, row.shape[0] - 2))
            vis_rows.append(row)
        if vis_rows:
            pages = _compose_pages(vis_rows, target_max_height=1600)
            # Save individual pages and log them separately for better fidelity in wandb
            for pi, page in enumerate(pages):
                cv2.imwrite(f"grid_page_{j}_{pi:02d}.png", page)
        batch = next(data_iter)

if __name__ == "__main__":
    main(_config.cli())