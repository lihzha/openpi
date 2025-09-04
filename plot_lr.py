import matplotlib.pyplot as plt
import numpy as np


def lr_schedule_array(
    warmup_steps: int, peak_lr: float, decay_steps: int, decay_lr: float, total_steps: int
) -> np.ndarray:
    """
    Vectorized implementation of Optax-like warmup + cosine decay:
      - Linear warmup from init_value = peak_lr / (warmup_steps + 1) to peak_lr over `warmup_steps` steps.
      - Cosine decay from peak_lr to decay_lr over `decay_steps` steps.
      - Constant at decay_lr after warmup_steps + decay_steps.
    """
    steps = np.arange(total_steps, dtype=float)
    init_value = peak_lr / (warmup_steps + 1.0)

    # Allocate
    lr = np.empty_like(steps, dtype=float)

    # Warmup mask
    m_warm = steps < warmup_steps
    # Decay mask
    m_decay = (steps >= warmup_steps) & (steps < warmup_steps + decay_steps)
    # Post-decay mask
    m_post = steps >= warmup_steps + decay_steps

    # Warmup (linear)
    if warmup_steps > 0:
        t = steps[m_warm] / warmup_steps
        lr[m_warm] = init_value + (peak_lr - init_value) * t
    else:
        # no warmup
        lr[m_warm] = peak_lr

    # Cosine decay
    if decay_steps > 0:
        t = (steps[m_decay] - warmup_steps) / decay_steps  # in [0,1)
        lr[m_decay] = decay_lr + 0.5 * (peak_lr - decay_lr) * (1.0 + np.cos(np.pi * t))
    else:
        lr[m_decay] = decay_lr

    # After decay
    lr[m_post] = decay_lr

    return lr


def plot_lr_schedules(schedules, total_steps: int, title: str = "Cosine Decay LR Schedules with Warmup"):
    """
    schedules: list of dicts. Each dict can contain:
      - name: label for legend
      - warmup_steps, peak_lr, decay_steps, decay_lr
    """
    steps = np.arange(total_steps, dtype=float)

    # ---------- Figure 1: Full schedule ----------
    plt.figure(figsize=(9, 4.5))
    for cfg in schedules:
        name = cfg.get("name", "schedule")
        warmup_steps = int(cfg["warmup_steps"])
        peak_lr = float(cfg["peak_lr"])
        decay_steps = int(cfg["decay_steps"])
        decay_lr = float(cfg["decay_lr"])

        lr = lr_schedule_array(warmup_steps, peak_lr, decay_steps, decay_lr, total_steps)
        plt.plot(steps, lr, linewidth=2.0, label=name)

    # Highlight warmup region and mark its end
    max_warmup = max(int(cfg["warmup_steps"]) for cfg in schedules)
    plt.axvspan(0, max_warmup, alpha=0.12)
    plt.axvline(max_warmup, linestyle="--", linewidth=1.0)
    plt.text(max_warmup * 0.02, plt.gca().get_ylim()[1] * 0.92, "warmup", fontsize=10)

    plt.title(title)
    plt.xlabel("Steps")
    plt.ylabel("Learning Rate")
    plt.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.7)
    ax = plt.gca()
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.minorticks_on()
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig("lr_schedules_full.png", dpi=160, bbox_inches="tight")
    # plt.show()

    # ---------- Figure 2: Zoomed warmup (no subplots; separate figure) ----------
    plt.figure(figsize=(9, 4.5))
    for cfg in schedules:
        name = cfg.get("name", "schedule")
        warmup_steps = int(cfg["warmup_steps"])
        peak_lr = float(cfg["peak_lr"])
        decay_steps = int(cfg["decay_steps"])
        decay_lr = float(cfg["decay_lr"])

        lr = lr_schedule_array(warmup_steps, peak_lr, decay_steps, decay_lr, total_steps)
        plt.plot(steps, lr, linewidth=2.0, label=name)

    # Zoom to warmup
    right_border = max_warmup * 1.05
    plt.xlim(0, right_border)
    plt.axvspan(0, max_warmup, alpha=0.12)
    plt.axvline(max_warmup, linestyle="--", linewidth=1.0)
    plt.title(f"{title} — Warmup Zoom")
    plt.xlabel("Steps (warmup)")
    plt.ylabel("Learning Rate")
    plt.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.7)
    ax = plt.gca()
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.minorticks_on()
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig("lr_schedules_warmup_zoom.png", dpi=160, bbox_inches="tight")
    # plt.show()


# ----- Example usage (your configs) -----
warmup_steps = 1000
peak_lr = 1e-4
decay_steps = 1_000_000

total_steps = warmup_steps + decay_steps + 5000

schedules = [
    {
        "name": "previous",
        "warmup_steps": warmup_steps,
        "peak_lr": peak_lr,
        "decay_steps": decay_steps,
        "decay_lr": 1e-4,
    },
    {
        "name": "peak_lr=5e-4",
        "warmup_steps": warmup_steps,
        "peak_lr": 5e-4,
        "decay_steps": decay_steps,
        "decay_lr": 1e-4,
    },
    {
        "name": "peak_lr=1e-3",
        "warmup_steps": warmup_steps,
        "peak_lr": 1e-3,
        "decay_steps": decay_steps,
        "decay_lr": 1e-4,
    },
]

plot_lr_schedules(schedules, total_steps)
