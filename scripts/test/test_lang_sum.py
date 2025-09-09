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
    units = {k: "cm" for k in totals.keys()}
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
    return " and ".join(result)


def test():
    seq = [
        "move left 0.05 cm and move forward 0.03 cm and move up 0.21 cm",
        "move right 0.03 cm and move down 0.10 cm",
        "move right 0.02 cm and move up 0.05 cm",
    ]
    got = _sum_language_actions(seq)
    # expected = "move forward 0.04 cm and move up 0.16 cm"
    print(got)
    # print("OK" if got == expected else f"Mismatch: {got}")

if __name__ == "__main__":
    test()