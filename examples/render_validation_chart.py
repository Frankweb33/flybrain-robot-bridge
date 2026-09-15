"""Render assets/connectome-vs-shuffled.png from docs/validation/results.json.

Two panels on separate axes (no shared scale): peak Giant Fiber rate and mean turn
per synthetic scene, for the real MaleCNS wiring and the shuffled-wiring controls.
Requires the [demo] extra.
"""

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
BG, PANEL, GRID = "#08121d", "#122331", "#27405a"
INK, MUTED = "#eaf5fc", "#8fa9ba"
# Validated with the dataviz palette checker against the #122331 surface (dark mode).
REAL, CONTROL = "#2aa87f", "#8a7fc4"
W, H = 1080, 520
SCENES = ["static", "pan", "pass_left", "approach_center", "approach_left", "approach_right"]


def font(size):
    for path in ("C:/Windows/Fonts/arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):  # fmt: skip
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def rounded_bar(draw, x0, x1, y_base, y_end, color):
    """Bar anchored square at the baseline with a 4 px rounded data end."""
    top, bottom = sorted((y_base, y_end))
    if bottom - top < 1:
        draw.line((x0, y_base, x1, y_base), fill=color, width=2)
        return
    draw.rounded_rectangle((x0, top, x1, bottom), 4, fill=color)
    if y_end < y_base:
        draw.rectangle((x0, max(top, bottom - 4), x1, bottom), fill=color)
    else:
        draw.rectangle((x0, top, x1, min(bottom, top + 4)), fill=color)


def panel(draw, box, title, unit, ticks, real, control, fmt):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, 12, fill=PANEL)
    draw.text((x0 + 20, y0 + 16), title, font=font(15), fill=INK)
    draw.text((x0 + 20, y0 + 38), unit, font=font(12), fill=MUTED)
    left, right, top, bottom = x0 + 64, x1 - 16, y0 + 78, y1 - 64
    lo, hi = ticks[0], ticks[-1]

    def y(value):
        return bottom - (value - lo) / (hi - lo) * (bottom - top)

    for t in ticks:
        draw.line((left, y(t), right, y(t)), fill=GRID if t else MUTED, width=1)
        draw.text((left - 8, y(t)), f"{t:g}", font=font(11), fill=MUTED, anchor="rm")
    slot = (right - left) / len(SCENES)
    width = min(26, slot / 2 - 6)
    for i, scene in enumerate(SCENES):
        cx = left + slot * (i + 0.5)
        base = y(0)
        rounded_bar(draw, cx - width - 1, cx - 1, base, y(real[scene]), REAL)
        rounded_bar(draw, cx + 1, cx + width + 1, base, y(control[scene]), CONTROL)
        value = real[scene]
        if abs(value) > (hi - lo) * 0.02:
            label_y = y(value) - 10 if value > 0 else y(value) + 10
            draw.text((cx - width / 2 - 1, label_y), fmt(value), font=font(11), fill=INK,
                      anchor="mm")  # fmt: skip
        name = scene.replace("approach_", "approach\n").replace("_", " ")
        draw.multiline_text((cx, bottom + 30), name, font=font(11), fill=MUTED, anchor="mm",
                            align="center")  # fmt: skip


def main(path=ROOT / "docs" / "validation" / "results.json"):
    results = json.loads(Path(path).read_text(encoding="utf-8"))
    loop = results["closed_loop"]

    def gf(row):
        return max(row["peak_rates_hz"]["DNp01:L"], row["peak_rates_hz"]["DNp01:R"])

    real = {r["scene"]: r for r in loop["connectome"]}
    control_rows = {s: [r for r in loop["shuffled"] if r["scene"] == s] for s in SCENES}
    seeds = sorted({r["seed"] for r in loop["shuffled"]})

    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    draw.text((32, 22), "Same neurons, same inputs, different wiring", font=font(24), fill=INK)
    draw.text((32, 56), "Whole-MaleCNS closed loop on synthetic scenes. Shuffled controls keep "
              "every neuron's in/out-degree, sign and synapse counts; only targets are permuted.",
              font=font(13), fill=MUTED)  # fmt: skip
    for k, (color, label) in enumerate(((REAL, "MaleCNS wiring"),
                                        (CONTROL, f"shuffled wiring (max of seeds "
                                                  f"{', '.join(map(str, seeds))})"))):  # fmt: skip
        lx = 32 + k * 190
        draw.rounded_rectangle((lx, 86, lx + 14, 100), 3, fill=color)
        draw.text((lx + 22, 93), label, font=font(13), fill=INK, anchor="lm")

    panel(draw, (24, 118, 532, 500), "Peak Giant Fiber (DNp01) rate", "Hz, higher side",
          [0, 100, 200, 300], {s: gf(real[s]) for s in SCENES},
          {s: max(gf(r) for r in control_rows[s]) for s in SCENES},
          lambda v: f"{v:.0f}")  # fmt: skip
    control_turn = {
        s: max((r["mean_turn"] for r in control_rows[s]), key=abs) for s in SCENES
    }
    panel(draw, (548, 118, 1056, 500), "Mean turn command", "+ = left, - = right (full turn = 1)",
          [-1, -0.5, 0, 0.5, 1], {s: real[s]["mean_turn"] for s in SCENES}, control_turn,
          lambda v: f"{v:+.2f}")  # fmt: skip
    out = ROOT / "assets" / "connectome-vs-shuffled.png"
    image.save(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
