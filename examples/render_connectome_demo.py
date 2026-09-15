"""Render assets/connectome-demo.gif from a real whole-MaleCNS run over the synthetic tour.

Every number and bar is read from the running MaleCNSBackend; nothing is animated
by hand. Needs the compiled connectome and the [demo,connectome] extras:

    python examples/render_connectome_demo.py --data data/malecns-v1.0
"""

import argparse
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

from flybrain_robot.brain.malecns import MaleCNSBackend
from flybrain_robot.motor_decoder import MotorDecoder
from flybrain_robot.telemetry import Telemetry
from flybrain_robot.vision import SCENES, TOUR_SEGMENT, VisionEncoder, scene_frame

ROOT = Path(__file__).resolve().parents[1]
BG, PANEL, TRACK = "#08121d", "#122331", "#203b4d"
TEXT, MUTED, GREEN, BLUE, RED = "#eaf5fc", "#8fa9ba", "#65e8ba", "#72cce8", "#ff8a7a"
W, H = 1080, 600
MAX_HZ = 300.0
INPUTS = [("LPLC2", "looming"), ("LC4", "looming"), ("LC10a", "moving object")]
OUTPUTS = [("DNp01", "Giant Fiber: escape"), ("DNa02", "steering"), ("DNa01", "steering")]


def font(size):
    for path in (
        "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def bar(draw, x, y, width, value, color, height=9):
    draw.rounded_rectangle((x, y, x + width, y + height), 4, fill=TRACK)
    if value > 0.003:
        draw.rounded_rectangle((x, y, x + width * min(value, 1.0), y + height), 4, fill=color)


def neuron_rows(draw, x, y, groups, rates, color):
    for name, role in groups:
        draw.text((x, y), name, font=font(15), fill=TEXT)
        draw.text((x, y + 19), role, font=font(11), fill=MUTED)
        for k, side in enumerate("LR"):
            hz = rates[f"{name}:{side}"]
            yy = y + 2 + k * 17
            draw.text((x + 128, yy - 3), side, font=font(12), fill=MUTED)
            bar(draw, x + 144, yy, 190, hz / MAX_HZ, color)
            draw.text((x + 342, yy - 3), f"{hz:3.0f} Hz", font=font(12), fill=TEXT)
        y += 46
    return y


def frame_image(scene, step, frame, sensory, rates, activity, command, realtime):
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    draw.text((32, 20), "WHOLE-CNS SPIKING SIMULATION / ACTUAL MODEL OUTPUT", font=font(15),
              fill=GREEN)  # fmt: skip
    draw.text((32, 50), "166,700 MaleCNS neurons turn camera motion into commands.",
              font=font(28), fill=TEXT)  # fmt: skip
    draw.text((32, 92), "LIF model of Shiu et al. 2024 on the MaleCNS v1.0 connectome · synthetic "
              "scenes · no hardware connected", font=font(15), fill=MUTED)  # fmt: skip

    draw.rounded_rectangle((24, 130, 386, 560), 12, fill=PANEL)
    draw.text((42, 146), "01  CAMERA", font=font(13), fill=BLUE)
    draw.text((370, 146), scene.replace("_", " "), font=font(13), fill=GREEN, anchor="ra")
    preview = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((320, 240))
    image.paste(preview, (45, 174))
    draw.line((205, 174, 205, 413), fill=GREEN, width=1)
    y = 430
    for label, left, right in (
        ("looming", sensory.looming_left, sensory.looming_right),
        ("object", sensory.object_left, sensory.object_right),
    ):
        draw.text((45, y), label, font=font(13), fill=MUTED)
        bar(draw, 110, y + 3, 110, left, GREEN)
        bar(draw, 250, y + 3, 110, right, GREEN)
        y += 26
    draw.text((110, y), "left half", font=font(11), fill=MUTED)
    draw.text((250, y), "right half", font=font(11), fill=MUTED)
    draw.text((45, 530), "encoder features → Poisson drive", font=font(11), fill=MUTED)

    draw.rounded_rectangle((402, 130, 1056, 560), 12, fill=PANEL)
    draw.text((420, 146), "02  NEURONS (smoothed firing rate)", font=font(13), fill=BLUE)
    draw.text((420, 174), "driven visual projection neurons", font=font(12), fill=MUTED)
    y = neuron_rows(draw, 420, 196, INPUTS, rates, GREEN)
    draw.text((420, y + 4), "read-out descending neurons", font=font(12), fill=MUTED)
    neuron_rows(draw, 420, y + 26, OUTPUTS, rates, BLUE)

    x0 = 846
    draw.text((x0, 174), "03  COMMAND", font=font(13), fill=BLUE)
    left, right = command
    for k, (label, value) in enumerate((("LEFT", left), ("RIGHT", right))):
        yy = 204 + k * 64
        draw.text((x0, yy), label, font=font(13), fill=MUTED)
        draw.text((x0, yy + 18), f"{value:+03d}", font=font(30), fill=RED if value < 0 else GREEN)
    escaping = activity.escape >= 0.3
    draw.rounded_rectangle((x0, 340, 1036, 380), 8, fill=RED if escaping else TRACK)
    draw.text((941, 360), "ESCAPE" if escaping else "cruise", font=font(16),
              fill=BG if escaping else MUTED, anchor="mm")  # fmt: skip
    turn = (activity.right - activity.left) / 2
    draw.text((x0, 400), "turn", font=font(13), fill=MUTED)
    draw.rounded_rectangle((x0, 424, 1036, 434), 4, fill=TRACK)
    centre = (x0 + 1036) / 2
    end = centre - turn * 95
    draw.rounded_rectangle((min(centre, end), 424, max(centre, end), 434), 4, fill=BLUE)
    draw.text((x0, 440), "left", font=font(11), fill=MUTED)
    draw.text((1036, 440), "right", font=font(11), fill=MUTED, anchor="ra")
    draw.text((x0, 480), "UDP OFF / DRY RUN", font=font(12), fill=MUTED)
    draw.text((x0, 500), f"brain {realtime:.2f}× real time", font=font(12), fill=MUTED)

    draw.text((32, 572), f"t = {step / 30:05.2f} s", font=font(12), fill=MUTED)
    draw.text((1048, 572), "Software visualization of simulation output. Not a physical robot "
              "recording.", font=font(12), fill=MUTED, anchor="ra")  # fmt: skip
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="data/malecns-v1.0")
    parser.add_argument("--every", type=int, default=3, help="Keep every Nth frame")
    args = parser.parse_args()
    import time

    brain = MaleCNSBackend(args.data)
    encoder, decoder = VisionEncoder(), MotorDecoder()
    images = []
    for step in range(TOUR_SEGMENT * len(SCENES)):
        scene = SCENES[step // TOUR_SEGMENT]
        if step % TOUR_SEGMENT == 0:  # each scene starts from a resting brain and a new view
            brain.reset()
            encoder.reset()
        frame = scene_frame(scene, step % TOUR_SEGMENT)
        sensory = encoder.encode(frame)
        started = time.perf_counter()
        activity = brain.step(sensory, Telemetry(), 1 / 30)
        realtime = (1 / 30) / (time.perf_counter() - started)
        command = decoder.update(activity, now=step / 30)
        if step % args.every == 0:
            rates = brain.named_rates()
            images.append(
                frame_image(scene, step, frame, sensory, rates, activity, command, realtime)
            )
    palette = [im.quantize(colors=96, method=Image.Quantize.MEDIANCUT) for im in images]
    out = ROOT / "assets" / "connectome-demo.gif"
    palette[0].save(out, save_all=True, append_images=palette[1:], duration=33 * args.every,
                    loop=0, optimize=True)  # fmt: skip
    for name, index in (("connectome-demo.png", 5 * TOUR_SEGMENT - 10),):
        images[index // args.every].save(ROOT / "assets" / name)
    print(f"Wrote {out} ({out.stat().st_size // 1024} KiB, {len(images)} frames)")


if __name__ == "__main__":
    main()
