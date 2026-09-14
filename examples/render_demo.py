"""Render real mock-backend output for the README. Install the [demo] extra first."""

from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

from flybrain_robot.brain.mock import MockBrain
from flybrain_robot.motor_decoder import MotorDecoder
from flybrain_robot.telemetry import synthetic_telemetry
from flybrain_robot.vision import VisionEncoder, synthetic_frame

ROOT = Path(__file__).resolve().parents[1]
BG, PANEL, TEXT, MUTED, GREEN, BLUE = (
    "#08121d",
    "#122331",
    "#eaf5fc",
    "#8fa9ba",
    "#65e8ba",
    "#72cce8",
)


def font(size):
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def main():
    encoder, brain, decoder = VisionEncoder(), MockBrain(), MotorDecoder()
    images = []
    for step in range(120):
        frame = synthetic_frame(step)
        sensory = encoder.encode(frame)
        telemetry = synthetic_telemetry(step / 30)
        activity = brain.step(sensory, telemetry, 1 / 30)
        left, right = decoder.update(activity, now=step / 30)
        if step % 2:
            continue
        image = Image.new("RGB", (1080, 490), BG)
        draw = ImageDraw.Draw(image)
        draw.text((32, 22), "SYNTHETIC DEMO / ACTUAL MODEL OUTPUT", font=font(16), fill=GREEN)
        draw.text((32, 58), "See the signal become a command.", font=font(30), fill=TEXT)
        draw.text(
            (32, 106),
            "Deterministic input · Eight mock populations · No hardware connected",
            font=font(16),
            fill=MUTED,
        )
        draw.rounded_rectangle((24, 150, 465, 453), 12, fill=PANEL)
        draw.rounded_rectangle((485, 150, 1056, 453), 12, fill=PANEL)
        draw.text((42, 166), "01  CAMERA INPUT", font=font(14), fill=BLUE)
        preview = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((320, 240))
        image.paste(preview, (84, 198))
        draw.line((244, 198, 244, 438), fill=GREEN, width=1)
        draw.text((508, 166), "02  MOCK NEURAL ACTIVITY", font=font(14), fill=BLUE)
        for i, (name, value) in enumerate(zip(brain.names, brain.state)):
            y = 201 + i * 23
            draw.text((508, y), name, font=font(13), fill=MUTED)
            draw.rounded_rectangle((656, y + 2, 922, y + 12), 4, fill="#203b4d")
            if value > 0.003:
                draw.rounded_rectangle(
                    (656, y + 2, 656 + 266 * float(value), y + 12), 4, fill=GREEN if i < 5 else BLUE
                )
            draw.text((942, y - 2), f"{value:.2f}", font=font(13), fill=TEXT)
        draw.text((508, 402), f"LEFT {left:+03d}    RIGHT {right:+03d}", font=font(22), fill=GREEN)
        draw.text((810, 411), "UDP OFF / DRY RUN", font=font(12), fill=MUTED)
        draw.text(
            (32, 464),
            f"t = {step / 30:0.2f} s     looming = {sensory.looming:.2f}",
            font=font(12),
            fill=MUTED,
        )
        draw.text(
            (642, 464),
            "Software visualization. Not a physical robot recording.",
            font=font(12),
            fill=MUTED,
        )
        images.append(image)
    out = ROOT / "assets" / "mock-demo.gif"
    images[0].save(out, save_all=True, append_images=images[1:], duration=67, loop=0, optimize=True)
    images[25].save(ROOT / "assets" / "mock-demo.png")
    print(f"Wrote {out} ({out.stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    main()
