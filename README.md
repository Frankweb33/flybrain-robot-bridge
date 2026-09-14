<div align="center">

<img src="assets/hero.svg" alt="FlyBrain Robot Bridge — from visual signals to physical motion" width="100%">

# FlyBrain Robot Bridge

**Connect a Drosophila connectome simulation to a physical robot.**

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-79cce8?style=flat-square)
[![Tests](https://github.com/Himas1211/flybrain-robot-bridge/actions/workflows/test.yml/badge.svg)](https://github.com/Himas1211/flybrain-robot-bridge/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-60dfb3?style=flat-square)](LICENSE)
![Stage: proof of concept](https://img.shields.io/badge/Stage-proof_of_concept-b4bfd0?style=flat-square)

[Quick start](#quick-start) · [Demo](#mock-demo) · [Architecture](#architecture) · [Roadmap](#roadmap)

</div>

An experimental interface between a camera, a neural backend and a physical robot.
Visual motion and IMU signals feed a small demonstration model; a decoder turns
its activity into commands for the left and right sides of a robot.

**Try it in 30 seconds.** The synthetic demo runs locally without a robot, camera,
network connection or connectome download.

> [!NOTE]
> This repository does not contain a biological brain or a complete emulation of consciousness. The default backend is a small demonstrator. MaleCNS support is an experimental integration target.

## What it does

- Runs locally with synthetic frames, a webcam or a video file.
- Estimates motion in two image halves and approximate center-relative expansion.
- Updates eight hand-designed leaky activity groups using vision and IMU features.
- Limits, smooths and optionally inverts two motor commands; looming triggers reverse motion.
- Validates JSON/UDP messages and ignores malformed or out-of-order telemetry.
- Defaults to dry-run. Physical command transmission requires `--send`.

## Architecture

![Signal flow and IMU feedback](assets/architecture.svg)

`Camera → VisionEncoder → BrainBackend → MotorDecoder → UDP → robot`

`Robot IMU → UDP telemetry → BrainBackend`

See [architecture and protocol](docs/ARCHITECTURE.md). The eight groups are
`left_motion`, `right_motion`, `looming`, `balance_left`, `balance_right`,
`left_motor`, `right_motor`, and `escape`. These names describe engineering signals,
not identified biological neurons. The model has a small forward-motion bias.

## Quick start

Requires Python 3.11+.

```bash
git clone https://github.com/Himas1211/flybrain-robot-bridge.git
cd flybrain-robot-bridge
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m flybrain_robot.main --backend mock --synthetic
```

The default run lasts 300 frames (about 10 seconds) and uses no network or camera.

## Mock demo

![Synthetic camera, mock activity and decoded commands](assets/mock-demo.gif)

*Generated from the actual synthetic encoder, mock backend and motor decoder.
This is a visualization of software output, not a physical robot recording.*

```bash
python -m flybrain_robot.main --synthetic --dry-run --steps 90
python -m flybrain_robot.main --backend mock --camera 0 --dry-run
python -m flybrain_robot.main --backend mock --video /path/to/your/video.mp4 --dry-run
```

Bring your own video for the video-file mode. Terminal output includes
motion, looming, IMU freshness, motor activity, commands and watchdog state.
Synthetic mode uses deterministic frames, fixed simulation time steps and a
synthetic IMU oscillation. It is a signal-flow demo, not a robot physics simulation.
The printed synthetic Hz is the simulation rate, not a performance benchmark.

## Connecting a physical robot

```bash
cp config.example.yaml config.yaml
# Set robot_ip, pc_port, robot_port and calibrated motor limits.
python -m flybrain_robot.main --camera 0 --config config.yaml --send
```

The [firmware scaffold](firmware/atom_matrix/README.md) requires board-specific
servo and IMU hooks before it can move a robot. It is intentionally disarmed until
those hooks are implemented. No hardware test has been performed.

The PC sends zero commands while telemetry is absent or older than 500 ms.
The receiver must independently stop motors after 500 ms without a fresh command.
Ctrl+C and normal exit send a best-effort stop packet. UDP delivery is not guaranteed.

## MaleCNS integration

`MaleCNSBackend` checks the configured dataset path and then exits with
`Not implemented yet`. No graph is loaded and no simulated MaleCNS result is fabricated.
See [integration notes](docs/MALECNS_INTEGRATION.md) for the proposed extension points.

## Repository structure

```text
assets/                       Cover, architecture diagram and recorded mock output
src/flybrain_robot/            CLI, vision, protocol, decoder, configuration
src/flybrain_robot/brain/      Backend interface, working mock, MaleCNS stub
firmware/atom_matrix/          Disarmed ESP32 integration scaffold
examples/                     Synthetic demo and GIF rendering script
tests/                        Protocol, model, decoder, watchdog and CLI checks
docs/                         Architecture, hardware and integration notes
.github/workflows/test.yml    Ruff, pytest and synthetic smoke run
```

## Current limitations

This is an early proof of concept. The model is hand-designed and does not use connectome data.
Motion signals are magnitudes in each image half, not a biological directional
vision model. Looming is a center-relative optical-flow heuristic and is sensitive
to camera motion, lighting and frame rate. It is not collision avoidance.
IMU feedback currently uses gyro yaw only. There is no gait generator, physical
simulation, interactive dashboard or recorded robot demonstration. UDP has no
authentication, reliability or replay protection across process restarts.
Firmware is a scaffold and has not been compiled or tested on a board.

## Roadmap

- [x] Working mock neural backend
- [x] OpenCV optical-flow encoder (synthetic validation; webcam hardware untested)
- [x] UDP bridge implementation (physical link untested)
- [ ] Complete and test Atom Matrix firmware on hardware
- [ ] Live motor activity visualization
- [ ] Partial MaleCNS graph loading
- [ ] Map visual populations to motor populations
- [ ] GPU simulation backend
- [ ] Record a physical Strandbeest robot demo

## Scientific sources

- [Male CNS Connectome — Janelia](https://www.janelia.org/project-team/flyem/male-cns-connectome)
- [Google Research overview](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/)

## Inspiration and attribution

An independent experimental bridge implementation inspired by open MaleCNS
research and community experiments with digital Drosophila models. The connectome
and scientific models belong to their original researchers; this project claims
no authorship of their discoveries. No third-party project source code or datasets
are bundled. Related community projects for further reading:

- [nftechie/doomfly](https://github.com/nftechie/doomfly)
- [ornata/fly](https://github.com/ornata/fly)
- [eonsystemspbc/fly-brain](https://github.com/eonsystemspbc/fly-brain)
- [philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model)
- [DenisSergeevitch/desktop-fly](https://github.com/DenisSergeevitch/desktop-fly)

Check each project's license before reusing any material. Dependency licenses and
future dataset terms remain separate from this project's MIT license.

## Safety

Start with dry-run, then calibrate with the robot lifted off the ground. Use an
independent motor power cutoff and a receiver watchdog. An optical-flow heuristic
cannot protect people or equipment. Use a trusted isolated network; firmware
integration needs a hardware review before motion is enabled.

## Development

```bash
ruff check .
pytest -q
python -m flybrain_robot.main --synthetic --dry-run --steps 20
```

Rebuild the README animation with `pip install -e ".[demo]"` followed by
`python examples/render_demo.py`. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE). Scientific datasets and dependencies retain their own license terms.
