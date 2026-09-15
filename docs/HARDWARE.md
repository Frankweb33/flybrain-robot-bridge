# Hardware

Target: a Strandbeest-style linkage walker with two continuous-rotation servos,
driven by an M5Stack Atom Matrix (ESP32-PICO-D4, onboard MPU6886 IMU, button,
LED matrix).

| status | item |
|---|---|
| ✅ done | Receiver firmware with servo output, IMU telemetry, arming and watchdog ([firmware/atom_matrix](../firmware/atom_matrix/README.md)) |
| ✅ done | Compiles with esp32 core 3.3.11, M5Unified 0.2.22, ArduinoJson 7.4.3; checked in CI |
| ✅ done | PC side: UDP protocol, stale-telemetry stop, final stop packet (unit-tested) |
| ✅ done | Run recording: camera video with burned-in frame numbers, per-frame log, metadata, plot (`--record`, unit-tested) |
| ⬜ open | Flash and run on a physical Atom Matrix |
| ⬜ open | Servo calibration, direction check, watchdog and Wi-Fi-loss test on the bench |
| ⬜ open | Recorded closed-loop run of the robot with `--backend malecns` |

Until the open rows are done and recorded here, no claim of a working physical
robot is made by this repository.

## Wiring

- Servo signals: G22 (left) and G19 (right) by default; confirm against your board.
- Power servos from a separate, adequately rated supply. Share ground with the
  Atom. Never power servos from the Atom's 5 V pin or a GPIO.
- The ESP32 outputs 3.3 V logic; most hobby servos accept it, check yours.
- Fit a physical switch that cuts servo power.

## Bring-up checklist

1. Flash with `MOTORS_CALIBRATED = false`. The LED is blue until Wi-Fi connects,
   then red (disarmed). Pressing the button toggles green (armed), but servos stay
   at neutral because the robot is not calibrated.
2. Run `python -m flybrain_robot.main --backend mock --synthetic --send --config config.yaml`
   and confirm the PC reports `IMU=fresh`. Check the axes in the status line: level
   shows `tilt(roll/pitch)=+0/+0deg`, nose down gives positive pitch, left side down
   positive roll. If not, set `imu_axes` in `config.yaml` (for example `-y,x,z`).
3. Robot lifted: adjust `LEFT_NEUTRAL_US` / `RIGHT_NEUTRAL_US` until the wheels do
   not creep, and `*_REVERSED` until positive speed drives forward.
4. Set `MOTORS_CALIBRATED = true`, flash, arm with the button, and verify: stop on
   Ctrl+C, stop 500 ms after unplugging the PC's network, stop on button disarm.
5. Only then put it on the ground, starting with a low `max_speed`.

## Recording a run

Record every run meant as evidence, including failed ones:

```bash
python -m flybrain_robot.main --backend malecns --camera 0 --config config.yaml --send --record runs
python -m flybrain_robot.recording runs/<run>   # plot.png; needs pip install -e ".[demo]"
```

`--record` writes `runs/<UTC start time>/` with the camera video as the brain saw it
(frame number, run time and motor command burned into each frame), `log.jsonl` (one
line per frame: camera features, IMU telemetry and whether it was fresh, tilt, Giant
Fiber, DNa02 and other group rates, motor activity, the command sent, the watchdog
stop flag, `brain_x_realtime`) and `meta.json` (commit, marked `+dirty` with
uncommitted changes, command line, configuration and package versions). The plot
shows Giant Fiber rates, motor commands and the gyro's yaw rate on one time axis.

Also film the robot from outside with a phone, and at the start film the laptop
screen: the status line prints `frame=` every ten frames, which lines up the phone
video with the log. A run counts when the external video, `video.avi`, `log.jsonl` and
`meta.json` are published together, and the commit in `meta.json` is not `+dirty`.

## Recorded runs

None yet.

| date | commit | backend and wiring | scene | external video | recording | result |
|---|---|---|---|---|---|---|
