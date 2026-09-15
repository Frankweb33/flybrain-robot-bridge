# Hardware

Target: a Strandbeest-style linkage walker with two continuous-rotation servos,
driven by an M5Stack Atom Matrix (ESP32-PICO-D4, onboard MPU6886 IMU, button,
LED matrix).

| status | item |
|---|---|
| ✅ done | Receiver firmware with servo output, IMU telemetry, arming and watchdog ([firmware/atom_matrix](../firmware/atom_matrix/README.md)) |
| ✅ done | Compiles with esp32 core 3.3.11, M5Unified 0.2.22, ArduinoJson 7.4.3; checked in CI |
| ✅ done | PC side: UDP protocol, stale-telemetry stop, final stop packet (unit-tested) |
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
