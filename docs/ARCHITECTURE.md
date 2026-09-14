# Architecture

The CLI acquires frames at a nominal 30 Hz. VisionEncoder resizes to 160 × 120,
converts to grayscale and computes Farneback optical flow. Average flow magnitude
in each half becomes a normalized motion input. Mean projection onto radial
vectors estimates expansion; this is a coarse looming cue.

MockBrain moves eight state values toward bounded sensory-driven targets with
an exponential leak (120 ms time constant). Contralateral motion excites the two
motor groups; yaw feedback reduces activity on the corresponding side.
MotorDecoder scales to a configurable maximum, applies a dead zone to the target,
and exponentially smooths commands. Escape reverses both targets through the same
smoother. Emergency stop bypasses smoothing. Each side may be inverted independently.
Small residual values can occur while smoothing decays toward a zero target.

The decoder returns zero when read after 500 ms without update. It is not a
background thread and cannot stop hardware when Python hangs. An independent
receiver watchdog is therefore mandatory. The CLI additionally sends stop commands
when actual telemetry is stale. The synthetic IMU is an independent waveform,
not feedback from simulated mechanics.

## Wire format

Commands and telemetry are UTF-8 JSON objects, one object per UDP datagram:

```json
{"type":"motor_command","sequence":42,"left":35,"right":28,"emergency_stop":false}
```

```json
{"type":"telemetry","sequence":42,"gyro":[0.1,0,-0.2],"accel":[0,0.1,0.98],"left_speed":34,"right_speed":27}
```

Sequence numbers are integers in 0..2147483647. Speeds are finite numbers in
-100..100. IMU vectors contain three finite numbers; gyro is degrees/second and
acceleration is g. The firmware scaffold reports commanded speeds, not measured
encoder speeds. Unknown types, missing fields, invalid JSON, booleans as numbers,
nonfinite values and oversized packets are rejected. Extra keys are allowed.

The PC binds port 9001 and sends to robot port 9000. Telemetry must originate
from the configured robot IP and port. Each sender increments its sequence;
non-increasing packets are ignored during one process session. Restart both
endpoints together to reset sequence state. There is no handshake or session ID.
Neither an IP check nor a sequence number provides authentication.
