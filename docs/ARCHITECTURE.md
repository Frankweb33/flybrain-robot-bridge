# Architecture

```text
camera / video / synthetic scene
  → VisionEncoder (vision.py)          optical flow → per-hemifield features
  → BrainBackend (brain/)              MaleCNSBackend: 166,700-neuron LIF model
                                       MockBrain: eight hand-made signals, no download
  → MotorDecoder (motor_decoder.py)    limits, dead zone, smoothing, escape, watchdog
  → UDPBridge (controller.py)          validated JSON datagrams, only with --send
  → ESP32 receiver (firmware/)         servo pulses, IMU telemetry, independent watchdog
```

## Vision

`VisionEncoder` resizes frames to 160 × 120, converts to grayscale and computes
Farneback optical flow. For each image half it reports:

- `looming_left/right`: mean positive flow divergence inside moving regions, at
  least 4 px from a motion boundary. Summed over a region, divergence only measures
  flow across the region's border, so an object on a still background nets zero;
  measured inside the object, expansion is positive and translation or rotation is
  about zero.
- `object_left/right`: mean of the strongest 10 % of flow that differs from that
  half's median flow, i.e. something moving relative to the rest of the view.
- `left_motion/right_motion`: mean flow magnitude (used by the mock backend).
- `yaw_flow`: median horizontal flow, a whole-view rotation estimate.

`scene_frame` renders deterministic test scenes (`static`, `pan`, `pass_left`,
`approach_center`, `approach_left`, `approach_right`, or `tour` through all of
them). `tests/test_core.py` checks that pans and sideways passes do not register
as looming and that approaches do, on the correct side.

## Brain backends

Both implement `BrainBackend.step(sensory, telemetry, dt) → MotorActivity(left,
right, escape)`.

**MaleCNSBackend** loads the compiled connectome, drives visual projection neurons
with Poisson spikes whose rate follows the encoder features, simulates the whole
graph for `dt`, and reads smoothed rates of descending neurons: Giant Fiber for
escape, DNa01 + DNa02 left minus right for steering, plus a constant forward
`cruise`. See [MALECNS_INTEGRATION.md](MALECNS_INTEGRATION.md).

The LIF kernel (`brain/lif.py`) keeps a list of neurons away from rest and
integrates only those, delivering each spike along its CSR row after the synaptic
delay. The operation order within a 0.1 ms step (state update, threshold,
synaptic and Poisson input, reset, with inputs to refractory neurons dropped)
matches Brian2, and `tests/test_lif_reference.py` compares both spike for spike.

**MockBrain** moves eight bounded values toward sensory targets with a 120 ms leak.
Contralateral motion excites the motor signals, looming drives escape, and gyro
yaw counter-steers (turning left slows the right side). It exists so the pipeline
runs without the 1.1 GB download; its names are engineering labels, not neurons.

## Timing

Synthetic scenes, video files and the MaleCNS backend advance a fixed simulated
time per frame (1/30 s, or the video's frame period). The connectome can compute
slower than real time under strong input; the status line shows
`brain_x_realtime`, and a live robot then receives commands at a lower rate. The
mock backend with a live camera uses measured frame intervals.

## Decoder and safety

`MotorDecoder` scales activity to `max_speed`, applies a dead zone, smooths
exponentially and inverts sides if configured. Escape activity at or above
`looming_threshold` targets full reverse through the same smoother. Emergency
stop bypasses smoothing. `read()` returns zero 500 ms after the last update, but
it is not a background thread and cannot stop hardware if Python hangs.

With `--send`, the CLI commands a stop whenever robot telemetry is missing or older
than 500 ms, and sends a final stop packet on exit. The firmware independently
returns servos to neutral 500 ms after the last valid command, on Wi-Fi loss, on
IMU failure, and whenever the robot is disarmed.

## Wire format

Commands and telemetry are UTF-8 JSON objects, one per UDP datagram:

```json
{"type":"motor_command","sequence":42,"left":35,"right":28,"emergency_stop":false}
```

```json
{"type":"telemetry","sequence":42,"gyro":[0.1,0,-0.2],"accel":[0,0.1,0.98],"left_speed":34,"right_speed":27}
```

Sequence numbers are integers in 0..2147483647. Speeds are finite numbers in
-100..100. IMU vectors contain three finite numbers; gyro is degrees/second and
acceleration is g. Telemetry speeds echo the last command, not encoder
measurements. Unknown types, missing fields, invalid JSON, booleans as numbers,
non-finite values and oversized packets are rejected; extra keys are allowed.

The PC binds port 9001 and sends to robot port 9000. Telemetry must come from the
configured robot IP and port. Each sender increments its sequence; non-increasing
packets are ignored within a process session, so restart both ends together.
There is no handshake, session ID or authentication: use a trusted, isolated
network.
