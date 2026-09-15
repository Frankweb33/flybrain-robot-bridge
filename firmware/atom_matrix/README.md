# M5Stack Atom Matrix receiver

`atom_matrix.ino` receives motor commands over UDP, drives two continuous-rotation
servos with 50 Hz LEDC pulses, and sends IMU telemetry from the onboard MPU6886
every 50 ms. It compiles in CI and locally; it has not been run on a robot yet
(see [docs/HARDWARE.md](../../docs/HARDWARE.md)).

## Build

```bash
arduino-cli core install esp32:esp32@3.3.11 \
  --additional-urls https://espressif.github.io/arduino-esp32/package_esp32_index.json
arduino-cli lib install "M5Unified@0.2.22" "ArduinoJson@7.4.3"
arduino-cli compile --fqbn esp32:esp32:m5stack_atom firmware/atom_matrix
arduino-cli upload  --fqbn esp32:esp32:m5stack_atom -p <port> firmware/atom_matrix
```

On Windows, keep the Arduino data directory on a short path (for example
`ARDUINO_DIRECTORIES_DATA=C:\ard`); the ESP32 toolchain fails with
`cannot execute 'cc1plus'` when paths get too long.

## Configure

Edit the block at the top of the sketch: Wi-Fi credentials, the PC's IP address,
servo pins, neutral pulse widths, direction and `MOTORS_CALIBRATED`. Do not commit
real credentials.

## Safety behaviour

- Servos hold neutral pulses unless `MOTORS_CALIBRATED` is true **and** the robot
  is armed with the front button. LED: blue = no Wi-Fi, red = disarmed, green = armed.
- Neutral immediately on: emergency stop, 500 ms without a valid command, Wi-Fi
  loss, IMU read failure or non-finite IMU values, disarm.
- Packets are accepted only from the configured PC IP, must be under 512 bytes,
  must pass the same field checks as the Python protocol, and must carry an
  increasing sequence number. This is filtering, not authentication.
- Sequence state resets on reboot; restart PC and robot together.
- Telemetry `left_speed`/`right_speed` echo the last command, not measured speed.
