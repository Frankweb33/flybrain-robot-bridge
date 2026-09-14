# ESP32 / Atom Matrix scaffold — incomplete

This Arduino sketch demonstrates the UDP receive/validate/watchdog structure.
It has NOT been compiled or tested. Wi-Fi and ArduinoJson (version 6 API) are
assumed. Servo and IMU library integration is intentionally left in three
hardware hooks; no claim of a working Atom Matrix driver is made.

1. Supply placeholder Wi-Fi credentials and the PC telemetry IP.
2. Implement `setupHardware`, `applyMotors` and `readIMU` for your board, using
   its vendor documentation and suitable servo/IMU libraries.
3. Calibrate neutral and direction, then set `HARDWARE_READY` to true.
4. Verify the 500 ms watchdog and emergency stop with the robot raised.

The scaffold never sends invented IMU readings. Until `readIMU` succeeds, it
emits no telemetry and the PC refuses motion. Speed telemetry reflects the last
command; it is not an encoder measurement. Sequence state resets on reboot;
restart the PC and board together. The receiver accepts only the configured PC
IP, but this is not authentication. Use a trusted local network.
