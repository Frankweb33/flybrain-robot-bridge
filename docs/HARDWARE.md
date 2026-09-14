# Hardware draft

Target: a Strandbeest-style linkage with two continuous-rotation servos, driven
by an ESP32 / M5Stack Atom Matrix. This design has not been tested on hardware.

Supply servos from a suitably rated independent supply, share signal ground and
verify voltage compatibility. Do not power servo motors from an ESP32 GPIO.
Choose pins from your specific board schematic; no wiring pinout is asserted here.
Calibrate neutral pulse widths and maximum travel with the robot raised.

Implement the three hardware hooks in the firmware scaffold: initialize outputs,
apply signed speed and read IMU. Verify the neutral state on boot, malformed
packets, telemetry loss, Wi-Fi loss, host crash and 500 ms command timeout.
Use a physical motor power cutoff. Only enable motion after these checks pass.
