# Local validation

Validated locally with Python 3.11.15 on macOS:

- Editable installation with development dependencies succeeded.
- `ruff check .`: passed.
- `pytest -q`: 26 passed, including module imports, malformed packets,
  out-of-order telemetry, speed limits, smoothing, escape and 500 ms watchdog.
- Synthetic mock CLI: 90 frames completed without camera or network.
- All SVG assets parse as valid XML; README file links resolve.
- README GIF generated from 120 actual mock-model steps; sample frame visually reviewed.

Not validated: webcam access, user-supplied video, physical UDP link, firmware
compilation or hardware behavior. MaleCNS is deliberately unimplemented.
The GitHub clone command targets Himas1211/flybrain-robot-bridge.
Remote publication and GitHub Actions are not yet verified.
No real credentials were added; Wi-Fi values remain placeholders; the owner is Himas1211.
