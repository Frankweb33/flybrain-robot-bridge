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

## Publication verification

- The public repository was cloned and all 37 published files matched the prepared copy.
- GitHub Actions passed on Python 3.11 and 3.12:
  [verified run](https://github.com/Himas1211/flybrain-robot-bridge/actions/runs/34838611127).
- The cover, architecture diagram and animated mock demo loaded on the GitHub README.

No real credentials were added; Wi-Fi values remain placeholders; the owner is Himas1211.
