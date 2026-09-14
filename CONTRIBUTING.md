# Contributing

Install Python 3.11+ and run `pip install -e ".[dev]"` in a virtual environment.
Before proposing a change, run `ruff check .`, `pytest -q`, and
`python -m flybrain_robot.main --synthetic --dry-run --steps 20`.

Keep the no-camera, no-network demo working. New backends should implement
BrainBackend, document model assumptions and dataset licenses, and include a
small deterministic test. Clearly label untested hardware changes and scientific
limitations. Never commit local configuration, Wi-Fi credentials or dataset files.
