# Contributing

Install Python 3.11+ and, in a virtual environment:

```bash
pip install -e ".[dev,connectome,reference]"
ruff check .
pytest -q
python -m flybrain_robot.main --synthetic --dry-run --steps 20
```

The test suite needs no dataset: the connectome code is tested on small synthetic
graphs, and the LIF kernel is compared spike for spike with Brian2.

Changes that touch the brain path (`vision.py`, `brain/`, `connectome/`) must also
regenerate the whole-connectome results and commit them together with the code:

```bash
python -m flybrain_robot.connectome.build download --data data/malecns-v1.0
python -m flybrain_robot.connectome.build compile --data data/malecns-v1.0
python -m flybrain_robot.validation --data data/malecns-v1.0
python examples/render_validation_chart.py
```

Rules of thumb:

- A claim in the README needs a test, a line in docs/VALIDATION.md, or a recorded
  hardware run. Mark everything else as untested.
- Pass conditions live in `validation/__main__.py`. If you change one after seeing
  results, say so in the code comment and in the report, as was done for the Giant
  Fiber lateralization check.
- New brain interfaces need a shuffled-wiring control.
- Keep the no-download mock demo working.
- Never commit datasets, local configuration or Wi-Fi credentials.
