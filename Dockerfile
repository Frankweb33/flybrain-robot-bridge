# CPU image for reproducing the validation report. The dataset is not baked in:
# mount a directory at /app/data (1.1 GB download plus 263 MB compiled graph).
#
#   docker build -t flybrain-robot-bridge .
#   docker run --rm -v "$PWD/data:/app/data" flybrain-robot-bridge \
#       flybrain_robot.connectome.build download --data data/malecns-v1.0
#   docker run --rm -v "$PWD/data:/app/data" flybrain-robot-bridge \
#       flybrain_robot.connectome.build compile --data data/malecns-v1.0
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/docs:/app/docs" flybrain-robot-bridge \
#       flybrain_robot.validation --data data/malecns-v1.0 --device cpu
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[connectome,reference]"
COPY tests ./tests
COPY docs ./docs

VOLUME ["/app/data"]
ENTRYPOINT ["python", "-m"]
CMD ["flybrain_robot.main", "--backend", "mock", "--synthetic", "--dry-run", "--steps", "30"]
