# Contributing

## Development setup

This project uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev      # create .venv and install the project + dev tools
uv run pytest            # run the unit tests
uv run ruff check .      # lint
uv run ruff format .     # format
```

## Conventions

- Python 3.12+. Keep the harness importable and unit-testable in isolation;
  service dependencies (TiTiler, object storage) belong in the deployment, not
  in the unit tests.
- CI never runs a benchmark. It lints, unit-tests the harness, builds the
  runner image, and proves the stack deploys. Anything that needs real data or
  real infrastructure runs on a deployed stack, not in CI.
- Datasets and runs are configuration. Adding a dataset or a target format
  should not require touching CI or the deployment manifests.

## Building the runner image

```bash
docker build -f docker/Dockerfile.runner -t cng-benchmark-runner:dev .
docker run --rm cng-benchmark-runner:dev version
```