# Simulator container - run scenarios against MLOps stack
FROM python:3.11-slim

WORKDIR /app

# Install project and dev deps (including simulator deps)
COPY requirements.txt requirements-dev.txt pyproject.toml ./
COPY common/ common/
COPY simulator/ simulator/
COPY services/ services/
RUN pip install --no-cache-dir -e . -r requirements-dev.txt

ENV PYTHONPATH=/app
WORKDIR /app

# Default: keep container running for interactive use
CMD ["python", "-m", "simulator.cli", "list-scenarios"]
