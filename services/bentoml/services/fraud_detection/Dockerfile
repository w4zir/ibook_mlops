FROM python:3.12-slim

WORKDIR /workspace

# Install build deps so packages like shap can compile C extensions
RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (cached unless requirements change)
COPY requirements.txt requirements.txt
COPY requirements-dev.txt requirements-dev.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt \
    && pip uninstall -y multipart || true \
    && pip install --no-cache-dir --force-reinstall python-multipart

# Copy application code (changes here don't invalidate pip cache)
COPY . /workspace
# Remove local multipart shim if present so it does not shadow python-multipart at runtime
RUN rm -rf /workspace/multipart

EXPOSE 7001

ENV BENTOML_HOME=/workspace

CMD ["bentoml", "serve", "services.bentoml.services.fraud_detection.service:svc", "--host", "0.0.0.0", "--port", "7001"]

