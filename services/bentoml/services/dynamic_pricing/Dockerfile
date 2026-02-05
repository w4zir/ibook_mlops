FROM python:3.12-slim

WORKDIR /workspace

# Install build deps so packages like shap can compile C extensions (then remove to keep image small)
RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY requirements-dev.txt requirements-dev.txt
COPY . /workspace

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

EXPOSE 7002

ENV BENTOML_HOME=/workspace

CMD ["bentoml", "serve", "services.bentoml.services.dynamic_pricing.service:svc", "--host", "0.0.0.0", "--port", "7002"]

