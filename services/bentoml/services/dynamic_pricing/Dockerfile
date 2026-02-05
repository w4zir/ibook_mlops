FROM python:3.12-slim

WORKDIR /workspace

COPY requirements.txt requirements.txt
COPY requirements-dev.txt requirements-dev.txt
COPY . /workspace

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

EXPOSE 7002

ENV BENTOML_HOME=/workspace

CMD ["bentoml", "serve", "services.bentoml.services.dynamic_pricing.service:svc", "--host", "0.0.0.0", "--port", "7002"]

