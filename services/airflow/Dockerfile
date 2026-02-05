FROM apache/airflow:2.8.1

# Extend the official Airflow image with the ibook-mlops project code, DAGs,
# and Python dependencies. This image is intended for future Kubernetes
# deployment; the local Docker Compose stack can continue to use the upstream
# image with volume-mounted DAGs.

USER root

WORKDIR /opt/airflow

# Copy dependency manifests and install project dependencies plus dev tools
# required by the DAGs (pandas, numpy, mlflow, xgboost, etc.).
COPY requirements.txt requirements-dev.txt /opt/airflow/
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

# Copy the full repository so Airflow can import `common.*`, `services.*`, etc.
COPY . /opt/airflow/ibook-mlops
WORKDIR /opt/airflow/ibook-mlops

RUN pip install --no-cache-dir -e .

# Make sure the DAGs and plugins end up in the conventional Airflow locations.
RUN mkdir -p /opt/airflow/dags /opt/airflow/plugins && \
    cp -r services/airflow/dags/* /opt/airflow/dags/ && \
    cp -r services/airflow/plugins/* /opt/airflow/plugins/ || true

WORKDIR /opt/airflow

USER airflow

# Entrypoint and default CMD are inherited from the base image.

