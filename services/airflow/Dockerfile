FROM apache/airflow:2.8.1-python3.10

# Extend the official Airflow image with the ibook-mlops project code, DAGs,
# and Python dependencies. This image is intended for future Kubernetes
# deployment; the local Docker Compose stack can continue to use the upstream
# image with volume-mounted DAGs.

USER root

WORKDIR /opt/airflow

# Copy DAG dependency manifest and full repo so Airflow can import `common.*`, etc.
COPY services/airflow/requirements-dags.txt /opt/airflow/
COPY . /opt/airflow/ibook-mlops
RUN chown -R 50000:0 /opt/airflow

# Install as airflow user (base image rejects pip run as root).
USER airflow
RUN pip install --no-cache-dir -r /opt/airflow/requirements-dags.txt

WORKDIR /opt/airflow/ibook-mlops
RUN pip install --no-cache-dir -e .

# Put DAGs and plugins in conventional Airflow locations.
RUN mkdir -p /opt/airflow/dags /opt/airflow/plugins && \
    cp -r services/airflow/dags/* /opt/airflow/dags/ && \
    cp -r services/airflow/plugins/* /opt/airflow/plugins/ || true

WORKDIR /opt/airflow

# Entrypoint and default CMD are inherited from the base image.

