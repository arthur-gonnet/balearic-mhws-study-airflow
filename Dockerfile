FROM apache/airflow:3.3.1

USER root

RUN apt-get update \
    && apt-get install -y \
        libeccodes-dev \
        libproj-dev \
        proj-data \
        proj-bin \
        libgeos-dev \
        gcc \
        g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

COPY requirements.txt /requirements.txt

RUN pip install --no-cache-dir -r /requirements.txt
