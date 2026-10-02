# Imagen de la aplicación: API, ingesta y analítica comparten la misma base.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/huggingface

WORKDIR /app

# PyTorch: variante CPU por defecto (liviana, funciona en cualquier máquina). Para usar la GPU
# del reranker se construye con TORCH_INDEX=https://download.pytorch.org/whl/cu126
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --index-url ${TORCH_INDEX} torch

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY scripts ./scripts

# Los datos (data/raw, data/clean) se montan como volumen; no forman parte de la imagen
CMD ["python", "scripts/ingest.py"]


# Imagen con herramientas de desarrollo para ejecutar las pruebas dentro de Docker
FROM base AS dev
RUN pip install pytest pytest-cov ruff
COPY tests ./tests
CMD ["pytest", "-q"]
