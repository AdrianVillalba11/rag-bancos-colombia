#!/bin/sh
# Descarga los modelos necesarios en el servicio de Ollama (idempotente).
set -eu

echo "Esperando a Ollama en ${OLLAMA_HOST}..."
until ollama list >/dev/null 2>&1; do sleep 2; done

for modelo in "$@"; do
  echo "Descargando modelo: ${modelo}"
  ollama pull "${modelo}"
done
echo "Modelos listos."
