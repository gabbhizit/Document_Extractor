#!/bin/bash
set -e

echo "Starting FastAPI backend on port 8000..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!

# Wait for FastAPI to be ready before starting Streamlit
echo "Waiting for FastAPI to be ready..."
for i in $(seq 1 40); do
  if curl -s http://localhost:8000/api/v1/health > /dev/null 2>&1; then
    echo "FastAPI is ready."
    break
  fi
  echo "  attempt $i/40..."
  sleep 5
done

# Fail fast if FastAPI never started — surfaces crash in Render logs immediately
if ! curl -s http://localhost:8000/api/v1/health > /dev/null 2>&1; then
  echo "ERROR: FastAPI failed to start after 200s. Check logs above. Aborting."
  exit 1
fi

echo "Starting Streamlit UI on port ${PORT:-8501}..."
streamlit run app/ui/streamlit_app.py \
  --server.port "${PORT:-8501}" \
  --server.address "0.0.0.0" \
  --server.headless true \
  --server.enableCORS false

# If Streamlit exits, kill FastAPI too
kill $FASTAPI_PID
