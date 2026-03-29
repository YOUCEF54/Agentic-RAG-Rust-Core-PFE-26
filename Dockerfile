FROM rust:1.78-slim AS builder

WORKDIR /app

# System deps for Python + build tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    gcc g++ make pkg-config libssl-dev \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt

COPY rag_rust /app/rag_rust
RUN python3 -m pip install --no-cache-dir maturin \
  && python3 -m maturin build --release -m /app/rag_rust/Cargo.toml

FROM python:3.10-slim

WORKDIR /app

# Install runtime deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Install the Rust extension wheel
COPY --from=builder /app/target/wheels /app/wheels
RUN pip install --no-cache-dir /app/wheels/*.whl

COPY api.py /app/api.py
COPY README.md /app/README.md

EXPOSE 8000

# Use Render's $PORT when provided, otherwise default to 8000.
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
