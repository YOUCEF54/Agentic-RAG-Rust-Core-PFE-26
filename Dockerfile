FROM rust:1.92-slim-bookworm AS builder

WORKDIR /app

# System deps for Python + build tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    gcc g++ make pkg-config libssl-dev protobuf-compiler libprotobuf-dev \
  && rm -rf /var/lib/apt/lists/*

# Ensure protoc can find well-known types (e.g., google/protobuf/empty.proto)
ENV PROTOC_INCLUDE=/usr/include

COPY requirements.txt /app/requirements.txt
RUN python3 -m venv /opt/venv \
  && /opt/venv/bin/pip install --no-cache-dir -r /app/requirements.txt

COPY rag_rust /app/rag_rust
ENV PYO3_PYTHON=/opt/venv/bin/python
RUN /opt/venv/bin/pip install --no-cache-dir maturin \
  && /opt/venv/bin/python -m maturin build --release -m /app/rag_rust/Cargo.toml

FROM python:3.11-slim

WORKDIR /app

# Install runtime deps
COPY requirements.txt /app/requirements.txt
RUN python -m venv /opt/venv \
  && /opt/venv/bin/pip install --no-cache-dir -r /app/requirements.txt

# Install the Rust extension wheel
COPY --from=builder /app/rag_rust/target/wheels /app/wheels
RUN /opt/venv/bin/pip install --no-cache-dir /app/wheels/*.whl

COPY api.py /app/api.py
COPY README.md /app/README.md

ENV PATH="/opt/venv/bin:$PATH"

EXPOSE 8000

# Use Render's $PORT when provided, otherwise default to 8000.
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
