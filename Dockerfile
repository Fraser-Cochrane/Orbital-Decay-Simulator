# Build a compact Python runtime for the website and numerical service.
FROM python:3.12-slim

# Keep logs immediate and give Matplotlib a writable configuration directory.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    VELOX_WEB_HOST=0.0.0.0 \
    VELOX_DECAY_OUTPUT_DIR=/app/outputs

# Install fonts used by Matplotlib without retaining operating-system indexes.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Install the application from the exact source copied into the image.
WORKDIR /app
COPY . /app
RUN python -m pip install --no-cache-dir . \
    && useradd --create-home --uid 10001 velox \
    && mkdir -p /app/outputs /tmp/matplotlib \
    && chown -R velox:velox /app/outputs /tmp/matplotlib

# Run without root privileges and keep one worker for model-state isolation.
USER velox
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn velox_decay.web:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
