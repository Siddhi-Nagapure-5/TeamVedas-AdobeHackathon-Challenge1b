# Use a lightweight Python base image
FROM python:3.10-slim

# Avoid interactive prompts during system install
ENV DEBIAN_FRONTEND=noninteractive

# Install system packages for OCR, PDF parsing, and image handling
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    gcc \
    pkg-config \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libfontconfig1 \
    libgl1 \
    python3-dev \
    build-essential \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create virtual environment
RUN python -m venv adobe_env

# Use absolute paths to avoid relying on shell activation
ENV VENV_PATH=/app/adobe_env
ENV PATH="$VENV_PATH/bin:$PATH"

# Install transformers and sentence-transformers inside venv
RUN $VENV_PATH/bin/pip install --no-cache-dir transformers sentence-transformers

# Copy requirements and install them
COPY requirements.txt .
RUN $VENV_PATH/bin/pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy all necessary folders (handle spaces using JSON-style COPY)
COPY ["Collection 1", "Collection 1"]
COPY ["Collection 2", "Collection 2"]
COPY ["Collection 3", "Collection 3"]
COPY models models
COPY scripts scripts

# Set Tesseract environment variables
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata
ENV TESSERACT_CMD=/usr/bin/tesseract

# Final command: run full pipeline from venv
CMD ["bash", "-c", "/app/adobe_env/bin/python scripts/download_models.py && /app/adobe_env/bin/python scripts/extract_text.py && /app/adobe_env/bin/python scripts/generate_embeddings.py && /app/adobe_env/bin/python scripts/semantic_search.py"]
