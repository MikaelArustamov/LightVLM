# ==============================================================================
# Runtime Stage: High-performance Python environment powered by Astral UV
# ==============================================================================
FROM astral-sh/uv:0.6-python3.11-slim AS runner

# Set the working directory inside the container
WORKDIR /app

# Install system-level shared libraries required for UI rendering,
# audio decoding (Whisper), and native code compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1-mesa-glx \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Optimize UV package management behavioral settings for containerized builds
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Copy dependency metadata first to maximize Docker layer caching efficiency
COPY pyproject.toml /app/

# Target CPU-only architecture for PyTorch to optimize image size and build speed.
# Remove the custom index-url line if NVIDIA GPU/CUDA support is strictly required.
RUN uv pip install --index-url https://pytorch.org torch

# Synchronize project dependencies into an isolated virtual environment (.venv)
RUN uv sync --frozen --no-install-project

# Copy the remaining application source code into the workspace
COPY . /app

# Prepend the UV virtual environment binary directory to the system PATH
ENV PATH="/app/.venv/bin:$PATH"

# Redirect heavy downstream model assets into persistent cache volume paths
ENV HF_HOME=/root/.cache/huggingface
ENV FASTEMBED_CACHE_DIR=/root/.cache/fastembed

# Expose the network port utilized by the FastAPI uvicorn production server
EXPOSE 8000

# Execute the main application entry point within the virtual environment context
CMD ["python", "app.py"]
