# Base stage with CUDA and system dependencies
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    python3.8 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    libgeos-dev \
    libcairo2-dev \
    poppler-utils \
    libtiff5-dev \
    libwebp-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Conda installation stage
FROM base AS conda-install
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh && \
    /bin/bash ~/miniforge.sh -b -p /opt/conda && \
    rm ~/miniforge.sh

ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda init bash && \
    conda install -y mamba -n base -c conda-forge

# Environment setup stage
FROM conda-install AS conda-env
COPY environment.yaml .
RUN mamba env create -f environment.yaml

# Python requirements stage
FROM conda-env AS python-deps
COPY requirements.txt .
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate sd && \
    pip install -r requirements.txt

# Final stage
FROM python-deps AS final
ENV PATH /opt/conda/envs/sd/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CUDA_VISIBLE_DEVICES=0

# Create working directory
WORKDIR /app

# Verify installation
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); \
    print(f'CUDA available: {torch.cuda.is_available()}'); \
    print(f'CUDA version: {torch.version.cuda}')"

# Default command
CMD ["python"]
