# Start with NVIDIA CUDA 12.4 base image with Python 3.10
FROM nvidia/cuda:12.4.0-cudnn8-devel-ubuntu22.04

# Set up environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TZ=UTC

# Install essential packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    vim \
    tmux \
    curl \
    wget \
    python3-dev \
    python3-pip \
    python3-setuptools \
    tzdata \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for Python
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Create a working directory
WORKDIR /app

# Install PyTorch with CUDA 12.2 support (closest available to 12.4)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install required packages
RUN pip install \
    numpy \
    pandas \
    matplotlib \
    scikit-learn \
    opencv-python \
    tqdm \
    timm \
    Pillow

# Set Python path to include Swin Transformer V2 repository
ENV PYTHONPATH="${PYTHONPATH}:/app/swin_transformer_v2_repo"

RUN mkdir -p /app/food-101 && \
    curl -L -o /app/food-101.tar.gz \
    http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz && \
    tar -xzf /app/food-101.tar.gz -C /app && \
    rm /app/food-101.tar.gz

# Copy your code into the container
COPY . /app/

# Default command to run when the container starts
CMD ["bash"]