FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip python3-venv \
    build-essential git cmake ninja-build \
    wget curl ca-certificates \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*


RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

RUN pip install --upgrade pip

RUN pip install "drake>=1.33.0"

RUN pip install \
    "numpy>=1.24" \
    "matplotlib>=3.8" \
    "scipy>=1.11" \
    "pandas>=2.1"

RUN pip install "manipulation==2025.10.20" || \
    pip install "git+https://github.com/RussTedrake/manipulation.git@2025.10.20"

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN pip install psutil
RUN pip install "transformers==4.49.0"
RUN pip install --no-build-isolation flash_attn 
RUN pip install timm einops

WORKDIR /workspace

CMD ["/bin/bash"]
