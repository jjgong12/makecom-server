FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Download LaMa model directly
RUN mkdir -p /lama_models && \
    wget -O /lama_models/big-lama.zip https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip && \
    cd /lama_models && unzip big-lama.zip && rm big-lama.zip

# Copy handler
COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
